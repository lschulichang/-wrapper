import torch
import numpy as np
import open3d as o3d
import scipy
import time

from polytopes.polytopes_utils import h_rep_minimal, find_interior, compute_segment_in_polytope
from initialization.grid_utils import GSplatVoxel
from polytopes.collision_set import GSplatCollisionSet, ellipsoid_halfspace_intersection
from polytopes.decomposition import compute_polytope
from ellipsoids.intersection_utils import compute_intersection_linear_motion


class SplatPlan():
    def __init__(self, gsplat, robot_config, env_config, spline_planner, device, seed_planner=None, collision_set=None):
        # gsplat: GSplat object

        self.gsplat = gsplat
        self.device = device

        # Robot configuration
        self.radius = robot_config['radius']
        self.vmax = robot_config['vmax']
        self.amax = robot_config['amax']
        self.collision_set = collision_set or GSplatCollisionSet(self.gsplat, self.vmax, self.amax, self.radius, self.device)

        # Environment configuration (specifically voxel)
        self.lower_bound = env_config['lower_bound']
        self.upper_bound = env_config['upper_bound']
        self.resolution = env_config['resolution']

        self.gsplat_voxel = None
        if seed_planner is None:
            tnow = time.time()
            torch.cuda.synchronize()
            self.gsplat_voxel = GSplatVoxel(self.gsplat, lower_bound=self.lower_bound, upper_bound=self.upper_bound, resolution=self.resolution, radius=self.radius, device=device)
            torch.cuda.synchronize()
            print('Time to create GSplatVoxel:', time.time() - tnow)
            self.seed_planner = self.gsplat_voxel
        else:
            self.seed_planner = seed_planner

        # Spline planner
        self.spline_planner = spline_planner

        # Save the mesh
        # gsplat_voxel.create_mesh(save_path=save_path)
        # gsplat.save_mesh(scene_name + '_gsplat.obj')

        # Record times
        self.times_cbf = []
        self.times_qp = []
        self.times_prune = []

        # Keep failures analyzable in batch experiments by default.
        self.raise_on_infeasible = False

    def _build_segment_debug_summary(self, segment_debug):
        created = [row for row in segment_debug if row.get('created_polytope', False)]

        if not created:
            return {
                'segment_total': len(segment_debug),
                'polytope_build_calls': 0,
                'candidate_count_sum': 0,
                'candidate_count_mean': 0.0,
                'candidate_count_max': 0,
                'candidate_filter_time_sum_sec': 0.0,
                'candidate_filter_time_mean_sec': 0.0,
                'exact_hit_sum': 0,
                'exact_hit_mean': 0.0,
                'exact_hit_max': 0,
                'exact_eval_sum': 0,
                'exact_eval_mean': 0.0,
            }

        cand = [int(row.get('candidate_count', 0)) for row in created]
        filt_t = [float(row.get('candidate_filter_time_sec', 0.0)) for row in created]
        exact_hits = [int(row.get('exact_hit_count', 0)) for row in created]
        exact_eval = [int(row.get('exact_eval_count', 0)) for row in created]

        return {
            'segment_total': len(segment_debug),
            'polytope_build_calls': len(created),
            'candidate_count_sum': int(np.sum(cand)),
            'candidate_count_mean': float(np.mean(cand)),
            'candidate_count_max': int(np.max(cand)),
            'candidate_filter_time_sum_sec': float(np.sum(filt_t)),
            'candidate_filter_time_mean_sec': float(np.mean(filt_t)),
            'exact_hit_sum': int(np.sum(exact_hits)),
            'exact_hit_mean': float(np.mean(exact_hits)),
            'exact_hit_max': int(np.max(exact_hits)),
            'exact_eval_sum': int(np.sum(exact_eval)),
            'exact_eval_mean': float(np.mean(exact_eval)),
        }

    def generate_path(self, x0, xf):
        # Part 1: Computes the path seed using A*
        tnow = time.time()
        torch.cuda.synchronize()

        path = self.seed_planner.create_path(x0, xf)

        torch.cuda.synchronize()
        time_astar = time.time() - tnow

        return self.generate_from_seed(path, fixed_z=None, time_seed=time_astar)

    def build_corridor_from_seed(self, path):
        path_np = path.detach().cpu().numpy() if isinstance(path, torch.Tensor) else np.asarray(path)
        if path_np.ndim != 2 or path_np.shape[1] != 3 or len(path_np) < 2:
            raise ValueError('seed path must have shape (N, 3), N >= 2')

        times_collision_set = 0
        times_polytope = 0

        polytopes = []      # List of polytopes (A, b)
        segments = torch.tensor(np.stack([path_np[:-1], path_np[1:]], axis=1), device=self.device)

        segment_debug = []

        for it, segment in enumerate(segments):

            if it > 0:
                is_in_polytope = compute_segment_in_polytope(polytope[0], polytope[1], segment)
            else:
                #If we haven't created a polytope yet, so we set it to False.
                is_in_polytope = False

            should_create_polytope = (it == 0) or (it == len(segments) - 1) or (not is_in_polytope)

            seg_debug = {
                'segment_idx': int(it),
                'segment_start': segment[0].detach().cpu().tolist(),
                'segment_end': segment[1].detach().cpu().tolist(),
                'segment_length': float(torch.linalg.norm(segment[1] - segment[0]).item()),
                'created_polytope': bool(should_create_polytope),
                'contained_in_previous_polytope': bool(is_in_polytope),
                'candidate_count': 0,
                'candidate_filter_time_sec': 0.0,
                'polytope_build_time_sec': 0.0,
                'exact_eval_count': 0,
                'exact_hit_count': 0,
                'polytope_cut_halfspaces': 0,
                'polytope_total_halfspaces': 0,
            }

            # If this is the first line segment, we always create a polytope. Or subsequently, we only instantiate a polytope if the line segment
            if should_create_polytope:

                # Part 2: Computes the collision set
                tnow = time.time()
                torch.cuda.synchronize()

                output = self.collision_set.compute_set_one_step(segment)

                torch.cuda.synchronize()
                filter_time = time.time() - tnow
                times_collision_set += filter_time

                seg_debug['candidate_filter_time_sec'] = float(filter_time)
                seg_debug['candidate_count'] = int(output['primitive_ids'].numel())

                # Part 3: Computes the polytope
                tnow = time.time()
                torch.cuda.synchronize()

                polytope, poly_debug = self.get_polytope_from_outputs(output, return_debug=True)

                torch.cuda.synchronize()
                poly_time = time.time() - tnow
                times_polytope += poly_time

                seg_debug['polytope_build_time_sec'] = float(poly_time)
                seg_debug['exact_eval_count'] = int(poly_debug.get('exact_eval_count', 0))
                seg_debug['exact_hit_count'] = int(poly_debug.get('exact_hit_count', 0))
                seg_debug['polytope_cut_halfspaces'] = int(poly_debug.get('polytope_cut_halfspaces', 0))
                seg_debug['polytope_total_halfspaces'] = int(poly_debug.get('polytope_total_halfspaces', 0))

                polytopes.append(polytope)

            segment_debug.append(seg_debug)

        return polytopes, segments, {
            'times_collision_set': times_collision_set,
            'times_polytope': times_polytope,
            'segment_debug': segment_debug,
            'segment_debug_summary': self._build_segment_debug_summary(segment_debug),
        }

    def generate_from_seed(self, path, fixed_z=None, time_seed=0.0):
        polytopes, segments, corridor_data = self.build_corridor_from_seed(path)
        if not polytopes:
            raise RuntimeError('No corridor polytopes generated')

        # Step 4: Perform Bezier spline optimization
        tnow = time.time()
        torch.cuda.synchronize()

        traj, feasible = self.spline_planner.optimize_b_spline(
            polytopes, segments[0][0], segments[-1][-1], fixed_z=fixed_z
        )
        if not feasible:
            # Persist an infeasible corridor snapshot for debugging.
            self.save_polytope(polytopes, 'infeasible.obj')
            print(compute_segment_in_polytope(polytopes[-1][0], polytopes[-1][1], segments[-1]))
            if self.raise_on_infeasible or fixed_z is not None:
                raise RuntimeError('Spline optimization infeasible')
            traj = torch.stack([segments[0][0], segments[-1][-1]], dim=0)

        torch.cuda.synchronize()
        times_opt = time.time() - tnow

        # Save outgoing information
        path_np = path.detach().cpu().numpy() if isinstance(path, torch.Tensor) else np.asarray(path)
        traj_data = {
            'path': path_np.tolist(),
            'polytopes': [torch.cat([polytope[0], polytope[1].unsqueeze(-1)], dim=-1).tolist() for polytope in polytopes],
            'num_polytopes': len(polytopes),
            'traj': traj.tolist(),
            'times_astar': time_seed,
            'times_collision_set': corridor_data['times_collision_set'],
            'times_polytope': corridor_data['times_polytope'],
            'times_opt': times_opt,
            'feasible': feasible,
            'qp_status': self.spline_planner.last_solver_status,
            'failure_reason': None if feasible else 'Spline optimization infeasible',
            'fixed_z': fixed_z,
            'coeffs': None if self.spline_planner.coeffs is None else self.spline_planner.coeffs.tolist(),
            'segment_debug': corridor_data['segment_debug'],
            'segment_debug_summary': corridor_data['segment_debug_summary'],
        }

        # self.save_polytope(polytopes, 'feasible.obj')

        return traj_data

    def get_polytope_from_outputs(self, data, return_debug=False):
        # For every single line segment, we always create a polytope at the first line segment,
        # and then we subsequently check if future line segments are within the polytope before creating new ones.
        gs_ids = data['primitive_ids']

        A_bb = data['A_bb']
        b_bb = data['b_bb_shrunk']
        segment = data['path']
        delta_x = segment[1] - segment[0]

        midpoint = data['midpoint']

        debug = {
            'candidate_count': int(gs_ids.numel()),
            'exact_eval_count': 0,
            'exact_hit_count': 0,
            'polytope_cut_halfspaces': 0,
            'polytope_total_halfspaces': int(A_bb.shape[0]),
        }

        if len(gs_ids) == 0:
            polytope = (A_bb, b_bb)
            if return_debug:
                return polytope, debug
            return polytope

        elif len(gs_ids) == 1:
            rots = data['rots']
            scales = data['scales']
            means = data['means']

        else:
            rots = data['rots']
            scales = data['scales']
            means = data['means']

        # Perform the intersection test
        intersection_output = compute_intersection_linear_motion(segment[0], delta_x, rots, scales, means,
                                R_B=None, S_B=self.radius, collision_type='sphere',
                                mode='bisection', N=10)

        # With the intersections computed, we can iterate through them and keep a minimal amount of halfspaces

        A = []
        b = []

        # Loop until we have no more Gaussian intersections.
        # The idea here is very similar to that done in SFC. For them, they use the Mahalanobis distance to scale their ellipsoid until it
        # reaches the first intersection point, create a halfplane there, segment out the points on the wrong side of the halfplane, and then repeat.

        # Instead, we use K_opt as this scaling factor, calculate the halfplane, then inflate the halfplane by the radius of the robot. If the halfplane
        # does not contain these ellipsoids, then we can safely ignore them. If it does, then we keep them in the queue.

        deltas = intersection_output['deltas']
        Q_opt = intersection_output['Q_opt']
        K_opt = intersection_output['K_opt']
        mu_A = intersection_output['mu_A']

        debug['exact_eval_count'] = int(K_opt.numel())
        debug['exact_hit_count'] = int((K_opt < 1.0).sum().item())

        if K_opt.numel() == 1:
            A_cut, b_cut, _ = compute_polytope(deltas, Q_opt.unsqueeze(0), K_opt.unsqueeze(0), mu_A)
            A.append(A_cut)
            b.append(b_cut)

        else:
            while len(K_opt) > 0:

                # Find the minimum distance point
                min_K, min_idx = torch.min(K_opt, dim=0)

                # Compute the halfspace for the min distance ellipsoid
                A_cut, b_cut, _ = compute_polytope(deltas[min_idx].unsqueeze(0), Q_opt[min_idx].unsqueeze(0), min_K.unsqueeze(0), mu_A[min_idx].unsqueeze(0))

                # Find all ellipsoids that are inside the halfspace. Remember that this halfspace is inflated!
                A_cut_inflated = A_cut / torch.linalg.norm(A_cut, dim=-1, keepdim=True)
                b_cut_inflated = b_cut / torch.linalg.norm(A_cut, dim=-1)
                b_cut_inflated = b_cut_inflated + self.radius

                keep_gaussians = ellipsoid_halfspace_intersection(means, rots, scales, A_cut_inflated.unsqueeze(0), b_cut_inflated)

                # Keep track of the mask with segmented out ellipsoids and the min distance point!
                keep_gaussians[min_idx] = False

                # TODO: I think means is the same as mu_A, so we can remove one of them.
                deltas = deltas[keep_gaussians]
                Q_opt = Q_opt[keep_gaussians]
                K_opt = K_opt[keep_gaussians]
                mu_A = mu_A[keep_gaussians]
                rots = rots[keep_gaussians]
                scales = scales[keep_gaussians]
                means = means[keep_gaussians]

                # Append the halfspace to the list
                A.append(A_cut)
                b.append(b_cut)

        A = torch.stack(A, dim=0).reshape(-1, deltas.shape[-1])
        b = torch.stack(b, dim=0).reshape(-1, )
        debug['polytope_cut_halfspaces'] = int(A.shape[0])

        # The full polytope is a concatenation of the intersection polytope and the bounding box polytope
        A = torch.cat([A, A_bb], dim=0)
        b = torch.cat([b, b_bb], dim=0)

        norm_A = torch.linalg.norm(A, dim=-1, keepdims=True)
        A = A / norm_A
        b = b / norm_A.squeeze()

        debug['polytope_total_halfspaces'] = int(A.shape[0])

        polytope = (A, b)
        if return_debug:
            return polytope, debug
        return polytope

    def save_polytope(self, polytopes, save_path):
        # Initialize mesh object
        mesh = o3d.geometry.TriangleMesh()

        for (A, b) in polytopes:
            # Transfer all tensors to numpy
            A = A.cpu().numpy()
            b = b.cpu().numpy()

            pt = find_interior(A, b)

            halfspaces = np.concatenate([A, -b[..., None]], axis=-1)
            hs = scipy.spatial.HalfspaceIntersection(halfspaces, pt, incremental=False, qhull_options=None)
            qhull_pts = hs.intersections

            pcd_object = o3d.geometry.PointCloud()
            pcd_object.points = o3d.utility.Vector3dVector(qhull_pts)
            bb_mesh, qhull_indices = pcd_object.compute_convex_hull()
            mesh += bb_mesh

        success = o3d.io.write_triangle_mesh(save_path, mesh, print_progress=True)

        return success
