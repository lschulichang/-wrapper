import numpy as np
import scipy
import torch
import open3d as o3d

from ellipsoids.mesh_utils import create_gs_mesh
from ellipsoids.covariance_utils import quaternion_to_rotation_matrix
from ellipsoids.sphere_utils import fibonacci_ellipsoid

# 【修改后】说明：本文件新增了中文注释；所有带“【修改后】”前缀的注释均为本次修改内容，未改变原有算法逻辑。

def _point_to_segment_distance(points, segment_start, segment_end):
    # 【修改后】计算多个点到一条线段的最短欧氏距离（支持批量点输入）。
    # points: N x 3, segment_start/end: 3
    seg = segment_end - segment_start
    seg_norm2 = torch.dot(seg, seg)

    if seg_norm2 <= 1e-12:
        # 【修改后】退化情形：线段长度近似为 0，距离退化为点到起点距离。
        return torch.linalg.norm(points - segment_start[None, :], dim=-1)

    rel = points - segment_start[None, :]
    t = (rel @ seg) / seg_norm2
    t = torch.clamp(t, 0.0, 1.0)
    proj = segment_start[None, :] + t[:, None] * seg[None, :]
    return torch.linalg.norm(points - proj, dim=-1)


def _build_covariance(rots, scales):
    # 【修改后】根据旋转矩阵和主轴尺度构造协方差矩阵：Sigma = R * diag(scales^2) * R^T。
    # rots: N x 3 x 3, scales: N x 3 (principal-axis radii)
    scales2 = scales * scales
    return (rots * scales2[:, None, :]) @ rots.transpose(1, 2)
#无用注释#
# This function calculates the bounding box for all line segments, but this is not strictly necessary
# Since we will probably prune polytopes and so sequential one-at-a-time may be fine as well.
def compute_bounding_box(path, rs):
    # 【修改后】为路径的每个线段构造局部坐标系下的长方体包围盒，并返回半空间形式 A x <= b。
    # path: N+1 x 3

    # Find the bounding box hyperplanes of the path

    # Without loss of generality, let us assume the local coordinate system is at the midpoint of the path and local x is along the path
    # We artibrarily choose y and z accordingly.
    midpoints = 0.5 * (path[1:] + path[:-1])        # N x 3
    lengths = torch.linalg.norm(path[1:] - path[:-1], dim=1)        # N

    lengths_x = lengths/2 + rs
    lengths_y = rs*torch.ones_like(lengths_x)
    lengths_z = rs*torch.ones_like(lengths_x)
    # 【修改后】三个方向的半长度：x 方向包含半线段长度，y/z 方向由安全半径给出。

    local_x = (path[1:] - path[:-1]) / torch.linalg.norm(path[1:] - path[:-1], dim=-1, keepdim=True)      # This is the pointing direction of the path (N x 3)

    # TODO: May have to treat the case where the path is just two points.!!!
    # Do Gram-Schmidt to find the other two directions
    random_vec = torch.randn(local_x.shape[-1], device=local_x.device)  # take a random vector
    local_y = random_vec[None, :] -  torch.sum(random_vec[None, :]*local_x, dim=-1, keepdim=True) * local_x       # make it orthogonal to x
    local_y = local_y / torch.linalg.norm(local_y, dim=-1, keepdim=True)            # normalize it
    local_z = torch.cross(local_x, local_y)    # This is the direction perpendicular to the path and the y-axis

    rotation_matrix = torch.stack([local_x, local_y, local_z], dim=-1)    # This is the local x,y,z to world frame rotation (N x 3 x 3)
    # 【修改后】rotation_matrix 的列向量分别是局部 x/y/z 在世界坐标中的方向。

    # These vectors form the normal of the hyperplanes. We simply need to find their intercepts. 
    # We are basically trying to find a_i.T * (x_0 + l_i * a_i) = b_i = a_i.T * x_0 + l_i (since a_i.T * a_i = 1)
    xyz_lengths = torch.stack([lengths_x, lengths_y, lengths_z], dim=-1)
    intercepts_pos = torch.bmm(midpoints[..., None, :], rotation_matrix).squeeze() + xyz_lengths    # N x 3
    intercepts_neg = -(intercepts_pos) + 2*xyz_lengths    # N x 3

    # Represent as x.T A <= b
    A = torch.cat([rotation_matrix, -rotation_matrix], dim=-1).transpose(1, 2)    # N x 6 x 3
    b = torch.cat([intercepts_pos, intercepts_neg], dim=-1)    # N x 6

    return A, b

def ellipsoid_halfspace_intersection(means, rots, scales, A, b):
    # 【修改后】判断椭球是否与半空间集合相交（用于筛选满足 Ax<=b 的高斯/椭球）。
    # This comparison tensor will be N (num of Gaussians) x 6 (num of hyperplanes)
    a_times_mu = means @ A.T
    numerator = b[None, :] - a_times_mu        # N x 6

    a_times_R = rots.transpose(1, 2) @ A.T # N x 3 x 6 
    denominator = torch.linalg.norm( a_times_R * scales[..., None] , dim=1)    # N x 6

    distance = -numerator / denominator

    # A gaussian must satisfy the metric for all 6 hyperplanes
    keep_gaussian = (distance <= 1.).all(dim=-1)      # mask of length N

    return keep_gaussian

def save_bounding_box(path, A, b, save_path):
    # 【修改后】将每段路径对应的半空间包围盒可视化为 mesh 并保存到磁盘。
    # Initialize mesh object
    mesh = o3d.geometry.TriangleMesh()

    midpoints = 0.5 * (path[1:] + path[:-1])
    for A0, b0, mid_pt in zip(A, b, midpoints):
        # Transfer all tensors to numpy
        A0 = A0.cpu().numpy()
        b0 = b0.cpu().numpy()
        mid_pt = mid_pt.cpu().numpy()

        halfspaces = np.concatenate([A0, -b0[..., None]], axis=-1)
        hs = scipy.spatial.HalfspaceIntersection(halfspaces, mid_pt, incremental=False, qhull_options=None)
        qhull_pts = hs.intersections

        pcd_object = o3d.geometry.PointCloud()
        pcd_object.points = o3d.utility.Vector3dVector(qhull_pts)
        bb_mesh, qhull_indices = pcd_object.compute_convex_hull()
        mesh += bb_mesh
    
    success = o3d.io.write_triangle_mesh(save_path, mesh, print_progress=True)

    return success

class CollisionSet():
    # 【修改后】碰撞集合基类：保存公共参数与接口定义，具体实现由子类完成。
    def __init__(self, gsplat, vmax, amax, radius, device):
        self.gsplat = gsplat
        self.vmax = vmax
        self.amax = amax
        self.radius = radius        # robot radius
        self.device = device

        self.rs = vmax**2 / (2*amax) + self.radius    # Safety radius

        self.means = self.gsplat.means
        self.rots = quaternion_to_rotation_matrix(self.gsplat.rots)
        self.scales = self.gsplat.scales
        self.gaussian_ids = torch.arange(self.means.shape[0], device=self.device)

    def compute_set(self, path, save_path=None):
        # 【修改后】接口：计算整条路径的碰撞集合。
        pass

    def compute_set_one_step(self, segment):
        # 【修改后】接口：计算单个线段的碰撞集合。
        pass

    def save_collision_set(self, ids, save_path):
        # 【修改后】接口：将碰撞集合写入可视化文件。
        pass

    def reset_reuse_cache(self):
        # 【修改后】接口：重置增量复用相关缓存。
        pass

class PointCloudCollisionSet(CollisionSet):
    # 【修改后】点云版本碰撞集合：可直接用高斯中心，或先对椭球表面采样。
    def __init__(self, gsplat, vmax, amax, radius, device, sample_surface=0):
        super().__init__(gsplat, vmax, amax, radius, device)

        if sample_surface > 0:
            # 【修改后】在每个椭球表面采样点，作为点云进行碰撞筛选。
            # Apply scaling and rotation to the sphere samples, then apply translation of the mean
            ellipsoid_samples = fibonacci_ellipsoid(self.means, self.rots, self.scales, kappa=0., n=sample_surface)
            self.point_cloud = ellipsoid_samples.reshape(-1, 3)

        # Don't sample the surface, use the mean.
        elif sample_surface == 0:
            # 【修改后】不采样时直接使用高斯中心点。
            self.point_cloud = self.means

        self.ids = torch.arange(self.point_cloud.shape[0], device=self.device)
     
    # NOTE: THIS COMPUTES THE COLLISION SET FOR ALL LINE SEGMENTS IN THE PATH!!!
    def compute_set(self, path, save_path=None):
        # 【修改后】遍历路径每一段，返回每段对应的点云碰撞集合。

        # Compute the bounding box
        A, b = compute_bounding_box(path, self.rs)

        # Then compute the points that lie within the bounding box
        pcd_collision_set = []
        
        collision_set_ids = []
        for i, (A0, b0) in enumerate(zip(A, b)):
            # Ax <= b for every point in point cloud
            keep_point = torch.all( (A0 @ self.point_cloud.T - b0[..., None]) <= 0., dim=0)       # N, size of point cloud

            # Save the indices of the gaussians that are within the bounding box
            data = {
                'primitive_ids': self.ids[keep_point],
                'A_bb': A0,
                'b_bb': b0,
                'b_bb_shrunk': b0 - self.radius,
                'path': path[i:i+2],
                'midpoint': 0.5 * (path[i] + path[i+1]),
                'id': i
            }

            pcd_collision_set.append(data)

            collision_set_ids.append(self.ids[keep_point])

        # We want to save the bounding box as a viewable mesh using Open3d
        if save_path is not None:
            save_bounding_box(path, A, b, save_path + '_bounding_box.obj')
            self.save_collision_set(collision_set_ids, save_path + '_collision_set.ply')

        # Return a dictionary of the point cloud collision set and the shrunk bounding box constraints representing only the free space for the robot centroid.
        return pcd_collision_set
    
    # NOTE: THIS COMPUTES THE COLLISION SET FOR ONE LINE SEGMENT IN THE PATH!!!
    def compute_set_one_step(self, segment):
        # 【修改后】单步版本：仅针对一段路径筛选落入包围盒的点。

        # Compute the bounding box
        A, b = compute_bounding_box(segment, self.rs)
        A = A.squeeze()
        b = b.squeeze()

        # Ax <= b for every point in point cloud
        keep_point = torch.all( (A @ self.point_cloud.T - b[..., None]) <= 0., dim=0)       # N, size of point cloud
  
        # Save the indices of the gaussians that are within the bounding box
        output = {
            'primitive_ids': self.ids[keep_point],
            'A_bb': A,
            'b_bb': b,
            'b_bb_shrunk': b - self.radius,
            'path': segment,
            'midpoint': 0.5*(segment[0] + segment[1])
        }

        # Return a dictionary of the Gaussian collision set and the shrunk bounding box constraints representing only the free space for the robot centroid.
        return output
    
    def save_collision_set(self, ids, save_path):
        # 【修改后】将所有线段的点云碰撞集合合并去重后写入 .ply。

        # This saves the whole collision set as one mesh
        unique_ids = torch.unique(torch.cat(ids, dim=0))

        points = self.point_cloud[unique_ids]

        scene = o3d.geometry.PointCloud()
        scene.points = o3d.utility.Vector3dVector(points.cpu().numpy())
    
        success = o3d.io.write_point_cloud(save_path, scene, print_progress=True)
        return success

class GSplatCollisionSet(CollisionSet):
    # 【修改后】GSplat 版本碰撞集合：采用“粗筛 -> 代理筛选 -> 精确筛选”的分层流程。
    def __init__(
        self,
        gsplat,
        vmax,
        amax,
        radius,
        device,
        delta_s_scale=0.5,
        delta_g_scale=0.2,
        neighbor_margin_scale=0.0,
        enable_incremental_reuse=True,
        index_cell_size=None,
    ):
        super().__init__(gsplat, vmax, amax, radius, device)

        # Patent-style candidate filter parameters.
        # 【修改后】delta_s/delta_g 等参数用于控制候选集膨胀与邻段复用。
        self.delta_s = float(delta_s_scale) * float(self.rs)
        self.delta_g = float(delta_g_scale) * float(self.rs)
        self.neighbor_margin = float(neighbor_margin_scale) * float(self.rs)
        self.enable_incremental_reuse = bool(enable_incremental_reuse)
        self.query_radius = float(self.rs + self.delta_s)

        self.index_cell_size = float(index_cell_size) if index_cell_size is not None else float(self.query_radius)
        if self.index_cell_size <= 1e-9:
            self.index_cell_size = float(max(self.query_radius, 1e-3))

        # Build a lightweight uniform-grid index on CPU for coarse retrieval.
        self.means_cpu = self.means.detach().cpu()
        self.index_origin = torch.min(self.means_cpu, dim=0).values - 1e-6
        cell_coords = torch.floor((self.means_cpu - self.index_origin[None, :]) / self.index_cell_size).to(torch.int64)
        self.grid_index = {}
        for idx, c in enumerate(cell_coords.tolist()):
            key = (int(c[0]), int(c[1]), int(c[2]))
            if key not in self.grid_index:
                self.grid_index[key] = []
            self.grid_index[key].append(idx)

        self.reset_reuse_cache()

    def reset_reuse_cache(self):
        # 【修改后】缓存上一个线段的最终碰撞 id 和查询过的网格单元。
        self.prev_final_ids = None
        self.prev_query_cells = None

    def _bbox_to_cell_bounds(self, bbox_min, bbox_max):
        # 【修改后】将连续坐标系下的包围盒映射到网格索引范围。
        min_idx = torch.floor((bbox_min - self.index_origin) / self.index_cell_size).to(torch.int64)
        max_idx = torch.floor((bbox_max - self.index_origin) / self.index_cell_size).to(torch.int64)
        return min_idx, max_idx

    def _collect_ids_from_cells(self, cell_keys):
        # 【修改后】从命中的网格单元收集高斯 id 并去重。
        if not cell_keys:
            return torch.empty((0,), dtype=torch.long, device=self.device)
        ids = []
        for key in cell_keys:
            ids.extend(self.grid_index.get(key, []))
        if not ids:
            return torch.empty((0,), dtype=torch.long, device=self.device)
        ids_cpu = torch.tensor(ids, dtype=torch.long).unique()
        return ids_cpu.to(self.device)

    def _query_cells_for_segment(self, segment, query_radius):
        # 【修改后】根据线段和查询半径，获得与包围盒相交的网格单元集合。
        s = segment[0].detach().cpu()
        e = segment[1].detach().cpu()
        bbox_min = torch.minimum(s, e) - query_radius
        bbox_max = torch.maximum(s, e) + query_radius
        min_idx, max_idx = self._bbox_to_cell_bounds(bbox_min, bbox_max)

        cell_keys = set()
        for ix in range(int(min_idx[0]), int(max_idx[0]) + 1):
            for iy in range(int(min_idx[1]), int(max_idx[1]) + 1):
                for iz in range(int(min_idx[2]), int(max_idx[2]) + 1):
                    key = (ix, iy, iz)
                    if key in self.grid_index:
                        cell_keys.add(key)
        return cell_keys

    def _distance_filter(self, ids, segment, distance_threshold):
        # 【修改后】按“点到线段距离阈值”进行一次快速裁剪。
        if ids.numel() == 0:
            return ids
        means = self.means[ids]
        d = _point_to_segment_distance(means, segment[0], segment[1])
        return ids[d <= distance_threshold]

    def _exact_filter(self, ids, segment):
        # 【修改后】精确碰撞判定：使用膨胀椭球模型进行二次型最小值测试。
        if ids.numel() == 0:
            return ids

        means = self.means[ids]
        rots = self.rots[ids]
        scales = self.scales[ids]

        sigma = _build_covariance(rots, scales)
        eye = torch.eye(3, dtype=sigma.dtype, device=sigma.device)[None, :, :]
        M = torch.linalg.inv(sigma + (self.rs ** 2) * eye)

        s = segment[0]
        e = segment[1]
        v = e - s
        diff = s[None, :] - means

        Aj = torch.einsum("i,nij,j->n", v, M, v)
        Bj = 2.0 * torch.einsum("i,nij,nj->n", v, M, diff)
        Cj = torch.einsum("ni,nij,nj->n", diff, M, diff) - 1.0

        t_star = torch.zeros_like(Bj)
        stable = torch.abs(Aj) > 1e-12
        t_star[stable] = torch.clamp(-Bj[stable] / (2.0 * Aj[stable]), 0.0, 1.0)

        f_star = Aj * (t_star ** 2) + Bj * t_star + Cj
        return ids[f_star <= 0.0]

    def _build_output(self, segment, ids, A, b, debug_info):
        # 【修改后】统一输出格式，附带调试统计信息。
        return {
            'primitive_ids': ids,
            'A_bb': A,
            'b_bb': b,
            'b_bb_shrunk': b - self.radius,
            'path': segment,
            'midpoint': 0.5*(segment[0] + segment[1]),
            'means': self.means[ids],
            'rots': self.rots[ids],
            'scales': self.scales[ids],
            'candidate_debug': debug_info,
        }

    # NOTE: THIS COMPUTES THE COLLISION SET FOR ALL LINE SEGMENTS IN THE PATH!!!
    def compute_set(self, path, save_path=None):
        # 【修改后】整条路径流程：逐段调用单步筛选，并可选保存可视化结果。
        self.reset_reuse_cache()
        gaussian_collision_set = []

        collision_set_ids = []
        A_all, b_all = compute_bounding_box(path, self.rs)
        for i, segment in enumerate(torch.stack([path[:-1], path[1:]], dim=1)):
            out = self.compute_set_one_step(segment)
            out['A_bb'] = A_all[i]
            out['b_bb'] = b_all[i]
            out['b_bb_shrunk'] = b_all[i] - self.radius
            out['id'] = i
            data = out
            gaussian_collision_set.append(data)
            collision_set_ids.append(data['primitive_ids'])

        # We want to save the bounding box as a viewable mesh using Open3d
        if save_path is not None:
            save_bounding_box(path, A_all, b_all, save_path + '_bounding_box.obj')
            self.save_collision_set(collision_set_ids, save_path + '_collision_set.obj')

        # Return a dictionary of the Gaussian collision set and the shrunk bounding box constraints representing only the free space for the robot centroid.
        return gaussian_collision_set
    
    # NOTE: THIS COMPUTES THE COLLISION SET FOR ONE LINE SEGMENT IN THE PATH!!!
    def compute_set_one_step(self, segment):
        # 【修改后】单段流程：包围盒 -> 粗检索 -> 邻段复用 -> 代理筛选 -> 精确筛选。
        # Compute path-segment safety envelope (used both for output and search bounds)
        A, b = compute_bounding_box(segment, self.rs)
        A = A.squeeze()
        b = b.squeeze()

        # Step 3: coarse retrieval from spatial index (capsule envelope Qi)
        current_cells = self._query_cells_for_segment(segment, self.query_radius)
        coarse_ids = self._collect_ids_from_cells(current_cells)
        coarse_ids = self._distance_filter(coarse_ids, segment, self.query_radius + self.neighbor_margin)

        # Step 6 (optional): incremental reuse between neighboring segments
        if self.enable_incremental_reuse and (self.prev_final_ids is not None):
            reuse_ids = self.prev_final_ids

            # Reuse condition: center inside current capsule envelope OR still proxy-near
            reuse_ids = self._distance_filter(reuse_ids, segment, self.query_radius)
            if reuse_ids.numel() == 0:
                prev_means = self.means[self.prev_final_ids]
                prev_scales = self.scales[self.prev_final_ids]
                prev_rho = torch.max(prev_scales, dim=-1).values + self.delta_g
                prev_dist = _point_to_segment_distance(prev_means, segment[0], segment[1])
                reuse_mask = prev_dist <= (prev_rho + self.rs)
                reuse_ids = self.prev_final_ids[reuse_mask]

            # Delta-query approximation: only query newly touched cells.
            if self.prev_query_cells is None:
                new_cells = current_cells
            else:
                new_cells = current_cells.difference(self.prev_query_cells)
            new_ids = self._collect_ids_from_cells(new_cells)
            new_ids = self._distance_filter(new_ids, segment, self.query_radius + self.neighbor_margin)

            init_ids = torch.unique(torch.cat([reuse_ids, new_ids], dim=0))
            reuse_count = int(reuse_ids.numel())
            delta_count = int(new_ids.numel())
        else:
            init_ids = coarse_ids
            reuse_count = 0
            delta_count = int(coarse_ids.numel())

        # Step 4: proxy conservative pruning with spherical over-approximation
        if init_ids.numel() > 0:
            means_init = self.means[init_ids]
            scales_init = self.scales[init_ids]
            rho = torch.max(scales_init, dim=-1).values + self.delta_g
            d = _point_to_segment_distance(means_init, segment[0], segment[1])
            keep_proxy = d <= (rho + self.rs)
            ids_proxy = init_ids[keep_proxy]
        else:
            ids_proxy = init_ids

        # Step 5: exact check with inflated ellipsoid model
        ids_exact = self._exact_filter(ids_proxy, segment)

        debug_info = {
            'n_initial': int(init_ids.numel()),
            'n_coarse': int(coarse_ids.numel()),
            'n_reuse': reuse_count,
            'n_delta': delta_count,
            'n_after_proxy': int(ids_proxy.numel()),
            'n_after_exact': int(ids_exact.numel()),
        }

        # Cache for next segment incremental reuse.
        self.prev_final_ids = ids_exact
        self.prev_query_cells = current_cells

        return self._build_output(segment, ids_exact, A, b, debug_info)
    
    def save_collision_set(self, ids, save_path):
        # 【修改后】将最终高斯碰撞集合转换为 mesh 后写入 .obj。

        # This saves the whole collision set as one mesh
        unique_ids = torch.unique(torch.cat(ids, dim=0))

        means = self.gsplat.means[unique_ids]
        rots = quaternion_to_rotation_matrix(self.gsplat.rots[unique_ids])
        scales = self.gsplat.scales[unique_ids]
        colors = self.gsplat.colors[unique_ids]
    
        scene = create_gs_mesh(means.cpu().numpy(), rots.cpu().numpy(), scales.cpu().numpy(), colors.cpu().numpy(), res=4, transform=None, scale=None)
        success = o3d.io.write_triangle_mesh(save_path, scene, print_progress=True)
        return success
