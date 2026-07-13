from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "splatnav-official"
sys.path.insert(0, str(UPSTREAM))

from ground_nav.ground_grid import GroundGrid  # noqa: E402
from ground_nav.path_utils import circumscribed_sphere_radius, gaussian_height_mask, simplify_ground_path  # noqa: E402
from polytopes.collision_set import GSplatCollisionSet, compute_bounding_box  # noqa: E402
from splatplan.splatplan import SplatPlan  # noqa: E402
from splatplan.spline_utils import SplinePlanner  # noqa: E402


def empty_voxel(n=10):
    lower, upper = torch.tensor([0., 0., 0.]), torch.tensor([10., 10., 2.])
    cell_sizes = (upper - lower) / torch.tensor([n, n, 2])
    x, y, z = torch.meshgrid(torch.linspace(.5, 9.5, n), torch.linspace(.5, 9.5, n), torch.tensor([.5, 1.5]), indexing="ij")
    return SimpleNamespace(non_navigable_grid=torch.zeros((n, n, 2), dtype=torch.bool), grid_centers=torch.stack([x, y, z], -1), cell_sizes=cell_sizes, radius=0.)


def fake_gsplat():
    return SimpleNamespace(
        means=torch.tensor([[0., 0., .1], [0., 0., 1.]], dtype=torch.float32),
        covs=torch.eye(3).unsqueeze(0).repeat(2, 1, 1) * .01,
        rots=torch.tensor([[1., 0., 0., 0.]]).repeat(2, 1),
        scales=torch.ones((2, 3)) * .1,
        colors=torch.ones((2, 3)),
    )


class Milestone2Test(unittest.TestCase):
    def test_radius_and_height_mask(self):
        self.assertAlmostEqual(circumscribed_sphere_radius(.15, .10), np.sqrt(.025))
        self.assertEqual(gaussian_height_mask(fake_gsplat(), .05, .20).tolist(), [True, False])

    def test_simplification(self):
        grid = GroundGrid(empty_voxel(), 0., 1., 0., project_occupied_endpoints=False)
        path = np.array([[.5,.5,.5],[1.5,.5,.5],[2.5,.5,.5],[2.5,1.5,.5],[3.5,1.5,.5]], dtype=np.float32)
        result = simplify_ground_path(path, grid, 3.1, lambda _: True)
        np.testing.assert_allclose(result[0], path[0]); np.testing.assert_allclose(result[-1], path[-1])
        self.assertLess(len(result), len(path))
        self.assertTrue(np.all(np.linalg.norm(np.diff(result[:, :2], axis=0), axis=1) <= 3.1 + 1e-6))

    def test_deterministic_bounding_box(self):
        path = torch.tensor([[0.,0.,0.],[1.,2.,0.]])
        a1,b1 = compute_bounding_box(path,.2); a2,b2 = compute_bounding_box(path,.2)
        self.assertTrue(torch.equal(a1,a2)); self.assertTrue(torch.equal(b1,b2))

    def test_collision_margin(self):
        gs = fake_gsplat()
        legacy = GSplatCollisionSet(gs,.2,.5,.1,torch.device('cpu'))
        explicit = GSplatCollisionSet(gs,99.,99.,.1,torch.device('cpu'),corridor_margin=.3)
        self.assertAlmostEqual(legacy.rs,.14); self.assertAlmostEqual(explicit.rs,.4)

    def test_fixed_z_qp(self):
        A = torch.tensor([[1.,0.,0.],[-1.,0.,0.],[0.,1.,0.],[0.,-1.,0.],[0.,0.,1.],[0.,0.,-1.]])
        planner = SplinePlanner(spline_deg=3,N_sec=8,device='cpu')
        traj, feasible = planner.optimize_b_spline([(A,torch.ones(6)*10)],torch.tensor([0.,0.,.5]),torch.tensor([1.,1.,.5]),fixed_z=.5)
        self.assertTrue(feasible)
        self.assertEqual(planner.last_solver_status, 'Solved')
        self.assertTrue(np.allclose(planner.coeffs[:,2,:],.5,atol=1e-6))
        self.assertTrue(np.allclose(traj[:,2],.5,atol=1e-6))
        self.assertTrue(np.allclose(traj[:,[5,8,11]],0.,atol=1e-6))

    def test_external_seed_skips_voxel(self):
        class Seed:
            def create_path(self,x0,xf): return np.array([[0.,0.,.5],[1.,0.,.5]],dtype=np.float32)
        planner = SplatPlan(fake_gsplat(),{'radius':.1,'vmax':.1,'amax':.1},{'lower_bound':torch.zeros(3),'upper_bound':torch.ones(3),'resolution':5},SplinePlanner(spline_deg=3,N_sec=5),torch.device('cpu'),seed_planner=Seed(),collision_set=object())
        self.assertIsNone(planner.gsplat_voxel)


if __name__ == '__main__': unittest.main()
