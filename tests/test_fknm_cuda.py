import pytest
import importlib
import roboticstoolbox as rtb
from roboticstoolbox.tools.profiler import Profiler
from roboticstoolbox.tools.utils import *
from spatialmath import SE3
from spatialgeometry import *
import numpy as np
import trimesh
import time
import math
from numba import vectorize 


def test_vec_ts():
    seed = 0
    np.random.seed(seed)
    robot = rtb.models.DH.Panda() # load Mesh version (for collisions)
    T = robot.fkine(robot.qz)  # forward kinematics
    T = SE3(0.7, 0.2, 0.1) * SE3.OA([0, 1, 0], [0, 0, -1])
    sol = robot.ikine_LM(T)         # solve IK
    q_pickup = sol.q
    qtraj = rtb.jtraj(robot.qz, q_pickup, 50)
    robot = rtb.models.URDF.Panda()  # load URDF version of the Panda
    obstacle = Box([0.3, 0.3, 0.3], base=SE3(0.5, 0.5, 0.8))
    pt = robot.closest_point(robot.q, obstacle)
    meshes = {}
    for link in robot.links:
        if len(link.geometry) != 1:
            print(len(link.geometry))
            continue
        kwargs = trimesh.exchange.dae.load_collada(link.geometry[0].filename)
        # kwargs = trimesh.exchange.dae.load_collada(filename)
        mesh = trimesh.exchange.load.load_kwargs(kwargs)
        meshes[link.name] = mesh.dump(concatenate=True)
    # Hyperparameters
    dt = 1
    nq = 50
    lmbda = 1000
    eta = 1000
    iters = 4
    num_pts = 10
    # Make cost field, starting & end points
    cdim = len(qtraj.q[0])
    xidim = cdim * nq
    AA = np.zeros((xidim, xidim))
    xi = np.zeros(xidim)
    # robot._set_link_fk(qtraj.q[1])
    qs = np.zeros(cdim)
    qe = q_pickup
    bb = np.zeros(xidim)
    bb[: cdim] = qs
    bb[-cdim:] = qe
    xi[: cdim] = qs
    bb /= - dt * dt * (nq + 1)
    # Make a matrix
    for i in range(nq):
        AA[cdim * i: cdim * (i+1), cdim * i: cdim * (i+1)] = 2 * np.eye(cdim)
        if i > 0:
            AA[cdim * (i-1): cdim * i, cdim * i: cdim * (i+1)] = -1 * np.eye(cdim)
            AA[cdim * i: cdim * (i+1), cdim * (i-1): cdim * i] = -1 * np.eye(cdim)
    AA /= dt * dt * (nq + 1)
    Ainv = np.linalg.pinv(AA)
    for t in range(iters):
        nabla_smooth = AA.dot(xi) + bb
        nabla_obs = np.zeros(xidim)
        xidd = AA.dot(xi)
        total_cost = 0

        link_base_all = []
        pts_all = []
        pts_xyz_all = []
        qd_all = []
        for i in range(nq): # timestep
            qt = xi[cdim * i: cdim * (i+1)]
            if i == 0:
                qd = 0.5 * (xi[cdim * (i+1): cdim * (i+2)] - qs)
            elif i == nq - 1:
                qd = 0.5 * (qe - xi[cdim * (i-1): cdim * (i)])
            else:
                qd = 0.5 * (xi[cdim * (i+1): cdim * (i+2)] - xi[cdim * (i-1): cdim * (i)])
            qd_all.append(qd)
            for il, link in enumerate(robot.links): # bodypart
                k = link.jindex
                if k is None:
                    continue
                link_base = robot.fkine(qt, end=link) # x_current, (4, 4)
                link_base_all.append(link_base.A)
                delta_nabla_obs = np.zeros(k + 1)
                mesh = meshes[link.name]
                idxs = np.random.choice(np.arange(len(mesh.vertices)), num_pts, replace=False)
                pts = mesh.vertices[idxs]
                pts_all.append(pts)
                pts_se3 = vrepeat(np.eye(4), num_pts)
                pts_se3[:, :3, 3] = pts
                pts_xyz = link_base.A.dot(pts_se3).swapaxes(0, 1)[:, :3, 3]
                pts_xyz_all.append(pts_xyz)

        link_base_all = np.stack(link_base_all) # (nq * cdim, 4, 4)
        pts_all = np.concatenate(pts_all) # (nq * cdim * num_pts, 3)
        pts_xyz_all = np.concatenate(pts_xyz_all) # (nq * cdim * num_pts, 3)
        qd_all = np.stack(qd_all) # (nq, cdim)
        path, njoints, _ = robot.get_path()
        assert njoints == cdim
        jacobs = np.zeros((len(pts_all), 6, cdim)) # (nq * cdim * num_pts, 6, cdim)
        jacob0_vec(robot, nq, cdim, link_base_all, pts_all, jacobs, qt)
        jacobs = jacobs.reshape(nq, cdim * num_pts, 6, cdim)
        jacobs_ = jacobs.reshape(nq, cdim, num_pts, 6, cdim) # (nq, cdim, num_pts, 6, cdim)
        xd = vmmatmul(jacobs_, qd_all).reshape(-1, 6) # x_delta, (nq * cdim * num_pts, 6)
        jacobs = jacobs_.reshape(-1, 6, cdim) # (nq * cdim * num_pts, 6, cdim)
        vel = np.linalg.norm(xd, axis=-1, keepdims=True) # (nq * cdim * npoints, 1)
        xdn = xd / (vel + 1e-3) # speed normalized, (nq * cdim * num_pts, 6)
        xidd = xidd.reshape(nq, cdim) # (nq, cdim)
        xdd = vmmatmul(jacobs_, xidd).reshape(-1, 6) # second derivative of xi, (nq * cdim * num_pts, 6)
        prj = vrepeat(np.eye(6), nq * cdim * num_pts)
        #import pdb; pdb.set_trace()
        prj -= np.matmul(xdn[:, :, None], xdn[:, None, :]) # curvature vector (nq * cdim * num_pts, 6, 6)
        kappa = (vmatmul(prj, xdd) / (vel ** 2 + 1e-3)) # (nq * cdim * num_pts, 6)
        cost = np.sum(pts_xyz_all, axis=1, keepdims=True) # (nq * cdim * num_pts, 1)
        delta = -1 * np.concatenate([[1, 1, 0], np.zeros(3)])
        delta = vrepeat(delta, nq * cdim * num_pts) # (nq * cdim * num_pts, 6)
        part1 = jacobs.swapaxes(1, 2) # (nq * cdim * num_pts, cdim, 6)
        part2 = vmatmul(prj, delta) - cost * kappa # (nq * cdim * num_pts, 6)
        delta_nabla_obs = vmatmul(part1, part2) * vel # (nq * cdim * num_pts, cdim)
        total_cost += cost.mean() * cdim * nq

        delta_nabla_obs = delta_nabla_obs.reshape(nq, cdim, num_pts, -1)
        nabla_obs = delta_nabla_obs.mean(axis=(1, 2)).flatten() * cdim
        

        nabla_obs2 = np.zeros(xidim)
        for i in range(nq): # timestep
            qt = xi[cdim * i: cdim * (i+1)]
            # qt = qe
            if i == 0:
                qd = 0.5 * (xi[cdim * (i+1): cdim * (i+2)] - qs)
            elif i == nq - 1:
                qd = 0.5 * (qe - xi[cdim * (i-1): cdim * (i)])
            else:
                qd = 0.5 * (xi[cdim * (i+1): cdim * (i+2)] - xi[cdim * (i-1): cdim * (i)])

            ## Parallel_links
            link_base_t = []
            pts_t = []
            pts_xyz_t = []
            for il, link in enumerate(robot.links): # bodypart
                k = link.jindex
                if k is None:
                    continue
                link_base = robot.fkine(qt, end=link) # x_current, (4, 4)
                link_base_t.append(link_base.A)
                mesh = meshes[link.name]
                idxs = np.random.choice(np.arange(len(mesh.vertices)), num_pts, replace=False)
                pts = mesh.vertices[idxs]
                pts_t.append(pts)
                pts_se3 = vrepeat(np.eye(4), num_pts)
                pts_se3[:, :3, 3] = pts
                pts_xyz = link_base.A.dot(pts_se3).swapaxes(0, 1)[:, :3, 3]
                pts_xyz_t.append(pts_xyz)
            link_base_t = np.stack(link_base_t) # (cdim, 4, 4)
            pts_t = np.concatenate(pts_t) # (cdim * npoints, 3)
            pts_xyz_t = np.concatenate(pts_xyz_t)
            path, njoints, _ = robot.get_path()
            jacobs1 = np.zeros((len(pts_t), 6, njoints))
            jacob0_vec(robot, 1, njoints, link_base_t, pts_t, jacobs1, qt)
            
            try:
                assert np.allclose(jacobs_[i].reshape(-1, 6, cdim), jacobs1)
            except:
                import pdb; pdb.set_trace()

            xd = jacobs1.dot(qd) # x_delta, (N, 6)
            vel = np.linalg.norm(xd, axis=1, keepdims=True) # (N, 1)
            xdn = xd / (vel + 1e-3) # speed normalized, (N, 6)
            xdd = jacobs1.dot(xidd[cdim * i: cdim * (i + 1)]) # second derivative of xi, (N, 6)
            prj = vrepeat(np.eye(6), num_pts * njoints)
            prj -= np.matmul(xdn[:, :, None], xdn[:, None, :]) # curvature vector (N, 6, 6)
            kappa = (vmatmul(prj, xdd) / (vel ** 2 + 1e-3)) # (N, 6)
            cost = np.sum(pts_xyz_t, axis=1, keepdims=True) # (N, 1)
            delta = -1 * np.concatenate([[1, 1, 0], np.zeros(3)])
            delta = vrepeat(delta, num_pts * njoints) # (N, 3, 6)
            part1 = jacobs1.swapaxes(1, 2) # (N, cdim, 6)
            part2 = vmatmul(prj, delta) - cost * kappa # (N, 6)
            delta_nabla_obs = vmatmul(part1, part2) * vel # (N, cdim)
            nabla_obs1[cdim * i: cdim * (i + 1)] = delta_nabla_obs.mean(axis=0) * njoints




def test_vec_links():
    seed = 0
    np.random.seed(seed)
    robot = rtb.models.DH.Panda() # load Mesh version (for collisions)
    T = robot.fkine(robot.qz)  # forward kinematics
    T = SE3(0.7, 0.2, 0.1) * SE3.OA([0, 1, 0], [0, 0, -1])
    sol = robot.ikine_LM(T)         # solve IK
    q_pickup = sol.q
    qtraj = rtb.jtraj(robot.qz, q_pickup, 50)
    robot = rtb.models.URDF.Panda()  # load URDF version of the Panda
    obstacle = Box([0.3, 0.3, 0.3], base=SE3(0.5, 0.5, 0.8))
    pt = robot.closest_point(robot.q, obstacle)
    meshes = {}
    for link in robot.links:
        if len(link.geometry) != 1:
            print(len(link.geometry))
            continue
        kwargs = trimesh.exchange.dae.load_collada(link.geometry[0].filename)
        # kwargs = trimesh.exchange.dae.load_collada(filename)
        mesh = trimesh.exchange.load.load_kwargs(kwargs)
        meshes[link.name] = mesh.dump(concatenate=True)
    # Hyperparameters
    dt = 1
    nq = 50
    lmbda = 1000
    eta = 1000
    iters = 4
    num_pts = 10
    # Make cost field, starting & end points
    cdim = len(qtraj.q[0])
    xidim = cdim * nq
    AA = np.zeros((xidim, xidim))
    xi = np.zeros(xidim)
    # robot._set_link_fk(qtraj.q[1])
    qs = np.zeros(cdim)
    qe = q_pickup
    bb = np.zeros(xidim)
    bb[: cdim] = qs
    bb[-cdim:] = qe
    xi[: cdim] = qs
    bb /= - dt * dt * (nq + 1)
    # Make a matrix
    for i in range(nq):
        AA[cdim * i: cdim * (i+1), cdim * i: cdim * (i+1)] = 2 * np.eye(cdim)
        if i > 0:
            AA[cdim * (i-1): cdim * i, cdim * i: cdim * (i+1)] = -1 * np.eye(cdim)
            AA[cdim * i: cdim * (i+1), cdim * (i-1): cdim * i] = -1 * np.eye(cdim)
    AA /= dt * dt * (nq + 1)
    Ainv = np.linalg.pinv(AA)
    for t in range(iters):
        nabla_smooth = AA.dot(xi) + bb
        nabla_obs1 = np.zeros(xidim)
        nabla_obs2 = np.zeros(xidim)
        xidd = AA.dot(xi)
        total_cost = 0
        jacobs2_all = []
        
        for i in range(nq): # timestep
            qt = xi[cdim * i: cdim * (i+1)]
            # qt = qe
            if i == 0:
                qd = 0.5 * (xi[cdim * (i+1): cdim * (i+2)] - qs)
            elif i == nq - 1:
                qd = 0.5 * (qe - xi[cdim * (i-1): cdim * (i)])
            else:
                qd = 0.5 * (xi[cdim * (i+1): cdim * (i+2)] - xi[cdim * (i-1): cdim * (i)])

            ## Parallel_links
            link_base_t = []
            pts_t = []
            pts_xyz_t = []
            for il, link in enumerate(robot.links): # bodypart
                k = link.jindex
                if k is None:
                    continue
                link_base = robot.fkine(qt, end=link) # x_current, (4, 4)
                link_base_t.append(link_base.A)
                mesh = meshes[link.name]
                idxs = np.random.choice(np.arange(len(mesh.vertices)), num_pts, replace=False)
                pts = mesh.vertices[idxs]
                pts_t.append(pts)
                pts_se3 = vrepeat(np.eye(4), num_pts)
                pts_se3[:, :3, 3] = pts
                pts_xyz = link_base.A.dot(pts_se3).swapaxes(0, 1)[:, :3, 3]
                pts_xyz_t.append(pts_xyz)
            link_base_t = np.stack(link_base_t) # (cdim, 4, 4)
            pts_t = np.concatenate(pts_t) # (cdim * npoints, 3)
            pts_xyz_t = np.concatenate(pts_xyz_t)
            path, njoints, _ = robot.get_path()
            jacobs1 = np.zeros((len(pts_t), 6, njoints))
            jacob0_vec(robot, 1, njoints, link_base_t, pts_t, jacobs1, qt)
            # import pdb; pdb.set_trace()
            xd = jacobs1.dot(qd) # x_delta, (N, 6)
            vel = np.linalg.norm(xd, axis=1, keepdims=True) # (N, 1)
            xdn = xd / (vel + 1e-3) # speed normalized, (N, 6)
            xdd = jacobs1.dot(xidd[cdim * i: cdim * (i + 1)]) # second derivative of xi, (N, 6)
            prj = vrepeat(np.eye(6), num_pts * njoints)
            prj -= np.matmul(xdn[:, :, None], xdn[:, None, :]) # curvature vector (N, 6, 6)
            kappa = (vmatmul(prj, xdd) / (vel ** 2 + 1e-3)) # (N, 6)
            cost = np.sum(pts_xyz_t, axis=1, keepdims=True) # (N, 1)
            delta = -1 * np.concatenate([[1, 1, 0], np.zeros(3)])
            delta = vrepeat(delta, num_pts * njoints) # (N, 3, 6)
            part1 = jacobs1.swapaxes(1, 2) # (N, cdim, 6)
            part2 = vmatmul(prj, delta) - cost * kappa # (N, 6)
            delta_nabla_obs = vmatmul(part1, part2) * vel # (N, cdim)
            nabla_obs1[cdim * i: cdim * (i + 1)] = delta_nabla_obs.mean(axis=0) * njoints


            # Non-parallel
            for il, link in enumerate(robot.links): # bodypart
                k = link.jindex
                if k is None:
                    continue
                link_base = robot.fkine(qt, end=link) # x_current, (4, 4)
                delta_nabla_obs = np.zeros(k + 1)
                mesh = meshes[link.name]
                pts = pts_t[k * num_pts: (k + 1) * num_pts]
                pts_se3[:, :3, 3] = pts
                pts_xyz = link_base.A.dot(pts_se3).swapaxes(0, 1)[:, :3, 3]
                path, njoints, _ = robot.get_path(end=link)
                jacobs2 = np.zeros((num_pts, 6, njoints))
                jacob0_pts_vec(robot, link, pts, jacobs2, qt)
                jacobs2_all.append(jacobs2)

                try:
                    assert np.allclose(jacobs1[num_pts * k: num_pts * (k + 1), :, :k + 1], jacobs2)
                except:
                    import pdb; pdb.set_trace()
                # print(np.allclose(jacobs1[num_pts * k: num_pts * (k + 1), :, k: k + 1], jacobs2))
                    
                # with Profiler("Step 2"):
                xd2 = jacobs2.dot(qd[:k+1]) # x_delta, (N, 6)
                try:
                    assert np.allclose(xd[num_pts * k: num_pts * (k + 1)], xd2)
                except:
                    import pdb; pdb.set_trace()
                vel2 = np.linalg.norm(xd2, axis=1, keepdims=True) # (N, 1)
                assert np.allclose(vel[num_pts * k: num_pts * (k + 1)], vel2)
                xdn2 = xd2 / (vel2 + 1e-3) # speed normalized, (N, 6)
                assert np.allclose(xdn[num_pts * k: num_pts * (k + 1)], xdn2)
                xdd2 = jacobs2.dot(xidd[cdim * i: cdim * i + k + 1]) # second derivative of xi, (N, 6)
                # assert np.allclose()
                prj2 = vrepeat(np.eye(6), num_pts)
                prj2 -= np.matmul(xdn2[:, :, None], xdn2[:, None, :]) # curvature vector (N, 6, 6)
                kappa2 = (vmatmul(prj2, xdd2) / (vel2 ** 2 + 1e-3)) # (N, 6)
                cost2 = np.sum(pts_xyz, axis=1, keepdims=True) # (N, 1)
                # delta := negative gradient of obstacle cost in work space, (6, cdim) 
                delta2 = -1 * np.concatenate([[1, 1, 0], np.zeros(3)])
                delta2 = vrepeat(delta2, num_pts) # (N, 3, 6)
                # for link in robot.links:
                #     cost = get_link_cost(robot, meshes, link)
                part12 = jacobs2.swapaxes(1, 2) # (N, k + 1, 6)
                part22 = vmatmul(prj2, delta2) - cost2 * kappa2 # (N, 6)
                delta_nabla_obs2 = vmatmul(part12, part22) * vel2 # (N, k + 1)
                nabla_obs2[cdim * i: cdim * i + k + 1] += delta_nabla_obs2.mean(axis=0)

            assert np.allclose(nabla_obs1, nabla_obs2)

def test_vec_points():
    seed = 0
    np.random.seed(seed)
    robot = rtb.models.DH.Panda() # load Mesh version (for collisions)
    T = robot.fkine(robot.qz)  # forward kinematics
    T = SE3(0.7, 0.2, 0.1) * SE3.OA([0, 1, 0], [0, 0, -1])
    sol = robot.ikine_LM(T)         # solve IK
    q_pickup = sol.q
    qtraj = rtb.jtraj(robot.qz, q_pickup, 50)
    robot = rtb.models.URDF.Panda()  # load URDF version of the Panda
    obstacle = Box([0.3, 0.3, 0.3], base=SE3(0.5, 0.5, 0.8))
    pt = robot.closest_point(robot.q, obstacle)
    meshes = {}
    for link in robot.links:
        if len(link.geometry) != 1:
            print(len(link.geometry))
            continue
        kwargs = trimesh.exchange.dae.load_collada(link.geometry[0].filename)
        # kwargs = trimesh.exchange.dae.load_collada(filename)
        mesh = trimesh.exchange.load.load_kwargs(kwargs)
        meshes[link.name] = mesh.dump(concatenate=True)
    # Hyperparameters
    dt = 1
    nq = 50
    lmbda = 1000
    eta = 1000
    iters = 4
    num_pts = 10
    # Make cost field, starting & end points
    cdim = len(qtraj.q[0])
    xidim = cdim * nq
    AA = np.zeros((xidim, xidim))
    xi = np.zeros(xidim)
    # robot._set_link_fk(qtraj.q[1])
    qs = np.zeros(cdim)
    qe = q_pickup
    bb = np.zeros(xidim)
    bb[: cdim] = qs
    bb[-cdim:] = qe
    xi[: cdim] = qs
    bb /= - dt * dt * (nq + 1)
    # Make a matrix
    for i in range(nq):
        AA[cdim * i: cdim * (i+1), cdim * i: cdim * (i+1)] = 2 * np.eye(cdim)
        if i > 0:
            AA[cdim * (i-1): cdim * i, cdim * i: cdim * (i+1)] = -1 * np.eye(cdim)
            AA[cdim * i: cdim * (i+1), cdim * (i-1): cdim * i] = -1 * np.eye(cdim)
    AA /= dt * dt * (nq + 1)
    Ainv = np.linalg.pinv(AA)

    for t in range(iters):
        nabla_smooth = AA.dot(xi) + bb
        nabla_obs = np.zeros(xidim)
        xidd = AA.dot(xi)
        total_cost = 0
        
        for i in range(nq): # timestep
            qt = xi[cdim * i: cdim * (i+1)]
            # qt = qe
            if i == 0:
                qd = 0.5 * (xi[cdim * (i+1): cdim * (i+2)] - qs)
            elif i == nq - 1:
                qd = 0.5 * (qe - xi[cdim * (i-1): cdim * (i)])
            else:
                qd = 0.5 * (xi[cdim * (i+1): cdim * (i+2)] - xi[cdim * (i-1): cdim * (i)])

            for il, link in enumerate(robot.links): # bodypart
                k = link.jindex
                if k is None:
                    continue

                link_base = robot.fkine(qt, end=link) # x_current, (4, 4)
                delta_nabla_obs = np.zeros(k + 1)
                mesh = meshes[link.name]
                idxs = np.random.choice(np.arange(len(mesh.vertices)), num_pts, replace=False)
                pts = mesh.vertices[idxs]
                pts_se3 = vrepeat(np.eye(4), num_pts)
                pts_se3[:, :3, 3] = pts
                pts_xyz = link_base.A.dot(pts_se3).swapaxes(0, 1)[:, :3, 3]

                path, njoints, _ = robot.get_path(end=link)
                jacobs1 = np.zeros((num_pts, 6, njoints))
                jacobs2 = np.zeros((num_pts, 6, njoints))
                jacob0_pts_vec(robot, link, pts, jacobs1, qt)
                jacob0_pts_loop(robot, link, pts, jacobs2, qt) # (N, 6, k + 1)
                assert np.allclose(jacobs1, jacobs2)
