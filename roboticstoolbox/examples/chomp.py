

## Make chomp optimization

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


MODES = {"PTS", "LINKS", "TS", "NONE"}
"""
Speed:

PTS: num_pts=50, block_size=768, nq=50 
    Iteration PTS: complete: 0.90 fps, 1.10855 seconds
    Before: complete: 31.35 fps, 0.03190 seconds
    Main: complete: 6.25 fps, 0.16009 seconds
    After: complete: 73.19 fps, 0.01366 seconds
LINKS: num_pts=50, block_size=768, nq=50 
    Iteration LINKS: complete: 3.28 fps, 0.30511 seconds
    50 *
    Before: complete: 1538.63 fps, 0.00065 seconds
    Main: complete: 258.97 fps, 0.00386 seconds
    After: complete: 2145.42 fps, 0.00047 seconds
LINKS: num_pts=50, block_size=768, nq=50 
    Iteration TS: complete: 7.45 fps, 0.13419 seconds
    350 *
    Before: complete: 10699.76 fps, 0.00009 seconds
    Main: complete: 411.97 fps, 0.00243 seconds
    After: complete: 6403.52 fps, 0.00016 seconds


"""

fknm_ = None

def chomp(mode, seed=0, num_pts=1, nq=10):
    np.random.seed(seed)
    assert mode in MODES
    print(f"CHOMP Mode {mode}")

    robot = rtb.models.DH.Panda() # load Mesh version (for collisions)
    # robot = rtb.models.URDF.Panda()  # load URDF version of the Panda (for visuals)
    # print(robot)

    T = robot.fkine(robot.qz)  # forward kinematics
    T = SE3(0.7, 0.2, 0.1) * SE3.OA([0, 1, 0], [0, 0, -1])
    sol = robot.ikine_LM(T)         # solve IK
    # print(sol)
    q_pickup = sol.q
    # print(robot.fkine(q_pickup))    # FK shows that desired end-effector pose was achieved
    qtraj = rtb.jtraj(robot.qz, q_pickup, 50)
    # robot.plot(qt.q, movie='panda1.gif')
    robot = rtb.models.URDF.Panda()  # load URDF version of the Panda
    obstacle = Box([0.3, 0.3, 0.3], base=SE3(0.5, 0.5, 0.8))
    pt = robot.closest_point(robot.q, obstacle)

    meshes = {}
    if use_mesh:
        for link in robot.links:
            if len(link.geometry) != 1:
                continue
            kwargs = trimesh.exchange.dae.load_collada(link.geometry[0].filename)
            # kwargs = trimesh.exchange.dae.load_collada(filename)
            mesh = trimesh.exchange.load.load_kwargs(kwargs)
            meshes[link.name] = mesh.dump(concatenate=True)
            if verbose:
                print(link.name, mesh)


    # Hyperparameters
    dt = 1
    lmbda = 1000
    eta = 1000
    iters = 4

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
        with Profiler(f"Mode {mode} iteration"):
            nabla_smooth = AA.dot(xi) + bb
            nabla_obs = np.zeros(xidim)
            xidd = AA.dot(xi)
            total_cost = 0

            pts_all = []
            for il, link in enumerate(robot.links):
                k = link.jindex
                if k is None:
                    continue
                mesh = meshes[link.name]
                idxs = np.random.choice(np.arange(len(mesh.vertices)), num_pts, replace=False)
                pts = mesh.vertices[idxs]
                pts_all.append(pts)

            if mode == "TS":
                with Profiler(f"Before", enable=verbose):
                    link_base_all = []
                    pts_xyz_all = []
                    qd_all = []
                    jacobs2s = []
                    robot_idxs = {}
                    pts_all_ = []
                                
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
                            pts_se3 = vrepeat(np.eye(4), num_pts)
                            pts = pts_all[k]
                            pts_all_.append(pts)
                            pts_se3[:, :3, 3] = pts
                            pts_xyz = link_base.A.dot(pts_se3).swapaxes(0, 1)[:, :3, 3]
                            pts_xyz_all.append(pts_xyz)

                    link_base_all = np.stack(link_base_all) # (nq * cdim, 4, 4)
                    # import pdb; pdb.set_trace()
                    pts_all_ = np.concatenate(pts_all_) # (nq * cdim * num_pts, 3)
                    pts_xyz_all = np.concatenate(pts_xyz_all) # (nq * cdim * num_pts, 3)
                    qd_all = np.stack(qd_all) # (nq, cdim)
                    path, njoints, _ = robot.get_path()
                    assert njoints == cdim
                    jacobs = np.zeros((len(pts_all_), 6, cdim)) # (nq * cdim * num_pts, 6, cdim)
                with Profiler(f"Main", enable=verbose):
                    jacob0_vec(robot, nq, cdim, link_base_all, pts_all_, jacobs, xi) # IMPORTANT: pass xi
                with Profiler(f"After", enable=verbose):
                    jacobs = jacobs.reshape(nq, cdim * num_pts, 6, cdim)
                    jacobs_ = jacobs.reshape(nq, cdim, num_pts, 6, cdim) # (nq, cdim, num_pts, 6, cdim)
                    xd = vmmatmul(jacobs_, qd_all).reshape(-1, 6) # x_delta, (nq * cdim * num_pts, 6)
                    jacobs = jacobs_.reshape(-1, 6, cdim) # (nq * cdim * num_pts, 6, cdim)
                    vel = np.linalg.norm(xd, axis=-1, keepdims=True) # (nq * cdim * npoints, 1)
                    xdn = xd / (vel + 1e-3) # speed normalized, (nq * cdim * num_pts, 6)
                    xidd_ = xidd.reshape(nq, cdim) # (nq, cdim)
                    xdd = vmmatmul(jacobs_, xidd_).reshape(-1, 6) # second derivative of xi, (nq * cdim * num_pts, 6)
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
                    # import pdb; pdb.set_trace()
                    nabla_obs = delta_nabla_obs.mean(axis=(1, 2)).flatten() * cdim

            else:
                for i in range(nq): # timestep
                    qt = xi[cdim * i: cdim * (i+1)]
                    # qt = qe
                    if i == 0:
                        qd = 0.5 * (xi[cdim * (i+1): cdim * (i+2)] - qs)
                    elif i == nq - 1:
                        qd = 0.5 * (qe - xi[cdim * (i-1): cdim * (i)])
                    else:
                        qd = 0.5 * (xi[cdim * (i+1): cdim * (i+2)] - xi[cdim * (i-1): cdim * (i)])

                    if mode == "LINKS": ## Concate all links together
                        with Profiler("Before", enable=verbose):
                            link_base_t = []
                            pts_t = []
                            pts_xyz_t = []
                            for il, link in enumerate(robot.links): # bodypart
                                k = link.jindex
                                if k is None:
                                    continue
                                link_base = robot.fkine(qt, end=link) # x_current, (4, 4)
                                link_base_t.append(link_base.A)
                                pts = pts_all[k]
                                pts_t.append(pts)
                                pts_se3 = vrepeat(np.eye(4), num_pts)
                                pts_se3[:, :3, 3] = pts
                                pts_xyz = link_base.A.dot(pts_se3).swapaxes(0, 1)[:, :3, 3]
                                pts_xyz_t.append(pts_xyz)
                            link_base_t = np.stack(link_base_t) # (cdim, 4, 4)
                            pts_t = np.concatenate(pts_t) # (cdim * npoints, 3)
                            pts_xyz_t = np.concatenate(pts_xyz_t)
                            path, njoints, _ = robot.get_path()
                            jacobs = np.zeros((len(pts_t), 6, njoints))
                        with Profiler("Main", enable=verbose):
                            jacob0_vec(robot, 1, njoints, link_base_t, pts_t, jacobs, qt)
                        with Profiler("After", enable=verbose):
                            xd = jacobs.dot(qd) # x_delta, (N, 6)
                            vel = np.linalg.norm(xd, axis=1, keepdims=True) # (N, 1)
                            xdn = xd / (vel + 1e-3) # speed normalized, (N, 6)
                            xdd = jacobs.dot(xidd[cdim * i: cdim * (i + 1)]) # second derivative of xi, (N, 6)
                            prj = vrepeat(np.eye(6), num_pts * njoints)
                            prj -= np.matmul(xdn[:, :, None], xdn[:, None, :]) # curvature vector (N, 6, 6)
                            kappa = (vmatmul(prj, xdd) / (vel ** 2 + 1e-3)) # (N, 6)
                            cost = np.sum(pts_xyz_t, axis=1, keepdims=True) # (N, 1)
                            delta = -1 * np.concatenate([[1, 1, 0], np.zeros(3)])
                            delta = vrepeat(delta, num_pts * njoints) # (N, 3, 6)
                            part1 = jacobs.swapaxes(1, 2) # (N, cdim, 6)
                            part2 = vmatmul(prj, delta) - cost * kappa # (N, 6)
                            delta_nabla_obs = vmatmul(part1, part2) * vel # (N, cdim)

                            total_cost += cost.mean() * njoints
                            nabla_obs[cdim * i: cdim * (i + 1)] = delta_nabla_obs.mean(axis=0) * njoints

                    else: ## Calculate links in for loop
                        
                        for il, link in enumerate(robot.links): # bodypart
                            k = link.jindex
                            if k is None:
                                continue

                            with Profiler("Before", enable=verbose):
                                link_base = robot.fkine(qt, end=link) # x_current, (4, 4)
                                delta_nabla_obs = np.zeros(k + 1)
                                pts = pts_all[k]
                                pts_se3 = vrepeat(np.eye(4), num_pts)
                                pts_se3[:, :3, 3] = pts
                                pts_xyz = link_base.A.dot(pts_se3).swapaxes(0, 1)[:, :3, 3]

                            with Profiler("Main", enable=verbose):
                                path, njoints, _ = robot.get_path(end=link)
                                jacobs = np.zeros((num_pts, 6, njoints))
                                if mode == "PTS":
                                    jacob0_pts_vec(robot, link, pts, jacobs, qt)
                                else:
                                    jacob0_pts_loop(robot, link, pts, jacobs, qt) # (N, 6, k + 1)
                                
                            with Profiler("After", enable=verbose):
                                xd = jacobs.dot(qd[:k+1]) # x_delta, (N, 6)
                                vel = np.linalg.norm(xd, axis=1, keepdims=True) # (N, 1)
                                xdn = xd / (vel + 1e-3) # speed normalized, (N, 6)
                                xdd = jacobs.dot(xidd[cdim * i: cdim * i + k + 1]) # second derivative of xi, (N, 6)
                                prj = vrepeat(np.eye(6), num_pts)
                                prj -= np.matmul(xdn[:, :, None], xdn[:, None, :]) # curvature vector (N, 6, 6)
                                kappa = (vmatmul(prj, xdd) / (vel ** 2 + 1e-3)) # (N, 6)
                                cost = np.sum(pts_xyz, axis=1, keepdims=True) # (N, 1)
                                # delta := negative gradient of obstacle cost in work space, (6, cdim) 
                                delta = -1 * np.concatenate([[1, 1, 0], np.zeros(3)])
                                delta = vrepeat(delta, num_pts) # (N, 3, 6)
                                # for link in robot.links:
                                #     cost = get_link_cost(robot, meshes, link)
                                part1 = jacobs.swapaxes(1, 2) # (N, k + 1, 6)
                                part2 = vmatmul(prj, delta) - cost * kappa # (N, 6)
                                delta_nabla_obs = vmatmul(part1, part2) * vel # (N, k + 1)
                                total_cost += cost.mean()

                            nabla_obs[cdim * i: cdim * i + k + 1] += delta_nabla_obs.mean(axis=0)

            # dxi = Ainv.dot(lmbda * nabla_smooth)
            dxi = Ainv.dot(nabla_obs + lmbda * nabla_smooth)
            xi -= dxi / eta
            if verbose:
                print(f"Iteration {t} total cost {total_cost}")


    if visualize:
        from roboticstoolbox.backends.swift import Swift  # instantiate 3D browser-based visualizer
        backend = Swift()
        backend.launch()            # activate it

        robot.q = qs
        backend.add(robot)          # add robot to the 3D scene
        backend.add(obstacle)
        backend.step()
        
        print("start")
        dt = 0.07
        marker = None
        marker_pos = None

        for i in range(nq):
            if verbose:
                print(i, xi[cdim * i: cdim * (i+1)])
            robot.q = xi[cdim * i: cdim * (i+1)]          # update the robot state

            if i == 0:
                link = robot.links[3]
                mesh = meshes[link.name]
                marker_pos = mesh.vertices[0]
                tool = SE3(marker_pos)
                xx_base = robot.fkine(robot.q, end=link) # x_current, (4, 4)
                xx = xx_base @ tool
                # xx = robot.fkine(robot.q, end=link, tool=tool) # x_current, (4, 4)
                # xx = get_link_cost(robot, meshes, link)
                marker = Sphere(0.02, base=SE3(xx))
                backend.add(marker)
                backend.step()
            else:
                tool = SE3(marker_pos)
                # xx = robot.fkine(robot.q, end=link, tool=tool) # x_current, (4, 4)
                xx_base = robot.fkine(robot.q, end=link) # x_current, (4, 4)
                xx = xx_base @ tool
                # xx = get_link_cost(robot, meshes, link)
                marker.base = SE3(xx)

            backend.step()        # update visualization
            time.sleep(dt)
        robot.q = qe
        backend.step()
        time.sleep(dt)
        backend.close()
        # for qk in qtraj.q:             # for each joint configuration on trajectory
        #     robot.q = qk
        #     backend.step()
        #     time.sleep(dt)


def test_parallel(seed=0):
    np.random.seed(seed)
    robot = rtb.models.DH.Panda() # load Mesh version (for collisions)
    T = robot.fkine(robot.qz)  # forward kinematics
    T = SE3(0.7, 0.2, 0.1) * SE3.OA([0, 1, 0], [0, 0, -1])
    sol = robot.ikine_LM(T)         # solve IK
    q_pickup = sol.q
    robot = rtb.models.URDF.Panda()  # load URDF version of the Panda
    meshes = {}
    for link in robot.links:
        if len(link.geometry) != 1:
            # print(len(link.geometry))
            continue
        kwargs = trimesh.exchange.dae.load_collada(link.geometry[0].filename)
        # kwargs = trimesh.exchange.dae.load_collada(filename)
        mesh = trimesh.exchange.load.load_kwargs(kwargs)
        meshes[link.name] = mesh.dump(concatenate=True)
        print(link.name, mesh)

    # Hyperparameters
    dt = 1
    nq = 50
    lmbda = 1000
    eta = 1000
    iters = 4
    num_pts = 2000


    # Make cost field, starting & end points
    cdim = len(robot.q)
    xidim = cdim * nq
    AA = np.zeros((xidim, xidim))
    xi = np.zeros(xidim)

    # for idx in [2, 3, 4, 5, 6, 7]:
    for idx in [2]:
        print(f"Iidx {idx}")
        link = robot.links[idx]
        k = link.jindex


        # Non-parallel, use for loop
        mesh = meshes[link.name]
        idxs = np.random.choice(np.arange(len(mesh.vertices)), num_pts, replace=False)
        pts = mesh.vertices[idxs]        
        path, njoints, _ = robot.get_path(end=link)


        t_start = time.time()
        jacobs1 = np.zeros((num_pts, 6, njoints))
        jacob0_pts_loop(robot, link, pts, jacobs1, q_pickup)
        print(f"\nNon-parallel time: {time.time() - t_start:.3f}\n")
        # print(jacobs1)
        # Parallel, use cuda
        t_start = time.time()
        jacobs2 = np.zeros((num_pts, 6, njoints))
        jacob0_pts_vec(robot, link, pts, jacobs2, q_pickup, verbose=True)
        print(f"\nParallel time: {time.time() - t_start:.3f}\n")
        print(jacobs2)
        # print(f"Max diff 0 {np.max(np.abs(jacobs1[0] - jacobs2[0]))}")
        # print(f"Max diff 0 {np.argmax(np.abs(jacobs1 - jacobs2))}")
        # print(f"Max diff {np.max(np.abs(jacobs1 - jacobs2))}")
        # import pdb; pdb.set_trace()
        assert np.all(np.isclose(jacobs1, jacobs2))




if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--pts', type=int, default=50)
    parser.add_argument('--nq', type=int, default=10)
    parser.add_argument('--vb', action="store_true", default=False)

    opt = parser.parse_args()
    visualize = True
    verbose = opt.vb
    num_pts = opt.pts
    nq = opt.nq
    use_mesh = True

    chomp(mode="NONE", num_pts=num_pts, nq=nq)
    chomp(mode="PTS", num_pts=num_pts, nq=nq)
    chomp(mode="LINKS", num_pts=num_pts, nq=nq)
    chomp(mode="TS", num_pts=num_pts, nq=nq)
    # test_parallel()