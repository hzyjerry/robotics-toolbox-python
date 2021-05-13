

## Make chomp optimization

import importlib
import roboticstoolbox as rtb
from roboticstoolbox.tools.profiler import Profiler
from spatialmath import SE3
from spatialgeometry import *
import numpy as np
import trimesh
import time
import math
from numba import vectorize 

visualize = True
use_mesh = True
parallel_pts = False
parallel_links = False
parallel_Ts = True
# parallel_pts = False
# parallel_links = True
# parallel_Ts = False

fknm_ = None


## TODO: parallelize timesteps
## TODO: parallelize links

def vmatmul(mat1, mat2):
    # mat1: (N, a, b)
    # mat2: (N, b)
    # out: (N, a)
    return np.matmul(mat1, mat2[:, :, None]).squeeze(-1)

def vmmatmul(mat1, mat2):
    # mat1: (N, a, ..., b)
    # mat2: (N, b)
    # out: (N, a, ...)
    mat2 = mat2[:, None, :, None]
    while len(mat2.shape) < len(mat1.shape):
        mat2 = mat2[:, None]
    return np.matmul(mat1, mat2).squeeze(-1)

def vrepeat(vec, N):
    # vec: (...)
    # out: (N, ...)
    return np.repeat(vec[None, :], N, axis=0)


def jacob0_pts_loop(robot, link, pts, jacob_vec, qt=None):
    """ 
    Non-parallel, use for loop
    :param pts: (num, 3) xyz positions of pts
    :param q: (cdim,) joint configurations
    :param jacob_vec: (num, 6, njoints)
    """
    if qt is None:
        qt = robot.q
    for ip, pt in enumerate(pts): 
        JJ = robot.jacob0(qt, end=link, tool=SE3(pt)) # (k, 6)                
        jacob_vec[ip] = JJ

def jacob0_pts_vec(robot, link, pts, jacob_vec, qt=None, verbose=False):
    """ 
    Parallel, use CUDA
    :param pts: (num, 3) xyz positions of pts
    :param q: (cdim,) joint configurations
    :param jacob_vec: (num, 6, njoints)
    """
    import ctypes as ct
    global fknm_
    if qt is None:
        qt = robot.q
    if fknm_ is None:
        fknm_=np.ctypeslib.load_library('roboticstoolbox/cuda/fknm','.')
    # Parallel, use cuda
    N = len(pts)
    pts_tool = vrepeat(np.eye(4), N)
    pts_tool[:, :3, 3] = pts
    link_base = robot.fkine(qt, end=link)
    # pts_mat = np.array((link_base @ se3_pts).A)
    # pts_mat = np.array(link_base.A.dot(pts_tool).swapaxes(0, 1), order='C')
    # pts_mat = np.einsum('ij,ljk->lik', link_base.A, pts_tool)
    # pts_mat = np.ascontiguousarray(link_base.A.dot(pts_tool).swapaxes(0, 1))
    pts_mat = np.matmul(vrepeat(link_base.A, N), pts_tool)

    # e_pts = np.zeros((N, 3))
    # pts_etool = np.array(SE3(e_pts).A)
    pts_etool = vrepeat(np.eye(4), N)
    link_As = []
    link_axes = []
    link_isjoint = []
    path, njoints, _ = robot.get_path(end=link)
    nlinks = len(path)
    for il, link in enumerate(path):
        axis = get_axis(link)
        link_As.append(link.A(qt[link.jindex]).A)
        link_axes.append(axis)
        link_isjoint.append(link.isjoint)
    # print(nlinks)
    nlinks_pt = np.ones(N, dtype=int) * nlinks
    link_As = np.array(link_As)
    link_axes = np.array(link_axes, dtype=int)
    link_isjoint = np.array(link_isjoint, dtype=int)

    if verbose:
        print(f"pts shape {pts_mat.shape}")
        print(f"pts_tool shape {pts_tool.shape}")
        print(f"pts_etool shape {pts_etool.shape}")
        print(f"link_As shape {link_As.shape}")
        print(f"link_axes shape {link_axes.shape}")
        print(f"link_isjoint shape {link_isjoint.shape}")
        print(f"jacob_vec shape {jacob_vec.shape}")
        print(f"nlinks {nlinks}")
        print(f"njoints {njoints}")
        print(f"N {N}")

    with Profiler("vec 2", enable=False):
    # with Profiler("vec 2"):
        fknm_.jacob0(pts_mat.ctypes.data_as(ct.c_void_p),
                     pts_tool.ctypes.data_as(ct.c_void_p),
                     pts_etool.ctypes.data_as(ct.c_void_p),
                     # link_As.ctypes.data_as(ct.c_void_p),t 
                     ct.c_void_p(link_As.ctypes.data), 
                     nlinks_pt.ctypes.data_as(ct.c_void_p),
                     link_axes.ctypes.data_as(ct.c_void_p),
                     link_isjoint.ctypes.data_as(ct.c_void_p),
                     ct.c_int(N),
                     ct.c_int(nlinks),
                     ct.c_int(njoints),
                     jacob_vec.ctypes.data_as(ct.c_void_p))


def jacob0_vec(robot, nq, njoints, link_bases, pts, jacob_vec, qt=None, verbose=False):
    """ 
    Parallel, use CUDA
    :param link_bases: (nq * njoints, 4, 4)
    :param pts: (nq * njoints * num_pts, 3) xyz positions of pts
    :param q: (cdim,) joint configurations
    :param jacob_vec: (nq * njoints * num_pts, 6, njoints)
    """
    import ctypes as ct
    global fknm_
    if qt is None:
        qt = robot.q
    if fknm_ is None:
        fknm_=np.ctypeslib.load_library('roboticstoolbox/cuda/fknm','.')
    # Parallel, use cuda
    N = len(pts)
    num_pts = int(N / len(link_bases))
    pts_tool = vrepeat(np.eye(4), N)
    pts_tool[:, :3, 3] = pts
    r_bases = np.repeat(link_bases[:, None], num_pts, axis=1) # (nq * njoints, num_pts, 4, 4)
    r_bases = r_bases.reshape(-1, 4, 4) # (nq * njoints * num_pts, 4, 4)
    pts_mat = np.matmul(r_bases, pts_tool) # (nq * njoints * num_pts, 4, 4)
    pts_etool = vrepeat(np.eye(4), N)
    link_As = []
    link_axes = []
    link_isjoint = []
    path, njoints, _ = robot.get_path()
    nlinks = len(path)
    nlinks_pt = []
    for il, link in enumerate(path):
        axis = get_axis(link)
        link_As.append(link.A(qt[link.jindex]).A)
        link_axes.append(axis)
        link_isjoint.append(link.isjoint)
    # print(nlinks)
    link_As = np.array(link_As)
    link_axes = np.array(link_axes, dtype=int)
    link_isjoint = np.array(link_isjoint, dtype=int)
    nlinks_pt = np.tile(np.arange(1, njoints + 1, dtype=int).repeat(num_pts), nq)

    if verbose:
        print(f"pts shape {pts_mat.shape}")
        print(f"pts_tool shape {pts_tool.shape}")
        print(f"pts_etool shape {pts_etool.shape}")
        print(f"link_As shape {link_As.shape}")
        print(f"link_axes shape {link_axes.shape}")
        print(f"link_isjoint shape {link_isjoint.shape}")
        print(f"jacob_vec shape {jacob_vec.shape}")
        print(f"nlinks {nlinks}")
        print(f"njoints {njoints}")
        print(f"N {N}")

    with Profiler("vec 2", enable=False):
    # with Profiler("vec 2"):
        fknm_.jacob0(pts_mat.ctypes.data_as(ct.c_void_p),
                     pts_tool.ctypes.data_as(ct.c_void_p),
                     pts_etool.ctypes.data_as(ct.c_void_p),
                     ct.c_void_p(link_As.ctypes.data),  
                     nlinks_pt.ctypes.data_as(ct.c_void_p),
                     link_axes.ctypes.data_as(ct.c_void_p),
                     link_isjoint.ctypes.data_as(ct.c_void_p),
                     ct.c_int(N),
                     ct.c_int(nlinks),
                     ct.c_int(njoints),
                     jacob_vec.ctypes.data_as(ct.c_void_p))

def chomp(seed=0):
    np.random.seed(seed)

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
                print(len(link.geometry))
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
    num_pts = 200
    # num_pts = 1

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

        if parallel_Ts:
            link_base_all = []
            pts_all = []
            pts_xyz_all = []
            for i in range(nq): # timestep
                qt = xi[cdim * i: cdim * (i+1)]
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
            pts_all = np.concatenate(pts_all) # (nq * cdim * npoints, 3)
            pts_xyz_all = np.concatenate(pts_xyz_all) # (nq * cdim * npoints, 3)
            path, njoints, _ = robot.get_path()
            jacobs = np.zeros((len(pts_all), 6, njoints)) # (nq * cdim * npoints, 6, njoints)
            jacob0_vec(robot, nq, njoints, link_base_all, pts_all, jacobs, qt)
            xd = jacobs.dot(qd) # x_delta, (nq * cdim * npoints, 6)
            vel = np.linalg.norm(xd, axis=1, keepdims=True) # (nq * cdim * npoints, 1)
            xdn = xd / (vel + 1e-3) # speed normalized, (nq * cdim * npoints, 6)
            jacobs_ = jacobs.reshape(nq, njoints * num_pts, 6, njoints)
            xidd = xidd.reshape(nq, njoints) # (nq, njoints)
            xdd = vmmatmul(jacobs_, xidd).reshape(-1, 6) # second derivative of xi, (nq * cdim * npoints, 6)
            prj = vrepeat(np.eye(6), nq * num_pts * njoints)
            prj -= np.matmul(xdn[:, :, None], xdn[:, None, :]) # curvature vector (nq * cdim * npoints, 6, 6)
            kappa = (vmatmul(prj, xdd) / (vel ** 2 + 1e-3)) # (nq * cdim * npoints, 6)
            cost = np.sum(pts_xyz_all, axis=1, keepdims=True) # (nq * cdim * npoints, 1)
            delta = -1 * np.concatenate([[1, 1, 0], np.zeros(3)])
            delta = vrepeat(delta, nq * num_pts * njoints) # (nq * cdim * npoints, 3, 6)
            part1 = jacobs.swapaxes(1, 2) # (nq * cdim * npoints, cdim, 6)
            part2 = vmatmul(prj, delta) - cost * kappa # (nq * cdim * npoints, 6)
            delta_nabla_obs = vmatmul(part1, part2) * vel # (nq * cdim * npoints, cdim)
            total_cost += cost.mean() * njoints * nq

            delta_nabla_obs = delta_nabla_obs.reshape(nq, cdim, num_pts, -1)
            # import pdb; pdb.set_trace()
            nabla_obs = delta_nabla_obs.mean(axis=(1, 2)).flatten() * njoints

            
            for i in range(nq): # timestep
                qt = xi[cdim * i: cdim * (i+1)]
                # qt = qe
                if i == 0:
                    qd = 0.5 * (xi[cdim * (i+1): cdim * (i+2)] - qs)
                elif i == nq - 1:
                    qd = 0.5 * (qe - xi[cdim * (i-1): cdim * (i)])
                else:
                    qd = 0.5 * (xi[cdim * (i+1): cdim * (i+2)] - xi[cdim * (i-1): cdim * (i)])
           
                link_base_t = []
                pts_t = []
                pts_xyz_t = []
                for il, link in enumerate(robot.links): # bodypart
                    k = link.jindex
                    if k is None:
                        continue
                    link_base = robot.fkine(qt, end=link) # x_current, (4, 4)
                    link_base_t.append(link_base.A)
                    delta_nabla_obs = np.zeros(k + 1)
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
                jacobs2 = np.zeros((len(pts_t), 6, njoints))
                jacob0_vec(robot, 1, njoints, link_base_t, pts_t, jacobs2, qt)
                # print(np.allclose(jacobs_[i], jacobs2))
                # print(i)
                try:
                    assert np.allclose(jacobs_[i], jacobs2)
                except:
                    print(np.max(np.abs(jacobs_[i] - jacobs2)))
                    import pdb; pdb.set_trace()
                print(np.max(np.abs(jacobs_[i] - jacobs2)))
                # # import pdb; pdb.set_trace()
                # xd = jacobs.dot(qd) # x_delta, (N, 6)
                # vel = np.linalg.norm(xd, axis=1, keepdims=True) # (N, 1)
                # xdn = xd / (vel + 1e-3) # speed normalized, (N, 6)
                # xdd = jacobs.dot(xidd[cdim * i: cdim * (i + 1)]) # second derivative of xi, (N, 6)
                # prj = vrepeat(np.eye(6), num_pts * njoints)
                # prj -= np.matmul(xdn[:, :, None], xdn[:, None, :]) # curvature vector (N, 6, 6)
                # kappa = (vmatmul(prj, xdd) / (vel ** 2 + 1e-3)) # (N, 6)
                # cost = np.sum(pts_xyz_t, axis=1, keepdims=True) # (N, 1)
                # delta = -1 * np.concatenate([[1, 1, 0], np.zeros(3)])
                # delta = vrepeat(delta, num_pts * njoints) # (N, 3, 6)
                # part1 = jacobs.swapaxes(1, 2) # (N, cdim, 6)
                # part2 = vmatmul(prj, delta) - cost * kappa # (N, 6)
                # delta_nabla_obs = vmatmul(part1, part2) * vel # (N, cdim)

                # total_cost += cost.mean() * njoints
                # nabla_obs[cdim * i: cdim * (i + 1)] = delta_nabla_obs.mean(axis=0) * njoints

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

                if parallel_links: ## Concate all links together

                    link_base_t = []
                    pts_t = []
                    pts_xyz_t = []
                    for il, link in enumerate(robot.links): # bodypart
                        k = link.jindex
                        if k is None:
                            continue
                        link_base = robot.fkine(qt, end=link) # x_current, (4, 4)
                        link_base_t.append(link_base.A)
                        delta_nabla_obs = np.zeros(k + 1)
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
                    jacobs = np.zeros((len(pts_t), 6, njoints))
                    jacob0_vec(robot, 1, njoints, link_base_t, pts_t, jacobs, qt)
                    # import pdb; pdb.set_trace()
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

                        link_base = robot.fkine(qt, end=link) # x_current, (4, 4)
                        delta_nabla_obs = np.zeros(k + 1)
                        mesh = meshes[link.name]
                        idxs = np.random.choice(np.arange(len(mesh.vertices)), num_pts, replace=False)
                        pts = mesh.vertices[idxs]
                        pts_se3 = vrepeat(np.eye(4), num_pts)
                        pts_se3[:, :3, 3] = pts
                        pts_xyz = link_base.A.dot(pts_se3).swapaxes(0, 1)[:, :3, 3]

                        # with Profiler("Step 1"):
                        with Profiler("Step 1", enable=False):
                            path, njoints, _ = robot.get_path(end=link)
                            jacobs = np.zeros((num_pts, 6, njoints))
                            if parallel_pts:
                                jacob0_pts_vec(robot, link, pts, jacobs, qt)
                            else:
                                jacob0_pts_loop(robot, link, pts, jacobs, qt) # (N, 6, k + 1)
                            
                        # with Profiler("Step 2"):
                        with Profiler("Step 2", enable=False):
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
        import pdb; pdb.set_trace()
        print(f"\nParallel time: {time.time() - t_start:.3f}\n")
        print(jacobs2)
        # print(f"Max diff 0 {np.max(np.abs(jacobs1[0] - jacobs2[0]))}")
        # print(f"Max diff 0 {np.argmax(np.abs(jacobs1 - jacobs2))}")
        # print(f"Max diff {np.max(np.abs(jacobs1 - jacobs2))}")
        # import pdb; pdb.set_trace()
        assert np.all(np.isclose(jacobs1, jacobs2))


def get_axis(link):
    axis = None
    if not link.isjoint:
        axis = -1
    elif link._v.axis == "Rx":
        axis = 0
    elif link._v.axis == "Ry":
        axis = 1
    elif link._v.axis == "Rz":
        axis = 2
    elif link._v.axis == "tx":
        axis = 3
    elif link._v.axis == "ty":
        axis = 4
    elif link._v.axis == "tz":
        axis = 5
    else:
        raise NotImplementedError
    return axis


if __name__ == "__main__":
    chomp()
    # test_parallel()