

## Make chomp optimization

import importlib
import roboticstoolbox as rtb
from spatialmath import SE3
from spatialgeometry import *
import numpy as np
import trimesh
import time
import math
from numba import vectorize 

visualize = True
use_mesh = True
parallel = False
fknm_ = None


def jacob0_loop(robot, link, pts, qt=None):
    """ 
    Non-parallel, use for loop
    :param pts: (num, 3) xyz positions of pts
    :param q: (cdim,) joint configurations
    """
    if qt is None:
        qt = robot.q
    jacob_vec = []
    for pt in pts: 
        JJ = robot.jacob0(qt, end=link, tool=SE3(pt)) # (k, 6)                
        jacob_vec.append(JJ)
    return np.array(jacob_vec)

def jacob0_vec(robot, link, pts, qt=None, verbose=False):
    """ 
    Parallel, use CUDA
    :param pts: (num, 3) xyz positions of pts
    :param q: (cdim,) joint configurations
    """
    import ctypes as ct
    global fknm_
    if qt is None:
        qt = robot.q
    if fknm_ is None:
        fknm_=np.ctypeslib.load_library('roboticstoolbox/cuda/fknm','.')
    # Parallel, use cuda
    t_start = time.time()
    num_pts = len(pts)
    se3_pts = SE3(pts)    
    pts_tool = np.array(se3_pts.A)
    link_base = robot.fkine(qt, end=link)
    print(f"1: {time.time() - t_start:.3f}\n")
    t_start = time.time()
    # pts_mat = np.array((link_base @ se3_pts).A)
    pts_mat = np.array(link_base.A.dot(np.array(se3_pts.A)).swapaxes(0, 1), order='C')
    e_pts = np.zeros((num_pts, 3))
    pts_etool = np.array(SE3(e_pts).A)    
    print(f"2: {time.time() - t_start:.3f}\n")
    t_start = time.time()
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
    link_As = np.array(link_As)
    link_axes = np.array(link_axes, dtype=int)
    link_isjoint = np.array(link_isjoint, dtype=int)
    jacob_vec = np.zeros((num_pts, 6, njoints))

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
        print(f"num_pts {num_pts}")

    fknm_.jacob0(pts_mat.ctypes.data_as(ct.c_void_p),
                 pts_tool.ctypes.data_as(ct.c_void_p),
                 pts_etool.ctypes.data_as(ct.c_void_p),
                 # link_As.ctypes.data_as(ct.c_void_p),
                 ct.c_void_p(link_As.ctypes.data),
                 link_axes.ctypes.data_as(ct.c_void_p),
                 link_isjoint.ctypes.data_as(ct.c_void_p),
                 ct.c_int(num_pts),
                 ct.c_int(nlinks),
                 ct.c_int(njoints),
                 jacob_vec.ctypes.data_as(ct.c_void_p))
    return jacob_vec

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
    obstacle = Box([0.3, 0.3, 0.3], base=SE3(0.5, 0.5, 0.5))
    pt = robot.closest_point(robot.q, obstacle)
    # robot.links[2].collision[0].get_data()
    # print(robot)    # display the model

    # print(len(qt.q))
    # for qk in qt.q:             # for each joint configuration on trajectory
    #     print(qk)


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
    num_pts = 5


    # Make cost field, starting & end points
    cdim = len(qtraj.q[0])
    xidim = cdim * nq
    AA = np.zeros((xidim, xidim))
    xi = np.zeros(xidim)
    robot._set_link_fk(qtraj.q[1])

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
            robot.q = xi[cdim * i: cdim * (i+1)]
            # print(t, i, xi[cdim * i: cdim * (i+1)])
            qt = xi[cdim * i: cdim * (i+1)]
            if i == 0:
                qd = 0.5 * (xi[cdim * (i+1): cdim * (i+2)] - qs)
            elif i == nq - 1:
                qd = 0.5 * (qe - xi[cdim * (i-1): cdim * (i)])
            else:
                qd = 0.5 * (xi[cdim * (i+1): cdim * (i+2)] - xi[cdim * (i-1): cdim * (i)])

            # import pdb; pdb.set_trace()
            for link in robot.links: # bodypart
                k = link.jindex
                if k is None:
                    continue

                link_base = robot.fkine(robot.q, end=link) # x_current, (4, 4)
                delta_nabla_obs = np.zeros(k + 1)
                mesh = meshes[link.name]
                idxs = np.random.choice(np.arange(len(mesh.vertices)), num_pts, replace=False)
                pts = mesh.vertices[idxs]

                if not parallel:
                    jacobs = jacob0_loop(robot, link, pts, qt)
                else:
                    jacobs = jacob0_vec(robot, link, pts, qt)

                # TODO: vectorize cost calculation
                for j in range(num_pts): 
                    # For each point: compute Jacobian, compute cost, compute cost gradient
                    pt_rel = mesh.vertices[j]
                    pt_tool = link_base @ SE3(pt_rel)
                    pt_pos = pt_tool.t

                    JJ = robot.jacob0(qt, end=link, tool=SE3(pt_rel)) # (k, 6)                
                    xd = JJ.dot(qd[:k+1]) # x_delta
                    vel = np.linalg.norm(xd)                 
                    xdn = xd / (vel + 1e-3) # speed normalized
                    xdd = JJ.dot(xidd[cdim * i: cdim * i + k + 1]) # second derivative of xi
                    prj = np.eye(6) - xdn[:, None].dot(xdn[:, None].T) # curvature vector (6, 6)
                    kappa = (prj.dot(xdd) / (vel ** 2 + 1e-3)) # (6,)

                    cost = np.sum(pt_pos)
                    total_cost += cost / num_pts
                    # delta := negative gradient of obstacle cost in work space, (6, cdim) 
                    delta = -1 * np.concatenate([[1, 1, 0], np.zeros(3)])
                    # for link in robot.links:
                    #     cost = get_link_cost(robot, meshes, link)
                    delta_nabla_obs += JJ.T.dot(vel).dot(prj.dot(delta) - cost * kappa)
                
                nabla_obs[cdim * i: cdim * i + k + 1] += (delta_nabla_obs / num_pts)


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


def get_vertices_xyz(robot, meshes, link, num=-1, parallel=True):
    """Return (num, 3) array of xyz positions of vertices on the mesh
    """
    if link.name not in meshes:
        return None
    mesh = meshes[link.name]
    base_se3 = robot.fkine(robot.q, end=link) # x_current, (4, 4) 
    if num < 0:
        num = len(mesh.vertices)
        idxs = np.arange(num)
    else:
        idxs = np.random.choice(np.arange(num, replace=True))

    offsets = np.repeat(np.eye(4)[np.newaxis], num, axis=0)
    xs = mesh.vertices[idxs]
    offsets[:, :3, 3] = xs
    if parallel:
        costs = []
        out_xs = np.dot(base_se3.data[0], offsets)[:3, :, -1].T
    else:
        pass
    # print(f"Out shape {out_xs.shape}")
    return out_xs


def get_link_cost(robot, meshes, link, num=-1, parallel=True):
    """Return gradient of the costs.
    """
    vertice_xyz = get_vertices_xyz(robot, meshes, link, num, parallel)

def test_parallel():
    robot = rtb.models.DH.Panda() # load Mesh version (for collisions)
    T = robot.fkine(robot.qz)  # forward kinematics
    T = SE3(0.7, 0.2, 0.1) * SE3.OA([0, 1, 0], [0, 0, -1])
    sol = robot.ikine_LM(T)         # solve IK
    q_pickup = sol.q
    robot = rtb.models.URDF.Panda()  # load URDF version of the Panda
    robot.q = q_pickup
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


    for idx in [2, 3, 4, 5, 6, 7]:
        print(f"Iidx {idx}")
        link = robot.links[idx]
        k = link.jindex


        # Non-parallel, use for loop
        mesh = meshes[link.name]
        pts = mesh.vertices[:num_pts]

        t_start = time.time()
        jacobs1 = jacob0_loop(robot, link, pts)
        print(f"\nNon-parallel time: {time.time() - t_start:.3f}\n")
        # print(jacobs1)
        # Parallel, use cuda
        t_start = time.time()
        jacobs2 = jacob0_vec(robot, link, pts, verbose=True)
        print(f"\nParallel time: {time.time() - t_start:.3f}\n")
        # print(jacobs2)
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
    # chomp()
    test_parallel()