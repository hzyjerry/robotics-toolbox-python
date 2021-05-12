

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


def chomp():
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

            for link in robot.links: # bodypart
                k = link.jindex
                if k is None:
                    continue

                link_base = robot.fkine(robot.q, end=link) # x_current, (4, 4)
                delta_nabla_obs = np.zeros(k + 1)
                if not parallel:
                    mesh = meshes[link.name]
                    for j in range(num_pts): 
                        # For each point: compute Jacobian, compute cost, compute cost gradient

                        pt_rel = mesh.vertices[j]
                        pt_tool = link_base @ SE3(pt_rel)
                        pt_pos = pt_tool.t

                        JJ = robot.jacob0(qt, end=link, tool=SE3(pt_rel)) # (k, 6)                
                        import pdb; pdb.set_trace()
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
                else:
                    pass
                
        # dxi = Ainv.dot(lmbda * nabla_smooth)
        dxi = Ainv.dot(nabla_obs + lmbda * nabla_smooth)
        xi -= dxi / eta
        print(f"Iteration {t} total cost {total_cost}")

    # @vectorize(['float32(float32)'], target='cuda')
    # def vec_jacob0()

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
                xx = get_link_cost(robot, meshes, link)
                marker = Sphere(0.02, base=SE3(xx))
                backend.add(marker)
                backend.step()
            else:
                tool = SE3(marker_pos)
                # xx = robot.fkine(robot.q, end=link, tool=tool) # x_current, (4, 4)
                xx_base = robot.fkine(robot.q, end=link) # x_current, (4, 4)
                xx = xx_base @ tool
                xx = get_link_cost(robot, meshes, link)
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



if __name__ == "__main__":
    chomp()