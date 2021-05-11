

## Make chomp optimization


import roboticstoolbox as rtb
from spatialmath import SE3
from spatialgeometry import *
import numpy as np

visualize = True

robot = rtb.models.DH.Panda()
# print(robot)

T = robot.fkine(robot.qz)  # forward kinematics
# print(T)
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
# import pdb; pdb.set_trace()
# robot.links[2].collision[0].get_data()
# import pdb; pdb.set_trace()
# print(robot)    # display the model

# print(len(qt.q))
# for qk in qt.q:             # for each joint configuration on trajectory
#     print(qk)


# Make a matrix


# Make cost field
dt = 1
nq = 50
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

for i in range(nq):
    AA[cdim * i: cdim * (i+1), cdim * i: cdim * (i+1)] = 2 * np.eye(cdim)
    if i > 0:
        AA[cdim * (i-1): cdim * i, cdim * i: cdim * (i+1)] = -1 * np.eye(cdim)
        AA[cdim * i: cdim * (i+1), cdim * (i-1): cdim * i] = -1 * np.eye(cdim)
AA /= dt * dt * (nq + 1)

Ainv = np.linalg.pinv(AA)

# import pdb; pdb.set_trace()

lmbda = 1
eta = 1
iters = 4
# import pdb; pdb.set_trace()

for t in range(iters):
    nabla_smooth = AA.dot(xi) + bb
    nabla_obs = np.zeros(xidim)
    xidd = AA.dot(xi)
    total_cost = 0
    for i in range(nq): # timestep
        robot.q = xi[cdim * i: cdim * (i+1)]
        # print(t, i, xi[cdim * i: cdim * (i+1)])
        for link in robot.links: # bodypart
            k = link.jindex
            if k is None:
                continue
            
            qt = xi[cdim * i: cdim * (i+1)]
            JJ = robot.jacob0(qt, end=link) # (k, 6)
            # print(i, JJ)
            tool = None # relative transformation to the last joint

            qq = xi[cdim * i: cdim * i + k + 1]
            if i == 0:
                qd = 0.5 * (xi[cdim * (i+1): cdim * (i+2)] - qs)
            elif i == nq - 1:
                qd = 0.5 * (qe - xi[cdim * (i-1): cdim * (i)])
            else:
                qd = 0.5 * (xi[cdim * (i+1): cdim * (i+2)] - xi[cdim * (i-1): cdim * (i)])

            xx = robot.fkine(qt, end=link) # x_current, (4, 4)
            ## TODO: get mesh data, vertices
            xd = JJ.dot(qd[:k+1]) # x_delta
            vel = np.linalg.norm(xd)
            # if vel < 1e-3:
            #     continue
     
            xdn = xd / (vel + 1e-3) # speed normalized
            xdd = JJ.dot(xidd[cdim * i: cdim * i + k + 1]) # second derivative of x
            prj = np.eye(6) - xdn[:, None].dot(xdn[:, None].T) # curvature vector (6, 6)
            kappa = (prj.dot(xdd) / (vel ** 2 + 1e-3)) # (6,)

            # for link in robot.links:
            xyz = xx.data[0][:3, -1]
            cost = np.sum(xyz)
            # print(i, qt, qe)
            total_cost += cost
            # delta := gradient of obstacle cost in work space, (6, cdim) 
            delta =  np.concatenate([[1, 1, 0], np.zeros(3)])

            # if t == 0:
                # print(JJ.T.dot(vel).dot(prj.dot(delta) - cost * kappa))
            # print(xdd)
            try:
                nabla_obs[cdim * i: cdim * i + k + 1] += JJ.T.dot(vel).dot(prj.dot(delta) - cost * kappa)
            except:
                import pdb; pdb.set_trace()
    dxi = Ainv.dot(lmbda * nabla_smooth)
    # dxi = Ainv.dot(nabla_obs + lmbda * nabla_smooth)
    xi -= dxi / eta
    print(f"Iteration {t} total cost {total_cost}")

# Make cost matrix

if visualize:
    from roboticstoolbox.backends.swift import Swift  # instantiate 3D browser-based visualizer
    backend = Swift()
    backend.launch()            # activate it
    backend.add(robot)          # add robot to the 3D scene
    backend.add(obstacle)
    for i in range(nq):
        print(i, xi[cdim * i: cdim * (i+1)])
        robot.q = xi[cdim * i: cdim * (i+1)]          # update the robot state
        backend.step()        # update visualization
    robot.q = qe
    backend.step()
    # for qk in qtraj.q:             # for each joint configuration on trajectory
    #     robot.q = qk
    #     backend.step()