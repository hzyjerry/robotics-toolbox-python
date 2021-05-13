

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

fknm_ = None

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

def jacob0_pts_vec(robot, end, pts, jacob_vec, qt=None, verbose=False):
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
    link_base = robot.fkine(qt, end=end)
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
    path, njoints, _ = robot.get_path(end=end)
    nlinks = len(path)
    for il, link in enumerate(path):
        axis = get_axis(link)
        # print(il, link.isjoint)
        if link.isjoint:
            link_As.append(link.A(qt[link.jindex]).A)
        else:
            link_As.append(link.A().A)
        link_axes.append(axis)
        link_isjoint.append(link.isjoint)
    # print(nlinks)
    nlinks_pt = np.ones(N, dtype=int) * nlinks
    link_As = np.array(link_As)[None, :].repeat(N, axis=0)
    link_axes = np.array(link_axes, dtype=int)
    link_isjoint = np.array(link_isjoint, dtype=int)

    # import pdb; pdb.set_trace()
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
    pts_tool = vrepeat(np.eye(4), N) # (nq * njoints * num_pts, 4, 4)
    pts_tool[:, :3, 3] = pts
    r_bases = np.repeat(link_bases[:, None], num_pts, axis=1) # (nq * njoints, num_pts, 4, 4)
    r_bases = r_bases.reshape(-1, 4, 4) # (nq * njoints * num_pts, 4, 4)
    pts_mat = np.matmul(r_bases, pts_tool) # (nq * njoints * num_pts, 4, 4)
    pts_etool = vrepeat(np.eye(4), N) # (nq * njoints * num_pts, 4, 4)
    link_As = []
    link_axes = []
    link_isjoint = []
    end = None
    # None, 1, 2, ..., 7 (last link), ....
    for il, link in enumerate(robot.links):
        if link.jindex is not None:
            end = link
    path, njoints, _ = robot.get_path(end)
    nlinks = len(path)
    nlinks_pt = []
    curr_links, curr_link_A = 0, SE3()
    j = 0
    for il, link in enumerate(path):
        axis = get_axis(link)
        curr_links += 1
        if link.isjoint:
            link_As.append(link.A(qt[link.jindex]).A)
            nlinks_pt.append(curr_links)
        else:
            link_As.append(link.A().A)
        link_axes.append(axis)
        link_isjoint.append(link.isjoint)
    # print(nlinks)
    link_As = np.array(link_As)[None, :].repeat(N, axis=0)
    link_axes = np.array(link_axes, dtype=int)
    link_isjoint = np.array(link_isjoint, dtype=int)
    nlinks_pt = np.tile(np.array(nlinks_pt, dtype=int).repeat(num_pts), nq)
    # import pdb; pdb.set_trace()
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
