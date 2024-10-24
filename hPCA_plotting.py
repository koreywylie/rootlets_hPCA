#!/usr/bin/env python3

# Plotting functions for output rootlets hPCA algorithm from hPCA.py,
#  projection on Poincare Disk 
#    (if package 'hyperbolic' available from https://pypi.org/project/hyperbolic/),
#  or projection onto Klein disk otherwise.

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from drawsvg import Drawing, Rectangle, Text
from hyperbolic import euclid
from hyperbolic.poincare import *

def flatten_list(L):
    """Recursively flattens list of list(s)."""
    if L == []:
        return L
    if isinstance(L[0], list):
        return flatten_list(L[0]) + flatten_list(L[1:])
    return L[:1] + flatten_list(L[1:])

def reorder_leaves(hPCA_results):
    """
    Returns re-ordering of indices input to rootlets hPCA,
      used to display correlation matrices showing hierarchical clustering.
    """
    B = hPCA_results['branches']
    L = len(B)
    v_inds = [B[-1][1]]  # top-level output index
    v_inds_ord = [0]     # order of above
    for b in range(L-1, -1, -1):  # iterate from L-1 to 0
        ij_br = B[b][0]  # indices to lower-level branches
        k_br = B[b][1]   # index of current branch
        
        # re-order inds. based on 2nd eigenvector
        eigv2 = B[b][2][0][:,1]
        ij_eig_ord = sorted(range(len(ij_br)), key=lambda k: eigv2[k])
        ij_br = [ij_br[v] for v in ij_eig_ord]
        
        # insert lower-level branch indices at location k
        k = v_inds.index(k_br)
        v_inds = v_inds[0:k] + ij_br + v_inds[(k+1):]

        # update ordering of indices
        k_ord = v_inds_ord[k]
        v1_inds_ord = v_inds_ord[0:k]
        v_ij_ord = list(range(len(ij_br)))
        v_ij_ord = [k_ord + o for o in v_ij_ord]
        v2_inds_ord = [o + len(ij_br) - 1 for o in v_inds_ord[(k+1):]]
        v_inds_ord = v1_inds_ord + v_ij_ord + v2_inds_ord
    
    return v_inds

def rotate_R2(u, v):
    """
    Finds rotation matrix in R^2 that sends vector u to v.
    
    Theory:
      If e=[1,0]^T, u=[x1,x2]^T, v=[y1,y2]^T,
        X=[[x1,-x2], and Y=[[y1,-y2],
           [x2, x1]]        [y2, y1]].
      Then u=Xe, v=Ye, e=(X^-1)u, and v=Ru, with R=YX^-1.
    """
    assert len(u) == 2
    assert len(u) == len(v)
    assert not all(v == 0)
    assert not all(u == 0)
    u_ = u / (sum(u**2))**0.5
    v_ = v / (sum(v**2))**0.5
    R = np.array([[u_[0]*v_[0] + u_[1]*v_[1], v_[0]*u_[1] - u_[0]*v_[1]],
                  [u_[0]*v_[1] - v_[0]*u_[1], u_[0]*v_[0] + u_[1]*v_[1]]])
    assert all(np.isclose(v_, np.matmul(R, u_)))
    
    return R
    
def align_and_rotate_PCs(U, v):
    """
    Aligns leading eigenvector w/ vector v within tan. space of H^n center, 
    & rotates all other eigs. in columns of U to maintain orthogonality.
    """
    if all(v == 0): return U  # nothing to do, directionless input
    assert hasattr(v, 'shape')
    assert hasattr(U, 'shape')
    assert v.shape[0] == U.shape[0]
    e = np.array([-1,0])
    R = rotate_R2(e, v[1:])
    U_ = U[1:,:]  # coords. in tangent space of center of H^n 
    u_L2 = np.apply_along_axis(lambda u: sum(u**2)**0.5, 0, U_)
    U_ = U_ / u_L2
    U_ = np.matmul(R, U_)
    U_ = U_ * u_L2
    U[1:,:] = U_
    assert all(U[0,:] == 0)
    
    return U

def Lorentz_prod(p, q=None):
    """
    Lorentz scalar product of vectors on hyperboloid, 
    Minkowski scalar product of vectors in Minkowski space.
    """
    if q is None: q = p
    assert len(p) == len(q)
    L_prod = [a*b for a,b in zip(p, q)]
    L_prod[0] = -L_prod[0]  # assumes 1st axis is time-like
    
    return(sum(L_prod))

def PT_pq(u, p, q):
    """
    Parallel Transport of vector u from tangent space of p, to tan. space of q.
    """
    assert np.isclose(Lorentz_prod(p), -1)
    assert np.isclose(Lorentz_prod(q), -1)
    assert np.isclose(Lorentz_prod(p, u), 0)
    if all(p == q): return u
    v = u + (p + q) * Lorentz_prod(q,u) / (1 - Lorentz_prod(p,q))
    assert np.isclose(Lorentz_prod(q, v), 0)
    
    return v

def Exp_p(v, p):
    """
    Exponential Map for vector v from tangent space at p to H^n.
    """
    assert np.isclose(Lorentz_prod(p), -1)
    assert np.isclose(Lorentz_prod(p, v), 0)
    v_norm = (Lorentz_prod(v))**0.5
    if v_norm == 0: return p
    q = np.cosh(v_norm) * p + np.sinh(v_norm) * v / v_norm
    assert np.isclose(Lorentz_prod(q), -1)
    
    return q

def Log_p(q, p):
    """
    Logarimic Map for point q from H^n to tangent space of p.
    """
    assert np.isclose(Lorentz_prod(p), -1)
    assert np.isclose(Lorentz_prod(q), -1)
    if all(p == q): return p * 0
    a = -Lorentz_prod(q, p)
    u = np.arccosh(a) * (q - a * p) / (a**2 - 1)**0.5
    assert np.isclose(Lorentz_prod(p, u), 0)
    
    return u

def diffeo_hyperboloid2ball(x):
    """
    Diffeomorphism/Mobius transform from hyperboloid to Poincare ball.
    """
    x = np.asarray(x)
    assert x[0] > 0
    y = x[1:] / (x[0] + 1)  # Nickel & Kiela, 2018, eq. (11)
    assert sum(y**2)**0.5 < 1
    
    return y
    
def diffeo_ball2hyperb(x):
    """
    Diffeomorphism/Mobius transform from Poincare ball to hyperboloid.
    """
    x = np.asarray(x)
    x2 = sum(x**2)
    assert x2**0.5 < 1
    y = [1 + x2, 2 * x] / (1 - x2)  # Nickel & Kiela, 2018, eq. (11)
    assert y[0] > 0
    
    return y

def diffeo_ball2halfplane(x):
    """
    Diffeomorphism/Mobius transform from Poincare ball to upper 1/2 plane.
    """
    x = np.asarray(x)
    assert sum(x**2)**0.5 < 1
    x2 = sum(x**2)
    s = 2 / (1 - 2*x[-1] + sum(x**2))
    y = np.zeros(x.shape)
    y[0:-1] = s * x[0:-1]
    y[-1] = s * (1 - x[-1]) - 1
    assert y[-1] > 0
    
    return y

def diffeo_hyperboloid2halfplane(x):
    """
    Diffeomorphism/Mobius transform from hyperboloid to upper 1/2 plane.
    """
    x = np.asarray(x)
    assert x[0] > 0
    x2 = sum(x**2)
    s = 2 / (1 + x[0] - 2*x[-1] + sum(x[1:]**2)) / (1 + x[0])
    y = np.zeros(x.shape)
    y[1:-1] = s * x[1:-1]
    y[-1] = s * (1 + x[0] - x[-1]) - 1
    assert y[-1] > 0
    
    return y[1:]
    
def get_displacement_from_PC1(V, d, norm_d=True, scale_d=True):
    """
    Gets coordinates of data points relative to coords. of leading PC.
    """
    V_ = V.copy()
    V_[:,0] = V_[:,0] - np.ones(V.shape[0])
    d_ = d.copy()
    # norming & scaling both strongly recommended for numerical stability c
    if norm_d: d_ = d_ / sum(d_)
    if scale_d: d_ = d_ / V.shape[0]  # scale to dim. of local PCA
    d_ = d_**0.5
    
    return V_ * d_
    

def embed_Tan_o(U_, n, inds=None):
    """
    Embeds matrix of vectors U_ from R^n in tangent space of center of H^n.
    """
    if not inds:
        z0 = np.zeros([U_.shape[0], 1])
        Z0 = np.zeros([U_.shape[0], n - (1 + U_.shape[1])])
        U = np.hstack((z0, U_, Z0))
    else:
        assert all([i > 0 for i in inds])
        U = np.zeros([U_.shape[0], n + 1])
        U[:,inds] = U_
    
    return U

def embed_H(p, V, d, rotate_PC1=True, root_embed=False, root_embed_halfplane=False):
    """
    Embeds input vars. from local PCA into H^k around low-dim projection of pt. p,
      using eigenvectors matrix V (in columns) & eiqenvalues d.
    """
    k = len(p)
    k_ = k - 1
    o = np.array([1] + [0] * k_)
    if not root_embed:
        Z = get_displacement_from_PC1(V[:,0:k_], d[0:k_])  # coords. relative to leading PC
    else:
        r = 0.5
        # r = 0.1 * V.shape[0]
        if root_embed_halfplane:
            th = np.linspace(np.pi/2 - 0.4*V.shape[0], np.pi/2 + 0.4*V.shape[0], V.shape[0])
        else:
            th = np.linspace(0, 2*np.pi, V.shape[0]+1)[0:-1]
        S1 = np.c_[r*np.cos(th), r*np.sin(th)]
        Z = S1
        
    Uo = embed_Tan_o(Z, k)
    if rotate_PC1 and not all(p == o):
        op = Log_p(p, o) # embed p in tangent space of o
        Uo = align_and_rotate_PCs(Uo.T, op).T
    Up = np.apply_along_axis(PT_pq, 1, Uo, o, p)  # axis=1 to apply to rows
    Q = np.apply_along_axis(Exp_p, 1, Up, p)
    
    return Q

def embed_hPCA(hPCA_results, k=2, rotate_PC1=True):
    """
    Embeds hPCA output in H^k, maps onto Poincare disk, 
      & returns coordinates with indices of geodesics connecting points.
    """
    B = hPCA_results['branches']
    L = len(B)
    if k is None:
        k_l = [len(br[2][1]) for br in B]
        k = max(k_l)
    if k > 2:
        rotate_PC1 = False  # avoid complications of high-dim. rotations
    
    o = np.array([1] + [0] * k) # starting point at center of H^k
    H_pts = [o]
    H_pts_inds = [B[-1][1]]
    D2_pts = o[1:]
    D2_pts_inds = H_pts_inds.copy()
    D2_geodesics_i = np.empty([0,2], dtype=int)
    D2_roots_i = []
    
    
    for b in range(L-1, -1, -1):  # iterate from L-1 to 0
        root_embed = False
        if (k <= 2) and (b+1 == L) and not hPCA_results['level_final_sig']:
            root_embed = True # embed spherical corr. matrix as sphere in R^2
        
        p_i = B[b][1]
        i = H_pts_inds.index(p_i)
        _ = H_pts_inds.pop(i)
        p = H_pts.pop(i)
        
        V = B[b][2][0]
        d = B[b][2][1]
        v_inds = flatten_list(B[b][0])

        Q = embed_H(p, V, d, rotate_PC1, root_embed)
        H_pts += [q for q in Q]  # append coords. as list of np.ndarrays
        H_pts_inds += v_inds
        
        Qb = np.apply_along_axis(diffeo_hyperboloid2ball, 1, Q)
        d2_i = len(D2_pts_inds) - 1 - D2_pts_inds[::-1].index(p_i) # find last prev. occurence of p_i
        D2_pts = np.vstack((D2_pts, Qb[:,0:2]))
        D2_pts_inds += v_inds
        if b+1 < L:
            d2_gd = np.c_[[d2_i] * len(v_inds), range(len(v_inds))]  # concats lists horizontally
            d2_gd[:,1] += len(D2_pts_inds) - len(v_inds)
            D2_geodesics_i = np.vstack((D2_geodesics_i, d2_gd))
        elif (b+1 == L) and hPCA_results['level_final_sig']:
            d2_gd = np.c_[[d2_i] * len(v_inds), range(len(v_inds))]
            d2_gd[:,1] += len(D2_pts_inds) - len(v_inds)
            D2_geodesics_i = np.vstack((D2_geodesics_i, d2_gd))
            D2_roots_i = [0]
            D2_roots_labels = [B[b][1]]
        else: # embed as disconnected points in circle, without geodesics
            D2_pts[0,:] = (None, None)
            D2_roots_i = [r for r in range(D2_pts.shape[0]) if r > 0]
            D2_roots_labels = B[b][0]
    
    return D2_pts, D2_geodesics_i, D2_roots_i, D2_roots_labels

def plot_disk_simple(D2_pts, D2_geodesics_i):
    """
    Simple plot of hyperbolic points & geodesics on Klein disk,
      geodesics as straight lines with distorted distances and angles.
    """
    a = 0.7 # alpha transparency
    fig, ax = plt.subplots()
    disk = plt.Circle((0,0), 1, facecolor=(1,1,1), edgecolor=(0,0,0))
    ax.add_patch(disk)
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_aspect('equal')
    ax.scatter(D2_pts[:,0], D2_pts[:,1], alpha=a)
    for gd_i in D2_geodesics_i:
        gd_ends = D2_pts[gd_i,:]
        gd = plt.Line2D(gd_ends[:,0], gd_ends[:,1], alpha=a)
        ax.add_line(gd)

def plot_disk_Poincare(D2_pts, D2_geodesics_i=None, D2_roots_i=None, D2_roots_labels=None,
                       save_fname=None):
    """
    Plot of hyperbolic points & geodesics on Poincare disk.
    """
    trans = Transform.rotation(rad = -np.pi/2) # x-axis oriented downward in display
    r_euclid = 1  # disk radius
    a = 0.8 # alpha transparency
    line_style = dict(hwidth=0.03, fill='firebrick', opacity=a)
    node_style = dict(hradius=0.05, fill=line_style['fill'], opacity=0.5)
    
    d = Drawing(2*r_euclid, 2*r_euclid, origin='center')  # plotting canvas & disk edge
    d.draw(euclid.Circle(0, 0, r_euclid), stroke='black', stroke_width=0.002, fill='none')
    
    if D2_geodesics_i is not None:
        N = D2_geodesics_i.shape[0]
        if N > 0:
            for n in range(N):
                i1, i2 = D2_geodesics_i[n,:]
                p1 = Point(*D2_pts[i1,:])
                p2 = Point(*D2_pts[i2,:])
                p1 = trans.apply_to_point(p1)
                p2 = trans.apply_to_point(p2)
                l12 = Line.from_points(*p1, *p2, segment=True)
                d.draw(l12, **line_style)
    if D2_roots_i is not None:
        if len(D2_roots_i) > 0:
            for l,i in enumerate(D2_roots_i):
                p = Point(*D2_pts[i,:])
                p = trans.apply_to_point(p)
                d.draw(p, **node_style)
                if D2_roots_labels is not None:
                    d.draw(Text(str(D2_roots_labels[l]), 0.1, *p))
    d.set_render_size(w=600)
    if save_fname:
        d.save_png(save_fname)
    
    return d

def plot_halfplane_Poincare(D2_pts, D2_geodesics_i=None, D2_roots_i=None, D2_roots_labels=None,
                            save_fname=None):
    """
    Plot of hyperbolic points & geodesics on Poincare half-plane.
    """
    # All points & geodesics specified on Poincare disk, transformed to half-plane.
    trans = Transform.merge(Transform.mirror((1, 0)),
                            Transform.disk_to_half(),
                            Transform.mirror((1, 0)))
    
    # Get plotting ranges
    U = np.zeros(D2_pts.shape)
    if all(np.isnan(D2_pts[0,:])):
        U[0,:] = np.nan
    else:
        U[0,:] = diffeo_ball2halfplane(D2_pts[0,:])
    U[1:,:] = np.apply_along_axis(diffeo_ball2halfplane,1,D2_pts[1:,:])
    w = np.nanmax(U[:,0]) - np.nanmin(U[:,0])
    h = np.nanmax(U[:,1]) - np.nanmin(U[:,1])
    o = [-w/2, -h/2]
    
    # # manual plotting ranges, by trial & error
    # w = 1.7  # for SimTB 3L3Ddirich plots
    # h = 0.85
    # o = (-0.85, -0.85)
    
    a = 0.5 # alpha transparency
    line_style = dict(hwidth=0.02, fill='firebrick')
    node_style = dict(hradius=0.05, fill=line_style['fill'], opacity=a)
    d = Drawing(w, h, origin=o)  # plotting canvas & disk edge
    
    if D2_geodesics_i is not None:
        N = D2_geodesics_i.shape[0]
        if N > 0:
            for n in range(N):
                i1, i2 = D2_geodesics_i[n,:]
                p1 = Point(*D2_pts[i1,:])
                p2 = Point(*D2_pts[i2,:])
                l12 = Line.from_points(*p1, *p2, segment=True)
                d.draw(l12, transform=trans, **line_style)
    if D2_roots_i is not None:
        if len(D2_roots_i) > 0:
            for l,i in enumerate(D2_roots_i):
                p = Point(*D2_pts[i,:])
                d.draw(p, transform=trans, **node_style)
                if D2_roots_labels is not None:
                    p1 = trans.apply_to_point(p)
                    d.draw(Text(str(D2_roots_labels[l]), 0.1, *p1))
    d.set_render_size(w=600)
    if save_fname:
        d.save_png(save_fname)
    
    return d


#########################################
data_path = os.getcwd()
data_fname = 'hPCA_results-complex_hier_example.pkl'  # default save filename from hPCA.py script
save_fname = os.path.splitext(data_fname)[0] + '.png'

data_fname = os.path.join(data_path, data_fname)
save_fname = os.path.join(data_path, save_fname)
assert os.path.isfile(data_fname)
with open(data_fname, 'rb') as f:
    hPCA_results = pickle.load(f)

D2_pts, D2_geodesics_i, D2_roots_i, D2_roots_labels = embed_hPCA(hPCA_results)
d0 = plot_disk_Poincare(D2_pts, D2_geodesics_i, D2_roots_i, D2_roots_labels, save_fname)
d0
#########################################
