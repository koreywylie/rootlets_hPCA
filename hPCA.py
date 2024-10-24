#!/usr/bin/env python3

# Rootlets Hierarchical Principle Components Analysis algorithm
#
# Loading saved output:
#   with open(data_fname, 'rb') as f:  hPCA_results = pickle.load(f)


import os
import pickle
import numpy as np
from numpy.linalg import svd, det
from scipy.stats import norm, chi2, zscore


def ESET(d, N, m=1, pval=None):
    """
    Equality of Smallest Eigenvalues Test.
        Anderson, T.W. 2003. An Introduction to Multivariate Statistical Analysis.
            Wiley, Hoboken, NJ.
    
    Args:
        d: Vector of eigenvalues.
        N: Number of observations in sample.
        m: Start of indice of repeated eig. sequence to test. 
        pval: P-value cutoff used to calculate critical point.
    """
    assert m >= 0
    if len(d) == (m+1):  return 1, None, None  # last eig. equal to itself, H0 true w/ prob. 1.
    if sum(d[m:]) < 1e-17: return 1, None, None  # numerical precision near log(0) = -Inf
    if any(np.isclose(d[m:], 0)) and not all(np.isclose(d[m:], 0)):
        return 0, None, None  # numerical precision near log(0) = -Inf
    assert len(d) - m - 1 > 0  # chi^2 w/ df = 0 only relevant for non-central chi^2
    p = len(d)
    n = N - 1
    
    test_stat = -n * sum(np.log(d[m:p])) + n * (p - m) * np.log(sum(d[m:p]) / (p - m))
    df = (1 / 2) * (p - m + 2) * (p - m - 1)
    pvalue_exact = 1 - chi2.cdf(test_stat, df)  # upper-tailed sig. point
    critical_pt = chi2.isf(pval, df) if pval else None  # upper-tailed critical point
    
    return pvalue_exact, test_stat, critical_pt

def flatten_list(L):
    """
    Recursively flattens list of list(s).
    """
    if L == []:
        return L
    if isinstance(L[0], list):
        return flatten_list(L[0]) + flatten_list(L[1:])
    return L[:1] + flatten_list(L[1:])

def cluster_merge(A, i, j):
    """
    Mergers indices of two active clusters containing nearly identical vars.
    """
    c_i, c_j = A[i], A[j]
    c_ij_old = [c_i, c_j]
    A[i], A[j] = [], [] # placeholders preserve unchanged ordinal indices
    c_i_new = sorted(c_i + c_j)  # merge branches / clusters
    i_new = i  # update in place, to allow for local updates in rows of A_PC1
    A[i_new] = c_i_new
    
    return A, c_i_new, i_new, c_ij_old

def cluster_create_new(A_b, i, j, A):
    """
    Creates new cluster & inserts in active clusters set, removes old indices.
    """
    c_i, c_j = A[i], A[j]
    c_ij_old = [c_i, c_j]
    c_i_new = [min(flatten_list(c_i)), min(flatten_list(c_j))]
    i_new, j_old = i, j  # update in place, for local updates in rows of A_PC1
    A[i], A[j] = [], [] # placeholders preserve unchanged ordinal indices
    A[i_new] = c_i_new
    A_b[i_new] += 1  # increment branch count for cluster
    
    return A_b, A, c_i_new, i_new, j_old, c_ij_old

def cluster_revise_ordering(A):
    """
    Clean up active cluster set by removing empty entries.
    """
    A_i_old_new = {}  # changes tracking old list inds : new inds
    j = -1
    for i, a_i in enumerate(A):
        if a_i:
            j += 1
            A_i_old_new[i] = j
    A = [a for i, a_i in enumerate(A) if a_i]
    
    return A, A_i_old_new
    
def cluster_merge_update(A, i, j, PC1, A_PC1, R):
    """
    Mergers cluster indices & updates Active clusters set, PC1s, sims.
    """
    A, c_merged, i_new, _ = cluster_merge(A, i, j)
    pc1_new, _, _, _, _ = get_local_PCA(PC1, [c_merged])
    A_PC1, i_new, _ = PC1_update_w_new(A_PC1, pc1_new, i, j, i_new)
    R = sims_update_w_newPC1(R, A, A_PC1, pc1_new, i_new, i, j)
    
    return A, A_PC1, R

def cluster_create_branch(A_b, i, j, A, B, R, PC1, A_PC1, l, pc2_orient_ref=None, save_all_pcs=False):
    """
    Creates new branch from clusters, updates active clusters set, PC1s, etc.
    """
    A_b, A, c_new, i_new, j_old, c_ij = cluster_create_new(A_b, i, j, A)
    pc1s_ij, V_ij, d_ij, Wt_ij, _ = get_local_PCA(PC1, c_ij, pc2_orient_ref)
    if len(c_ij[0]) > 1:
        B, _ = branch_create(B, c_ij[0], V_ij[0], d_ij[0], Wt_ij[0], l, c_new[0], save_all_pcs)
    if len(c_ij[1]) > 1:
        B, _ = branch_create(B, c_ij[1], V_ij[1], d_ij[1], Wt_ij[1], l, c_new[1], save_all_pcs)
    PC1, i_new, _ = PC1_update_w_new(PC1, pc1s_ij, c_ij[0], c_ij[1], i_new, j_old)
    pc1_new, _, _, _, _ = get_local_PCA(PC1, [c_new])
    A_PC1, i_new, _ = PC1_update_w_new(A_PC1, pc1_new, i, j, i_new)
    R = sims_update_w_newPC1(R, A, A_PC1, pc1_new, i_new, i, j)
    
    return A_b, A, B, R, PC1, A_PC1

def balance_branches(A_b, i, j, A, B, l, R, PC1, A_PC1, save_all_pcs=False):
    """
    Balances individual branches of multiway merger, 
      by fixing less-merged branch & creating new cluster from its leading PC.
    """
    i_, j_, c_full = False, False, []
    if A_b[i] != A_b[j]:
        c_i, c_j = A[i], A[j]
        if A_b[i] < A_b[j]:
            ind_, i_, c_full = i, True, c_i
        else:
            ind_, j_, c_full = j, True, c_j
        ij_new = min(c_full)
        A[ind_] = [ij_new]
        A_b[ind_] += 1

        if len(c_full) > 1:
            pc1_new, V_ij, d_ij, Wt_ij, _ = get_local_PCA(PC1, [c_full])
            B, _ = branch_create(B, c_full, V_ij[0], d_ij[0], Wt_ij[0], l, ij_new, save_all_pcs)
            PC1, _, _ = PC1_update_w_new(PC1, pc1_new, c_full, i_new=ij_new)
            A_PC1, _, _ = PC1_update_w_new(A_PC1, pc1_new, c_full, i_new=ij_new)
            R = sims_update_w_newPC1(R, A, A_PC1, pc1_new, ij_new, c_full)
    
    return A_b, A, B, R, PC1, A_PC1, i_, j_, c_full

def branch_create(B, c_ij, V_ij, d_ij, Wt_ij, l=None, c_out=None, save_all_pcs=False):
    """
    Creates new branch, formats & updates branch set B.
    """
    if c_out is None:
        c_out = min(flatten_list(c_ij))
    if not save_all_pcs:
        Wt_ij = Wt_ij[0,:]
    br_details = [V_ij, d_ij, Wt_ij]
    br_new = [c_ij, c_out, br_details, l]
    b = len(B)
    B.append(br_new)
    
    return B, b

def align_pc1(X, Wt, V=None):
    """
    Removes +/- ambiguity of pc1, orients pos. direction w/ input vars. X.
    """
    M = X.shape[0]
    r_pc1 = np.zeros(M)
    pc1 = Wt[0,:] if hasattr(Wt, 'shape') else Wt
    for m in range(M):
        r_pc1[m] = np.corrcoef(X[m,:], pc1)[0,1]
    sign_adj_pc1 = False
    if sum(r_pc1) < 0:
        sign_adj_pc1 = True
        if hasattr(Wt, 'shape'):
            Wt[0,:] = -Wt[0,:]
            if V is not None:
                V[:,0] = -V[:,0]
        else:
            Wt = pc1 * -1
    
    return Wt, V, sign_adj_pc1

def align_pc2(pc2_orient_ref, Wt, V):
    """
    Orients pos. dir. of 2nd eigvector with vector pc2_orient_ref.
    """
    pc2_m_realigned = False
    if np.corrcoef(Wt[1,:], pc2_orient_ref) < 0:
        pc2_m_realigned = True
        Wt[1,:] = -Wt[1,:]
        V[:,1] = -V[:,1]
    if V.shape[1] > 2:
        if det(V) < 0:
            V[:,2] = -V[:,2]
            assert det(V) > 0
    
    return Wt, V, pc2_m_realigned


def get_local_PCA(X, A, pc2_orient_ref=None, pc1_orient=True, standardize=True):
    """
    Calculates local Principal Comps of cluster(s) of active variables.
    
    Args:
        X: Matrix of current variables in rows.
        A: Active clusters set, list of indices used to parcellate X.
        pc2_orient_ref: Vector used to orient +/- dir. of 2nd eigenvector.
        pc1_orient: Orient +/- of 1st eigenvector relative to inputs.
        standardize: Subtract mean & scale to s.d. for all vars.
    
    Returns:
        localPC1: Matrix of leading PCs for clusters in A.
        V: List of rotation matrices for local PCAs, eigenvectors in columns.
        d: List of eigenvalues for local PCAs.
        Wt: List of right-singular matrices for local PCAs, transposed.
        pc1s_realigned: List of changes in +/- orientations of leading PCs.
    """
    M = len(A)
    K, N = X.shape
    if standardize:
        X = zscore(X, ddof=1, axis=1)
    localPC1 = np.zeros([M, N])
    V, d, Wt, pc1s_realigned = ([], [], [], [])
    for m,a in enumerate(A):
        if a:
            a = flatten_list(a)
            if len(a) == 1:
                localPC1[m,:] = X[a,:]
                pc1_m_realigned = False
                V_m, d_m, Wt_m = ([], [], [])
            else:
                V_m, d_m, Wt_m = svd(X[a,:], full_matrices=False)
                
                d_m = d_m**2 / (N - 1)  # scale to eigenvalues of cov. matrix
                if standardize:  # format right singular vector as iid sample of signal w/ var=1
                    Wt_m = zscore(Wt_m, ddof=1, axis=1)
                if pc1_orient:
                    Wt_m, V_m, pc1_m_realigned = align_pc1(X[a,:], Wt_m, V_m)
                if pc2_orient_ref is not None:
                    Wt_m, V_m, pc2_m_realigned = align_pc2(pc2_orient_ref, Wt_m, V_m)
                pc1_new = Wt_m[0,:] * d_m[0]**0.5  # ~ np.matmul(V[:,0], X[a,:] / X.shape[1]**0.5)
                localPC1[m,:] = pc1_new
        V.append(V_m)
        d.append(d_m)
        Wt.append(Wt_m)
        pc1s_realigned.append(pc1_m_realigned)
    
    return localPC1, V, d, Wt, pc1s_realigned

def PC1_update_w_new(PC1, pc1s_new, i_old=None, j_old=None, i_new=None, j_new=None):
    """
    Zereos-out leading PCs with active cluster indices i_old, j_old,
    & inserts new leading PC(s) at i_new, j_new.
    """
    ij_new = []
    if i_old is not None:  PC1[i_old,:] = PC1[i_old,:] * 0
    if j_old is not None:  PC1[j_old,:] = PC1[j_old,:] * 0
    if i_new is None:  # update in place
        i_new = min(flatten_list(i_old))
    ij_new.append(i_new)
    if  j_new is not None:
        assert hasattr(pc1s_new, "shape")
        assert pc1s_new.shape[0] > 1
        j_new = min(flatten_list(j_old))
        ij_new.append(j_new)
    PC1[ij_new,:] = pc1s_new
    
    return PC1, i_new, j_new

def sim_inds_old2new(ij_old, ij_old_new):
    """
    Replaces old active cluster set indices i,j according to dict ij_old_new.
    """
    assert len(ij_old) == 2
    if any(ij_old in ij_old_new.keys()):
        i,j = ij_old
        i_new, j_new = ij_old
        if i in ij_old_new.keys():
            i_new = ij_old_new[i]
            i = i_new
        if j in ij_old_new.keys():
            j_new = ij_old_new[j]
            j = j_new
    
    return nd.array([i,j])
            
def sims_update_list(R, i=None, j=None, ij_old_new=None):
    """
    Removes active cluster indices i,j from simarities list R,
     or updates sim. list w/ changes of indices in dict ij_old_new.
    """
    assert ((ij_old_new is None and (i is not None or j is not None)) or 
            (ij_old_new is not None and i is None and j is None))
    if (i is not None) or (j is not None):
        ij_list = flatten_list([i,j])
        m_old0 = [m for m,h in enumerate(R[1][:,0]) if h in ij_list]
        m_old1 = [m for m,h in enumerate(R[1][:,1]) if h in ij_list]
        m_old = np.unique(m_old0 + m_old1)
        R[0] = np.delete(R[0], m_old, axis=0)
        R[1] = np.delete(R[1], m_old, axis=0)
    elif ij_old_new:
        R[1] = np.apply_along_axis(sim_inds_old2new, 1, R[1], ij_old_new)
    
    return R

def sims_update_w_newPC1(R, A, A_PC1, pc1_new, A_i_pc1_new, 
                         A_i_old=None, A_j_old=None):
    """
    Updates similarities list R between rows of A_PC1,
      by removing old active set indices A_i_old, A_j_old,
      appending sims. w/ new leading principal comp. pc1_new
      w/ associated A_PC1 row index A_i_pc1_new.
    """
    if (A_i_old or A_j_old):
        R = sims_update_list(R, A_i_old, A_j_old)
    M = A_PC1.shape[0]
    r_new = np.zeros(M)
    r_new_inds = np.zeros([M, 2], dtype=int)
    r_keep = []
    for m in range(M):
        if m == A_i_pc1_new: continue
        if all(A_PC1[m,:] == 0): continue
        a = len(A[m])  # scale sim. measure to cluster sizes
        b = len(A[A_i_pc1_new])
        r_new[m] = np.cov(A_PC1[m,:], pc1_new)[0,1] / (a*b)**0.5  # equal to av. of off-diag. corr. block
        # r_new[m] = np.corrcoef(A_PC1[m,:], pc1_new)[0,1]  # biased by block sizes, not eq to off-diag. corrs
        r_new_inds[m,:] = [m, A_i_pc1_new]
        r_keep.append(m)
    r_new = r_new[r_keep]
    r_new_inds = r_new_inds[r_keep,:]
    R[0] = np.append(R[0], r_new, axis=0)
    R[1] = np.append(R[1], r_new_inds, axis=0)
    
    return R
    

def init_active_clusters(X0):
    """
    Initializes list w/ sets of indices of active vars.
    """
    A = [[a] for a in range(X0.shape[0]) if not (np.isnan(X0[a,:]).any() or 
                                                 (X0[a,:] == 0).all())]
    return A

def init_sim_list(PC1):
    """
    Initializes list of similarities as corrs. between leading Principal Comps.
    """
    R_ = np.corrcoef(PC1)
    n = R_.shape[0]
    R_inds = np.triu_indices(n, k=1) # strictly upper-tri. w/o diag.
    R_r = R_[R_inds]
    R_inds = np.stack(R_inds, axis=1) # format as matrix of pairs in rows
    r_ij_nan = [i for r,i in enumerate(R_r) if np.isnan(r)]
    R_r = np.delete(R_r, r_ij_nan)
    R_inds = np.delete(R_inds, r_ij_nan, axis=0)
    
    return [R_r, R_inds]

def rootlets_init(X0, verbose=False):
    if verbose: print("...initializing hierarchy leaves w/ data & calc. similarities...")
    A = init_active_clusters(X0)  # active clusters set
    A_b = [0] * X0.shape[0]  # branch merge count for active clusters
    B = []  # branch list
    PC1, _, _, _, _ = get_local_PCA(X0, A) # leading PCs for clusters in A
    R = init_sim_list(PC1) # formatted as [corrs., A-element ind. pairs]
    A_PC1 = PC1.copy()  # rows indexed by A-element indices for efficient updates
    
    return R, A_PC1, PC1, A, A_b, B

def display_merged_inds_seq(A_by_level):
    print()
    print("Active clusters set by level, with indices of observed variables:")
    for l,a in enumerate(A_by_level):
        if l == 0:
            print("  Init:    ", end='')
        else:
            print("  level " + str(l) + ": ", end='')
        print(a)

def rootlets_level(R, A_b, A, B, A_PC1, PC1, l, p_thresh, 
                   pc2_orient_ref=None, stop_neg_sim=False,
                   save_all_pcs=False, verbose=False): 
    """
    Single level of rootlets hPCA algorithm.
    
    Args:
        R: List of similarities, formatted as [corrs., ind. pairs]
        A: Active clusters set.
        A_b: previously branching count for active clusters.
        B: Branch set. 
        l: Current hPCA level.
        p_thresh: p-value threshold for ESET & cluster-defining criteria.
        A_PC1: Matrix of leading PCs for clusters in A.
        PC1: Matrix of leading PCs (before grouping by A).
        pc2_orient_ref: Vector used to orient +/- dir. of 2nd eigenvalues.
        save_all_pcs: if False, only save/return leading PC for each branch.
    
    Returns:
        R, A, B, A_PC1, PC1: Updated input args.
        d_ESET: Eigenvalues input to ESET, identical to local PCA eigs. iff n.s. by ESET.
        pval_exact: Exact p-value calculated by ESET.
        r_max: Maximum similarity at level.
    """
    r_max_ind = R[0].argmax()
    r_max = R[0][r_max_ind]
    if stop_neg_sim:
        if r_max < 0:
            if verbose: print('All pos. similarities exhausted, r_max = %0.3f' % r_max)
            return R, A_b, A, B, A_PC1, PC1, None, None, r_max
    
    i, j = R[1][r_max_ind,:]
    if j < i: j, i = i, j
    A_b, A, B, R, PC1, A_PC1, i_, j_, c_full = balance_branches(A_b, i, j, A, B, l, R, 
                                                                PC1, A_PC1, save_all_pcs)
    c_i, c_j = A[i], A[j]
    if verbose: 
        msg = 'i = %d' % int(flatten_list(c_i)[0] + 1)
        if len(c_i) > 1:  msg += '+'
        if i_: msg += '-'  # indicates cluster condensed to leading PC
        msg += ',  j = %d' % int(flatten_list(c_j)[0] + 1)
        if len(c_j) > 1:  msg += '+'
        if j_: msg += '-'
        msg += ', r_max = %0.3f' % r_max
    
    X_l = np.append(PC1[flatten_list(c_i),:], PC1[flatten_list(c_j),:], axis=0)
    X_l = zscore(X_l, ddof=1, axis=1)
    K, N = X_l.shape
    d_ESET = svd(X_l, compute_uv=False)
    d_ESET = d_ESET**2 / (N - 1)  # scale to eigenvalues of cov. matrix
    
    pval_exact, _, _ = ESET(d_ESET, N)
    if pval_exact >= p_thresh:
        # Case 1: merge clusters w/ neglible diffs. in vars.
        #  Updates active clusters set A, 
        #  calculates newly-merged leading PC for cluster, 
        #  inserts into A_PC1 & updates sims., 
        #  leaves PC1 unchanged.
        if verbose: print(msg)
        A, A_PC1, R = cluster_merge_update(A, i, j, PC1, A_PC1, R)
    else:
        # Case 2: fixes individual clusters & associated local PCAs,
        #   creates new vars./cluster as leading PCs of fixed clusters,
        #   updates active clusters set A, 
        #   updates both matrices PC1, A_PC1, and similarity list accordingly.
        if verbose: print(msg + ' *')
        A_b, A, B, R, PC1, A_PC1 = cluster_create_branch(A_b, i, j, A, B, R, PC1, A_PC1, 
                                                         l, pc2_orient_ref, save_all_pcs)
    
    return R, A_b, A, B, A_PC1, PC1, d_ESET, pval_exact, r_max


def rootlets_final(A, B, PC1, l, p_thresh, 
                   pc2_orient_ref=None, save_all_pcs=False, verbose=False):
    """
    Final level of rootlets algorithm.
      Tests all remaining active clusters for any corrs. & branches if appropriate,
      assembles all remaining active vars. into cluster & creates branch, applies ESET to all eigenvalues.
    """
    pvals_active_c = []
    c_ij = [c_i for c_i in A if len(c_i) > 1]
    if len(c_ij) > 1:
        if verbose: print('...testing all remaining active cluster(s) for similar vars...')
        _, V_ij, d_ij, Wt_ij, _ = get_local_PCA(PC1, c_ij, pc2_orient_ref)
        for i in range(len(c_ij)):
            pval, _, _ = ESET(d_ij[i], PC1.shape[1], m=0)
            pvals_active_c.append(pval)
        if any(np.array(pvals_active_c) < p_thresh):
            if verbose: print('.....fixing active clusters with indices: i = ', end='')
            for i,pval in enumerate(pvals_active_c):
                if pval < p_thresh:
                    i_c = min(c_ij[i])
                    if verbose:
                        msg = str(i_c + 1)
                        if len(A[i_c]) > 1: msg += '+'
                        print(msg, end=', ')
                    B, _ = branch_create(B, c_ij[i], V_ij[i], d_ij[i], Wt_ij[i], l, 
                                         i_c, save_all_pcs)
                    A[i_c] = [i_c]
            if verbose: print()
        elif verbose:
            print('.....no similarities/corrs. between vars. in active cluster(s) found')
    
    c_ij = flatten_list(A)
    c_ij.sort()
    if verbose:
        msg = "...final, remaining vars. located in cluster:  "
        msg += 'i = %d' % int(c_ij[0] + 1)
        if len(c_ij) > 1:  msg += '+'
        print(msg)
    _, V_ij, d_ij, Wt_ij, _ = get_local_PCA(PC1, [c_ij], pc2_orient_ref)
    B, _ = branch_create(B, c_ij, V_ij[0], d_ij[0], Wt_ij[0], l+1, 
                         c_ij[0], save_all_pcs)
    pval_final, _, _ = ESET(d_ij[0], PC1.shape[1], m=0)
    
    return A, B, c_ij, pvals_active_c, pval_final

def rootlets_hPCA(X0, p_thresh=0.05, 
                  pc2_orient_ref_i=None, stop_neg_sim=False,
                  save_fname='hPCA_results.pkl', save_debug_info=False, 
                  display_level_inds=False, verbose=True):
    """
    Rootlets Hierarchical Principal Components Algorithm.
    
    Args:
        X0: Data matrix, variables in rows.
        p_thresh: p-value threshold for ESET & cluster-defining criteria.
        pc2_orient_ref_i: Indice of vector used to orient +/- dir. of 2nd eigenvalues.
        stop_neg_sim: Stop algorithm once all positive similarities are exhausted.
        save_fname: If none, skip saving hPCA output.
        save_debug_info: Include additional info in save/returned results.
        display_level_inds: Quick display of indices merged at each level.
    Returns:
        Dictionary with keys:
            levels_pvalues: Exact p-values calc. by ESET by level.
            levels_sig: List of significant levels by ESET & p_thresh.
            level_final_pvalue: Exact p-value of final cluster by ESET with m=0.
            level_final_sig: Signifance of above final cluster.
            branches: List of branched clusters, indices, eigs., etc.
            branches_field_names: Field names for above list elements.
            pvalue_cutoff: Applied upper bound on ESET p-values for significance.
            save_fname: Path to saved file.
    """
    if verbose: print("Applying hPCA with rootlets algorithm...")
    assert isinstance(X0, np.ndarray)
    assert p_thresh > 0

    levels_sims = []        # similarities for all levels
    levels_pvalues = []     # p-values for all levels from ESET
    levels_sig = []         # significant levels by ESET
    levels_eigvals_tested = []  # eigenvalues tested by ESET for all levels
    levels_active_clusters = [] # indices of active clusters at current level
    
    R, A_PC1, PC1, A, A_b, B = rootlets_init(X0, verbose)
    R_element_names = ['corr.', 'ind pairs.']
    B_field_names = ['cluster_indices', 'cluster_i_out', 'cluster_eigs', 'cluster_levels']
    if save_debug_info or display_level_inds:
        levels_active_clusters.append(A.copy())
    save_all_pcs = True if save_debug_info else False
    
    
    pc2_orient_ref = None
    if pc2_orient_ref_i:
        assert pc2_orient_ref_i < X0.shape[0]
        pc2_orient_ref = X0[pc2_orient_ref_i,:]
    
    L = X0.shape[0] - 1
    for l in range(L):
        if verbose: print("...level " + str(l+1) + ":", end='  ')
        if (len(R[0]) == 0) or (PC1.shape[0] == 0) or (len(A) == 0): break
        
        R, A_b, A, B, A_PC1, PC1, d_test, pval, r = rootlets_level(R, A_b, A, B, A_PC1, PC1, l, p_thresh, 
                                                                   pc2_orient_ref, stop_neg_sim,
                                                                   save_all_pcs, verbose)
        levels_pvalues.append(pval)
        if stop_neg_sim and r < 0: 
            break
        if pval <= p_thresh:
            levels_sig.append(l)
        if save_debug_info or display_level_inds:
            levels_sims.append(r)
            levels_eigvals_tested.append(d_test)
            levels_active_clusters.append(A.copy())
        elif display_level_inds:
            levels_active_clusters.append(A.copy())

    if verbose and (l+1 == L):
        print('......all levels of hierarchy constructed.')
    elif verbose and stop_neg_sim:
        print('......all non-negative similarities exhausted, hierarchy construction stopped.')
    
    A, B, c_ij_final, pval_remain, pval_final = rootlets_final(A, B, PC1, l, p_thresh, 
                                                               pc2_orient_ref, save_all_pcs, verbose=verbose)
    if pval_remain is not None:
        levels_pvalues.append(pval_remain)
        if any(np.array(pval_remain) < p_thresh):
            levels_sig.append(l)
    final_sig = pval_final < p_thresh
    
    if verbose:
        if (len(levels_sig) == 0) and not final_sig:
            msg = ".......no hierarchical or clustering structure found in data!"
        elif (len(levels_sig) > 0) and not final_sig:
            msg = ".......multiple uncorrelated hierarchies/clusters found in data!"
        elif (len(levels_sig) == 0) and final_sig:
            msg = ".......only singleton cluster found in data, without hierarchy!"
        else:
            msg = None
        if msg:  print(msg)
    
    hPCA_results = {'levels_pvalues' : levels_pvalues,
                    'levels_sig' : levels_sig,
                    'level_final_pvalue' : pval_final,
                    'level_final_sig' : final_sig,
                    'branches' : B,
                    'branches_field_names' : B_field_names,
                    'pvalue_cutoff' : p_thresh,
                    'save_fname' : save_fname}
    if save_debug_info:
        hPCA_results['levels_similarities'] = levels_sims
        hPCA_results['levels_eigvals_tested'] = levels_eigvals_tested
        hPCA_results['levels_active_clusters'] = levels_active_clusters
    
    if isinstance(save_fname, str):
        if verbose: print("...saving results as:  " + save_fname)

        with open(save_fname, 'wb') as f:
            pickle.dump(hPCA_results, f)
        
    if display_level_inds:
        display_merged_inds_seq(levels_active_clusters)

    return hPCA_results


def create_test_data(T=20, standardize=True, SNR=5,
                     mixing_matrix_fname='simple_hier_example0.csv',
                     save_fname='hPCA_input.npz', verbose=True):
    if mixing_matrix_fname:
        if os.path.isfile(mixing_matrix_fname):
            W = np.genfromtxt(mixing_matrix_fname, delimiter=',')
        else:
            print('ERROR: could not find: ' + str(mixing_matrix_fname))
    else:
        W = np.identity(9)
    K, N = W.shape
    if verbose: 
        msg = "Creating toy hierarchy with " + str(N) + " observed vars., "
        msg += str(K) + " latent vars."
        print(msg)
        print(W)
    if verbose:
        print("Creating time series of length " + str(T) + " for latent vars.")
    Y = norm.rvs(size = K * T).reshape([K, T])
    X = np.matmul(W.T, Y)
    if SNR > 0:
        if verbose:
            print("Adding Gaussian noise (SNR=" + str(SNR) +") to observed vars.")
            X = X * SNR + norm.rvs(size = N * T).reshape([N, T])
    if standardize:
        X = zscore(X, ddof=1, axis=1)
    if verbose: 
        print()
        print("Correlation matrix of observed vars:")
        print(np.around(np.corrcoef(X), decimals=2))
        print()
    if isinstance(save_fname, str):
        if verbose: print("......saving test data as:  " + save_fname)
        hPCA_test_data = {'X0': X,
                          'W' : W,
                          'Y' : Y}
        np.savez_compressed(save_fname, **hPCA_test_data)
        if verbose: print()
    
    return X, W, Y


#########################################
### Create new datasets ###
# X0, W, Y = create_test_data()

data_matrix_path = os.getcwd()
# data_matrix_fname = 'simple_hier_example0.csv'
# data_matrix_fname = 'simple_hier_example1.csv'
# data_matrix_fname = 'simple_hier_example2.csv'
data_matrix_fname = 'complex_hier_example.csv'

save_path = os.getcwd()
data_save_fname = os.path.splitext(data_matrix_fname)[0] + '.npz'
hier_save_fname = 'hPCA_results-' + os.path.splitext(data_matrix_fname)[0] + '.pkl'

data_matrix_fname = os.path.join(data_matrix_path, data_matrix_fname)
data_save_fname = os.path.join(save_path, data_save_fname)
hier_save_fname = os.path.join(save_path, hier_save_fname)

X0, W, Y = create_test_data(T=1200, SNR=2,
                            mixing_matrix_fname=data_matrix_fname,
                            save_fname=data_save_fname, verbose=True)


hPCA_results = rootlets_hPCA(X0, p_thresh=0.001 / X0.shape[0], 
                             save_fname=hier_save_fname)

#########################################



    