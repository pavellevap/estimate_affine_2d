#cython: language_level=3, boundscheck=False

import numpy as np
cimport numpy as np
from scipy.linalg.cython_lapack cimport sgels
from libc.string cimport memset, memcpy
from libc.stdlib cimport rand, RAND_MAX, malloc, calloc, free
from libc.math cimport log, sqrt
from cpython.array cimport array


# auxiliary parameters for sgels
cdef float work[4198]
cdef int lwork = 4198

cdef double PI = 3.14159265359


cdef void fit(float* p1, float* p2, int matches_count, int* matches, float* model) nogil:
    """
    Estimates affine transformation by solving overdetermined system of linear equations.
    Sensitive to noise.

    in:  p1 - n x 2 array of source points
    in:  p2 - m x 2 array of destination points
    in:  matches_count - number of matches
    in:  matches - matches_count x 2 array of correspondences. 
         p1[matches[i][0]], p2[matches[i][1]] - corresponding points
    out: res - 2 x 3 estimated affine matrix
    """
    cdef int i, j
    cdef int n = 6, m = matches_count * 2
    cdef int ms[6]
    for i in range(6):
        ms[i] = m * i
    cdef float* a = <float *> calloc(m * 6, sizeof(float))
    cdef float* b = <float *> malloc(m * sizeof(float))
    cdef int info, nrhs = 1, lda = m, ldb = m
    for i in range(matches_count):
        j = i * 2
        a[ms[0] + j] = a[ms[3] + j + 1] = p1[matches[j] * 2 + 0]
        a[ms[1] + j] = a[ms[4] + j + 1] = p1[matches[j] * 2 + 1]
        a[ms[2] + j] = a[ms[5] + j + 1] = 1
        b[j + 0] = p2[matches[j + 1] * 2 + 0]
        b[j + 1] = p2[matches[j + 1] * 2 + 1]
    sgels('N', &m, &n, &nrhs, a, &lda, b, &ldb, work, &lwork, &info)
    memcpy(model, b, 6 * sizeof(b[0]))
    free(a)
    free(b)


cdef void compute_intervals(int n1, int* l1, int n2, int* l2, int* intervals_count, int* intervals) nogil:
    """
    Finds indices to l1 and l2 arrays containing the same values.

    in:  n1 - size of l1 array
    in:  l1 - sorted array of numbers (labels)
    in:  n2 - size of l2 array
    in:  l2 - sorted array of numbers (labels)
    out: intervals_count - number of common labels in l1 and l2 arrays
    out: intervals - intervals_count x 4 array of intervals.
         i1, j1, i2, j2 = intervals[k, :]
         l1[i1: i2], l2[j1: j2] - subarrays containing the same label
    """
    cdef int i = 0, j = 0
    cdef int count = 0
    while i < n1 and j < n2:
        if l1[i] < l2[j]:
            i += 1
            continue
        elif l1[i] > l2[j]:
            j += 1
            continue
        
        intervals[count + 0] = i
        intervals[count + 1] = j
        i += 1
        while i < n1 and l1[i] == l1[i - 1]:
            i += 1
        j += 1
        while j < n2 and l2[j] == l2[j - 1]:
            j += 1
        intervals[count + 2] = i
        intervals[count + 3] = j
        count += 4
    intervals_count[0] = count >> 2
    

cdef void sample_labels(int count, int* indices) nogil:
    """
    Samples three distinct numbers from 0 to count
    
    in:  count - upper limit for sampled numbers (not inclusive)
    out: indices - sampled numbers
    """
    cdef int i, i1, i2, i3, r
    cdef int aux[16]
    if count <= 16:
        # fill auxiliary array with numbers from 0 to count
        for i in range(count):
            aux[i] = i
        # sample three random indices to auxiliary array
        r = rand()
        i1 = (r & 15) % count 
        r >>= 4
        i2 = (r & 15) % (count - 1)
        r >>= 4
        i3 = (r & 15) % (count - 2)
        # pick number from i1 position
        indices[0] = aux[i1]
        # and swap it with the last element of array so that it will not be picked again
        aux[i1] = aux[count-1]
        indices[1] = aux[i2]
        aux[i2] = aux[count-2]
        indices[2] = aux[i3]
    else:
        # pick random number from 1 to count
        indices[0] = rand() % count
        # pick the secont number
        indices[1] = rand() % count
        # if it is the same as first, pick again
        while indices[1] == indices[0]:
            indices[1] = rand() % count
        indices[2] = rand() % count
        while indices[2] == indices[0] or indices[2] == indices[1]:
            indices[2] = rand() % count
    

cdef void sample_matches(int intervals_count, int* intervals, int* indices) nogil:
    """
    Samples three pair of indices. Each pair corresponds to points with the same labels.
    Labels do not repeat.

    in:  intervals_count - number of intervals (common labels)
    in:  intervals - intervals_count x 4 array of intervals.
         i1, j1, i2, j2 = intervals[k, :]
         (i1..i2), (j1..j2) correspond to point indices with the same label
    out: indices - 3 x 2 array of sampled matches
    """
    # pick 3 distinct labels
    cdef int label_indices[3]
    sample_labels(intervals_count, label_indices)
    
    cdef int i, j, l1, r1, l2, r2
    for i in range(3):
        # pick random number from each interval
        j = label_indices[i] * 4
        l1 = intervals[j + 0]
        l2 = intervals[j + 1]
        r1 = intervals[j + 2]
        r2 = intervals[j + 3]
        j = i * 2
        indices[j + 0] = l1 + rand() % (r1 - l1)
        indices[j + 1] = l2 + rand() % (r2 - l2)
        

cdef void find_inliers(float* p1, float* p2, int intervals_count, int* intervals, 
                       float threshold, float* model, int* inliers_count, int* inliers) nogil:
    """
    Find matches that satisfy model (affine transformation). 
    For each point in p1 array there will be at most 1 correspondence.

    in:  p1 - n x 2 array of point coordinates 
    in:  p2 - m x 2 array of point coordinates 
    in:  intervals_count - number of intervals (common labels)
    in:  intervals - intervals_count x 4 array of intervals.
         i1, j1, i2, j2 = intervals[k, :]
         points p1[i1: i2], p2[j1: j2] have the same labels (visual words)
    in:  threshold - matches with reprojection error higher than this threshold will be considered outliers
    in:  model - 2 x 3 affine transformation matrix
    out: inliers_count - number of detected inliers
    out: inliers - inliers_count x 2 array of matches. 
         p1[inliers[i][0]], p2[inliers[i][1]] - corresponding points
    """
    cdef int l1, r1, l2, r2
    cdef int i, j1, j2, best_j2
    cdef float x, y, d, best_d
    cdef float t2 = threshold * threshold
    cdef int count = 0
    for i in range(intervals_count):
        l1 = intervals[i * 4 + 0]
        l2 = intervals[i * 4 + 1]
        r1 = intervals[i * 4 + 2]
        r2 = intervals[i * 4 + 3]
        for j1 in range(l1, r1):
            # find point projection
            x = model[0] * p1[j1 * 2 + 0] + model[1] * p1[j1 * 2 + 1] + model[2]
            y = model[3] * p1[j1 * 2 + 0] + model[4] * p1[j1 * 2 + 1] + model[5]
            best_d = t2 + 1
            best_j2 = 0
            for j2 in range(l2, r2):
                # find reprojection error
                d =  (x - p2[j2 * 2 + 0]) * (x - p2[j2 * 2 + 0])
                d += (y - p2[j2 * 2 + 1]) * (y - p2[j2 * 2 + 1])
                if d < best_d:
                    best_d = d
                    best_j2 = j2
            if best_d < t2:
                inliers[count + 0] = j1
                inliers[count + 1] = best_j2
                count += 2
    inliers_count[0] = count >> 1

        
cdef void ransac(int n1, float* p1, int* l1, int n2, float* p2, int* l2, 
                 int max_iter, float threshold, int stop_match, float stop_prob,
                 int* inliers_count, int* inliers, float* model):
    """
    Estimation of affine transformation with RANSAC.

    in:  n1 - number of source points and size of p1 and l1 arrays
    in:  p1 - n1 x 2 array of point coordinates
    in:  l1 - sorted array of point labels of size n1. l1[i] - label of point p1[i]
    in:  n2 - number of destination points and size of p2 and l2 arrays
    in:  p2 - n2 x 2 array of point coordinates
    in:  l2 - sorted array of point labels of size n2. l2[i] - label of point p2[i]
    in:  max_iter - maximum number of ransac iterations
    in:  threshold - maximum reprojection error for inliers
    in:  stop_match - minimal number of detected inliers after which algorithms stops
    in:  stop_prob - algorithm stops after probability of randomly selecting all-inlier 
         triple at least ones becomes larger than this threshold
    out: inliers_count - number ofdetected inliers
    out: inliers - inliers_count x 2 array of detected inliers
    out: model - 2 x 3 affine transformation matrix
    """
    cdef int max_inliers = n1
    cdef int* intervals = <int *> malloc(4 * max_inliers * sizeof(int))
    cdef int intervals_count
    compute_intervals(n1, l1, n2, l2, &intervals_count, intervals)
    
    inliers_count[0] = 0
    if intervals_count < 3:
        return
    
    cdef int i, j
    cdef int matches[6]
    cdef float _model[6]
    cdef int* _inliers = <int *> malloc(2 * max_inliers * sizeof(int))
    cdef int _inliers_count = 0
    cdef float eps, iter_estimate
    cdef int matches_count = 0
    for i in range(intervals_count):
        j = i * 4
        matches_count += (intervals[j + 2] - intervals[j + 0]) * (intervals[j + 3] - intervals[j + 1])
    i = 0
    while i < max_iter and inliers_count[0] < stop_match:
        sample_matches(intervals_count, intervals, matches)
        fit(p1, p2, 3, matches, _model)
        find_inliers(p1, p2, intervals_count, intervals, threshold, _model, &_inliers_count, _inliers)
        if _inliers_count > inliers_count[0]:
            # estimate model on all inliers
            fit(p1, p2, _inliers_count, _inliers, model)
            find_inliers(p1, p2, intervals_count, intervals, threshold, model, inliers_count, inliers)
            if _inliers_count > inliers_count[0]:
                inliers_count[0] = _inliers_count
                memcpy(inliers, _inliers, _inliers_count * 2 * sizeof(inliers[0]))
                memcpy(model, _model, 9 * sizeof(model[0]))

            # update number of iterations based on number of found inliers
            eps = inliers_count[0] / matches_count
            eps = eps * eps * eps
            iter_estimate = 1 + log(1.0 - stop_prob) / log(1.0 - eps)
            if iter_estimate < max_iter:
                max_iter = <int>iter_estimate
        i += 1
        
    free(intervals)
    free(_inliers)
    

def estimate_affine_ransac(points1, labels1, points2, labels2, max_iter=1000, 
                           threshold=1, stop_match=2**31-1, stop_prob=0.9999):
    """
    Estimation of affine transformation with RANSAC.

    points - point coordinates
    labels - point labels (wisual words)
    max_iter - maximum number of ransac iterations
    threshold - maximum reprojection error for inliers
    stop_match - minimal number of detected inliers after which algorithms stops
    stop_prob - algorithm stops after probability of randomly selecting all-inlier 
                triple at least ones becomes larger than this threshold

    returns: model, inliers
    """
    perm1 = np.argsort(labels1)
    points1_sorted = points1[perm1].copy(order='C')
    labels1_sorted = labels1[perm1].copy(order='C')
    perm2 = np.argsort(labels2)
    points2_sorted = points2[perm2].copy(order='C')
    labels2_sorted = labels2[perm2].copy(order='C')
    
    cdef int n1 = len(points1_sorted)
    cdef float[:, :] p1 = points1_sorted
    cdef int[:] l1 = labels1_sorted
    cdef int n2 = len(points2_sorted)
    cdef float[:, :] p2 = points2_sorted
    cdef int[:] l2 = labels2_sorted
    
    inliers = np.zeros((min(len(points1), len(points2)), 2), dtype=np.int32)
    cdef int[:, :] inliers_view = inliers
    cdef int inliers_count
    
    model = np.zeros((2, 3), dtype=np.float32)
    cdef float[:, :] model_view = model

    count = ransac(n1, &p1[0][0], &l1[0], n2, &p2[0][0], &l2[0], 
                   max_iter, threshold, stop_match, stop_prob, 
                   &inliers_count, &inliers_view[0][0], &model_view[0][0])
    
    inliers[:inliers_count, 0] = perm1[inliers[:inliers_count, 0]] 
    inliers[:inliers_count, 1] = perm2[inliers[:inliers_count, 1]] 
    
    return model, inliers[:inliers_count]
    

def estimate_affine(points1, points2):
    """
    Estimation of affine transformation by solvin a system of linear equations.
    Sensitive to noise.

    points1[i], points2[i] - corresponding points
    returns: model - matrix of affine transformation
    """
    model = np.zeros((2, 3), dtype=np.float32)
    matches = np.array([[i, i] for i in range(len(points1))], dtype=np.int32)
    cdef float[:, :] p1 = points1
    cdef float[:, :] p2 = points2
    cdef int[:, :] mat = matches
    cdef float[:, :] mod = model
    fit(&p1[0][0], &p2[0][0], 3, &mat[0][0], &mod[0][0])
    return model

