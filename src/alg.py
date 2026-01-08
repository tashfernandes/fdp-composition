from .core import *
import numpy as np
import math

# Implementation of algorithms/theorems from the paper.

#
# canonical_channel
#
# Returns the canonical channel for a given epsilon and delta.
# See Equation (9) and Corollary 3. 
#

def canonical_channel(eps, delta):
  e_eps = math.e ** eps
  C = np.array([
      [ delta, (1 - delta) * e_eps / (1 + e_eps), (1 - delta) / (1 + e_eps), 0],
      [ 0, (1 - delta) / (1 + e_eps), (1 - delta) * e_eps / (1 + e_eps), delta]
  ])
  return C


#
# tradeoff_channel
#
# Implements Algorithm 2 (Sec 4.1).
#
# Takes as input a trade-off function f described as an 
# array of [alpha, f(alpha)] facet points. (See compute_f).
# 
# Returns a 2xn channel whose trade-off function is f.
#

def tradeoff_channel(f):
    # Pre: f is an array of [alpha, f(alpha)] values.
    # 1. Make sure that f is sorted on increasing alpha
    sorted_f = f[:, f[0].argsort()]
    # 2. Compute first column
    res = np.array([sorted_f[0][0], 1-sorted_f[1][0]])
    # 3. Iterate for each i < N
    for i in range(0, len(sorted_f[0])-1):
        res = np.vstack((res, [sorted_f[0][i+1] - sorted_f[0][i], sorted_f[1][i] - sorted_f[1][i+1]]))
    return np.transpose(res)

#
# compute_min
#
# Implements Proposition 1 (See appendix B).
#

def compute_min(f1, f2):
    # Compute a channel which is the min of trade-off channels from the same f
    if np.shape(f1) != (2, 2) or np.shape(f2) != (2, 2):
        print("Error: Inputs must be 2x2 channels")
        return None

    if alpha(f1) > alpha(f2):
        res = np.array([ [1 - alpha(f1), alpha(f1) - alpha(f2), alpha(f2)], [f_alpha(f1), f_alpha(f2) - f_alpha(f1), 1-f_alpha(f2)]])
        return res
    else:
        res = np.array([ [1 - alpha(f2), alpha(f2) - alpha(f1), alpha(f1)], [f_alpha(f2), f_alpha(f1) - f_alpha(f2), 1-f_alpha(f1)]])
        return res

#
# minCol
#
# Implementation supporting Definition 15 and Proposition 2 
# (See appendix B).
# 
# Takes as input a 2x2 channel C, and returns the minimum
# column, identified by the normalised ratio of the elements
# in the top row.
# 

def minCol(C):
  # Pre: C is a 2x2 (partial) channel
  a = C[0][0]
  ab = a + C[1][0]
  c = C[0][1]
  cd = c + C[1][1]
  if a/ab <= c/cd:
    # first col is the smallest
    return np.array( [ a, ab-a ] )
  else:
    return np.array([ c, cd-c ])

#
# min_order
#
# Implements the partial order of Definition 15. 
#
# Takes as input 2 2x2 (partial) channels and returns the minimum
# in the minCol order.
#

def min_order(C, D):
  # Pre: C, D are 2x2 (partial) channels
  Cmin = minCol(C)
  Dmin = minCol(D)

  if Cmin[0]/np.sum(Cmin) <= Dmin[0]/np.sum(Dmin):
    return C
  return D

