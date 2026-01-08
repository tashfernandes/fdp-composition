import numpy as np
import math

#
# uniform
#
# Returns an array representing a uniform distribution of
# the given size.
#

def uniform(size):
  return np.array([ 1/size ] * size)

# 
# hyper
#
# Given a channel C and a prior pi, returns an array 
# containing the marginal on y and the hyper [pi > C]
#

def hyper(C, pi):
  J = C.T * pi
  # J is transposed, y_marginals are now on "x" axis.
  y_marginal = np.sum(J, axis=1)
  H = J.T / y_marginal
  return tuple([ y_marginal, H ])

#
# vg_prior
#
# Given a prior pi and a gain function G represented as a 
# matrix (WxG), returns the maximum expected gain (i.e. the
# prior g-vulnerability).
#

def vg_prior(pi, G):
  return np.max(G @ pi)

#
# vg_posterior
#
# Given a channel C, prior pi and gain function G represented
# as a matrix (WxG), returns the posterior g-vulnerability.
#

def vg_posterior(C, pi, G):
  H = hyper(C, pi)
  max_vg = sum(
    w * vg_prior(h, G) for w, h in zip(H[0], H[1].T)
  )
  return max_vg

#
# parallel
#
# Computes the parallel composition of 2 channels assuming that
# they have the same X dimension.
#

def parallel(C, D):
  # Pre: C and D have the same number of rows
  return np.einsum("xy,xz->xyz", C, D).reshape(C.shape[0], -1)

#
# randomized_response
#
# Produces a randomized response channel using the given epsilon
# and channel size. The produced channel is square (size x size).
#

def randomized_response(size, eps):
  e_eps = math.e ** eps
  b = 1 / (e_eps + size - 1)
  a = b * e_eps

  return np.identity(size) * (a - b) + np.ones((size, size)) * b

#
# truncated_geometric
#
# Produces a channel representing the truncated geometric
# mechanism with epsilon=eps. The produced channel is square
# of dim size x size.
#

def truncated_geometric(size, eps):
  # Pre: size > 0, eps >= 0

  # The truncated geometric can be calculated as follows:
  # G_{x,y} = (1-a)/(1+a) * a^d(x,y) for 0 < y < size-1
  # G_{x,y} = 1/(1+a) * a^d(x,y) for y = 0, size-1
  # where a = e ** (-eps)
  a = math.e ** (-eps)
  G = np.zeros((size, size))
  for x in range(size):
    for y in range(size):
      if y == 0 or y == (size-1):
        G[x][y] = a ** (abs(x-y)) / (1+a)
      else:
        G[x][y] = a ** (abs(x-y)) * (1-a) / (1+a)
  return G
