import numpy as np
import math

#
# sortby_f
#
# Takes a 2xn channel as input and produces a 2xn channel
# with columns sorted by decreasing ratios.
# This allows the trade-off function f to be (more easily)
# computed from the channel by collapsing columns.
#

def sortby_f(ch):
    # Pre: ch is a 2xn channel
    P = ch[0]
    Q = ch[1]
    # allow 0 values for Q
    values = []
    for r in range(len(P)):
        if Q[r] == 0:
            values.append(np.inf)
        else:
            values.append(P[r]/Q[r])
    sorted = np.flip(np.argsort(values))
    newCH = np.array([P[sorted], Q[sorted]])
    # Post: newCH is a channel sorted by decreasing column ratios.
    return newCH

#
# abstract_channel
#
# Takes a 2xn channel as input, and produces a 2xk
# abstract channel as output, by collapsing columns
# which have the same ratio.
# (See definition of abstract channel: WHERE).
#

def abstract_channel(C):
    # Pre: C is a 2xn channel
    C_sorted = sortby_f(C)
    # Now we can collapse adjacent columns
    P = C_sorted[0]
    Q = C_sorted[1]
    # Pf, Qf are output arrays
    Pf = [P[0]]
    Qf = [Q[0]]
    i = 0
    for r in range(len(P)):
        if r == 0:
            continue
        # Need to allow 0 values in Q
        if (Q[r] == 0) and (Q[r-1] == 0):
            Pf[i] += P[r]
        elif (Q[r] != 0) and (Q[r-1] != 0) and (P[r] / Q[r]) == (P[r-1]/Q[r-1]):
            # Collapse adjacent columns which have the same ratio
            Pf[i] += P[r]
            Qf[i] += Q[r]
        else:
            i += 1
            Pf.append(P[r])
            Qf.append(Q[r])
    # Post: [ Pf, Qf ] is an abstract channel sorted by decreasing column ratios.
    return np.array([Pf, Qf])

#
# compute_alpha
#
# Compute the alpha value of a channel given an h.
#

def compute_alpha(C,h):
    # Pre: C is a 2xn channel
    # First we need to sort the channel
    C_sorted = np.flip(sortby_f(C), axis=1)
    alpha = 0
    f_alpha = 0
    for i in range(len(C_sorted[0])):
        if C_sorted[1][i] - h * C_sorted[0][i] >= 0:
            alpha += C_sorted[0][i]
            f_alpha += C_sorted[1][i]
    return [alpha, 1-f_alpha]


# 
# compute_f
#
# Takes a 2xn channel as input and returns an 
# array of (alpha, f(alpha)) values corresponding
# to the "facet" points of the trade-off function f.
#
# See Section 4.1 (facet points).
#

def compute_f(C):
    # Pre: C is a 2xn channel
    C_sorted = abstract_channel(sortby_f(C))
    # C_sorted is ordered by f, so we can sum adjacent columns
    # to read off the face points.
    P = C_sorted[0]
    Q = C_sorted[1]
    alpha = []
    f = []
    for i in range(len(P)+1):
        alpha.append(np.sum(P[i:]))
        f.append(np.sum(Q[:i]))
    # Post: [alpha, f] are the coords of the facet points
    # for the trade-off function corresponding to C.
    return np.array([alpha, f])

