from .core import *
from .alg import canonical_channel
from . import qif

import math
import numpy as np
import matplotlib.pyplot as plt

# Functions to generate graphs from the paper.

COLORS = ['#898989', '#116594', '#fdae61', '#1a9850', '#4daf4a', '#984ea3']
CM = 1/2.54  # centimeters in inches
FS = 16
PTSIZE = 70

def setup_plot():
  plt.rcParams.update({"text.usetex": True, "font.family": "Helvetica"})
  plt.rcParams.update({'pgf.preamble': r'\usepackage{amsmath}'})
  plt.rc('text.latex', preamble=r'\usepackage{sfmath, amssymb}')
  plt.rcParams['font.sans-serif'] = ['Tahoma', 'Lucida Grande', 'Verdana']
  plt.rcParams['xtick.labelsize'] = FS 
  plt.rcParams['ytick.labelsize'] = FS

def save_plot(filename):
  plt.tight_layout()
  plt.savefig(filename, format="pdf")

def show_plot():
  plt.show()

# 
# tradeoff_graph
#
# Generates Fig. 2.
# Given an epsilon and delta, draw the trade-off function f_{\epsilon, \delta}.
#

def tradeoff_graph(eps, delta):

  # We use the canonical trade-off channel from Eqn (9)
  C = canonical_channel(eps, delta)
  f = compute_f(C)
  
  # Now plot the trade-off function f
  fig, ax = plt.subplots(figsize=(12*CM, 12*CM))
  ax.set_xlim(0, 1)
  ax.set_ylim(0, 1)
  ax.grid(visible=True)
  ax.set_xlabel(r'$\alpha$', fontsize=FS)
  
  plt.xticks(np.arange(0, 1.1, step=0.25))
  plt.yticks(np.arange(0, 1.1, step=0.25))
  plt.plot(f[0], f[1], color=COLORS[1], label="$f_{\\epsilon, \\delta}, \\epsilon = " + str(eps) + ", \\delta = " + str(delta) + "$")
  plt.legend(fontsize="xx-large")

#
# barycentric_graph
#
# Generates Fig. 3.
# Given a 2x2 channel C, computes the barycentric representation
# for C under a uniform prior, showing the weighted
# average g-vulnerability over posteriors.
# 

def Vg(x):
    # Pre: x is a probability
    # We construct a g-leakage measure manually.
    G = np.array([[0.6, 0], [0, 1]])
    pi = np.array([x, 1-x])
    return qif.vg_prior(pi, G)

def midway(p1, p2):
    # Computes where line connecting these points hits x=1/2
    # Eqn is y = (x - x1) * (y2 - y1) / (x2 - x1) + y1
    return (1/2 - p1[0]) * (p2[1] - p1[1]) / (p2[0] - p1[0]) + p1[1]

def barycentric_graph(C):
  # Pre: C is a 2x2 channel
  fig, ax = plt.subplots(figsize=(16*CM, 12*CM))
  ax.set_xlim(0, 1)
  ax.set_ylim(-0.02, 1.2)
  ax.set_xlabel(r'Probability ${\delta}$', fontsize=FS, labelpad=10)
  ax.set_ylabel(r'Vulnerability $V_g$', fontsize=FS, labelpad=15)
  #plt.xticks(fontsize=FS)
  #plt.yticks(fontsize=FS)
  
  X = np.linspace(0, 1, 100)
  Vg_vec = np.vectorize(Vg)
  # Plot Vg
  plt.plot(X, Vg_vec(X), color=COLORS[1], label='$V_g({\delta})$', linewidth=2)
  
  # Now plot hypers for our channel C
  y_marginal = np.array([ C[0][0]+C[1][0], C[0][1]+C[1][1] ])
  hyper1_x = C[0][0] / y_marginal[0]
  hyper2_x = C[0][1] / y_marginal[1]
  # x_vals contains the x values for the hypers
  x_vals = np.array([ hyper1_x, hyper2_x ])
  intercept = Vg_vec(x_vals)
  plt.vlines(x_vals[0], 0, intercept[0], color=COLORS[2], linestyle='--')
  plt.vlines(x_vals[1], 0, intercept[1], color=COLORS[2], linestyle='--')
  
  plt.scatter(x_vals, [0, 0], s=50, color=COLORS[2], edgecolor='black', marker='o', label=r'Posteriors $\delta_0, \delta_1$', zorder=2)
  
  # Now compute the average
  plt.plot(x_vals, intercept, color=COLORS[0], linestyle='dashdot', linewidth=2)
  avgVg = midway([x_vals[0], intercept[0]], [x_vals[1], intercept[1]])
  plt.vlines(1/2, 0, avgVg, color=COLORS[0], linestyle='dotted', linewidth=2)
  plt.hlines(avgVg, 0, 1/2, color=COLORS[0], linestyle='dotted', linewidth=2)
  plt.scatter(1/2, avgVg, color=COLORS[0], s=50, marker='D', edgecolor='black', label=r'$V_g[u \triangleright C]$', zorder=2)
  plt.legend(fontsize="xx-large")


#
# hockeystick_refinement
#
# Generates Fig. 4.
# Constructs channels according to given alpha values, and computes
# the h-vulnerability on posteriors of each channel under a uniform
# prior, with the corresponding hockeystick function. This shows
# how the hockeystick function is related to refinement.
#

def hockeystick_refinement(alpha, f1_alpha, f2_alpha, h):
  # Compute the channels corresponding to the given alpha values
  C = np.array([ [1-alpha, alpha], [f1_alpha, 1-f1_alpha]])
  M = np.array([ [1-alpha, alpha], [f2_alpha, 1-f2_alpha]])
  
  # Compute the channel hypers wrt a uniform distribution
  HC = qif.hyper(C, qif.uniform(2))
  HM = qif.hyper(M, qif.uniform(2))

  # Use the given h value to create the h-vulnerability function
  h_function = np.array([[-(h-1), 1], [0, 0]])
  C_hvuln = qif.vg_posterior(C, qif.uniform(2), h_function)
  M_hvuln = qif.vg_posterior(M, qif.uniform(2), h_function)

  fig, ax = plt.subplots(figsize=(18*CM, 12*CM))
  ax.set_xlim(0, 1)
  ax.set_ylim(-0.05, 1)
  
  X = np.linspace(0, 1, 100)
  # hockey stick
  plt.plot(X, np.maximum(1-h*X, 0), color=COLORS[3], label='Hockey stick fn')
  # points of hypers
  plt.scatter(HC[1][0], [0, 0], s=PTSIZE, color=COLORS[1], edgecolor='black', label=r'Posteriors of $C^{\alpha}$', zorder=2)
  plt.scatter(HM[1][0], [0, 0], s=PTSIZE, color=COLORS[2], edgecolor='black', label=r'Posteriors of $M^{\alpha}$', zorder=2)
  
  # Vertical lines for HC
  H10_intercept = max(1-h*HC[1][0][0], 0)
  H11_intercept = max(1-h*HC[1][0][1], 0)
  plt.vlines(HC[1][0][0], 0, H10_intercept, color=COLORS[1], linestyle='--', linewidth=2)
  plt.vlines(HC[1][0][1], 0, H11_intercept, color=COLORS[1], linestyle='--', linewidth=2)
  
  # Vertical lines for HM
  H20_intercept = max(1-h*HM[1][0][0], 0)
  H21_intercept = max(1-h*HM[1][0][1], 0)
  plt.vlines(HM[1][0][0], 0, H20_intercept, color=COLORS[2], linestyle='--', zorder=1, linewidth=2)
  plt.vlines(HM[1][0][1], 0, H21_intercept, color=COLORS[2], linestyle='--', zorder=1, linewidth=2)
  
  # Connect lines and average
  plt.plot(HC[1][0], np.array([ H10_intercept, H11_intercept]), linestyle='--', color=COLORS[1], zorder=1, linewidth=2)
  plt.plot(HM[1][0], np.array([ H20_intercept, H21_intercept]), color=COLORS[2], linestyle='--', zorder=1, linewidth=2)
  
  # Print the means
  plt.vlines(0.5, 0, 1, color=COLORS[0], linestyle='--', zorder=1, linewidth=2)
  # Plot the value of the vulnerability function for C and M. Only one label required.
  plt.scatter(0.5, C_hvuln, color=COLORS[0], s=PTSIZE, marker='D', edgecolor='black', label=r'Posterior $V_{\underline{h}}$', zorder=2)
  plt.scatter(0.5, M_hvuln, color=COLORS[0], s=PTSIZE, marker='D', edgecolor='black', zorder=2)
  plt.legend(fontsize="xx-large")  


#
# hockeystick_posteriors
#
# Generates Fig. 5.
#

def hockeystick_posteriors(C, h):

  # Compute alpha from h
  [ alpha, f_alpha ] = compute_alpha(C, h)
  # Turn this into C^alpha
  C_alpha = np.array([ [1-alpha, alpha], [f_alpha, 1-f_alpha]])
  # Now we want to compute the h_vulnerability for the h that gave us C^alpha
  h_function = np.array([[-h, 1], [0, 0]])
  C_hvuln = qif.vg_posterior(C, qif.uniform(2), h_function)
  C_alpha_hvuln = qif.vg_posterior(C_alpha, qif.uniform(2), h_function)
  # Also compute the posteriors so we can plot them
  H = qif.hyper(C, qif.uniform(2))
  H_alpha = qif.hyper(C_alpha, qif.uniform(2))

  fig, (ax, ax2) = plt.subplots(1, 2, figsize=(30*CM, 10*CM))
  ax.set_xlim(0.2, 1)
  ax2.set_xlim(0.2, 1)
  ax.set_ylim(-0.05, 0.6)
  ax2.set_ylim(-0.05, 0.6)
  
  X = np.linspace(0, 1, 100)
  # hockey stick
  ax.plot(X,  np.maximum(1-X*(h+1), 0), color=COLORS[3], label='Hockey stick fn', zorder=1, linewidth=2)
  ax2.plot(X, np.maximum(1-X*(h+1), 0), color=COLORS[3], label='Hockey stick fn', zorder=1, linewidth=2)
  # points of hypers
  ax.scatter(H[1][0], [0]*len(H[1][0]), s=PTSIZE, color=COLORS[1], edgecolor='black', label='Posteriors of C', zorder=2)
  ax2.scatter(H_alpha[1][0], [0]*len(H_alpha[1][0]), s=PTSIZE, color=COLORS[2], edgecolor='black', label=r'Posteriors of $C^\alpha$', zorder=2)
  
  # Vertical lines for H
  H_intercepts = []
  for i in range(len(H[1][0])):
      H_intercept = max(1-H[1][0][i]*(h+1), 0)
      H_intercepts.append(H_intercept)
      ax.vlines(H[1][0][i], 0, H_intercept, color=COLORS[1], linestyle='--', zorder=1, linewidth=2)
  
  # Vertical lines for H_alpha
  H_alpha_intercepts = []
  for i in range(len(H_alpha[1][0])):
      H_intercept = max(1-H_alpha[1][0][i]*(h+1), 0)
      H_alpha_intercepts.append(H_intercept)
      ax2.vlines(H_alpha[1][0][i], 0, H_intercept, color=COLORS[2], linestyle='--', zorder=1, linewidth=2)
  
  ax2.plot(H_alpha[1][0], np.array(H_alpha_intercepts), color=COLORS[2], linestyle='--', linewidth=2)
  
  # Print the mean
  ax.vlines(0.5, 0, 1, color=COLORS[0], linestyle='--', zorder=1, linewidth=2)
  ax2.vlines(0.5, 0, 1, color=COLORS[0], linestyle='--', zorder=1, linewidth=2)
  # Plot the value of the vulnerability function
  ax.scatter(0.5, C_hvuln, color=COLORS[0], s=PTSIZE, marker='D', edgecolor='black', label=r'$V_{\underline{h}}[u \triangleright C]$', zorder=2)
  ax2.scatter(0.5, C_alpha_hvuln, color=COLORS[0], s=PTSIZE, marker='D', edgecolor='black', label=r'$V_{\underline{h}}[u \triangleright C^{\alpha}]$', zorder=2)
  
  # Weighted averages for h_alpha
  h_alpha_avg = 0
  h_alpha_x = 0
  for i in range(len(H_alpha_intercepts)):
      h_alpha_avg += H_alpha_intercepts[i] * H_alpha[0][i]
      h_alpha_x += H_alpha[0][i] * H_alpha[1][0][i]
  
  # Compute the weighted averages for H
  weighted_avg = 0
  weighted_sum = 0
  weighted_avg_zero = 0
  weighted_hvuln = 0
  #weighted_remains = 0
  for i in range(len(H_intercepts)):
      if H_intercepts[i] > 0:
          # This one contributes to v_h
          weighted_avg += H[0][i] * H[1][0][i]
          weighted_sum += H[0][i]
          weighted_hvuln += (H_intercepts[i] * H[0][i])
      else:
          weighted_avg_zero += H[0][i] * H[1][0][i]
  
  # Normalise!!
  weighted_avg = weighted_avg / weighted_sum
  weighted_hvuln = weighted_hvuln / weighted_sum
  weighted_avg_zero = weighted_avg_zero / (1 - weighted_sum)
  
  ax.scatter([weighted_avg], [weighted_hvuln], color=COLORS[2], s=PTSIZE, marker='o', edgecolor='black', label="Weighted avg")
  ax.plot([weighted_avg, weighted_avg_zero], [weighted_hvuln, 0], color=COLORS[2],  zorder=1, linestyle='--', linewidth=2)
  
  # Annotations - arrows
  ax.annotate("", xytext=(H[1][0][0], H_intercepts[0]), xy=(weighted_avg, weighted_hvuln), arrowprops=dict(arrowstyle="->", color='black', shrinkA=5, shrinkB=5, connectionstyle="angle3,angleA=0,angleB=90"))
  ax.annotate("", xytext=(H[1][0][1], H_intercepts[1]), xy=(weighted_avg, weighted_hvuln), arrowprops=dict(arrowstyle="->", color='black', shrinkA=5, shrinkB=5, connectionstyle="angle3,angleA=90,angleB=0"))
  ax.annotate("", xytext=(H[1][0][3], H_intercepts[3]), xy=(weighted_avg, weighted_hvuln), arrowprops=dict(arrowstyle="->", color='black', shrinkA=5, shrinkB=5, connectionstyle="angle3,angleA=0,angleB=90"))
  
  ax.legend(fontsize="xx-large")
  ax2.legend(fontsize="xx-large")



#
# parallel_composition
#
# Generates Fig. 6.
# Computes the trade-off functions for parallel composition
# against the trade-off function for traditional eps,delta composition.
# This shows that parallel composition gives a better approximation for
# the privacy loss than the (loose) epsilon,delta composition.
#

def parallel_composition(eps, delta):
  C = canonical_channel(eps, delta)
  Ced = abstract_channel(C)
  # Parallel composition
  CCed = qif.parallel(Ced, Ced)

  # Get the trade-off functions for composition
  fC = compute_f(Ced)
  fCC = compute_f(CCed)

  # Traditional eps, delta trade-off
  C2 = canonical_channel(2*eps, 2*delta)
  ftwoC = compute_f(C2)

  # Plot the different trade-off functions
  fig, ax = plt.subplots(figsize=(13*CM, 11*CM))
  ax.set_xlim(0, 1)
  ax.set_ylim(-0.02, 1)
  ax.set_xlabel(r'$\alpha$', fontsize=FS, labelpad=10)
  ax.set_ylabel(r'$f(\alpha)$', fontsize=FS, labelpad=15)
  
  plt.plot(fC[0], fC[1], color=COLORS[1], label=r'$\cal{T}(C_{\epsilon, \delta})$', linestyle="dashed", linewidth=2)
  plt.plot(fCC[0], fCC[1], color=COLORS[2], label=r'$\cal{T}(C_{\epsilon, \delta}~||~C_{\epsilon, \delta})$', linewidth=2)
  plt.plot(ftwoC[0], ftwoC[1], color=COLORS[3], label=r'$\cal{T}{(C_{\mathrm{2}\epsilon, \mathrm{2}\delta})}$', linestyle="dotted", linewidth=2)
  
  ax.legend(fontsize="xx-large")


#
# visible_choice
#
# Generates Fig. 7.
# Computes the visible probabilistic choice between channels
# C and D using probability r for channel C.
#

def visible_choice(C, D, r):

  fC = compute_f(C)
  fD = compute_f(D)

  # We need to flip the functions around for numpy interpolation
  Cflipped = np.flip(fC, axis=1)
  Dflipped = np.flip(fD, axis=1)

  X = np.linspace(0, 1, 100)
  Cy = np.interp(X, Cflipped[0], Cflipped[1])
  Dy = np.interp(X, Dflipped[0], Dflipped[1])

  XChoice = X
  YChoice = r * Cy + (1 - r) * Dy

  fig, ax = plt.subplots(figsize=(13*CM, 11*CM))
  ax.set_xlim(0, 1)
  ax.set_ylim(-0.02, 1)
  ax.set_xlabel(r'$\alpha$', fontsize=FS, labelpad=10)
  ax.set_ylabel(r'$f(\alpha)$', fontsize=FS, labelpad=15)
  
  plt.plot(fC[0], fC[1], color=COLORS[1], label=r'$\cal{T}(C_{\epsilon, \delta})$', linestyle="dashed", linewidth=2)
  plt.plot(fD[0], fD[1], color=COLORS[2], label=r"$\cal{T}(C_{\epsilon', \delta'})$", linewidth=2)

  plt.plot(XChoice, YChoice, color=COLORS[3], label=r"$\cal{T}(C_{\epsilon, \delta})~ {_\mathrm{r}}{\oplus}~ \cal{T}(C_{\epsilon', \delta'})$", linestyle="dotted", linewidth=2)
  ax.legend(fontsize="xx-large")


# 
# purification
#
# Generates Fig. 8.
#
# This shows the effect of the purification algorithm
# on (epsilon, delta) channels.
#

def purification(eps, delta, r, eps_prime, delta_prime, r_prime):
  # Pre: eps >= 0, 0 < delta < 1, 0 <= r <= 1

  # We first do the case where the output domains are the same.
  C = canonical_channel(eps, delta)
  # Create the uniform "channel" on 2 secrets
  UP = qif.uniform(4)
  U = np.array( [UP, UP] )
  # we first do internal choice assuming C, U have same range.
  D = r * C + (1-r) * U
  # And postprocess with a truncated geometric
  G = qif.truncated_geometric(4, eps_prime)
  Z = D @ G

  Cf = compute_f(C)
  Df = compute_f(D)
  Zf = compute_f(Z)

  # Now we do the case where the output domains differ
  I = np.identity(2)
  # We want to put the visible choice of identity around C
  # so as to simulate what happens for symmetric mechanisms
  dI = np.hsplit(delta_prime * I, 2)
  dC = (1 - delta_prime) * C
  C_prime = np.hstack( (dI[0], dC, dI[1]) )

  # Create the uniform "channel" on 2 secrets
  UP = qif.uniform(4)
  U = np.array( [UP, UP] )
  # U only partially overlaps with C_prime, we need to add some zero cols
  U_full = np.hstack( (np.zeros((2,1)), U, np.zeros((2,1))) )

  # now we can compute D as if it is a visible choice
  D_prime = r_prime * C_prime + (1 - r_prime) * U_full
  # And postprocess with a truncated geometric
  G_prime = qif.truncated_geometric(6, eps_prime)
  Z_prime = D_prime @ G_prime

  Dpf = compute_f(D_prime)
  Zpf = compute_f(Z_prime)

  # Now plot
  fig, (ax, ax2) = plt.subplots(1, 2, figsize=(30*CM, 10*CM))
  ax.set_xlim(0, 1)
  ax2.set_xlim(0, 1)
  ax.set_ylim(0, 1)
  ax2.set_ylim(0, 1)
  
  ax.set_xlabel(r'$\alpha$', fontsize=FS, labelpad=10)
  ax2.set_xlabel(r'$\alpha$', fontsize=FS, labelpad=10)
  ax.set_ylabel(r'$f(\alpha)$', fontsize=FS, labelpad=15)
  ax2.set_ylabel(r'$f(\alpha)$', fontsize=FS, labelpad=15)
  ax.grid(visible=True)
  ax2.grid(visible=True)
  
  ax.plot(Cf[0], Cf[1], color='black', label=r'Input $\cal{T}(M)$', linewidth=2, fillstyle='none', marker = 'o')
  ax2.plot(Cf[0], Cf[1], color='black', label=r'Input $\cal{T}(M)$', linewidth=2, fillstyle='none', marker = 'o')
  ax.plot(Df[0], Df[1], color='black', label=r"${\cal T}(M ~{_\mathrm{r}}\boxplus U[{\cal Y}])$", linewidth=2, marker = 'o', linestyle='dashed')
  ax2.plot(Dpf[0], Dpf[1], color='black', label=r"${\cal T}(M ~{_\mathrm{r}}\boxplus U[{\cal Y}'])$", linewidth=2, marker = 'o', linestyle='dashed')
  ax.plot(Zf[0], Zf[1], color='black', label=r"${\cal T}((M ~{_\mathrm{r}}\boxplus U[{\cal Y}])\cdot G_{\epsilon'})$", linewidth=2, fillstyle='none', marker = 'o', linestyle='dotted')
  ax2.plot(Zpf[0], Zpf[1], color='black', label=r"${\cal T}((M ~{_\mathrm{r}}{\boxplus} U[{\cal Y}'])\cdot G_{\epsilon'})$", linewidth=2, fillstyle='none', marker = 'o', linestyle='dotted')
  ax.legend(fontsize="xx-large")
  ax2.legend(fontsize="xx-large")


#
# subsampling
#
# Generates Fig. 9.
#

def subsampling(eps, delta, gamma):
  # Pre: eps >= 0, 0 < delta < 1, 0 <= gamma <= 1
  C = canonical_channel(eps, delta)
  f = compute_f(C)

  # P is the sub-sampling channel
  P = np.array([ [1, 0], [1-gamma, gamma]])
  PC = P @ C
  Pf = compute_f(PC)

  fig, ax = plt.subplots(figsize=(13*CM, 11*CM))
  ax.set_xlim(0, 1)
  ax.set_ylim(0, 1)
  ax.set_xlabel(r'$\alpha$', fontsize=FS, labelpad=10)
  ax.set_ylabel(r'$f(\alpha)$', fontsize=FS, labelpad=15)
  ax.grid(visible=True)
  
  plt.xticks(np.arange(0, 1.1, step=0.25))
  plt.yticks(np.arange(0, 1.1, step=0.25))
  plt.plot(f[0], f[1], color='blue', label=r'$\cal{T}(M)$', linewidth=2, marker = 'o')
  plt.plot(Pf[0], Pf[1], color='red', label=r"$\cal{T}(P{\cdot}M)$", linewidth=2, marker = 's')
  ax.legend(fontsize="xx-large")
