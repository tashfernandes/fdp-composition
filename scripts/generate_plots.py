from src.graphs import *
import numpy as np
import math

def fig2():
  eps = 1
  delta = 0.1

  filename = 'figures/fig2_f_epsilon_delta.pdf'

  tradeoff_graph(eps, delta)
  print("Writing ", filename)
  save_plot(filename)

def fig3():

  filename = 'figures/fig3_barycentric.pdf'

  C = np.array([ [2/5, 3/5], [4/5, 1/5] ])
  barycentric_graph(C)
  print("Writing ", filename)
  save_plot(filename)

def fig4():

  filename = 'figures/fig4_hockeystick.pdf'

  # Hockey sticks and hypers
  alpha = 0.1
  f1_alpha = 0.8
  f2_alpha = 0.5
  h = 2.3

  hockeystick_refinement(alpha, f1_alpha, f2_alpha, h)
  print("Writing ", filename)
  save_plot(filename)

def fig5():

  filename = 'figures/fig5_hockeystick_posteriors.pdf'

  C = np.array([ [1/4, 1/6, 1/3, 1/4], [1/2, 1/8, 1/8, 1/4]])
  h = 0.6

  hockeystick_posteriors(C, h)
  print("Writing ", filename)
  save_plot(filename)

def fig6():
  
  filename = 'figures/fig6_parallel_composition.pdf'

  eps = math.log(2, math.e)
  delta = 0.05

  parallel_composition(eps, delta)
  print("Writing ", filename)
  save_plot(filename)

def fig7():

  filename = 'figures/fig7_visible_choice.pdf'

  eps = math.log(6, math.e)
  R = qif.randomized_response(2, eps)

  eps2 = math.log(2, math.e)
  delta = 0.1
  C = canonical_channel(eps2, delta)

  p = 0.3

  visible_choice(R, C, p)
  print("Writing ", filename)
  save_plot(filename)

def fig8():

  filename = 'figures/fig8_purification.pdf'

  eps = math.log(3, math.e)
  delta = 0.1
  r = 0.75
  eps_prime = math.log(2, math.e)

  delta_prime = 0.05
  r_prime = 0.7595

  purification(eps, delta, r, eps_prime, delta_prime, r_prime)

  print("Writing ", filename)
  save_plot(filename)

def fig9():

  filename = 'figures/fig9_subsampling.pdf'

  eps = 1
  delta = 0.1
  gamma = 0.2

  subsampling(eps, delta, gamma)
  print("Writing ", filename)
  save_plot(filename)


if __name__ == "__main__":

  # Create the figures directory to store images
  from pathlib import Path
  Path("figures").mkdir(parents=True, exist_ok=True)

  setup_plot()
  fig2()
  fig3()
  fig4()
  fig5()
  fig6()
  fig7()
  fig8()
  fig9()
