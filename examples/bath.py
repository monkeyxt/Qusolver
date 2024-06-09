################################################################################
# File      bath.py
# Date      June 6, 2024
#
# Simple simulation of a system coupled to a thermal bath
################################################################################
import qusolver
import numpy as np
import matplotlib as mp

################################################################################
# System setup
################################################################################
hamiltonian = [[500.0+0j, 0.0+0j], [0+0j, -500.0+0j]]
density = [[0+0j, 0+0j], [0+0j, 0.0+0j]]

dissipation_ops1 = [[0+0j, 0+0j], [1+0j, 0.0+0j]]
dissipation_ops2 = [[0+0j, 1+0j], [0+0j, 0.0+0j]]
dissipation_ops = [dissipation_ops1, dissipation_ops2]

observation_ops1 = [[1+0j, 0+0j], [0+0j, -1.0+0j]]
observation_ops = [observation_ops1]

couplings = [2.0, 0.0]
steps = 10000
dt = 0.1

################################################################################
# Plot measurements
################################################################################
