"""Helper functions for (1) circuit simulation using qiskit and (2) optimizing the parameters beta, gamma."""
from qiskit import Aer, transpile
import numpy as np
import scipy.optimize as opt


class Simulation:

    backend = Aer.get_backend("aer_simulator_statevector")

    def get_statevector(self, transpiled_circuit, parameter_dict):
        bound_circuit = transpiled_circuit.bind_parameters(parameter_dict)
        result = self.backend.run(bound_circuit, shots = 1).result()
        statevector = result.get_statevector()
        return statevector


class Optimization:
    
    def optimize_angles(self, func, gamma_range, beta_range, p):
        """Optimize the parameters beta, gamma for a given function func, SHGO is searching for minimum"""
        bounds = np.array([gamma_range, beta_range] * p)
        result = opt.shgo(func, bounds, iters = 3)
        return result.x






