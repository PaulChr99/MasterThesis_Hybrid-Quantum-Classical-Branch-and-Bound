import numpy as np
from numpy.typing import NDArray
import scipy.optimize as opt
import random
import itertools
from typing import List, Union
from enum import Enum



class ProblemType(Enum):
    maximization = 0
    minimization = 1



class AuxiliaryFunctions():

    def find_binary_variable_representation_for_basis_state_index(self, index_of_one: int, register_sizes: List[int]):
        total_register_size = sum(register_sizes)
        binary_rep = bin(index_of_one)[2:]
        difference = total_register_size - len(binary_rep)
        full_binary_rep = "0"*int(difference) + binary_rep
        return full_binary_rep
    
    

class Measurement(AuxiliaryFunctions):

    num_type = np.cdouble

    def __init__(self, *register_sizes: int):
        self.register_sizes_list = list(register_sizes[0])
    
    def measure_state(self, state: NDArray[num_type]):
        """ Measure both the item and the capacity register """
        probs_dict = {}
        for s in range(len(state)):
            binary_variable_representation = self.find_binary_variable_representation_for_basis_state_index(s, self.register_sizes_list)
            probability = np.abs(state[s])**2
            probs_dict[f"{binary_variable_representation}"] = probability
        
        if not np.isclose(sum(probs_dict.values()), 1, rtol = 1e-05):
            raise ValueError("Even with tolerance all single probabilities don't sum to one.")
        
        return probs_dict
    


class QAOA():

    num_type = np.cdouble

    def __init__(self, problem_type: ProblemType, objective_function, apply_quasiadiabatic_evolution, depth, *register_sizes: int):
        self.measurement = Measurement(register_sizes)
        self.objective_function = objective_function
        self.apply_quasiadiabatic_evolution = apply_quasiadiabatic_evolution
        self.depth = depth
        self.factor_for_optimization = -1 if (problem_type == ProblemType.minimization) else 1
    
    def get_expectation_value(self, probs_dict: dict):
        bitstrings = list(probs_dict.keys())
        objective_function_values = np.array([self.objective_function(bitstring) for bitstring in bitstrings])
        probabilities = np.array(list(probs_dict.values()))
        return self.factor_for_optimization * np.dot(objective_function_values, probabilities)
    
    def angles_to_value(self, angles: List[Union[int, float]]):
        state = self.apply_quasiadiabatic_evolution(angles)
        probs_dict = self.measurement.measure_state(state)
        expectation_value = self.get_expectation_value(probs_dict)
        return - self.factor_for_optimization * expectation_value # Minus is needed for maximization due to Nelder-Mead optimization strategy
    
    def optimize(self):
        print("QAOA running...")
        gamma_range = (0, 2 * np.pi)
        beta_range = (0, np.pi)
        bounds_neldermead_format = (gamma_range, beta_range) * self.depth
        initial_guess = list(itertools.chain.from_iterable([[random.uniform(gamma_range[0], gamma_range[1])] + [random.uniform(beta_range[0], beta_range[1])] for _ in range(self.depth)]))
        optimization_result = opt.minimize(fun = self.angles_to_value, x0 = initial_guess, method = "Nelder-Mead", bounds = bounds_neldermead_format)
        optimal_angles = optimization_result.x
        optimal_value = optimization_result.fun
        return {"optimal angles": optimal_angles, "optimal value": optimal_value}
        #return {"optimal gamma values": optimal_angles[0::2], "optimal beta values": optimal_angles[1::2], "maximal expectation value": max_expectation_value}
        #return max_expectation_value
    
    def execute_qaoa(self):
        optimization_result = self.optimize()
        optimal_angles = optimization_result["optimal angles"]
        optimized_expectation_value = - self.factor_for_optimization * optimization_result["optimal value"]
        state = self.apply_quasiadiabatic_evolution(optimal_angles)
        probs_dict = self.measurement.measure_state(state)
        #print("probs dict = ", probs_dict)
        expectation_value = self.get_expectation_value(probs_dict)
        if not np.isclose(expectation_value, optimized_expectation_value, rtol = 1e-05):
            raise ValueError("Even with tolerance the double-checked expectation value does not match the optimized value.")
        return expectation_value