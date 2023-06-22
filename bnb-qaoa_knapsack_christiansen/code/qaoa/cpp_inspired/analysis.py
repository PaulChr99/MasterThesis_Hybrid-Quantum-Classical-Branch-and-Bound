import sys
sys.path.append("C:\\Users\\d92474\\Documents\\Uni\\Master Thesis\\GitHub\\QuBRA\\bnb-qaoa_knapsack_christiansen\\code")

import numpy as np
from numpy.typing import NDArray
import scipy
import scipy.optimize as opt
from typing import List, Union
import time

from circuit import AdiabaticEvolution, Measurement
from knapsack_problem import KnapsackProblem, exemplary_kp_instances



class ProblemRelatedFunctions:

    def __init__(self, problem_instance: KnapsackProblem):
        self.problem_instance = problem_instance

    def bitstring_to_choice(self, bitstring: str):
        bits = np.array(list(map(int, list(bitstring))))
        item_choice = np.array(bits[:self.problem_instance.number_items])
        return item_choice
    
    def objective_function(self, bitstring: str):
        item_choice = self.bitstring_to_choice(bitstring)
        value = np.dot(item_choice, self.problem_instance.profits)
        return value



class QAOA(ProblemRelatedFunctions):

    num_type = np.cdouble

    def __init__(self, problem_instance: KnapsackProblem, depth: int):
        self.problem_instance = problem_instance
        self.depth = depth
        self.circuit = AdiabaticEvolution(self.problem_instance, self.depth)
        self.measurement = Measurement(self.problem_instance)
    
    def get_expectation_value(self, probs_dict: dict):
        #probs_dict = Measurement(self.problem_instance).measure_state(state)
        #print("-----")
        #print("probs dict = ", probs_dict)
        bitstrings = list(probs_dict.keys())
        objective_function_values = np.array([self.objective_function(bitstring) for bitstring in bitstrings])
        probabilities = np.array(list(probs_dict.values()))
        return np.dot(objective_function_values, probabilities)
    
    def angles_to_value(self, angles: List[Union[int, float]]):
        state = self.circuit.apply_adiabatic_evolution(angles)
        probs_dict = self.measurement.measure_state(state)
        expectation_value = self.get_expectation_value(probs_dict)
        return - expectation_value
    
    def optimize(self):
        print("QAOA running...")
        gamma_range = (0, 2 * np.pi)
        beta_range = (0, np.pi)
        bounds_for_optimization = np.array([gamma_range, beta_range] * self.depth)
        #bounds = opt.Bounds([gamma_range[0], beta_range[0]] * self.depth, [gamma_range[1], beta_range[1]] * self.depth)
        bounds_cobyla = (gamma_range, beta_range) * self.depth
        optimization_result = opt.minimize(fun = self.angles_to_value, x0 = [0]*(2*self.depth), method = "Nelder-Mead", bounds = bounds_cobyla)
        optimal_angles = optimization_result.x
        optimal_value = optimization_result.fun
        return {"optimal angles": optimal_angles, "optimal value": optimal_value}
        #return {"optimal gamma values": optimal_angles[0::2], "optimal beta values": optimal_angles[1::2], "maximal expectation value": max_expectation_value}
        #return max_expectation_value
    
    def execute_qaoa(self):
        optimization_result = self.optimize()
        optimal_angles = optimization_result["optimal angles"]
        optimized_expectation_value = - optimization_result["optimal value"]
        state = self.circuit.apply_adiabatic_evolution(optimal_angles)
        probs_dict = Measurement(self.problem_instance).measure_state(state)
        #print(probs_dict)
        expectation_value = self.get_expectation_value(probs_dict)
        if not np.isclose(expectation_value, optimized_expectation_value, rtol = 1e-05):
            raise ValueError("Even with tolerance the double-checked expectation value does not match the optimized value.")
        return expectation_value




def main():
    problem = exemplary_kp_instances["C"]
    #print(ProblemRelatedFunctions(problem).objective_function("10010000000"))
    start_time = time.time()
    print("QAOA result = ", QAOA(problem_instance = problem, depth = 1).execute_qaoa())
    print("Elapsed time = ", time.time() - start_time)


if __name__ == "__main__":
    main()