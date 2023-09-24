import sys
sys.path.append("C:\\Users\\d92474\\Documents\\Uni\\Master Thesis\\GitHub\\MasterThesis_Hybrid-Quantum-Classical-Branch-and-Bound\\bnb-qaoa_knapsack_christiansen\\code")


import numpy as np
from numpy.typing import NDArray
from typing import List, Union
import itertools
import random
import scipy.optimize as opt
from copy import deepcopy
import time

from knapsack_problem import KnapsackProblem, GenerateKnapsackProblemInstances, exemplary_kp_instances
from classical_ingredients import SortingProfitsAndWeights, FeasibilityAndPruning, EvaluatingProfitsAndWeights


class QTG:

    def __init__(self, kp_instance: KnapsackProblem):
        self.kp_instance = KnapsackProblem(
            profits = SortingProfitsAndWeights(kp_instance.profits, kp_instance.weights).sorting_profits_weights()["profits"],
            weights = SortingProfitsAndWeights(kp_instance.profits, kp_instance.weights).sorting_profits_weights()["weights"],
            capacity = kp_instance.capacity
        )
        self.feasibility = FeasibilityAndPruning(kp_instance)


    def quantum_tree_generator(self):

        feasible_solutions = ["0" * self.kp_instance.number_items]
        feasible_solutions_amplitudes = [1]
        binary_values_of_feasible_solutions = [0]

        for item_idx in range(self.kp_instance.number_items):
            #print("Item nr. ", item_idx)
            feasible_solutions_at_item_idx = []
            for bitstring in feasible_solutions:
                #print("Bitstring = ", bitstring)
                candidate = bitstring[:item_idx] + "1" + bitstring[(item_idx + 1):]
                #print("Candidate = ", candidate)
                #print("Weight of candidate = ", self.feasibility.calculate_weight(candidate))
                #print("Profit of candidate = ", self.feasibility.calculate_profit(candidate))
                if self.feasibility.is_feasible(candidate):
                    feasible_solutions_at_item_idx.append(candidate)
                    binary_values_of_feasible_solutions.append(int(candidate, 2))
                    feasible_solutions_amplitudes[feasible_solutions.index(bitstring)] *= 1/np.sqrt(2)
                    feasible_solutions_amplitudes.append(feasible_solutions_amplitudes[feasible_solutions.index(bitstring)])
            feasible_solutions += feasible_solutions_at_item_idx
        #print("KP instance = ", self.kp_instance)
        #print("Feasible solutions = ", feasible_solutions)
        print("QTG done")
        return {"binary values of feasible solutions": binary_values_of_feasible_solutions, "feasible solution amplitudes": feasible_solutions_amplitudes}
    


class QuasiAdiabaticEvolution(QTG):

    def __init__(self, kp_instance: KnapsackProblem, depth: int):
        QTG.__init__(self, kp_instance)
        self.depth = depth
        qtg_result = self.quantum_tree_generator()
        self.binary_values_of_feasible_solutions = qtg_result["binary values of feasible solutions"]
        self.feasible_solutions_amplitudes = np.array(qtg_result["feasible solution amplitudes"], dtype = "complex_")
        #print("Sum of feasible solution amplitudes = ", sum([np.abs(amplitude)**2 for amplitude in self.feasible_solutions_amplitudes]))
        evaluation = EvaluatingProfitsAndWeights(self.kp_instance)
        #print("Binary = ", self.binary_values_of_feasible_solutions)
        #print("Translations = ", [bin(binary_value)[2:].zfill(self.kp_instance.number_items) for binary_value in self.binary_values_of_feasible_solutions])
        self.feasible_solutions_profits = [evaluation.calculate_profit(bin(binary_value)[2:].zfill(self.kp_instance.number_items)) for binary_value in self.binary_values_of_feasible_solutions]
        #print("Profits of feasible solutions = ", self.feasible_solutions_profits)
    

    def apply_phase_separation_unitary(self, state: NDArray, gamma: Union[int, float]):
        for idx in range(len(state)):
            state[idx] = np.exp(-1j * gamma * self.feasible_solutions_profits[idx]) * state[idx]
        return state
    

    def apply_mixing_unitary(self, state: NDArray, beta: Union[int, float]):
        scalar_product = sum([self.feasible_solutions_amplitudes[idx].conjugate() * state[idx] for idx in range(len(state))])
        for idx in range(len(state)):
            state[idx] = state[idx] + (np.exp(-1j * beta) - 1) * scalar_product * self.feasible_solutions_amplitudes[idx]
        return state

    
    def apply_quasiadiabatic_evolution(self, angles: List[Union[float, int]]):
        
        if len(angles) != 2 * self.depth:
            raise ValueError("Number of provided values for gamma and beta parameters need to be consistent with specified circuit depth.")
        
        gamma_values = angles[0::2]
        beta_values = angles[1::2]
        
        state = deepcopy(self.feasible_solutions_amplitudes)
        #print("Sum of initial state probabilities = ", sum([np.abs(amplitude)**2 for amplitude in state]))
        for j in range(self.depth):
            state = self.apply_phase_separation_unitary(state, gamma_values[j])
            #print("State after phase separation = ", sum([np.abs(amplitude)**2 for amplitude in state]))
            state = self.apply_mixing_unitary(state, beta_values[j])
            #print("State after mixing = ", sum([np.abs(amplitude)**2 for amplitude in state]))

        return state
    


class QAOA(QuasiAdiabaticEvolution):

    def calculate_expectation_value(self, state: NDArray):
        return sum([self.feasible_solutions_profits[idx] * np.abs(state[idx])**2 for idx in range(len(state))])
    
    def angles_to_value(self, angles: List[Union[float, int]]):
        #print("Initial feasible solution amplitudes = ", self.feasible_solutions_amplitudes)
        angle_state = self.apply_quasiadiabatic_evolution(angles)
        #print("Sum of angle state probabilities = ", sum([np.abs(amplitude)**2 for amplitude in angle_state]))
        #print("Expectation value = ", - self.calculate_expectation_value(angle_state))
        #print("Circuit executed")
        return - self.calculate_expectation_value(angle_state)
    
    def optimize(self):
        print("QAOA running...")
        gamma_range = (0, 2 * np.pi)
        beta_range = (0, np.pi)
        bounds_neldermead_format = (gamma_range, beta_range) * self.depth
        initial_guess = list(itertools.chain.from_iterable([[random.uniform(gamma_range[0], gamma_range[1])] + [random.uniform(beta_range[0], beta_range[1])] for _ in range(self.depth)]))
        optimization_result = opt.minimize(fun = self.angles_to_value, x0 = initial_guess, method = "Nelder-Mead", bounds = bounds_neldermead_format)
        optimal_angles = optimization_result.x
        optimal_value = - optimization_result.fun # Minus needed for optimization via Nelder-Mead minimum finding
        return optimal_value





def main():
    kp_instance = exemplary_kp_instances["D"]
    random_kp_instance = GenerateKnapsackProblemInstances.generate_random_kp_instance_for_capacity_ratio(
        size = 30,
        desired_capacity_ratio = 0.05,
        maximum_value = 100
    )
    print("Random KP instance = ", random_kp_instance)
    large_kp_instance_data = open(
        "C:\\Users\\d92474\\Documents\\Uni\\Master Thesis\\GitHub\\MasterThesis_Hybrid-Quantum-Classical-Branch-and-Bound\\bnb-qaoa_knapsack_christiansen\\code\\kp_instances_data\\uncorrelated\\100.txt", 
        "r"
    ).readlines()
    large_kp_instance = KnapsackProblem(
        profits = [int(line.split()[0]) for line in large_kp_instance_data[1:-1]],
        weights = [int(line.split()[1]) for line in large_kp_instance_data[1:-1]],
        capacity = int(large_kp_instance_data[0].split()[1])
    )
    #print(QTG(kp_instance).quantum_tree_generator())
    start_time = time.time()
    print("QAOA result = ", QAOA(random_kp_instance, depth = 5).optimize())
    print("Elapsed time = ", time.time() - start_time)


if __name__ == "__main__":
    main()
