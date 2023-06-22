"""Helper functions for the linear penalty based qaoa implementation."""
import sys
sys.path.append("C:\\Users\\d92474\\Documents\\Uni\\Master Thesis\\GitHub\\QuBRA\\bnb-qaoa_knapsack_christiansen\\code")

from functools import partial

import numpy as np
from qiskit import transpile
import time

from knapsack_problem import KnapsackProblem, exemplary_kp_instances
from circuits import LinQAOACircuit
from simulate_and_optimize import Simulation, Optimization


class AuxiliaryFunctions:

    def __init__(self, problem: KnapsackProblem, circuit: LinQAOACircuit, a: float):
        self.problem = problem
        self.circuit = circuit
        self.a_min = max(problem.profits)
        if not a > self.a_min:
            raise ValueError("Parameter a must be strictly greater than highest item value.")
        self.a = a

    def bitstring_to_choice(self, bitstring):
        """Convert a measurement bitstring to an item choice array."""
        bits = np.array(list(map(int, list(bitstring))))[::-1]
        choice = np.array(bits[:self.problem.number_items])
        return choice

    def value(self, choice):
        """Return the value of an item choice.
        Assumes choice is a numpy array of length problem.N"""
        return choice.dot(self.problem.profits)

    def weight(self, choice):
        """Return the weight of an item choice.
        Assumes choice is a numpy array of length problem.N"""
        return choice.dot(self.problem.weights)

    def is_choice_feasible(self, choice):
        """Returns whether an item choice is feasible.
        Assumes choice is a numpy array of length problem.N"""
        return self.weight(choice, self.problem) <= self.problem.capacity

    def to_parameter_dict(self, angles):
        """Create a circuit specific parameter dict from given parameters.
        angles = np.array([gamma0, beta0, gamma1, beta1, ...])"""
        if len(angles)%2 != 0:
            raise ValueError("List of beta and gamma angles must not have different length.")
        gammas = angles[0::2]
        betas = angles[1::2]
        parameters = {}
        for parameter, value in zip(self.circuit.betas, betas):
            parameters[parameter] = value
        for parameter, value in zip(self.circuit.gammas, gammas):
            parameters[parameter] = value
        parameters[self.circuit.a] = float(self.a)
        return parameters

    def average_value(self, probs_dict, func):
        """Calculate the average value of a function over a probability dict."""
        bitstrings = list(probs_dict.keys())
        values = np.array(list(map(func, bitstrings)))
        probs = np.array(list(probs_dict.values()))
        return sum(values * probs)


class LinearSoftConstraintQAOA(AuxiliaryFunctions, Simulation, Optimization):

    def objective_function(self, bitstring):
        """The objective function of the linear penalty based approach."""
        choice = self.bitstring_to_choice(bitstring)
        value = self.value(choice)
        weight = self.weight(choice)
        if weight > self.problem.capacity:
            penalty = self.a * (weight - self.problem.capacity)
        else:
            penalty = 0
        return value - penalty

    def get_probs_dict(self, transpiled_circuit, angles, choice_only = True):
        """Simulate circuit for given parameters and return probability dict."""
        parameter_dict = self.to_parameter_dict(angles)
        statevector = self.get_statevector(transpiled_circuit, parameter_dict)
        if choice_only:
            probs_dict = statevector.probabilities_dict(range(self.problem.number_items))
        else:
            probs_dict = statevector.probabilities_dict()
        return probs_dict

    def find_optimal_angles(self):
        """Optimize the parameters beta, gamma for given circuit and parameters."""
        transpiled_circuit = transpile(self.circuit, self.backend)

        def angles_to_value(angles):
            probs_dict = self.get_probs_dict(transpiled_circuit, angles)
            value = - self.average_value(probs_dict, self.objective_function)
            return value

        return self.optimize_angles(angles_to_value, self.circuit.gamma_range(self.a),
                                    self.circuit.beta_range(), self.circuit.p)

    def get_expectation_value(self, angles):
        """Return the expectation value of the objective function for given parameters."""
        transpiled_circuit = transpile(self.circuit, self.backend)
        probs_dict = self.get_probs_dict(transpiled_circuit, angles)
        print(probs_dict)
        return self.average_value(probs_dict, self.objective_function)

    def execute_algorithm(self):
        print("QAOA running...")
        optimal_angles = self.find_optimal_angles()
        return self.get_expectation_value(optimal_angles)



def main():
    a = 4
    problem = exemplary_kp_instances["B"]
    #offset = 4
    #problem = KnapsackProblem(profits=[2,1], weights=[2,2], capacity=2)
    print(problem)
    print("Building Circuit...")
    circuit = LinQAOACircuit(problem, 2)
    print("Done!")
    """
    print("Optimizing Angles...")
    angles = LinearSoftConstraintQAOA(problem = problem1, circuit = circuit, a = a).find_optimal_angles()
    
    print("Done!")
    print(f"Optimized Angles: {angles}")
    #probs = get_probs_dict(circuit, problem1, angles, a)
    #print(f"Probability dictionary: {probs}")
    final_expect_value = LinearSoftConstraintQAOA(problem = problem1, circuit = circuit, a = a).get_expectation_value(angles = angles)
    
    print(f"Final expectation value: {final_expect_value}")
    
    #visualization.hist(probs)
    """
    start_time = time.time()
    print(LinearSoftConstraintQAOA(problem = problem, circuit = circuit, a = a).execute_algorithm())
    end_time = time.time()
    print(f"Elapsed time: {end_time - start_time}")


if __name__ == "__main__":
    main()


"""
Since the order of items is reversed in qiskit, this "false" sorting is still active in
the probabilities dict.
"""