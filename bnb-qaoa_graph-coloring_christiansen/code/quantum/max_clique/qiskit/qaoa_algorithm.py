import sys
sys.path.append("C:\\Users\\d92474\\Documents\\Uni\\Master Thesis\\GitHub\\QuBRA\\bnb-qaoa_graph-coloring_christiansen\\code")
sys.path.append("C:\\Users\\d92474\\Documents\\Uni\\Master Thesis\\GitHub\\QuBRA\\bnb-qaoa_graph-coloring_christiansen\\code\\qaoa")


import math
import numpy as np
from qiskit import QuantumCircuit
from qiskit import transpile
import time
from typing import Union

from circuits_mc import QAOACircuitMC
from simulate_and_optimize import Simulation, Optimization

from graph import Graph, GraphRelatedFunctions, exemplary_gc_instances


class AuxiliaryFunctions:

    def __init__(self, graph: Graph, circuit: QuantumCircuit, a: Union[int, float]):
        self.graph = graph
        self.max_degree = GraphRelatedFunctions(self.graph).find_max_degree(self.graph)
        self.d = math.floor(math.log2(self.max_degree + 1))
        self.delta = (self.max_degree + 1) - (2**(self.d) - 1)
        self.circuit = circuit
        self.a = a
        self.b2 = 2 * self.a
        self.b1 = (self.max_degree + 2) * self.b2

    def bitstring_to_bits(self, bitstring):
        """ Convert a qiskit bitstring to two numpy arrays, one for each register """
        bits = np.array(list(map(int, list(bitstring))))[::-1]
        x = bits[:(self.graph.number_vertices)]
        y = bits[(self.graph.number_vertices):]
        return x, y
    
    def value(self, vertex_bits):
        return sum(vertex_bits)

    def penalty1(self, vertex_bits, auxiliary_bits):
        auxiliary_term = sum([2**j * auxiliary_bits[j] for j in range(len(auxiliary_bits) - 1)]) + self.delta * auxiliary_bits[-1]
        vertex_term = sum(vertex_bits)
        return (auxiliary_term - vertex_term)**2

    def penalty2(self, vertex_bits, auxiliary_bits):
        auxiliary_term = sum([2**j * auxiliary_bits[j] for j in range(len(auxiliary_bits) - 1)]) + self.delta * auxiliary_bits[-1]
        edge_term = 0
        for edge in self.graph.edges:
            index_left = GraphRelatedFunctions(self.graph).find_left_index_for_edge(edge)
            index_right = GraphRelatedFunctions(self.graph).find_right_index_for_edge(edge)
            edge_term += vertex_bits[index_left] * vertex_bits[index_right]
        return 1/2 * auxiliary_term * (auxiliary_term - 1) - edge_term
    
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
        parameters[self.circuit.b1] = float(self.b1)
        parameters[self.circuit.b2] = float(self.b2)
        return parameters

    def average_value(self, probs_dict, func):
        """ Calculate the average value of a function over a probability dict """
        colorings = list(probs_dict.keys())
        values = np.array(list(map(func, colorings)))
        probs = np.array(list(probs_dict.values()))
        return sum(values * probs)



class QAOAMC(AuxiliaryFunctions, Simulation, Optimization):

    def __init__(self, graph: Graph, circuit: QuantumCircuit, a: Union[int, float]):
        AuxiliaryFunctions.__init__(self, graph, circuit, a)
        self.A = 2 * self.a
        self.B1 = 2 * self.b1
        self.B2 = 4 * self.b2
    
    def objective_function(self, bitstring: str):
        """ The objective function of the quadratic-penalty based approach """
        x_bits, y_bits = self.bitstring_to_bits(bitstring)
        value = self.value(vertex_bits = x_bits)
        penalty1 = self.penalty1(vertex_bits = x_bits, auxiliary_bits = y_bits)
        penalty2 = self.penalty2(vertex_bits = x_bits, auxiliary_bits = y_bits)
        return self.A * value - self.B1 * penalty1 - self.B2 * penalty2

    def get_probs_dict(self, transpiled_circuit, angles, choices_only = False):
        """ Simulate circuit for given parameters and return probability dict """
        parameter_dict = self.to_parameter_dict(angles)
        statevector = self.get_statevector(transpiled_circuit, parameter_dict)
        if choices_only:
            probs_dict = statevector.probabilities_dict(range(self.graph.number_vertices))
        else:
            probs_dict = statevector.probabilities_dict()
        return probs_dict

    def find_optimal_angles(self):
        """ Optimize the parameters beta, gamma for given circuit and parameters """
        transpiled_circuit = transpile(self.circuit, self.backend)

        def angles_to_value(angles):
            probs_dict = self.get_probs_dict(transpiled_circuit, angles)
            value = - self.average_value(probs_dict, self.objective_function)
            return value

        return self.optimize_angles(angles_to_value, self.circuit.gamma_range(self.a, self.b1, self.b2),
                                    self.circuit.beta_range(), self.circuit.p)

    def get_expectation_value(self, angles):
        """Return the expectation value of the objective function for given parameters."""
        transpiled_circuit = transpile(self.circuit, self.backend)
        probs_dict = self.get_probs_dict(transpiled_circuit, angles)
        #print(probs_dict)
        return self.average_value(probs_dict, self.objective_function)

    def execute_algorithm(self):
        print("QAOA running...")
        optimal_angles = self.find_optimal_angles()
        return self.get_expectation_value(optimal_angles) / self.A


def main():
    a = 2
    graph = exemplary_gc_instances["B"]
    print(graph)
    print("Building Circuit...")
    circuit = QAOACircuitMC(graph, depth = 3)
    print("Done!")
    #print(circuit)
    #print(QAOAMC(graph, circuit, a).objective_function("100011") / a)
    start_time = time.time()
    print("QAOA result = ", QAOAMC(graph, circuit, a).execute_algorithm())
    end_time = time.time()
    print(f"Elapsed time: {end_time - start_time}")

"""
if __name__ == "__main__":
    main()



Since the order of items is reversed in qiskit, this "false" sorting is still active in
the probabilities dict.
"""
