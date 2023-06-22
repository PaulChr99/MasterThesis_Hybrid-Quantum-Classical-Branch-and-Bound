import sys
sys.path.append("C:\\Users\\d92474\\Documents\\Uni\\Master Thesis\\GitHub\\QuBRA\\bnb-qaoa_graph-coloring_christiansen\\code")
sys.path.append("C:\\Users\\d92474\\Documents\\Uni\\Master Thesis\\GitHub\\QuBRA\\bnb-qaoa_graph-coloring_christiansen\\code\\qaoa")


import numpy as np
from qiskit import QuantumCircuit
from qiskit import transpile
import time
from typing import Union

from circuits_gc import QAOACircuitGC, PhaseCircuitHelperFunctions
from simulate_and_optimize import Simulation, Optimization

from graph import Graph, GraphRelatedFunctions, exemplary_gc_instances


class AuxiliaryFunctions:

    def __init__(self, graph: Graph, circuit: QuantumCircuit, a: Union[int, float], b: Union[int, float]):
        self.graph = graph
        self.max_degree = GraphRelatedFunctions(self.graph).find_max_degree(self.graph)
        self.circuit = circuit
        self.a = a
        self.b_min = 1/2 * a
        if not b > self.b_min:
            raise ValueError("Penalty parameter a must be strictly greater than half the value parameter b.")
        self.b = b

    def bitstring_to_bits(self, bitstring):
        """ Convert a qiskit bitstring to two numpy arrays, one for each register """
        bits = np.array(list(map(int, list(bitstring))))[::-1]
        w = bits[:(self.max_degree + 1)]
        x = bits[(self.max_degree + 1):]
        return w, x
    
    def value(self, color_bits):
        return sum(color_bits)

    def penalty(self, color_bits, vertex_color_bits):
        
        def indices_for_fixed_vertex(vertex: int):
            return [PhaseCircuitHelperFunctions(self.graph).find_index_in_vertex_color_register(vertex, color) for color in range(1, self.max_degree + 2)]

        vertex_term = 0
        for vertex in self.graph.vertices:
            corresponding_indices = indices_for_fixed_vertex(vertex)
            vertex_term += (1 - sum([vertex_color_bits[i] for i in corresponding_indices]))**2
            vertex_term += sum([(1 - color_bits[j]) * vertex_color_bits[corresponding_indices[j]] for j in range(self.max_degree + 1)])

        edge_term = 0
        for edge in self.graph.edges:
            corresponding_indices_left = indices_for_fixed_vertex(edge[0])
            corresponding_indices_right = indices_for_fixed_vertex(edge[1])
            edge_term += sum([vertex_color_bits[corresponding_indices_left[j]] * vertex_color_bits[corresponding_indices_right[j]] for j in range(self.max_degree + 1)])

        return vertex_term + edge_term
    
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
        parameters[self.circuit.b] = float(self.b)
        return parameters

    def average_value(self, probs_dict, func):
        """ Calculate the average value of a function over a probability dict """
        colorings = list(probs_dict.keys())
        values = np.array(list(map(func, colorings)))
        probs = np.array(list(probs_dict.values()))
        return sum(values * probs)


class QAOAGC(AuxiliaryFunctions, Simulation, Optimization):

    def __init__(self, graph: Graph, circuit: QuantumCircuit, a: Union[int, float], b: Union[int, float]):
        AuxiliaryFunctions.__init__(self, graph, circuit, a, b)
        self.A = 2 * self.a
        self.B = 4 * self.b
    
    def objective_function(self, bitstring: str):
        """ The objective function of the quadratic-penalty based approach """
        w_bits, x_bits = self.bitstring_to_bits(bitstring)
        value = self.value(color_bits = w_bits)
        penalty = self.penalty(color_bits = w_bits, vertex_color_bits = x_bits)
        return self.A * value + self.B * penalty

    def get_probs_dict(self, transpiled_circuit, angles, choice_only = False):
        """ Simulate circuit for given parameters and return probability dict """
        parameter_dict = self.to_parameter_dict(angles)
        statevector = self.get_statevector(transpiled_circuit, parameter_dict)
        if choice_only:
            number_of_bits_in_solution = self.graph.number_vertices + self.graph.number_vertices * (self.max_degree + 1)
            probs_dict = statevector.probabilities_dict(range(self.max_degree + 1))
        else:
            probs_dict = statevector.probabilities_dict()
        return probs_dict

    def find_optimal_angles(self):
        """ Optimize the parameters beta, gamma for given circuit and parameters """
        transpiled_circuit = transpile(self.circuit, self.backend)

        def angles_to_value(angles):
            probs_dict = self.get_probs_dict(transpiled_circuit, angles)
            value = self.average_value(probs_dict, self.objective_function)
            return value

        return self.optimize_angles(angles_to_value, self.circuit.gamma_range(self.a, self.b),
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
        return self.get_expectation_value(optimal_angles) / self.A


def main():
    a = 2
    b = a
    #graph = Graph(vertices = [1,2], edges = [])
    graph = exemplary_gc_instances["A"]
    print(graph)
    print("Building Circuit...")
    circuit = QAOACircuitGC(graph, depth = 2)
    print("Done!")
    #print(circuit)
    #print(AuxiliaryFunctions(graph, circuit, a, b).value([1]))
    #print(AuxiliaryFunctions(graph, circuit, a, b).penalty([1], [0,1]))
    #print(QAOAGC(graph, circuit, a, b).objective_function("111") / (2*a))
    start_time = time.time()
    print("QAOA result = ", QAOAGC(graph, circuit, a, b).execute_algorithm())
    end_time = time.time()
    print(f"Elapsed time: {end_time - start_time}")

"""
if __name__ == "__main__":
    main()  



Since the order of items is reversed in qiskit, this "false" sorting is still active in
the probabilities dict.
"""
