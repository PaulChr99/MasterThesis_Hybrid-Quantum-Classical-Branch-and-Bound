import sys
sys.path.append("C:\\Users\\d92474\\Documents\\Uni\\Master Thesis\\GitHub\\MasterThesis_Hybrid-Quantum-Classical-Branch-and-Bound\\bnb-qaoa_graph-coloring_christiansen\\code")
sys.path.append("C:\\Users\\d92474\\Documents\\Uni\\Master Thesis\\GitHub\\MasterThesis_Hybrid-Quantum-Classical-Branch-and-Bound\\common-utils")
sys.path.append("C:\\Users\\d92474\\Documents\\Uni\\Master Thesis\\GitHub\\MasterThesis_Hybrid-Quantum-Classical-Branch-and-Bound\\common-utils\\cpp-configuration")
sys.path.append("C:\\Users\\d92474\\Documents\\Uni\\Master Thesis\\GitHub\\MasterThesis_Hybrid-Quantum-Classical-Branch-and-Bound\\common-utils\\quantum-gates")



import kron_dot
from single_qubit_gates import RotationOperators, BasicGates
from multiple_qubit_gates import TwoQubitRotations
from graph import Graph, GraphRelatedFunctions


import numpy as np
from numpy.typing import NDArray
from typing import Union, List



class MixingUnitary(RotationOperators):

    num_type = np.cdouble
    
    def __init__(self, graph: Graph):
        self.graph = graph
    
    def apply_mixing_unitary(self, beta: Union[float, int], state: NDArray[num_type]):
        for v in range(self.graph.number_vertices):
            rotation_gate = self.rotation_x(2*beta)
            kron_dot.kron_dot_dense(v, rotation_gate, state)
        return state
    


class PhaseSeparationUnitary(TwoQubitRotations, GraphRelatedFunctions):

    num_type = np.cdouble

    def __init__(self, graph: Graph, penalty: Union[float, int]):
        self.graph = graph
        
        if not penalty > 1:
            raise ValueError("Penalty is not chosen properly; p > 1 needs to hold in order to not corrupt the desired heighest-energy eigenstate of the softcoded objective Hamitlonian.")
        
        self.penalty = penalty


    def apply_vertex_term(self, gamma: Union[float, int], state: NDArray[num_type]):
        for v in range(self.graph.number_vertices):
            degree_of_vertex = self.degree(self.graph.vertices[v], self.graph)
            rotation_gate = self.rotation_z(gamma * (1 - degree_of_vertex * self.penalty / 2))
            kron_dot.kron_dot_dense(v, rotation_gate, state)
        return state
    

    def apply_edge_term(self, gamma: Union[float, int], state: NDArray[num_type]):
        for edge in self.graph.edges:
            index_left, index_right = self.find_left_index_for_edge(edge), self.find_right_index_for_edge(edge)
            angle = - self.penalty * gamma / 2
            state = self.apply_rotation_zz(index_left, index_right, angle, state)
        return state
    

    def apply_phase_separation_unitary(self, gamma: Union[float, int], state: NDArray[num_type]):
        
        """ Apply term iterating through vertices """
        state = self.apply_vertex_term(gamma, state)

        """ Apply term iterating through edges """
        state = self.apply_edge_term(gamma, state)

        return state
    


class QuasiAdiabaticEvolution(BasicGates):

    num_type = np.cdouble
    
    def __init__(self, graph: Graph, depth: int, penalty: Union[float, int]):
        self.depth = depth
        self.vertex_register_size = graph.number_vertices
        self.phase_separation_unitary = PhaseSeparationUnitary(graph, penalty)
        self.mixing_unitary = MixingUnitary(graph)


    def apply_quasiadiabatic_evolution(self, angles: List[Union[float, int]]):

        if len(angles) != 2 * self.depth:
            raise ValueError("Number of provided values for gamma and beta parameters need to be consistent with specified circuit depth.")
        
        gamma_values = angles[0::2]
        beta_values = angles[1::2]
        
        """ Initialize |0> = |0,...,0> state """
        state = np.zeros(2**(self.vertex_register_size), dtype = self.num_type)
        state[0] = 1

        """ Apply Hadamards to prepare heighest-energy eigenstate of mixer """
        for v in range(self.vertex_register_size):
            kron_dot.kron_dot_dense(v, self.hadamard(), state)

        """ Alternatingly apply phase separation and mixing unitary """
        for j in range(self.depth):
            state = self.phase_separation_unitary.apply_phase_separation_unitary(gamma_values[j], state)
            state = self.mixing_unitary.apply_mixing_unitary(beta_values[j], state)

        return state