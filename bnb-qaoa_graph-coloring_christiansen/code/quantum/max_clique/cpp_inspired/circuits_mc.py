import sys
sys.path.append("C:\\Users\\d92474\\Documents\\Uni\\Master Thesis\\GitHub\\MasterThesis_Hybrid-Quantum-Classical-Branch-and-Bound\\bnb-qaoa_graph-coloring_christiansen\\code")
sys.path.append("C:\\Users\\d92474\\Documents\\Uni\\Master Thesis\\GitHub\\MasterThesis_Hybrid-Quantum-Classical-Branch-and-Bound\\common-utils")
sys.path.append("C:\\Users\\d92474\\Documents\\Uni\\Master Thesis\\GitHub\\MasterThesis_Hybrid-Quantum-Classical-Branch-and-Bound\\common-utils\\cpp-configuration")
sys.path.append("C:\\Users\\d92474\\Documents\\Uni\\Master Thesis\\GitHub\\MasterThesis_Hybrid-Quantum-Classical-Branch-and-Bound\\common-utils\\quantum-gates")



import kron_dot
from single_qubit_gates import RotationOperators, BasicGates
from multiple_qubit_gates import TwoQubitRotations
from graph import Graph, GraphRelatedFunctions


import math
import numpy as np
from numpy.typing import NDArray
from typing import Union, List



num_type = np.cdouble



class MixingUnitary(RotationOperators):

    def __init__(self, graph: Graph):
        self.graph = graph
        self.d = math.floor(math.log2(GraphRelatedFunctions(graph).find_max_degree(graph) + 1))
    

    def apply_mixing_unitary(self, beta: Union[float, int], state: NDArray[num_type]):
        
        rotation_gate = self.rotation_x(2*beta)
        
        """ Single-qubit x-rotation on vertex register """
        for v in range(self.graph.number_vertices):
            kron_dot.kron_dot_dense(v, rotation_gate, state)

        """ Single-qubit x-rotation on auxiliary register """
        for j in range(self.d + 1):
            kron_dot.kron_dot_dense(self.graph.number_vertices + j, rotation_gate, state)

        return state
    


class PhaseSeparationUnitary(TwoQubitRotations, GraphRelatedFunctions):

    def __init__(self, graph: Graph, penalty: Union[float, int], penalty3: Union[float, int]):
        self.graph = graph
        
        self.max_degree = GraphRelatedFunctions(graph).find_max_degree(graph)
        #if not penalty > self.max_degree * penalty3 + 1:
        #    raise ValueError("Penalty is not chosen properly; p > p_3 * Delta(G) + 1 needs to hold for preserving the heighest-energy eigenstate of the softcoded objective Hamitlonian.")
        
        self.penalty = penalty
        self.penalty3 = penalty3
        self.d = math.floor(math.log2(self.max_degree + 1))
        self.delta = self.max_degree + 2 - 2**(self.d)

    
    def apply_X_term(self, gamma: Union[float, int], state: NDArray[num_type]):
        for v in range(self.graph.number_vertices):
            degree_of_vertex = self.degree(v, self.graph)
            angle = gamma * (self.penalty * (self.max_degree + 1 - self.graph.number_vertices) + degree_of_vertex * self.penalty3 / 2 + 1 )
            rotation_gate = self.rotation_z(angle)
            kron_dot.kron_dot_dense(v, rotation_gate, state)
        return state
    

    def apply_Y_term(self, gamma: Union[float, int], state: NDArray[num_type]):
        for j in range(self.d):
            angle = - 2**(self.d - j) * gamma * (self.penalty * (self.max_degree + 1 - self.graph.number_vertices) + self.penalty3 * (self.max_degree + 1/2))
            rotation_gate = self.rotation_z(angle)
            kron_dot.kron_dot_dense(self.graph.number_vertices + j, rotation_gate, state)
        return state
    

    def apply_XX_term(self, gamma: Union[float, int], state: NDArray[num_type]):
        for u in range(self.graph.number_vertices):
            for v in range(u + 1, self.graph.number_vertices):
                angle = - gamma * self.penalty
                if (u,v) in self.graph.edges or (v,u) in self.graph.edges:
                    angle += gamma * self.penalty3 / 2
                state = self.apply_rotation_zz(u, v, angle, state)
        return state
    

    def apply_XY_term(self, gamma: Union[float, int], state: NDArray[num_type]):
        for v in range(self.graph.number_vertices):
            for j in range(self.d):
                angle = 2**(self.d - j) * self.penalty * gamma
                state = self.apply_rotation_zz(v, self.graph.number_vertices + j, angle, state)
        return state 
    

    def apply_XY_end_term(self, gamma: Union[float, int], state: NDArray[num_type]):
        for v in range(self.graph.number_vertices):
            angle = self.delta * self.penalty * gamma
            state = self.apply_rotation_zz(v, self.graph.number_vertices + self.d, angle, state) # Remember index is shifted downwards by one unit, d + 1 -> d
        return state
    

    def apply_YY_term(self, gamma: Union[float, int], state: NDArray[num_type]):
        for k in range(self.d):
            for j in range(k + 1, self.d):
                angle = - gamma * (2**(2*self.d - (k+j)) * self.penalty + self.penalty3 )
                state = self.apply_rotation_zz(self.graph.number_vertices + k, self.graph.number_vertices + j, angle, state)
        return state
    
    
    def apply_YY_end_term(self, gamma: Union[float, int], state: NDArray[num_type]):
        for j in range(self.d):
            angle = - 2**(self.d - j) * self.delta * gamma * (self.penalty + self.penalty3)
            state = self.apply_rotation_zz(self.graph.number_vertices + j, self.graph.number_vertices + self.d, angle, state)
        return state
    
    
    def apply_phase_separation_unitary(self, gamma: Union[float, int], state: NDArray[num_type]):
        
        """ Single-qubit z-rotations on vertex register """
        state = self.apply_X_term(gamma, state)

        """ Single-qubit z-rotations on auxiliary register """
        state = self.apply_Y_term(gamma, state)

        """ Single-qubit z-rotation on last position of auxiliary register """
        angle = - self.delta * gamma * (self.penalty * (self.max_degree + 1 - self.graph.number_vertices) + self.penalty3 * (self.max_degree + 1/2))
        kron_dot.kron_dot_dense(self.graph.number_vertices + self.d, self.rotation_z(angle), state)

        """ Two-qubit z-rotations on vertex register """
        state = self.apply_XX_term(gamma, state)

        """ Two-qubit z-rotations on vertex and auxiliary registers mixed """
        state = self.apply_XY_term(gamma, state)

        """ Two-qubit z-rotations on vertex register and last position of auxiliary register """
        state = self.apply_XY_end_term(gamma, state)

        """ Two-qubit z-rotations on auxiliary register """
        state = self.apply_YY_term(gamma, state)

        """ Two-qubit z-rotations on auxiliary register with second qubit fixed to last position of the register """
        state = self.apply_YY_end_term(gamma, state)

        return state
    


class QuasiAdiabaticEvolution(BasicGates):

    def __init__(self, graph: Graph, depth: int, penalty: Union[float, int], penalty3: Union[float, int]):
        self.graph = graph
        self.depth = depth
        max_degree = GraphRelatedFunctions(graph).find_max_degree(graph)
        self.d = math.floor(math.log2(max_degree + 1))
        self.delta = max_degree + 2 - 2**(self.d)
        self.phase_separation_unitary = PhaseSeparationUnitary(graph, penalty, penalty3)
        self.mixing_unitary = MixingUnitary(graph)
        self.vertex_register_size = self.graph.number_vertices
        self.auxiliary_register_size = self.d + 1


    def apply_quasiadiabatic_evolution(self, angles: List[Union[float, int]]):

        if len(angles) != 2 * self.depth:
            raise ValueError("Number of provided values for gamma and beta parameters need to be consistent with specified circuit depth.")
        
        gamma_values = angles[0::2]
        beta_values = angles[1::2]
        
        """ Initialize |0> = |0,...,0> state """
        state = np.zeros(2**(self.vertex_register_size + self.auxiliary_register_size), dtype = num_type)
        state[0] = 1

        """ Apply Hadamards to prepare heighest-energy eigenstate of mixer """
        for v in range(self.graph.number_vertices):
            kron_dot.kron_dot_dense(v, self.hadamard(), state)
        for j in range(self.d + 1):
            kron_dot.kron_dot_dense(self.graph.number_vertices + j, self.hadamard(), state)
                
        """ Alternatingly apply phase separation and mixing unitary """
        for lap in range(self.depth):
            state = self.phase_separation_unitary.apply_phase_separation_unitary(gamma_values[lap], state)
            state = self.mixing_unitary.apply_mixing_unitary(beta_values[lap], state)

        return state