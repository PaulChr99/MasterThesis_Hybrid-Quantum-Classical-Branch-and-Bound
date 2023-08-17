""" """
import sys
sys.path.append("C:\\Users\\d92474\\Documents\\Uni\\Master Thesis\\GitHub\\MasterThesis_Hybrid-Quantum-Classical-Branch-and-Bound\\bnb-qaoa_graph-coloring_christiansen\\code")
sys.path.append("C:\\Users\\d92474\\Documents\\Uni\\Master Thesis\\GitHub\\MasterThesis_Hybrid-Quantum-Classical-Branch-and-Bound\\common-utils")
sys.path.append("C:\\Users\\d92474\\Documents\\Uni\\Master Thesis\\GitHub\\MasterThesis_Hybrid-Quantum-Classical-Branch-and-Bound\\common-utils\\cpp-configuration")
sys.path.append("C:\\Users\\d92474\\Documents\\Uni\\Master Thesis\\GitHub\\MasterThesis_Hybrid-Quantum-Classical-Branch-and-Bound\\common-utils\\quantum-gates")



import kron_dot
from single_qubit_gates import RotationOperators, BasicGates
from multiple_qubit_gates import TwoQubitRotations
from qaoa import Measurement
from graph import Graph, GraphRelatedFunctions


import numpy as np
from numpy.typing import NDArray
from typing import Union, List



num_type = np.cdouble



class AuxiliaryFunctions(GraphRelatedFunctions):
    
    def __init__(self, graph: Graph):
        self.graph = graph
    
    def find_index_in_combined_total_register(self, color: int, vertex: Union[int, None] = None):
        color_register_size = self.find_max_degree(self.graph) + 1
        if color > color_register_size:
            raise ValueError("Color must be contained in the list of valid colors.")
        if vertex == None:
            return color - 1 # Have to substract 1 at the end as list indices start at 0
        else:
            if vertex not in self.graph.vertices:
                raise ValueError("Vertex must be contained in graph.")
            vertex_index = self.graph.vertices.index(vertex)
            #print("(vertex index, vertex) = ", (vertex_index,vertex))
            #print("index in color vertex register = ", color_register_size + (vertex_index - 1) * (color_register_size) + color - 1)
            return color_register_size + vertex_index * (color_register_size) + color - 1 # Subtract 1 for proper list start and add offset of color register
            


class MixingUnitary(RotationOperators, AuxiliaryFunctions):

    def __init__(self, graph: Graph, valid_colors: List[int]):
        self.graph = graph
        self.valid_colors = valid_colors


    def apply_mixing_unitary(self, beta: Union[float, int], state: NDArray[num_type]):
        rotation_gate = self.rotation_x(2*beta)
       
        """ Single-qubit x-rotations on all qubits, iterating through color and vertex-color register simultaneously to save computing time"""
        for color in self.valid_colors:
            index_in_color_register = self.find_index_in_combined_total_register(color)
            kron_dot.kron_dot_dense(index_in_color_register, rotation_gate, state)
            for vertex in self.graph.vertices:
                index_in_vertex_color_register = self.find_index_in_combined_total_register(color, vertex)
                kron_dot.kron_dot_dense(index_in_vertex_color_register, rotation_gate, state)

        return state
    


class PhaseSeparationUnitary(TwoQubitRotations, AuxiliaryFunctions):

    def __init__(self, graph: Graph, valid_colors: List[int], penalty: Union[float, int]):
        self.graph = graph
        
        if not penalty > 1:
            raise ValueError("Penalty is not chosen properly; p > 1 needs to hold in order to not corrupt the desired heighest-energy eigenstate of the softcoded objective Hamitlonian.")
        
        self.penalty = penalty
        self.valid_colors = valid_colors

    
    def apply_W_term(self, gamma: Union[float, int], state: NDArray[num_type]):
        for color in self.valid_colors:
            index_in_color_register = self.find_index_in_combined_total_register(color)
            rotation_gate = self.rotation_z(gamma * (self.graph.number_vertices * self.penalty / 2 - 1))
            kron_dot.kron_dot_dense(index_in_color_register, rotation_gate, state)
        return state
    

    def apply_X_term(self, gamma: Union[float, int], state: NDArray[num_type]):
        max_degree = self.find_max_degree(self.graph)
        for vertex in self.graph.vertices:
            for color in self.valid_colors:
                index_in_vertex_color_register = self.find_index_in_combined_total_register(color, vertex)
                rotation_gate = self.rotation_z(- self.penalty * gamma / 2 * (self.degree(vertex, self.graph) + 2 * max_degree - 1))
                kron_dot.kron_dot_dense(index_in_vertex_color_register, rotation_gate, state)
        return state
    

    def apply_WX_term(self, gamma: Union[float, int], state: NDArray[num_type]):
        for vertex in self.graph.vertices:
            for color in self.valid_colors:
                index_in_color_register = self.find_index_in_combined_total_register(color)
                index_in_vertex_color_register = self.find_index_in_combined_total_register(color, vertex)
                angle = self.penalty * gamma / 2
                state = self.apply_rotation_zz(index_in_color_register, index_in_vertex_color_register, angle, state)
        return state
    

    def apply_XX_V_term(self, gamma: Union[float, int], state: NDArray[num_type]):
        for vertex in self.graph.vertices:
            for color1 in self.valid_colors:
                for color2 in range(color1 + 1, len(self.valid_colors) + 1):
                    index1_in_vertex_color_register = self.find_index_in_combined_total_register(color1, vertex)
                    index2_in_vertex_color_register = self.find_index_in_combined_total_register(color2, vertex)
                    angle = - self.penalty * gamma
                    state = self.apply_rotation_zz(index1_in_vertex_color_register, index2_in_vertex_color_register, angle, state)
        return state
    

    def apply_XX_E_term(self, gamma: Union[float, int], state: NDArray[num_type]):
        for edge in self.graph.edges:
            for color in self.valid_colors:
                index1_in_vertex_color_register = self.find_index_in_combined_total_register(color, edge[0])
                index2_in_vertex_color_regsiter = self.find_index_in_combined_total_register(color, edge[1])
                angle = - self.penalty * gamma / 2
                state = self.apply_rotation_zz(index1_in_vertex_color_register, index2_in_vertex_color_regsiter, angle, state)
        return state

    
    def apply_phase_separation_unitary(self, gamma: Union[float, int], state: NDArray[num_type]):
        """ Single-qubit z-rotations on color register """
        state = self.apply_W_term(gamma, state)

        """ Single-qubit z-rotations on register combining vertices and colors """
        state = self.apply_X_term(gamma, state)

        """ Two-qubit z-rotations on color register and register combining vertices and colors mixed """
        state = self.apply_WX_term(gamma, state)

        """ Two-qubit z-rotations on register combining vertices and colors iterating over vertices """
        state = self.apply_XX_V_term(gamma, state)

        """ Two-qubit z-rotations on register combining vertices and colors iterating over edges """
        state = self.apply_XX_E_term(gamma, state)

        return state
    


class QuasiAdiabaticEvolution(BasicGates, AuxiliaryFunctions):

    def __init__(self, graph: Graph, valid_colors: List[int], depth: int, penalty: Union[float, int]):
        self.graph = graph
        self.valid_colors = valid_colors
        self.depth = depth
        self.phase_separation_unitary = PhaseSeparationUnitary(graph, valid_colors, penalty)
        self.mixing_unitary = MixingUnitary(graph, valid_colors)
        self.color_register_size = len(valid_colors)
        self.color_vertex_register_size = graph.number_vertices * len(valid_colors)


    def apply_quasiadiabatic_evolution(self, angles: List[Union[float, int]]):

        if len(angles) != 2 * self.depth:
            raise ValueError("Number of provided values for gamma and beta parameters need to be consistent with specified circuit depth.")
        
        gamma_values = angles[0::2]
        beta_values = angles[1::2]
        
        """ Initialize |0> = |0,...,0> state """
        state = np.zeros(2**(self.color_register_size + self.color_vertex_register_size), dtype = num_type)
        state[0] = 1

        """ Apply Hadamards to prepare heighest-energy eigenstate of mixer """
        for color in self.valid_colors:
            index_in_color_register = self.find_index_in_combined_total_register(color)
            kron_dot.kron_dot_dense(index_in_color_register, self.hadamard(), state)
            for vertex in self.graph.vertices:
                index_in_vertex_color_register = self.find_index_in_combined_total_register(color, vertex)
                kron_dot.kron_dot_dense(index_in_vertex_color_register, self.hadamard(), state)
                
        """ Alternatingly apply phase separation and mixing unitary """
        for lap in range(self.depth):
            state = self.phase_separation_unitary.apply_phase_separation_unitary(gamma_values[lap], state)
            state = self.mixing_unitary.apply_mixing_unitary(beta_values[lap], state)

        return state
