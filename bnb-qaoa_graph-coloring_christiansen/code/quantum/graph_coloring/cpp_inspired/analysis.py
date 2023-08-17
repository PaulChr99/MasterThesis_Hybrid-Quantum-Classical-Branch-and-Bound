import sys
sys.path.append("C:\\Users\\d92474\\Documents\\Uni\\Master Thesis\\GitHub\\MasterThesis_Hybrid-Quantum-Classical-Branch-and-Bound\\bnb-qaoa_graph-coloring_christiansen\\code")
sys.path.append("C:\\Users\\d92474\\Documents\\Uni\\Master Thesis\\GitHub\\MasterThesis_Hybrid-Quantum-Classical-Branch-and-Bound\\common-utils")


from graph import Graph, GraphRelatedFunctions, exemplary_graph_instances
from quantum.graph_coloring.cpp_inspired.circuits_gc import QuasiAdiabaticEvolution, AuxiliaryFunctions
from qaoa import QAOA, ProblemType


import numpy as np
from typing import Union, List
import time



class ProblemRelatedFunctions(AuxiliaryFunctions):

    def __init__(self, graph: Graph, valid_colors: List[int], penalty: Union[float, int]):
        self.graph = graph
        self.valid_colors = valid_colors
        self.penalty = penalty


    def value(self, bits: List[bool]):
        color_bits = bits[:len(self.valid_colors)]
        return sum(color_bits)
    

    def constraint_penalty(self, bits: List[bool]):
        
        def vertex_term(bits: List[bool]):
            vertex_penalty = 0
            for vertex in self.graph.vertices:
                vertex_penalty += (1 - sum( [ bits[self.find_index_in_combined_total_register(color, vertex)] for color in self.valid_colors ] ))**2
                vertex_penalty += sum( [ 
                    (1 - bits[self.find_index_in_combined_total_register(color)]) * bits[self.find_index_in_combined_total_register(color, vertex)] 
                    for color in self.valid_colors 
                ] )
            return vertex_penalty
        
        def edge_term(bits: List[bool]):
            edge_penalty = 0
            for edge in self.graph.edges:
                edge_penalty += sum( [ 
                    bits[self.find_index_in_combined_total_register(color, edge[0])] * bits[self.find_index_in_combined_total_register(color, edge[1])]
                    for color in self.valid_colors 
                ] )
            return edge_penalty
        
        return self.penalty * vertex_term(bits) + self.penalty * edge_term(bits)


    def objective_function(self, bitstring: str):
        bits = list(map(int, list(bitstring)))
        return - self.value(bits) - self.constraint_penalty(bits)
    


class QAOAGraphColoring(ProblemRelatedFunctions):

    def __init__(self, graph: Graph, depth: int, penalty: Union[float, int]):
        self.graph = graph
        self.valid_colors = [c for c in range(1, GraphRelatedFunctions(graph).find_max_degree(graph) + 1 + 1)] # Need to add 1 at the end since we start with 1 instead of 0
        self.depth = depth
        self.penalty = penalty
        
        self.circuit = QuasiAdiabaticEvolution(self.graph, self.valid_colors, self.depth, self.penalty)
        self.color_register_size = self.graph.number_vertices
        self.color_vertex_register_size = self.graph.number_vertices * len(self.valid_colors)

        self.qaoa = QAOA(ProblemType.minimization, self.objective_function, self.circuit.apply_quasiadiabatic_evolution, depth, self.color_register_size, self.color_vertex_register_size)

    def execute_qaoa_gc(self):
        return self.qaoa.execute_qaoa()




def main():
    graph = exemplary_graph_instances["B"]
    print(graph)
    start_time = time.time()
    print("QAOA result = ", QAOAGraphColoring(graph, depth = 2, penalty = 1.01).execute_qaoa_gc())
    print("Elapsed time = ", time.time() - start_time)


if __name__ == "__main__":
    main()