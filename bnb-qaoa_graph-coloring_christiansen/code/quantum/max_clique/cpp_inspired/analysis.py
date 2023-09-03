import sys
sys.path.append("C:\\Users\\d92474\\Documents\\Uni\\Master Thesis\\GitHub\\MasterThesis_Hybrid-Quantum-Classical-Branch-and-Bound\\bnb-qaoa_graph-coloring_christiansen\\code")
sys.path.append("C:\\Users\\d92474\\Documents\\Uni\\Master Thesis\\GitHub\\MasterThesis_Hybrid-Quantum-Classical-Branch-and-Bound\\common-utils")


from graph import Graph, GraphRelatedFunctions, exemplary_graph_instances
from quantum.max_clique.cpp_inspired.circuits_mc import QuasiAdiabaticEvolution
from qaoa import QAOA, ProblemType


import math
import numpy as np
from typing import Union, List
import time



class ProblemRelatedFunctions(GraphRelatedFunctions):

    def __init__(self, graph: Graph, penalty: Union[float, int], penalty3: Union[float, int]):
        self.graph = graph
        self.penalty = penalty
        self.penalty3 = penalty3
        max_degree = self.find_max_degree(graph)
        self.d = math.floor(math.log2(max_degree + 1))
        self.delta = max_degree + 2 - 2**(self.d)

    def value(self, vertex_bits: List[bool]):
        return sum(vertex_bits)
    
    def constraint2_penalty(self, vertex_bits: List[bool], auxiliary_bits: List[bool]):
        auxiliary_term = sum([ 2**(self.d - (j+1)) * auxiliary_bits[j] for j in range(self.d) ]) + self.delta * auxiliary_bits[self.d]
        vertex_term = sum(vertex_bits)
        return (auxiliary_term - vertex_term)**2

    def constraint3_penalty(self, vertex_bits: List[bool], auxiliary_bits: List[bool]):
        auxiliary_term = sum([ 2**(self.d - (j+1)) * auxiliary_bits[j] for j in range(self.d) ]) + self.delta * auxiliary_bits[self.d]
        vertex_term = sum([ vertex_bits[self.find_left_index_for_edge(edge)] * vertex_bits[self.find_right_index_for_edge(edge)] for edge in self.graph.edges ])
        return 1/2 * auxiliary_term * (auxiliary_term - 1) - vertex_term 

    def objective_function(self, bitstring: str):
        bits = list(map(int, list(bitstring)))
        vertex_bits = bits[:self.graph.number_vertices]
        auxiliary_bits = bits[self.graph.number_vertices:]
        return self.value(vertex_bits) - self.penalty * self.constraint2_penalty(vertex_bits, auxiliary_bits) - self.constraint3_penalty(vertex_bits, auxiliary_bits)
    


class QAOAMaxClique(ProblemRelatedFunctions):

    def __init__(self, graph: Graph, depth: int, penalty: Union[float, int], penalty3: Union[float, int]):
        self.graph = graph
        self.depth = depth
        self.penalty, self.penalty3 = penalty, penalty3
        self.circuit = QuasiAdiabaticEvolution(self.graph, self.depth, self.penalty, self.penalty3)
        max_degree = self.find_max_degree(graph)
        self.d = math.floor(math.log2(max_degree + 1))
        self.delta = max_degree + 2 - 2**(self.d)
        self.vertex_register_size, self.auxiliary_register_size = graph.number_vertices, self.d + 1
        self.qaoa = QAOA(ProblemType.maximization, self.objective_function, self.circuit.apply_quasiadiabatic_evolution, depth, self.vertex_register_size, self.auxiliary_register_size)

    def execute_qaoa_mc(self):
        return self.qaoa.execute_qaoa()




def main():
    graph = exemplary_graph_instances["D"]
    max_degree = GraphRelatedFunctions(graph).find_max_degree(graph)
    penalty3 = 10**(-5) 
    penalty = max_degree * penalty3 + 1 + penalty3
    #print("Objective function = ", ProblemRelatedFunctions(graph, penalty, penalty3).objective_function("110001110"))
    start_time = time.time()
    print("QAOA result = ", QAOAMaxClique(graph, depth = 3, penalty = penalty, penalty3 = penalty3).execute_qaoa_mc())
    print("Elapsed time = ", time.time() - start_time)


if __name__ == "__main__":
    main()
    