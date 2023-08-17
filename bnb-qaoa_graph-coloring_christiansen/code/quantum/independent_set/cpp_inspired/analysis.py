import sys
sys.path.append("C:\\Users\\d92474\\Documents\\Uni\\Master Thesis\\GitHub\\MasterThesis_Hybrid-Quantum-Classical-Branch-and-Bound\\bnb-qaoa_graph-coloring_christiansen\\code")
sys.path.append("C:\\Users\\d92474\\Documents\\Uni\\Master Thesis\\GitHub\\MasterThesis_Hybrid-Quantum-Classical-Branch-and-Bound\\common-utils")


from graph import Graph, GraphRelatedFunctions, exemplary_graph_instances
from quantum.independent_set.cpp_inspired.circuits_is import QuasiAdiabaticEvolution
from qaoa import QAOA, ProblemType


import numpy as np
from typing import Union, List
import time



class ProblemRelatedFunctions(GraphRelatedFunctions):

    def __init__(self, graph: Graph, penalty: Union[float, int]):
        self.graph = graph
        self.penalty = penalty

    def value(self, bits: List[bool]):
        return sum(bits)
    
    def constraint_penalty(self, bits: List[bool]):
        constraint_penalty = 0 
        for edge in self.graph.edges:
            index_left = self.find_left_index_for_edge(edge)
            index_right = self.find_right_index_for_edge(edge)
            constraint_penalty += bits[index_left] * bits[index_right]
        return constraint_penalty

    def objective_function(self, bitstring: str):
        bits = list(map(int, list(bitstring)))
        return self.value(bits) - self.penalty * self.constraint_penalty(bits)
    


class QAOAMaxIndependentSet(ProblemRelatedFunctions):

    def __init__(self, graph: Graph, depth: int, penalty: Union[float, int]):
        self.graph = graph
        self.depth = depth
        self.penalty = penalty
        self.vertex_register_size = graph.number_vertices
        self.circuit = QuasiAdiabaticEvolution(self.graph, self.depth, self.penalty)
        self.qaoa = QAOA(ProblemType.maximization, self.objective_function, self.circuit.apply_quasiadiabatic_evolution, depth, self.vertex_register_size)

    def execute_qaoa_is(self):
        return self.qaoa.execute_qaoa()




def main():
    graph = exemplary_graph_instances["D"]
    #graph = GraphRelatedFunctions(graph).complement_graph(graph)
    print(graph)
    start_time = time.time()
    print("QAOA result = ", QAOAMaxIndependentSet(graph, depth = 7, penalty = 2).execute_qaoa_is())
    print("Elapsed time = ", time.time() - start_time)


if __name__ == "__main__":
    main()
    