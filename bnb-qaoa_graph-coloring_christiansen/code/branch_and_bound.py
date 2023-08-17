import sys
sys.path.append("C:\\Users\\d92474\\Documents\\Uni\\Master Thesis\\GitHub\\MasterThesis_Hybrid-Quantum-Classical-Branch-and-Bound\\common-utils")

import math
from typing import Union, Dict
import time

from graph import Graph, GraphRelatedFunctions, exemplary_graph_instances
from classical_ingredients import DSatur, BranchingSearchingBacktracking, FeasibilityAndPruning
from quantum.graph_coloring.cpp_inspired.analysis import QAOAGraphColoring
from quantum.max_clique.cpp_inspired.analysis import QAOAMaxClique
from quantum.independent_set.cpp_inspired.analysis import QAOAMaxIndependentSet



class BranchAndBound(DSatur, BranchingSearchingBacktracking, FeasibilityAndPruning):

    def __init__(self, graph: Graph, quantum_ub = False, quantum_lb = False):
        vertices_sorted = GraphRelatedFunctions(graph).sort_vertices_by_degree()
        self.graph = Graph(vertices = vertices_sorted, edges = graph.edges)
        self.quantum_ub = quantum_ub
        self.quantum_lb = quantum_lb


    def is_first_solution(self, incumbent: list):
        return True if (len(incumbent) == 0) else False


    def upper_bound(self, current_node: list, qaoa_properties_gc: Dict[str, Union[int, float, None]]):
        
        if self.quantum_ub and (qaoa_properties_gc["depth"] == None or qaoa_properties_gc["penalty"] == None):
            raise ValueError("Depth and penalty for GC QAOA must not be null when a quantum upper bound shall be computed.")
        
        def qaoa_gc_upper_bound():
            equivalent_subgraph = self.equivalent_subgraph(current_node)
            return QAOAGraphColoring(graph = equivalent_subgraph, depth = qaoa_properties_gc["depth"], penalty = qaoa_properties_gc["penalty"]).execute_qaoa_gc()
        
        upper_bound_by_degree = self.upper_bound_by_degree()
        dsatur_upper_bound = self.dsatur_upper_bound(current_node)
        print("Dsatur upper bound = ", dsatur_upper_bound)
        if not self.quantum_ub:
            return min(upper_bound_by_degree, dsatur_upper_bound)
        else:
            qaoa_gc_ub = qaoa_gc_upper_bound()
            print("QAOA Graph Coloring upper bound = ", qaoa_gc_ub)
            return min(upper_bound_by_degree, dsatur_upper_bound, qaoa_gc_ub)


    def lower_bound(self, current_node: list, qaoa_properties_mc: Dict[str, Union[int, float, None]], qaoa_properties_is: Dict[str, Union[int, float, None]]):
        
        if self.quantum_lb and (
            qaoa_properties_mc["depth"] == None or qaoa_properties_mc["penalty"] == None or qaoa_properties_mc["penalty3"] == None 
            or qaoa_properties_is["depth"] == None or qaoa_properties_is["penalty"] == None
        ):
            raise ValueError("Depths and penalties for MC and IS QAOAs must be specified if a quantum lower bound shall be computed.")

        hoffman_lower_bound = self.hoffman_lower_bound(current_node)
        print("Hoffman lower bound = ", hoffman_lower_bound)
        equivalent_subgraph = self.equivalent_subgraph(current_node) 

        if not self.quantum_lb:
            return hoffman_lower_bound
        
        else:
            max_degree_of_subgraph = self.find_max_degree(equivalent_subgraph)
            d_subgraph = math.floor(math.log2(max_degree_of_subgraph + 1))
            relation_qubit_amounts = 1 + (d_subgraph + 1) / equivalent_subgraph.number_vertices
            gate_amount_comparing = 1/3 * equivalent_subgraph.number_vertices**2 + 1/3 * d_subgraph**2 + d_subgraph + 2/3 * equivalent_subgraph.number_vertices * d_subgraph + 2/3
            complement_subgraph = self.complement_graph(equivalent_subgraph)
            
            if complement_subgraph.number_edges > relation_qubit_amounts * gate_amount_comparing:
                qaoa_mc_lb = QAOAMaxClique(
                    graph = equivalent_subgraph, 
                    depth = qaoa_properties_mc["depth"], 
                    penalty = qaoa_properties_mc["penalty"], 
                    penalty3 = qaoa_properties_mc["penalty3"]
                ).execute_qaoa_mc()
                print("QAOA Max Clique lower bound = ", qaoa_mc_lb)
                return max(hoffman_lower_bound, qaoa_mc_lb)
            
            else:
                qaoa_is_lb = QAOAMaxIndependentSet(graph = complement_subgraph, depth = qaoa_properties_is["depth"], penalty = qaoa_properties_is["penalty"]).execute_qaoa_is()
                print("QAOA Independent Set lower bound = ", qaoa_is_lb)
                return max(hoffman_lower_bound, qaoa_is_lb)


    def branch_and_bound_algorithm(self, 
            qaoa_properties_gc: Dict[str, Union[int, float, None]] = {"depth": None, "penalty": None},
            qaoa_properties_mc: Dict[str, Union[int, float, None]] = {"depth": None, "penalty": None, "penalty3": None},
            qaoa_properties_is: Dict[str, Union[int, float, None]] = {"depth": None, "penalty": None}
        ):
        
        print("Ordered vertices = ", self.graph.vertices)

        """ Instantiate the algorithm """
        stack = []
        incumbent = []
        best_upper_bound = self.dsatur_upper_bound(incumbent)
        stack = self.branching(stack)
        current_node = self.node_selection(stack)
        counter, leaf_counter, qaoa_ub_counter, qaoa_lb_counter = 0, 0, 0, 0

        while stack:
            print("stack = ", stack)
            counter += 1

            # Every node, i.e. (partial) solution, is first checked for feasibility
            if not self.is_feasible(current_node):
                stack.remove(current_node)
                # Can only backtrack to another node if stack is not empty after removing
                if stack:
                    current_node = self.backtracking(stack, current_node)
                continue

            # Computing bounds for (feasible) leafs is useless effort, so they are directly evaluated
            if len(current_node) == self.graph.number_vertices:
                leaf_counter += 1
                # No further checks are needed when arriving at first leaf - it will always become the new incumbent
                if len(incumbent) == 0:
                    incumbent = current_node
                    incumbent_value = self.objective_function(incumbent)
                else:
                    current_value = self.objective_function(current_node)
                    # Only update the incumbent if the current leaf is better than the best one found so far
                    if (current_value < incumbent_value):
                        incumbent = current_node
                        incumbent_value = current_value
                stack.remove(current_node)
                # Can only backtrack to another node if stack is not empty after removing
                if stack:
                    current_node = self.backtracking(stack, current_node)
                continue

            # To avoid never arriving at any leaf, as long as no leaf has been found, nodes are only pruned
            # if current upper bound is strictly smaller (not \leq, see Latex) than best lower bound 
            current_lower_bound = self.lower_bound(current_node, qaoa_properties_mc, qaoa_properties_is)
            if self.quantum_lb:
                qaoa_lb_counter += 1 
            if self.can_be_pruned(current_node, current_lower_bound, best_upper_bound, is_first_solution = self.is_first_solution(incumbent)):
                stack.remove(current_node)
                # Can only backtrack to another node if stack is not empty after removing
                if stack:
                    current_node = self.backtracking(stack, current_node)
                continue

            # Nodes can be further processed properly if being feasible, not being a leaf and not being prunable
            # In this case: best upper bound potentially updated, child nodes generated via branching, and next node selected
            current_upper_bound = self.upper_bound(current_node, qaoa_properties_gc)
            if self.quantum_ub:
                qaoa_ub_counter += 1 
            #print("Current upper bound = ", current_upper_bound)
            if current_upper_bound < best_upper_bound:
                #print(f"Best upper bound updated: from {best_upper_bound} to {current_upper_bound}")
                best_upper_bound = current_upper_bound
            stack = self.branching(stack, current_node)
            stack.remove(current_node)
            # No check for emptyness of stack needed here since branching always generates two further nodes
            current_node = self.node_selection(stack)

        optimal_solution = [{"v": self.graph.vertices[i], "c": incumbent[i]} for i in range(self.graph.number_vertices)]
        chromatic_number = self.objective_function(incumbent)
        result = {"optimal solution": optimal_solution, "chromatic number": chromatic_number, "number of explored nodes": counter, 
                    "number of leafs reached": leaf_counter, "number of QAOA executions for upper bounds": qaoa_ub_counter,
                    "number of QAOA executions for lower bounds": qaoa_lb_counter} 
        return result 




def main():
    graph = exemplary_graph_instances["C"]
    qaoa_properties_gc = {"depth": 1, "penalty": 1.01}
    
    max_degree = GraphRelatedFunctions(graph).find_max_degree(graph)
    penalty3_mc = 10**(-5)
    penalty_mc = max_degree * penalty3_mc + 1 + penalty3_mc
    qaoa_properties_mc = {"depth": 3, "penalty": penalty_mc, "penalty3": penalty3_mc}
    
    qaoa_properties_is = {"depth": 5, "penalty": 2}
    
    start_time = time.time()
    print(BranchAndBound(graph, quantum_ub = True, quantum_lb = True).branch_and_bound_algorithm(qaoa_properties_gc, qaoa_properties_mc, qaoa_properties_is))
    print("Elapsed time = ", time.time() - start_time)


if __name__ == "__main__":
    main()