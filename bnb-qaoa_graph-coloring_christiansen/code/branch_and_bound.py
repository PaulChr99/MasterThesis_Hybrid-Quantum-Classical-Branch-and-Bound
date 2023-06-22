import math
from typing import Union, Dict

from graph import Graph, GraphRelatedFunctions
from classical_ingredients import DSatur, BranchingSearchingBacktracking, FeasibilityAndPruning
from qaoa.graph_coloring.circuits_gc import QAOACircuitGC
from qaoa.graph_coloring.qaoa_algorithm import QAOAGC
from qaoa.max_clique.circuits_mc import QAOACircuitMC
from qaoa.max_clique.qaoa_algorithm import QAOAMC
from qaoa.independent_set.circuits_is import QAOACircuitIS
from qaoa.independent_set.qaoa_algorithm import QAOAIS


class BranchAndBound(DSatur, BranchingSearchingBacktracking, FeasibilityAndPruning):

    def __init__(self, graph: Graph, quantum_ub = False, quantum_lb = False):
        vertices_sorted = GraphRelatedFunctions(graph).sort_vertices_by_degree()
        self.graph = Graph(vertices = vertices_sorted, edges = graph.edges)
        self.quantum_ub = quantum_ub
        self.quantum_lb = quantum_lb


    def is_first_solution(self, incumbent: list):
        return True if (len(incumbent) == 0) else False


    def upper_bound(self, current_node: list, qaoa_gc_depth: Union[int, None], a_gc: Union[int, float, None], b_gc: Union[int, float, None]):
        
        def qaoa_gc_upper_bound():
            equivalent_subgraph = self.equivalent_subgraph(current_node)
            circuit_for_subgraph = QAOACircuitGC(equivalent_subgraph, qaoa_gc_depth)
            return QAOAGC(equivalent_subgraph, circuit_for_subgraph, a_gc, b_gc).execute_algorithm()
        
        upper_bound_by_degree = self.upper_bound_by_degree()
        dsatur_upper_bound = self.dsatur_upper_bound(current_node)
        print("Dsatur upper bound = ", dsatur_upper_bound)
        if not self.quantum_ub:
            return min(upper_bound_by_degree, dsatur_upper_bound)
        else:
            qaoa_gc_ub = qaoa_gc_upper_bound()
            print("QAOA Graph Coloring upper bound = ", qaoa_gc_ub)
            return min(upper_bound_by_degree, dsatur_upper_bound, qaoa_gc_ub)


    def lower_bound(self, current_node: list, qaoa_mc_depth: Union[int, None], a_mc: Union[int, float, None], 
                    qaoa_is_depth: Union[int, None], a_is: Union[int, float, None], b_is: Union[int, float, None]):
        
        hoffman_lower_bound = self.hoffman_lower_bound(current_node)
        print("Hoffman lower bound = ", hoffman_lower_bound)
        equivalent_subgraph = self.equivalent_subgraph(current_node)

        def qaoa_mc_lower_bound():
            circuit_for_subgraph = QAOACircuitMC(equivalent_subgraph, qaoa_mc_depth)
            return QAOAMC(equivalent_subgraph, circuit_for_subgraph, a_mc).execute_algorithm()

        complement_subgraph = self.complement_graph(equivalent_subgraph)

        def qaoa_is_lower_bound():
            circuit_for_complement_subgraph = QAOACircuitIS(complement_subgraph, qaoa_is_depth)
            return QAOAIS(complement_subgraph, circuit_for_complement_subgraph, a_is, b_is).execute_algorithm()

        if not self.quantum_lb:
            return hoffman_lower_bound
        else:
            max_degree_of_subgraph = self.find_max_degree(equivalent_subgraph)
            d_subgraph = math.floor(math.log2(max_degree_of_subgraph + 1))
            relation_qubit_amounts = 1 + (d_subgraph + 1) / equivalent_subgraph.number_vertices
            gate_amount_comparing = 1/3 * equivalent_subgraph.number_vertices**2 + 1/3 * d_subgraph**2 + d_subgraph + 2/3 * equivalent_subgraph.number_vertices * d_subgraph + 2/3
            if complement_subgraph.number_edges > relation_qubit_amounts * gate_amount_comparing:
                qaoa_mc_lb = qaoa_mc_lower_bound()
                print("QAOA Max Clique lower bound = ", qaoa_mc_lb)
                return max(hoffman_lower_bound, qaoa_mc_lb)
            else:
                qaoa_is_lb = qaoa_is_lower_bound()
                print("QAOA Independent Set lower bound = ", qaoa_is_lb)
                return max(hoffman_lower_bound, qaoa_is_lb)


    def branch_and_bound_algorithm(self, qaoa_depth_dict: Dict[str, Union[int, None]] = {"depth_gc": None, "depth_mc": None, "depth_is": None}, 
                                    a_dict: Dict[str, Union[int, float, None]] = {"a_gc": None, "a_mc": None, "a_is": None}, 
                                    b_dict: Dict[str, Union[int, float, None]] = {"b_gc": None, "b_is": None}):
        
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
            current_lower_bound = self.lower_bound(current_node, qaoa_depth_dict["depth_mc"], a_dict["a_mc"], qaoa_depth_dict["depth_is"], a_dict["a_is"], b_dict["b_is"])
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
            current_upper_bound = self.upper_bound(current_node, qaoa_depth_dict["depth_gc"], a_dict["a_gc"], b_dict["b_gc"])
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

