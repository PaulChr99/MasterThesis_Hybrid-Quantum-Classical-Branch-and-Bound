import sys
sys.path.append("C:\\Users\\d92474\\Documents\\Uni\\Master Thesis\\GitHub\\QuBRA\\bnb-qaoa_graph-coloring_christiansen\\code")
sys.path.append("C:\\Users\\d92474\\Documents\\Uni\\Master Thesis\\GitHub\\QuBRA\\bnb-qaoa_graph-coloring_christiansen\\code\\qaoa\\graph_coloring")
sys.path.append("C:\\Users\\d92474\\Documents\\Uni\\Master Thesis\\GitHub\\QuBRA\\bnb-qaoa_graph-coloring_christiansen\\code\\qaoa\\max_clique")
sys.path.append("C:\\Users\\d92474\\Documents\\Uni\\Master Thesis\\GitHub\\QuBRA\\bnb-qaoa_graph-coloring_christiansen\\code\\qaoa\\independent_set")


import time

from graph import exemplary_gc_instances
from classical_ingredients import DSatur, HoffmanBound
from branch_and_bound import BranchAndBound

dsatur_ub = DSatur(exemplary_gc_instances["B"]).dsatur_upper_bound([])
hoffman_lb = HoffmanBound(exemplary_gc_instances["B"]).hoffman_lower_bound([1,2,3])
#print("DSatur upper bound = ", dsatur_ub)
#print("Hoffman's lower bound = ", hoffman_lb)
#print(dsatur_ub == hoffman_lb)
#print(GraphRelatedFunctions(graph).sort_vertices_by_degree())
#print(BranchingSearchingBacktracking(graph).backtracking(stack=[[1],[1,2,1],[1,2,3]], current_node=[1,2,2]))
#print(BranchingSearchingBacktracking(graph).node_selection(stack=[[1],[1,2,1],[1,2,2],[1,2,3]]))
#print(FeasibilityAndPruning(graph).is_feasible([1,2,3,4]))
qaoa_depth_dict = {"depth_gc": 2, "depth_mc": 2, "depth_is": 2}
a_dict = {"a_gc": 2, "a_mc": 2, "a_is": 2}
b_dict = {"b_gc": 2, "b_is": 2.01}
start_time = time.time()
print(BranchAndBound(exemplary_gc_instances["B"], quantum_ub = True, quantum_lb = True).branch_and_bound_algorithm(qaoa_depth_dict, a_dict, b_dict))
end_time = time.time()
print(f"Elapsed time: {end_time - start_time}")
