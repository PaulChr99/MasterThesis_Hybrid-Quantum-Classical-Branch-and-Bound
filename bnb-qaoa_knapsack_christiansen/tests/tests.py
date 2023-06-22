import sys
sys.path.append("C:\\Users\\d92474\\Documents\\Uni\\Master Thesis\\GitHub\\QuBRA\\bnb-qaoa_knapsack_christiansen\\code")
sys.path.append("C:\\Users\\d92474\\Documents\\Uni\\Master Thesis\\GitHub\\QuBRA\\bnb-qaoa_knapsack_christiansen\\code\\qaoa\\qiskit")
sys.path.append("C:\\Users\\d92474\\Documents\\Uni\\Master Thesis\\GitHub\\QuBRA\\bnb-qaoa_knapsack_christiansen\\code\\qaoa\\cpp_inspired")


import numpy as np
import random

from knapsack_problem import KnapsackProblem, exemplary_kp_instances
from classical_ingredients import SortingProfitsAndWeights, GreedyBounds, FeasibilityAndPruning, BranchingSearchingBacktracking
from branch_and_bound import BranchAndBound

#print(BranchAndBound(first_exemplary_kp).greedy_lower_bound("0"))
#print(SortingProfitsAndWeights(first_exemplary_kp.profits, first_exemplary_kp.weights).sorting_profits_weights())
#print(BranchAndBound(first_exemplary_kp).calculate_weight("111"))
#print(BranchAndBound(first_exemplary_kp).is_feasible("111"))
#print(BranchAndBound(first_exemplary_kp).backtracking(["0", "10"]))
#print(BranchAndBound(first_exemplary_kp).branch_and_bound_algorithm())

#print(SortingProfitsAndWeights(second_exemplary_kp.profits, second_exemplary_kp.weights).sorting_profits_weights())
#print(BranchAndBound(second_exemplary_kp).is_feasible("111"))
print(BranchAndBound(problem_instance = exemplary_kp_instances["C"]).branch_and_bound_algorithm())
