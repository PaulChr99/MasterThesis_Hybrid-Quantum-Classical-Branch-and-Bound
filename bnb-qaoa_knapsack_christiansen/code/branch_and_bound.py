from dataclasses import dataclass, fields
import numpy as np
import math
from typing import Union, Dict

from knapsack_problem import KnapsackProblem
from classical_ingredients import BranchingSearchingBacktracking, DynamicalSubproblems
from qaoa.cpp_inspired.analysis import QAOA as HardQAOA
from qaoa.qiskit.circuits import LinQAOACircuit
from qaoa.qiskit.linear_soft_constraint import LinearSoftConstraintQAOA


class BranchAndBound(BranchingSearchingBacktracking, DynamicalSubproblems):
    """
    Main class for constructing the branch and bound algorithm.

    Attributes:
    problem_instance (KnapsackProblem): the instance of the knapsack problem at hand
    profits_sorted (list): profits sorted according to ratio of profit/weight
    weights_sorted (list): weights sorted according to ratio of profit/weight

    Methods: 
    greedy_lower_bound: lower bound obtained by greedy algorithm, i.e. selecting all
        items up to (but without) the critical one
    greedy_upper_bound: upper bound obtained by greedy algorithm, i.e. exact solution
        of LP relaxed knapsack problem
    branching: performing the next branching step, i.e. fixing the next item to be 
        either included or not
    """

    def __init__(self, problem_instance: KnapsackProblem, quantum_soft: bool = False, quantum_hard: bool = False):
        BranchingSearchingBacktracking.__init__(self, problem_instance)
        self.quantum_soft, self.quantum_hard = quantum_soft, quantum_hard


    def greedy_vs_qaoa(self, partial_choice: str, soft_qaoa_depth: Union[int, None], hard_qaoa_depth: Union[int, None]):
        
        greedy_lower_bound = self.greedy_lower_bound(partial_choice)
        print("Greedy lower bound = ", greedy_lower_bound)
        
        offset = self.calculate_profit(partial_choice)
        residual_subproblem = self.partial_choice_to_subproblem(partial_choice)
        print("Residual subproblem = ", residual_subproblem)
        
        def soft_qaoa_lower_bound():
            circuit_for_subproblem = LinQAOACircuit(problem = residual_subproblem, p = soft_qaoa_depth)

            def get_min_penalty_prefactor(problem_instance: KnapsackProblem):
                return max(problem_instance.profits)
            
            """ Currently using as value for a always a_min + 1, which however differs from problem
            to problem and even between subproblems of the same knapsack problem instance. """
            penalty_prefactor = get_min_penalty_prefactor(residual_subproblem) + 1
            qaoa_result = LinearSoftConstraintQAOA(residual_subproblem, circuit = circuit_for_subproblem, a = penalty_prefactor).execute_algorithm()
            return offset + qaoa_result
        
        def hard_qaoa_lower_bound():
            qaoa_result = HardQAOA(problem_instance = residual_subproblem, depth = hard_qaoa_depth).execute_qaoa()
            return offset + qaoa_result
        
        if self.quantum_soft and not self.quantum_hard:
            qaoa_lb = soft_qaoa_lower_bound()
            print("Soft QAOA lower bound = ", qaoa_lb)
            return max(greedy_lower_bound, qaoa_lb)
        elif not self.quantum_soft and self.quantum_hard:
            qaoa_lb = hard_qaoa_lower_bound()
            print("Hard QAOA lower bound = ", qaoa_lb)
            return max(greedy_lower_bound, qaoa_lb)
        elif self.quantum_soft and self.quantum_hard:
            soft_qaoa_lb = soft_qaoa_lower_bound()
            print("Soft QAOA lower bound = ", soft_qaoa_lb)
            hard_qaoa_lb = hard_qaoa_lower_bound()
            print("Hard QAOA lower bound = ", hard_qaoa_lb)
            return max(greedy_lower_bound, soft_qaoa_lb, hard_qaoa_lb)
        else:
            return greedy_lower_bound


    def branch_and_bound_algorithm(self, soft_qaoa_depth: Union[int, None] = None, hard_qaoa_depth: Union[int, None] = None):
        
        print("Problem with items sorted = ", KnapsackProblem(self.profits, self.weights, self.capacity))

        # Instantiate the algorithm
        stack = []
        incumbent = ""
        best_lower_bound = self.greedy_lower_bound(incumbent)
        stack = self.branching(stack)
        current_node = self.node_selection(stack)
        counter, leaf_counter, qaoa_counter = 0, 0, 0

        # Iterate until the stack of unexplored nodes is empty
        while stack:
            print("stack = ", stack)
            #print("current node = ", current_node)
            counter += 1

            # Every node, i.e. (partial) solution, is first checked for feasibility
            if not self.is_feasible(current_node):
                stack.remove(current_node)
                # Can only backtrack to another node if stack is not empty after removing
                if stack:
                    current_node = self.backtracking(stack)
                continue

            """ It reduces computational effort when detecting a node for which the capacity
            is exactly and fully consumed, since the only valid solution is obtained by filling
            up the remaining items with all 0s """
            if self.calculate_residual_capacity(current_node) == 0:
                leaf_counter += 1
                optimal_candidate = self.complete_partial_choice(current_node)
                print("optimal candidate = ", optimal_candidate)
                if len(incumbent) == 0:
                    incumbent = optimal_candidate
                    # Profit of current_node is same as profit of optimal_candidate since only added 0s
                    incumbent_profit = self.calculate_profit(current_node) 
                else:
                    current_profit = self.calculate_profit(current_node)
                    # Only update the incumbent if the current leaf is better than the best one found so far
                    if (current_profit > incumbent_profit):
                        incumbent = optimal_candidate
                        incumbent_profit = current_profit
                stack.remove(current_node)
                if stack:
                    current_node = self.backtracking(stack)
                continue
            
            # Computing bounds for (feasible) leafs is useless effort, so they are directly evaluated
            if len(current_node) == self.number_items:
                leaf_counter += 1
                # No further checks are needed when arriving at first leave, will always become incumbent
                if len(incumbent) == 0:
                    incumbent = current_node
                    incumbent_profit = self.calculate_profit(incumbent)
                else:
                    current_profit = self.calculate_profit(current_node)
                    # Only update the incumbent if the current leaf is better than the best one found so far
                    if (current_profit > incumbent_profit):
                        incumbent = current_node
                        incumbent_profit = current_profit
                stack.remove(current_node)
                # Can only backtrack to another node if stack is not empty after removing
                if stack:
                    current_node = self.backtracking(stack)
                continue
            
            # To avoid never arriving at any leaf, as long as no leaf has been found, nodes are only pruned
            # if current upper bound is really smaller (not \leq, see Latex) than best lower bound 
            if self.can_be_pruned(current_node, best_lower_bound, is_first_solution = True if (len(incumbent) == 0) else False):
                stack.remove(current_node)
                # Can only backtrack to another node if stack is not empty after removing
                if stack:
                    current_node = self.backtracking(stack)
                continue
            
            # Nodes can be further processed properly if being feasible, not being a leaf and not being prunable
            # In this case: best lower bound potentially updated, child nodes generated via branching, and next node selected
            current_lower_bound = self.greedy_vs_qaoa(current_node, soft_qaoa_depth, hard_qaoa_depth)
            if self.quantum_soft or self.quantum_hard:
                qaoa_counter += 1
            if current_lower_bound > best_lower_bound:
                best_lower_bound = current_lower_bound
            stack = self.branching(stack, current_node)
            stack.remove(current_node)
            # No check for emptyness of stack needed here since branching always generates two further nodes
            current_node = self.node_selection(stack) 

        optimal_solution, maximum_profit = incumbent, self.calculate_profit(incumbent)
        result = {"optimal solution": optimal_solution, "maximum profit": maximum_profit, "number of explored nodes": counter, 
                    "number of leafs reached": leaf_counter, "number of qaoa executions": qaoa_counter} 
        return result

"""
Now everything seems to work properly. However, due to choosing the next node randomly (e.g. when not both of the candidates are feasible)
the performance (i.e. number of explored nodes and reached leafs) may vary from one execution to another.
"""