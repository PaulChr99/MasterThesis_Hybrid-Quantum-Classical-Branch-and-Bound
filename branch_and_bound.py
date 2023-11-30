import numpy as np
import time
from typing import Union

from knapsack_problem import KnapsackProblem, GenerateKnapsackProblemInstances
from classical_ingredients import BranchingSearchingBacktracking, DynamicalSubproblems, SortingProfitsAndWeights
from qaoa import QAOA, QTG


class BranchAndBound(BranchingSearchingBacktracking, DynamicalSubproblems):
    """
    Main class for constructing the branch and bound algorithm.

    Attributes:
    profits (list): profits sorted according to ratio of profit/weight
    weights (list): weights sorted according to ratio of profit/weight
    capacity (int): the capacity of the KP instance
    number_items (int): the number of items of the KP instance
    problem_instance (KnapsackProblem): the instance of the knapsack problem at hand
    simulation (bool): indicating whether the B&B is run for simulation purposes, implying fewer print statements
    quantum (bool): indicating whether the quantum extension, i.e. the QAOA as alternative lower bound, shall be activated
    lb_simulation_raw_data (list): raw data for the ratio between QAOA and greedy lower bound that can be analyzed in simulations
    qaoa_counter (int): counting the number of QAOA applications throughout the B&B run

    Methods: 
    greedy_vs_qaoa: Calculates the greedy lower bound and compares it to the QAOA output for a specified depth if the quantum flag is set to true.
        If that is the case, the better (i.e. the larger) of the two results is returned. If not, the greedy solution is returned without a QAOA run.
    branch_and_bound_algorithm: The main method implementing the Branch and Bound, incorporating all single fragments, like the searching routines
        (node selection and backtracking), the pruning strategy (via upper & lower bounds) and the branching heuristic together with certain refinements 
    """
    
    def __init__(self, problem_instance: KnapsackProblem, simulation: bool = False, quantum: bool = False):
        BranchingSearchingBacktracking.__init__(self, problem_instance)
        self.problem_instance = problem_instance
        self.simulation = simulation
        self.quantum = quantum
        self.lb_simulation_raw_data = []
        self.qaoa_counter = 0
    

    def greedy_vs_qaoa(self, partial_choice: str, qaoa_depth: Union[int, None]):
        
        offset = self.calculate_profit(partial_choice)
        residual_subproblem = self.partial_choice_to_subproblem(partial_choice)
        if len(residual_subproblem.profits) == 0 or len(residual_subproblem.weights) == 0:
            if len(residual_subproblem.profits) == 0 == len(residual_subproblem.weights):
                return offset
            else:
                raise ValueError("The subproblem generation is broken; none of the remaining items being affordable must be reflected both in profits and weights!")

        greedy_lower_bound = self.greedy_lower_bound(residual_subproblem)
        full_greedy_lower_bound = offset + greedy_lower_bound
        if not self.simulation: 
            print("Greedy lower bound = ", full_greedy_lower_bound)
        
        def hard_qaoa_lower_bound():
            qtg_output = QTG(residual_subproblem).quantum_tree_generator()
            qaoa_result = QAOA(residual_subproblem, qtg_output, qaoa_depth).optimize()["qaoa result"]
            if self.simulation:
                qubit_number_for_residual_subproblem = residual_subproblem.number_items + int(np.ceil(np.log2(residual_subproblem.capacity)) + 1)
                self.lb_simulation_raw_data.append({"residual qubit size": qubit_number_for_residual_subproblem, "ratio": qaoa_result / greedy_lower_bound})
            return offset + qaoa_result
        
        if self.quantum:
            qaoa_lb = hard_qaoa_lower_bound()
            self.qaoa_counter += 1
            if not self.simulation: 
                print("Hard QAOA lower bound = ", qaoa_lb)
            return max(full_greedy_lower_bound, qaoa_lb)
        else:
            return full_greedy_lower_bound



    def branch_and_bound_algorithm(self, qaoa_depth: Union[int, None] = None):
        
        # Instantiate the algorithm
        stack = []
        incumbent = ""
        best_lower_bound = self.greedy_lower_bound(self.partial_choice_to_subproblem(incumbent))
        stack = self.branching(stack)
        current_node = self.node_selection(stack)
        counter, leaf_counter = 0, 0

        # Iterate until the stack of unexplored nodes is empty
        while stack:
            if not self.simulation: 
                print("stack = ", stack)
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
            
            # Computing bounds for (feasible) leaves is useless effort, so they are directly evaluated
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
            if self.can_be_pruned(current_node, best_lower_bound, is_first_solution = True if len(incumbent) == 0 else False):
                stack.remove(current_node)
                # Can only backtrack to another node if stack is not empty after removing
                if stack:
                    current_node = self.backtracking(stack)
                continue
            
            #print("Incumbent after = ", incumbent)

            # Nodes can be further processed properly if being feasible, not being a leaf and not being prunable
            # In this case: best lower bound potentially updated, child nodes generated via branching, and next node selected
            current_lower_bound = self.greedy_vs_qaoa(current_node, qaoa_depth)
            if current_lower_bound > best_lower_bound:
                best_lower_bound = current_lower_bound
            stack = self.branching(stack, current_node)
            stack.remove(current_node)
            # No check for emptyness of stack needed here since branching always generates two further nodes
            current_node = self.node_selection(stack) 

        optimal_solution, maximum_profit = incumbent, self.calculate_profit(incumbent)
        sorting_permutation = SortingProfitsAndWeights(self.profits, self.weights).sorting_permutation(self.problem_instance.profits, self.problem_instance.weights)
        optimal_solution_original_sorting = "".join([list(optimal_solution)[sorting_permutation.index(idx)] for idx in range(len(sorting_permutation))])
        result = {"optimal solution": optimal_solution_original_sorting, "maximum profit": maximum_profit, "number of explored nodes": counter, 
                    "number of leaves reached": leaf_counter, "number of qaoa executions": self.qaoa_counter} 
        return result


"""
Now everything seems to work properly. However, due to choosing the next node randomly (e.g. when not both of the candidates are feasible)
the performance (i.e. number of explored nodes and reached leaves) may vary from one execution to another.
"""



def main():
    random_kp_instance = GenerateKnapsackProblemInstances.generate_random_kp_instance_for_capacity_ratio_and_maximum_value(
        size = 3, 
        desired_capacity_ratio = 0.75,
        maximum_value = 1e18)
    check_kp_instance = KnapsackProblem(
        profits = [1184, 111, 4967, 1544, 1100],
        weights = [10885, 10950, 2267, 3385, 7808],
        capacity = 14393
    )
    bnb = BranchAndBound(check_kp_instance, simulation = False, quantum = True)
    start_time = time.time()
    bnb_result = bnb.branch_and_bound_algorithm(qaoa_depth = 2)
    print(bnb_result)
    print("Elapsed time = ", time.time() - start_time)



if __name__ == "__main__":
    main()