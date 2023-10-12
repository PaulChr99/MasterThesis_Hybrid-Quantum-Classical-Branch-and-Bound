from dataclasses import dataclass, fields
import numpy as np
import time
from typing import Union, Dict

from knapsack_problem import KnapsackProblem, GenerateKnapsackProblemInstances, exemplary_kp_instances
from classical_ingredients import BranchingSearchingBacktracking, DynamicalSubproblems, SortingProfitsAndWeights, AuxiliaryFunctions, EvaluatingProfitsAndWeights, FeasibilityAndPruning
#from quantum.cpp_inspired.analysis import QAOAKnapsack as HardQAOA
from quantum.classical_pretending.qaoa import QAOA as HardQAOA, QTG
from quantum.qiskit.circuits import LinQAOACircuit
from quantum.qiskit.linear_soft_constraint import LinearSoftConstraintQAOA


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
    
    
    def __init__(self, problem_instance: KnapsackProblem, simulation: bool = False, quantum_hard: bool = False, quantum_soft: bool = False):
        BranchingSearchingBacktracking.__init__(self, problem_instance)
        self.problem_instance = problem_instance
        self.simulation = simulation
        self.quantum_soft, self.quantum_hard = quantum_soft, quantum_hard
        self.lb_simulation_raw_data = []
        self.qaoa_counter = 0
    
    def greedy_vs_qaoa(self, partial_choice: str, hard_qaoa_depth: Union[int, None], soft_qaoa_depth: Union[int, None]):
        
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
        
                #print("Partial choice = ", partial_choice)
        #print("Residual subproblem = ", residual_subproblem)
        
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
            qtg_output = QTG(residual_subproblem).quantum_tree_generator()
            qaoa_result = HardQAOA(residual_subproblem, qtg_output, hard_qaoa_depth).optimize()["qaoa result"]
            if self.simulation:
                qubit_number_for_residual_subproblem = residual_subproblem.number_items + int(np.ceil(np.log2(residual_subproblem.capacity)) + 1)
                self.lb_simulation_raw_data.append({"residual qubit size": qubit_number_for_residual_subproblem, "ratio": qaoa_result / greedy_lower_bound})
            return offset + qaoa_result
        
        if self.quantum_soft and not self.quantum_hard:
            qaoa_lb = soft_qaoa_lower_bound()
            self.qaoa_counter += 1
            print("Soft QAOA lower bound = ", qaoa_lb)
            return max(full_greedy_lower_bound, qaoa_lb)
        elif not self.quantum_soft and self.quantum_hard:
            qaoa_lb = hard_qaoa_lower_bound()
            self.qaoa_counter += 1
            if not self.simulation: 
                print("Hard QAOA lower bound = ", qaoa_lb)
            return max(full_greedy_lower_bound, qaoa_lb)
        elif self.quantum_soft and self.quantum_hard:
            soft_qaoa_lb = soft_qaoa_lower_bound()
            print("Soft QAOA lower bound = ", soft_qaoa_lb)
            hard_qaoa_lb = hard_qaoa_lower_bound()
            print("Hard QAOA lower bound = ", hard_qaoa_lb)
            self.qaoa_counter += 1
            return max(full_greedy_lower_bound, soft_qaoa_lb, hard_qaoa_lb)
        else:
            return full_greedy_lower_bound


    def branch_and_bound_algorithm(self, hard_qaoa_depth: Union[int, None] = None, soft_qaoa_depth: Union[int, None] = None):
        
        #print("Problem with items sorted = ", KnapsackProblem(self.profits, self.weights, self.capacity))

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
                #print("current node = ", current_node)
            counter += 1

            # Every node, i.e. (partial) solution, is first checked for feasibility
            if not self.is_feasible(current_node):
                stack.remove(current_node)
                # Can only backtrack to another node if stack is not empty after removing
                if stack:
                    current_node = self.backtracking(stack)
                continue

            #print("Incumbent before = ", incumbent)

            """ It reduces computational effort when detecting a node for which the capacity
            is exactly and fully consumed, since the only valid solution is obtained by filling
            up the remaining items with all 0s """
            if self.calculate_residual_capacity(current_node) == 0:
                leaf_counter += 1
                optimal_candidate = self.complete_partial_choice(current_node)
                #print("optimal candidate = ", optimal_candidate)
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
                #print("leaf = ", AuxiliaryFunctions.find_selected_items(current_node))
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
            current_lower_bound = self.greedy_vs_qaoa(current_node, hard_qaoa_depth, soft_qaoa_depth)
            if current_lower_bound > best_lower_bound:
                best_lower_bound = current_lower_bound
            stack = self.branching(stack, current_node)
            stack.remove(current_node)
            # No check for emptyness of stack needed here since branching always generates two further nodes
            current_node = self.node_selection(stack) 

        optimal_solution, maximum_profit = incumbent, self.calculate_profit(incumbent)
        sorting_permutation = SortingProfitsAndWeights(self.profits, self.weights).sorting_permutation(self.problem_instance.profits, self.problem_instance.weights)
        #print("Length optimal solution == length sorting permutation ?: ", len(optimal_solution) == len(sorting_permutation))
        #print("Optimal solution = ", optimal_solution)
        #print("Incumbent profit = ", incumbent_profit)
        #print("Sorting permutation = ", sorting_permutation)
        optimal_solution_original_sorting = "".join([list(optimal_solution)[sorting_permutation.index(idx)] for idx in range(len(sorting_permutation))])
        result = {"optimal solution": optimal_solution_original_sorting, "maximum profit": maximum_profit, "number of explored nodes": counter, 
                    "number of leafs reached": leaf_counter, "number of qaoa executions": self.qaoa_counter} 
        return result

"""
Now everything seems to work properly. However, due to choosing the next node randomly (e.g. when not both of the candidates are feasible)
the performance (i.e. number of explored nodes and reached leafs) may vary from one execution to another.
"""



def main():
    problem_instance = exemplary_kp_instances["D"]
    kp_instance_data = open(
        "C:\\Users\\d92474\\Documents\\Uni\\Master Thesis\\GitHub\\MasterThesis_Hybrid-Quantum-Classical-Branch-and-Bound\\bnb-qaoa_knapsack_christiansen\\code\\kp_instances_data\\uncorrelated\\10000.txt", 
        "r"
    ).readlines()
    profits = [int(line.split()[0]) for line in kp_instance_data[1:-1]]
    weights = [int(line.split()[1]) for line in kp_instance_data[1:-1]]
    kp_instance = KnapsackProblem(
        profits, 
        weights, 
        capacity = int(kp_instance_data[0].split()[1]) #int(np.ceil(1/100 * sum(weights)))
    )
    #print("capacity ratio = ", kp_instance.capacity / sum(kp_instance.weights))
    #print("capacity = ", int(np.ceil(kp_instance.capacity)))
    random_kp_instance = GenerateKnapsackProblemInstances.generate_random_kp_instance_for_capacity_ratio_and_maximum_value(
        size = 10, 
        desired_capacity_ratio = 0.25,
        maximum_value = 1e5)
    #print("KP instance = ", random_kp_instance)
    bnb = BranchAndBound(random_kp_instance, simulation = True, quantum_hard = False)
    start_time = time.time()
    bnb_result = bnb.branch_and_bound_algorithm(hard_qaoa_depth = 3)
    print(bnb_result)
    #print("Selected items of bnb solution = ", AuxiliaryFunctions.find_selected_items(bnb_result["optimal solution"]))
    #optimal_solution = kp_instance_data[-1].replace(" ", "")
    #print("Selected items of bnb solution equal to optimal solution = ", AuxiliaryFunctions.find_selected_items(bnb_result["optimal solution"]) == AuxiliaryFunctions.find_selected_items(optimal_solution))
    #print("Optimal solution value = ", EvaluatingProfitsAndWeights(kp_instance).calculate_profit(optimal_solution))
    #print("Weight of optimal solution = ", EvaluatingProfitsAndWeights(kp_instance).calculate_weight(optimal_solution))
    print("Elapsed time = ", time.time() - start_time)



if __name__ == "__main__":
    main()