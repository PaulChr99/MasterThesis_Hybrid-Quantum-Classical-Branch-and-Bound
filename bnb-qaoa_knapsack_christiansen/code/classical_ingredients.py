import numpy as np
import random

from knapsack_problem import KnapsackProblem, exemplary_kp_instances


class AuxiliaryFunctions:
    
    def find_selected_items(bitstring: str):
        return [idx + 1 for idx, bit in enumerate(bitstring) if bit == "1"]



class SortingProfitsAndWeights:
    """
    Class for sorting profits and weights according to the ratio of profit/weight in
    ascending order.

    Attributes:
    profits (list): the profits (values) of the items 
    weights (list): the costs (weights) of the items

    Methods:
    sorting_profits_weights: sorting according to ratio of profit over weight
    """

    def __init__(self, profits: list, weights: list):
        self.profits = profits
        self.weights = weights

    def sorting_profits_weights(self):
        if len(self.profits) == 0 or len(self.weights) == 0:
            if not len(self.profits) == 0 == len(self.weights):
                raise ValueError("The generation of subproblems may be broken - both profits and weights should be empty in case of no remaining items being affordable!")
            return {"profits": [], "weights": []}
        sorted_profits, sorted_weights = zip(*sorted(zip(self.profits, self.weights), reverse = True, key = lambda k: k[0]/k[1]))
        return {"profits": sorted_profits, "weights": sorted_weights}
    
    def sorting_permutation(self, old_profits: list, old_weights: list):
        old_profit_weight_tuples = [(old_profits[idx], old_weights[idx]) for idx in range(len(old_profits))]
        new_profit_weight_tuples = [(self.profits[idx], self.weights[idx]) for idx in range(len(self.profits))]
        sorting_permutation = []
        for new_tuple in new_profit_weight_tuples:
            first_occurrence = old_profit_weight_tuples.index(new_tuple)
            if first_occurrence in sorting_permutation:
                remaining_occurrences_of_tuple = [idx for idx in range(first_occurrence + 1, len(old_profits)) if old_profit_weight_tuples[idx] == new_tuple]
                for occurrence_idx in remaining_occurrences_of_tuple:
                    if occurrence_idx not in sorting_permutation:
                        sorting_permutation.append(occurrence_idx)
            else: 
                sorting_permutation.append(first_occurrence)
        return sorting_permutation
    


class EvaluatingProfitsAndWeights:
    """
    Class for calculating the profit and weight associated to a given (partial)
    represented as bitstring.

    Attributes:
    problem_instance (KnapsackProblem): the instance of the knapsack problem at hand
    profits_sorted (list): profits sorted according to ratio of profit/weight
    weights_sorted (list): weights sorted according to ratio of profit/weight

    Methods:
    calculate_profit: determining the profit of a given bitstring
    calculate_weight: determining the weight of a given bitstring
    """

    def __init__(self, problem_instance: KnapsackProblem):
        self.profits = SortingProfitsAndWeights(profits = problem_instance.profits, weights = problem_instance.weights).sorting_profits_weights()["profits"]
        self.weights = SortingProfitsAndWeights(profits = problem_instance.profits, weights = problem_instance.weights).sorting_profits_weights()["weights"]

    def calculate_profit(self, bitstring: str):
        as_int_list = list(map(int, list(bitstring)))
        profit = np.dot(np.array(as_int_list), np.array(self.profits[:len(as_int_list)]))
        return profit
    
    def calculate_weight(self, bitstring: str):
        as_int_list = list(map(int, list(bitstring)))
        weight = np.dot(np.array(as_int_list), np.array(self.weights[:len(as_int_list)]))
        return weight


class DynamicalSubproblems(EvaluatingProfitsAndWeights):

    def __init__(self, problem_instance: KnapsackProblem):
        EvaluatingProfitsAndWeights.__init__(self, problem_instance)
        self.capacity = problem_instance.capacity
        self.number_items = problem_instance.number_items

    def calculate_residual_capacity(self, partial_choice: str):
        return self.capacity - self.calculate_weight(partial_choice)
    
    def partial_choice_to_subproblem(self, partial_choice: str):
        residual_capacity = self.calculate_residual_capacity(partial_choice)
        if len(partial_choice) > self.number_items:
            raise ValueError("There cannot be more items specified than existing.")
        residual_profits = []
        residual_weights = []
        for idx in range(len(partial_choice), self.number_items):
            if self.weights[idx] <= residual_capacity: # No need to consider items not even fitting in the residual knapsack alone
                residual_profits.append(self.profits[idx])
                residual_weights.append(self.weights[idx])
        return KnapsackProblem(residual_profits, residual_weights, residual_capacity)

    def complete_partial_choice(self, partial_choice: str):
        number_unspecified_items = self.number_items - len(partial_choice)
        return partial_choice + "0" * number_unspecified_items


class GreedyBounds(EvaluatingProfitsAndWeights):
    
    def greedy_lower_bound(self, problem_instance: KnapsackProblem):
        sorted_problem_instance = KnapsackProblem(
            profits = SortingProfitsAndWeights(problem_instance.profits, problem_instance.weights).sorting_profits_weights()["profits"],
            weights = SortingProfitsAndWeights(problem_instance.profits, problem_instance.weights).sorting_profits_weights()["weights"],
            capacity = problem_instance.capacity
        )
        current_weight, current_profit = 0, 0
        for j in range(sorted_problem_instance.number_items):
            if current_weight + sorted_problem_instance.weights[j] <= sorted_problem_instance.capacity:
                current_weight += sorted_problem_instance.weights[j]
                current_profit += sorted_problem_instance.profits[j]
                if current_weight == sorted_problem_instance.capacity:
                    break
        return current_profit
        """current_weight = self.calculate_weight(partial_solution)
        if current_weight > self.capacity:
            raise ValueError(f"The current partial solution {partial_solution} is not feasible, since its weight exceeds the capacity.")
        current_profit = self.calculate_profit(partial_solution)
        last_fixed_item = len(partial_solution) - 1 if (len(partial_solution) != 0) else -1
        for i in range(last_fixed_item + 1, self.number_items):
            if current_weight + self.weights[i] > self.capacity:
                continue
            current_weight += self.weights[i]
            current_profit += self.profits[i]
            if current_weight == self.capacity:
                break
        return current_profit"""

    def greedy_upper_bound(self, problem_instance: KnapsackProblem):
        sorted_problem_instance = KnapsackProblem(
            profits = SortingProfitsAndWeights(problem_instance.profits, problem_instance.weights).sorting_profits_weights()["profits"],
            weights = SortingProfitsAndWeights(problem_instance.profits, problem_instance.weights).sorting_profits_weights()["weights"],
            capacity = problem_instance.capacity
        )
        current_weight, current_profit = 0, 0
        for j in range(sorted_problem_instance.number_items):
            if current_weight + sorted_problem_instance.weights[j] > sorted_problem_instance.capacity:
                residual_capacity = sorted_problem_instance.capacity - current_weight
                current_profit += sorted_problem_instance.profits[j] / sorted_problem_instance.weights[j] * residual_capacity
                break
            else:
                current_weight += sorted_problem_instance.weights[j]
                current_profit += sorted_problem_instance.profits[j]
        return np.floor(current_profit)
        """current_weight = self.calculate_weight(partial_solution)
        if current_weight > self.capacity:
            raise ValueError("The current partial solution is not feasible, since its weight exceeds the capacity.")
        current_profit = self.calculate_profit(partial_solution)
        last_fixed_item = len(partial_solution) - 1 if (len(partial_solution) != 0) else -1
        for i in range(last_fixed_item + 1, self.number_items):
            if current_weight + self.weights[i] > self.capacity:
                residual_fraction = (self.capacity - current_weight) / self.weights[i]
                current_profit += residual_fraction * self.profits[i]
                break
            current_weight += self.weights[i]
            current_profit += self.profits[i]
        return current_profit"""


class FeasibilityAndPruning(GreedyBounds, DynamicalSubproblems):

    def __init__(self, problem_instance: KnapsackProblem):
        DynamicalSubproblems.__init__(self, problem_instance)
        self.problem_instance = problem_instance
    
    def is_feasible(self, current_node: str):
        weight = self.calculate_weight(current_node)
        return False if weight > self.capacity else True
    
    def can_be_pruned(self, partial_solution: str, best_lower_bound: int, is_first_solution: bool = False):
        if len(partial_solution) == 0:
            raise ValueError("There is no node to investigate.")
        offset = self.calculate_profit(partial_solution)
        upper_bound = offset + self.greedy_upper_bound(self.partial_choice_to_subproblem(partial_solution))
        if is_first_solution:
            return True if (upper_bound < best_lower_bound) else False
        else:
            return True if (upper_bound <= best_lower_bound) else False


class BranchingSearchingBacktracking(FeasibilityAndPruning):    
    
    def branching(self, stack: list, node_to_branch_from: str = ""):
        stack.append(node_to_branch_from + "0")
        stack.append(node_to_branch_from + "1")
        return stack

    def node_selection(self, stack: list):
        """
        Comparing lower bounds does not make sense for leafs to investigate or when
        at least one of the two candidates is not feasible anymore.
        So, in these cases, we randomly pick one.
        However, this should be refined for the sake of efficiency and avoiding 
        duplication or redundancy, since the selected node is afterwards (at beginning
        of next iteration) again checked for feasibility and (rather at the end) its 
        lower bound is computed another time.
        """
        if len(stack) == 0:
            raise ValueError("No node to select.")
        solution_next_0 = stack[-2]
        solution_next_1 = stack[-1]
        if len(solution_next_0) == len(solution_next_1) == self.number_items:
            return random.choice([solution_next_0, solution_next_1]) 
        if self.is_feasible(solution_next_0) and self.is_feasible(solution_next_1):
            greedy_lb_0, greedy_lb_1 = self.greedy_lower_bound(self.partial_choice_to_subproblem(solution_next_0)), self.greedy_lower_bound(self.partial_choice_to_subproblem(solution_next_1))
            if greedy_lb_0 > greedy_lb_1:
                return solution_next_0
            elif greedy_lb_1 > greedy_lb_0:
                return solution_next_1
        return random.choice([solution_next_0, solution_next_1])

    def backtracking(self, stack: list):
        """
        Here a very simple function: just selects the last node in the stack due
        to the chosen depth-first searching (DFS) heuristic. Works as intended since
        method "branching" always generates two new nodes of which only one can be
        further explored (depth first). However, nevertheless represented by an own
        function instead of including it directly in the algorithm to maintain 
        readibility in case the searching strategy shall be refined or adapted.
        May be merged with "node_selection" method by introducing a backtracking flag.
        """
        if len(stack) == 0:
            raise ValueError("No further node to explore.")
        next_node = stack[-1]
        return next_node
    



def main():
    kp_instance_data = open(
        "C:\\Users\\d92474\\Documents\\Uni\\Master Thesis\\GitHub\\MasterThesis_Hybrid-Quantum-Classical-Branch-and-Bound\\bnb-qaoa_knapsack_christiansen\\code\\kp_instances_data\\uncorrelated\\2000.txt", 
        "r"
    ).readlines()
    profits = [int(line.split()[0]) for line in kp_instance_data[1:-1]]
    weights = [int(line.split()[1]) for line in kp_instance_data[1:-1]]
    sorted_profits_weights = SortingProfitsAndWeights(profits, weights).sorting_profits_weights()
    sorted_profits = sorted_profits_weights["profits"]
    sorted_weights = sorted_profits_weights["weights"]
    sorting_permutation = SortingProfitsAndWeights(sorted_profits, sorted_weights).sorting_permutation(profits, weights)
    #print("Sorting permutation = ", max(sorting_permutation))
    for idx in range(len(sorting_permutation)):
        if idx not in sorting_permutation:
            print(idx)
    


if __name__ == "__main__":
    main()
