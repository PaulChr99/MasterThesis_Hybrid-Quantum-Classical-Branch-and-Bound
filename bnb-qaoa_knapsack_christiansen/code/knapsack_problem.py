from dataclasses import dataclass, fields


@dataclass
class KnapsackProblem:
    """
    Class for construction and representation of instances of the knapsack problem.

    Only intended for 0-1 knapsack problems, i.e. only one knapsack, one constraint,
    integer-valued weights/costs and values/profits as well as the decision "take it 
    vs. don't take it".

    Furthermore only such instances are permitted for which all profits and all costs
    are non-negative integers, the capacity of the knapsack is a positive integer, 
    the sum of all weights exceeds the capacity and no single weight is larger than the 
    capacity (otherwise the optimal solution would trivially be given be including
    all items or a certain item could never be selected, respectively). 

    Attributes:
    profits (list): the profits (values) of the items 
    weights (list): the costs (weights) of the items
    capacity (int): the capacity of the knapsack
    total_weight (int): the sum of all weights
    number_items (int): the number of items
    """

    profits: list
    weights: list
    capacity: int

    def __post_init__(self):
        # Check if profits, weights and capacity are of correct type
        """
        for f in fields(type(self)):
            fieldname_attribute = getattr(self, f.name)
            if not isinstance(fieldname_attribute, f.type):
                actual_type = type(fieldname_attribute)
                raise TypeError(f"The field '{f.name}' was assigned by '{actual_type}' instead of '{f.type}'.")
        """

        if any([p for p in self.profits if not type(p) == int]):
            raise TypeError("Profits must be integer-valued.")

        if any([w for w in self.weights if not type(w) == int]):
            raise TypeError("Weights must be integer-valued.")
        
        if len(self.profits) != len(self.weights):
            raise ValueError("Profits and weights have different length.")

        if any([p for p in self.profits if p < 0]):
            raise ValueError("Profits must not be negative.")

        if any([w for w in self.weights if w < 0]):
            raise ValueError("Weights must not be negative.")

        if self.capacity < 0:
            raise ValueError("Capacity of knapsack must not be negative.")
        
        self.total_weight = sum(self.weights)
        self.number_items = len(self.weights)


exemplary_kp_instances = {
    "A": KnapsackProblem(profits = [1, 2], weights = [1, 1], capacity = 2),
    "B": KnapsackProblem(profits = [1, 4, 2], weights = [1, 3, 2], capacity = 3),
    "C": KnapsackProblem(profits = [3, 1, 2, 1], weights = [1, 1, 2, 2], capacity = 4),
    "D": KnapsackProblem(profits = [6, 5, 8, 9, 6, 7, 3], weights = [2, 3, 6, 7, 5, 9, 4], capacity = 9)
}
