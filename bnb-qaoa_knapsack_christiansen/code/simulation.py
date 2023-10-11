#import sys
#sys.path.append("C:\\Users\\d92474\\Documents\\Uni\\Master Thesis\\GitHub\\MasterThesis_Hybrid-Quantum-Classical-Branch-and-Bound\\bnb-qaoa_knapsack_christiansen")

from knapsack_problem import KnapsackProblem, GenerateKnapsackProblemInstances, exemplary_kp_instances
from branch_and_bound import BranchAndBound
from quantum.classical_pretending.qaoa import QAOA

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Union





def simulate_number_of_explored_nodes_in_bnb_for_kp_instance_size(bnb_repitions: int):#, number_of_capacities_per_instance: int):
    
    numbers_of_items = [100, 200, 500, 1000, 2000, 5000, 10000]
    
    markers = ["o", "x", "^", "P"]

    #sample_data = []
    #for capacity_idx in range(1, number_of_capacities_per_instance + 1):
    #    print("New capacity index = ", capacity_idx)
    explored_nodes_for_capacity = []
    for number_of_items in numbers_of_items:
        print("New number of items = ", number_of_items)
        kp_instance_data = open(
            f"C:\\Users\\d92474\\Documents\\Uni\\Master Thesis\\GitHub\\MasterThesis_Hybrid-Quantum-Classical-Branch-and-Bound\\bnb-qaoa_knapsack_christiansen\\code\\kp_instances_data\\uncorrelated\\{number_of_items}.txt", 
            "r"
        ).readlines()
        profits = [int(line.split()[0]) for line in kp_instance_data]
        weights = [int(line.split()[1]) for line in kp_instance_data]
        kp_instance = KnapsackProblem(profits, weights, capacity = 1000)#capacity_idx / (number_of_capacities_per_instance + 1) * sum(weights))
        bnb = BranchAndBound(kp_instance, simulation = True)
        explored_nodes_for_kp_instance_size = []
        for _ in range(bnb_repitions):
            explored_nodes_for_kp_instance_size.append(bnb.branch_and_bound_algorithm()["number of explored nodes"])
        average_explored_nodes = sum(explored_nodes_for_kp_instance_size) / len(explored_nodes_for_kp_instance_size)
        explored_nodes_for_capacity.append(average_explored_nodes)
        #sample_data.append({
        #    "relative capacity":  f"{capacity_idx}/{number_of_capacities_per_instance + 1}", "explored nodes": explored_nodes_for_capacity
        #})
    #print(sample_data)
    #for data_series in sample_data:
    plt.scatter(
        numbers_of_items, 
        explored_nodes_for_capacity#, data_series["explored nodes"],
        #label = f"Relative Capacity = {data_series['relative capacity']}"#,
        #marker = markers[sample_data.index(data_series)]
    )
    plt.legend()
    plt.xlabel("Number of items")
    plt.ylabel("Number of explored nodes")
    plt.title(f"Average number of nodes needed to find optimum for {bnb_repitions} B&B repitions")
    plt.show()
        





def main():
    pass
    #simulate_approximation_ratio_vs_depth(qaoa_executions = 3)
    #simulate_lb_ratio_vs_residual_problem_size_and_depth(bnb_repitions = 5)
    #simulate_number_of_explored_nodes_in_bnb_for_kp_instance_size(bnb_repitions = 3)#, number_of_capacities_per_instance = 4)


if __name__ == "__main__":
    main()