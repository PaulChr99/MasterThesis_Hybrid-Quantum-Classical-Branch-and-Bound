#import sys
#sys.path.append("C:\\Users\\d92474\\Documents\\Uni\\Master Thesis\\GitHub\\MasterThesis_Hybrid-Quantum-Classical-Branch-and-Bound\\bnb-qaoa_knapsack_christiansen")

from knapsack_problem import KnapsackProblem, exemplary_kp_instances
from branch_and_bound import BranchAndBound

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Union


sample_depths = [1,2]





def simulate_lb_ratio_vs_residual_problem_size_and_depth(repitions: int):
    

    def transform_raw_data_to_functional(raw_data: List[Dict[str, Union[float, int]]]):
        functional_raw_data = []
        for raw_data_point in raw_data:
            residual_qubit_size = raw_data_point["residual qubit size"]
            if residual_qubit_size not in [data_point["residual qubit size"] for data_point in functional_raw_data]:
                ratios_for_same_size = [data_point["ratio"] for data_point in raw_data if data_point["residual qubit size"] == residual_qubit_size]
                average_ratio = sum(ratios_for_same_size) / len(ratios_for_same_size)
                functional_raw_data.append({"residual qubit size": residual_qubit_size, "ratio": average_ratio})
        return functional_raw_data
    

    def generate_data_for_kp_instance(kp_instance: KnapsackProblem):
        sample_data_for_kp_instance = []
        relative_residual_sizes = []
        for depth in sample_depths:
            bnb = BranchAndBound(kp_instance, simulation = True, quantum_hard = True)
            for _ in range(repitions):
                bnb.branch_and_bound_algorithm(hard_qaoa_depth = depth)
            functional_raw_data = transform_raw_data_to_functional(bnb.lb_simulation_raw_data)
            #print(functional_raw_data)
            if len(functional_raw_data) == 0:
                continue
            relative_residual_sizes_for_depth = [data_point["residual qubit size"] / qubit_number for data_point in functional_raw_data]
            #print(f"relative residual sizes for depth {depth} =", relative_residual_sizes_for_depth)
            relative_residual_sizes = list(set(relative_residual_sizes + relative_residual_sizes_for_depth))
            lb_ratios = [data_point["ratio"] for data_point in functional_raw_data]
            #print(f"lb ratios for depth {depth} = ", lb_ratios)
            sample_data_for_kp_instance.append({"depth": depth, "relative residual sizes": relative_residual_sizes_for_depth, "lb ratios": lb_ratios})
        return sample_data_for_kp_instance
    

    sample_data = []
    kp_instance: KnapsackProblem
    for kp_instance in exemplary_kp_instances.values():
        qubit_number = kp_instance.number_items + int(np.floor(np.log2(kp_instance.capacity)) + 1)
        sample_data_for_kp_instance = generate_data_for_kp_instance(kp_instance)
        #plt.scatter(sample_data_for_kp_instance["relative residual sizes"], sample_data_for_kp_instance["lb ratios"])
        #plt.figure()
        if len(sample_data_for_kp_instance) != 0:
            sample_data.append({
                "kp instance name": list(exemplary_kp_instances.keys())[list(exemplary_kp_instances.values()).index(kp_instance)], 
                "kp instance identifier": qubit_number, 
                "data": sample_data_for_kp_instance
            })
    print(sample_data)
    for data_series in sample_data:
        for data_per_depth in data_series["data"]:
            plt.scatter(data_per_depth["relative residual sizes"], data_per_depth["lb ratios"], label=f"p = {data_per_depth['depth']}")
            plt.legend(loc = "lower left")
            plt.xlabel("Relative qubit size of subproblem")
            plt.ylabel("Average ratio QAOA/Greedy")
            plt.title(f"Quantum vs classical lower bound for KP instance {data_series['kp instance name']}")
        if sample_data.index(data_series) != len(sample_data) - 1:
            plt.figure()
    plt.show()




def main():
    simulate_lb_ratio_vs_residual_problem_size_and_depth(repitions = 10)


if __name__ == "__main__":
    main()