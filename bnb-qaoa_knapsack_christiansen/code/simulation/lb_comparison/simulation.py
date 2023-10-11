import sys
sys.path.append("C:\\Users\\d92474\\Documents\\Uni\\Master Thesis\\GitHub\\MasterThesis_Hybrid-Quantum-Classical-Branch-and-Bound\\bnb-qaoa_knapsack_christiansen\\code")

from knapsack_problem import KnapsackProblem, GenerateKnapsackProblemInstances
from branch_and_bound import BranchAndBound

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Union



class AuxiliaryFunctions:

    
    def __init__(self, desired_qubit_numbers: Dict[tuple, int]):
        self.desired_qubit_numbers = desired_qubit_numbers

    
    def calculate_number_of_qubits(self, kp_instance: KnapsackProblem):
        return kp_instance.number_items + int(np.floor(np.log2(kp_instance.capacity)) + 1)
    

    def generate_random_kp_instance_for_desired_qubit_number(self, size: int, capacity_ratio: float, maximum_value: int):
        kp_instance = GenerateKnapsackProblemInstances.generate_random_kp_instance_for_capacity_ratio_and_maximum_value(size, capacity_ratio, maximum_value)
        qubit_number = self.calculate_number_of_qubits(kp_instance)
        if qubit_number == self.desired_qubit_numbers[(size, maximum_value)]:
            return kp_instance
        else:
            return self.generate_random_kp_instance_for_desired_qubit_number(size, capacity_ratio, maximum_value)


    def save_kp_instances_to_new_files(generated_random_kp_instances: Dict[tuple, List[KnapsackProblem]]):
        for ((size, capacity_ratio, maximum_value, qubit_number), equivalent_kp_instances) in generated_random_kp_instances.items():
            for kp_instance in equivalent_kp_instances:
                file = open(
                    f"C:\\Users\\d92474\\Documents\\Uni\\Master Thesis\\GitHub\\MasterThesis_Hybrid-Quantum-Classical-Branch-and-Bound\\bnb-qaoa_knapsack_christiansen\\code\\kp_instances_data\\simulation_data\\lb_comparison\\{size}.txt", 
                    "a"
                )
                file.writelines([
                    f"Number of qubits: {qubit_number} \n",
                    f"Profits: {kp_instance.profits} \n",
                    f"Weights: {kp_instance.weights} \n",
                    f"Capacity: {kp_instance.capacity} \n",
                    f"Capacity ratio [capacity/sum(weights)]: {capacity_ratio} \n",
                    f"Maximum value for profits & weights: {str(maximum_value)} \n"
                ])
                if equivalent_kp_instances.index(kp_instance) != len(equivalent_kp_instances) - 1:
                    file.write("-" * (2 * size + 20) + "\n")
            equivalent_kp_instances_for_number_of_items = {key: value for (key, value) in generated_random_kp_instances.items() if key[0] == size}
            index_in_kp_instances_of_same_item_number = list(equivalent_kp_instances_for_number_of_items.keys()).index((size, capacity_ratio, maximum_value, qubit_number))
            if index_in_kp_instances_of_same_item_number != len(equivalent_kp_instances_for_number_of_items) - 1:
                file.write("-" * (2 * size + 40) + "\n")
                file.write("-" * (2 * size + 40) + "\n")
            file.close()

    
    def save_figure_to_new_file(data_series: dict, number_of_bnb_repitions: int, number_of_equivalent_random_kp_instances: int):
        base_file_name = f"{data_series['kp instance size']}-Items_{data_series['kp instance identifier']}-Qubits_{number_of_bnb_repitions}-BnB-Repitions_{number_of_equivalent_random_kp_instances}-Equivalent-KP-Instances"
        plt.savefig(f"code/simulation/lb_comparison/results/{base_file_name}.png")
        plt.savefig(f"C:/Users/d92474/Documents/Uni/Master Thesis/Simulations/Lower Bound Comparison/{'LB-Comparison_' + base_file_name}.pdf")
    



class Visualization:

    def __init__(self, number_of_bnb_repitions: int, number_of_equivalent_random_kp_instances: int, desired_qubit_numbers: Dict[tuple, int]):
        self.markers = ["o", "x", "^", "P", "d"]
        self.colors = ["peru", "teal", "darkorchid", "maroon", "cornflowerblue"]
        self.number_of_bnb_repitions = number_of_bnb_repitions
        self.number_of_equivalent_random_kp_instances = number_of_equivalent_random_kp_instances
        self.desired_qubit_numbers = desired_qubit_numbers

    
    def generate_plot(self, sample_data: List[dict]):
        colors = ["peru", "teal", "darkorchid", "maroon", "cornflowerblue"]
        for data_series in sample_data:
            for data_per_depth in data_series["data"]:
                relative_residual_sizes_to_plot, lb_ratios_to_plot = map(list, zip(*sorted(zip(data_per_depth["relative residual sizes"], data_per_depth["lb ratios"]))))
                plt.plot(
                    relative_residual_sizes_to_plot, 
                    lb_ratios_to_plot, 
                    label = f"p = {data_per_depth['depth']}",
                    marker = self.markers[data_series["data"].index(data_per_depth)],
                    color = colors[data_series["data"].index(data_per_depth)]
                )
                plt.legend()
                plt.xlabel("Relative qubit size of subproblem")
                plt.ylabel("Average ratio QAOA/Greedy")
                #plt.title(f"Quantum vs classical lower bound for KP instance {data_series['kp instance name']} with {bnb_repitions} repitions")
            AuxiliaryFunctions.save_figure_to_new_file(data_series, self.number_of_bnb_repitions, self.number_of_equivalent_random_kp_instances)
            if sample_data.index(data_series) != len(sample_data) - 1:
                plt.figure()
        plt.show()




class LowerBoundComparison:

    def __init__(self, sample_kp_data: Dict[int, List[tuple]], desired_qubit_numbers: Dict[tuple, int], sample_depths: List[int], number_of_bnb_repitions: int, number_of_equivalent_random_kp_instances: int):
        self.sample_kp_data = sample_kp_data
        self.desired_qubit_numbers = desired_qubit_numbers
        self.sample_depths = sample_depths
        self.number_of_bnb_repitions = number_of_bnb_repitions
        self.number_of_equivalent_random_kp_instances = number_of_equivalent_random_kp_instances


    def transform_raw_data_to_functional(self, raw_data: List[Dict[str, Union[float, int]]]):
        functional_raw_data = []
        for raw_data_point in raw_data:
            residual_qubit_size = raw_data_point["residual qubit size"]
            if residual_qubit_size not in [data_point["residual qubit size"] for data_point in functional_raw_data]:
                ratios_for_same_size = [data_point["ratio"] for data_point in raw_data if data_point["residual qubit size"] == residual_qubit_size]
                average_ratio = sum(ratios_for_same_size) / len(ratios_for_same_size)
                functional_raw_data.append({"residual qubit size": residual_qubit_size, "ratio": average_ratio})
        return functional_raw_data
    

    def generate_data_for_kp_instance(self, kp_instance: KnapsackProblem):
        sample_data_for_kp_instance = []
        relative_residual_sizes = []
        for depth in self.sample_depths:
            print("New depth = ", depth)
            bnb = BranchAndBound(kp_instance, simulation = True, quantum_hard = True)
            for _ in range(self.number_of_bnb_repitions):
                bnb.branch_and_bound_algorithm(hard_qaoa_depth = depth)
            functional_raw_data = self.transform_raw_data_to_functional(bnb.lb_simulation_raw_data)
            if len(functional_raw_data) == 0:
                continue
            qubit_number = AuxiliaryFunctions(self.desired_qubit_numbers).calculate_number_of_qubits(kp_instance)
            relative_residual_sizes_for_depth = [data_point["residual qubit size"] / qubit_number for data_point in functional_raw_data]
            #print(f"relative residual sizes for depth {depth} =", relative_residual_sizes_for_depth)
            relative_residual_sizes = list(set(relative_residual_sizes + relative_residual_sizes_for_depth))
            lb_ratios = [data_point["ratio"] for data_point in functional_raw_data]
            #print(f"lb ratios for depth {depth} = ", lb_ratios)
            sample_data_for_kp_instance.append({"depth": depth, "relative residual sizes": relative_residual_sizes_for_depth, "lb ratios": lb_ratios})
        return sample_data_for_kp_instance
    

    def compute_average_data_for_kp_instance_configuration_data(self, sample_data_per_configuration_data: List[List[dict]]):
        averaged_data = []
        for idx in range(len(self.sample_depths)):
            sample_data_with_same_depths = [sample_data_for_kp_instance[idx] for sample_data_for_kp_instance in sample_data_per_configuration_data]
            if not all(depth == sample_data_with_same_depths[0]["depth"] for depth in [single_sample_data["depth"] for single_sample_data in sample_data_with_same_depths]):
                raise ValueError("Check averaging function as sample data dict objects in sample_data_with_sample_depths are supposed to have the same depth, but don't!")
            aggregated_data_per_depth = {}
            for sample_data_for_depth in sample_data_with_same_depths:
                dict_rel_res_size_to_lb_ratio = {rel_res_size: lb_ratio for (rel_res_size, lb_ratio) in zip(sample_data_for_depth["relative residual sizes"], sample_data_for_depth["lb ratios"])}
                for rel_res_size in dict_rel_res_size_to_lb_ratio.keys():
                    if rel_res_size in aggregated_data_per_depth.keys():
                        aggregated_data_per_depth[rel_res_size] += [dict_rel_res_size_to_lb_ratio[rel_res_size]]
                    else:
                        aggregated_data_per_depth[rel_res_size] = [dict_rel_res_size_to_lb_ratio[rel_res_size]]    
            #print("Aggregated data per depth = ", aggregated_data_per_depth)
            average_data_per_depth = {key: sum(value)/len(value) for (key, value) in aggregated_data_per_depth.items()}
            #print("Average data per depth = ", average_data_per_depth)
            averaged_data.append({
                "depth": sample_data_for_depth["depth"],
                "relative residual sizes": average_data_per_depth.keys(),
                "lb ratios": average_data_per_depth.values()
            })
        #print("Averaged data = ", averaged_data)
        return averaged_data
    

    def simulate_and_visualize(self):
        generated_random_kp_instances: Dict[tuple, KnapsackProblem] = {}
        for size in self.sample_kp_data.keys():
            for (capacity_ratio, maximum_value) in self.sample_kp_data[size]:
                generated_random_kp_instances_per_configuration_data = []
                for _ in range(self.number_of_equivalent_random_kp_instances):
                    kp_instance = AuxiliaryFunctions(self.desired_qubit_numbers).generate_random_kp_instance_for_desired_qubit_number(size, capacity_ratio, maximum_value)
                    generated_random_kp_instances_per_configuration_data.append(kp_instance)
                qubit_number = self.desired_qubit_numbers[(size, maximum_value)]
                generated_random_kp_instances[(size, capacity_ratio, maximum_value, qubit_number)] = generated_random_kp_instances_per_configuration_data
        AuxiliaryFunctions.save_kp_instances_to_new_files(generated_random_kp_instances)
        sample_data = []
        for ((size, capacity_ratio, maximum_value, qubit_number), equivalent_kp_instances) in generated_random_kp_instances.items():
            sample_data_per_configuration_data = []
            for kp_instance in equivalent_kp_instances:
                sample_data_per_configuration_data.append(self.generate_data_for_kp_instance(kp_instance))
            #print("Sample data per configuration data = ", sample_data_per_configuration_data)
            averaged_sample_data = self.compute_average_data_for_kp_instance_configuration_data(sample_data_per_configuration_data)
            if len(averaged_sample_data) != 0:
                sample_data.append({
                    "kp instance size": size,
                    "kp instance identifier": qubit_number,
                    "data": averaged_sample_data
                })
        print(sample_data)
        Visualization(self.number_of_bnb_repitions, self.number_of_equivalent_random_kp_instances, self.desired_qubit_numbers).generate_plot(sample_data)





def main():
    capacity_ratios = {5: 0.75, 20: 0.1, 40: 0.04, 60: 0.025}
    sample_kp_data = {
        5: [(capacity_ratios[5], 10), (capacity_ratios[5], 1e4)],
        20: [(capacity_ratios[20], 1e3), (capacity_ratios[20], 1e9)],
        40: [(capacity_ratios[40], 1e9), (capacity_ratios[40], 1e14)],
        60: [(capacity_ratios[60], 1e12), (capacity_ratios[60], 1e18)]
    }
    desired_qubit_numbers = {
        (5, 10): 10,
        (5, 1e4): 20,
        (20, 1e3): 30,
        (20, 1e9): 50,
        (40, 1e9): 70,
        (40, 1e14): 86,
        (60, 1e12): 100,
        (60, 1e18): 120
    }
    LowerBoundComparison(sample_kp_data, desired_qubit_numbers, sample_depths = [1, 3, 5, 7, 10], number_of_bnb_repitions = 4, number_of_equivalent_random_kp_instances = 10).simulate_and_visualize()


if __name__ == "__main__":
    main()