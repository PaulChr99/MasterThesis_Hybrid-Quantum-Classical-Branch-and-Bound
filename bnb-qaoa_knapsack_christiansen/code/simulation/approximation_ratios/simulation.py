from io import TextIOWrapper
import sys
sys.path.append("C:\\Users\\d92474\\Documents\\Uni\\Master Thesis\\GitHub\\MasterThesis_Hybrid-Quantum-Classical-Branch-and-Bound\\bnb-qaoa_knapsack_christiansen\\code")

from knapsack_problem import KnapsackProblem, GenerateKnapsackProblemInstances, exemplary_kp_instances
from quantum.classical_pretending.qaoa import QAOA, QTG
from branch_and_bound import BranchAndBound

import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from typing import List, Dict, Union



class AuxiliaryFunctions:

    def calculate_number_of_qubits(self, kp_instance: KnapsackProblem):
        return kp_instance.number_items + int(np.floor(np.log2(kp_instance.capacity)) + 1)
    
    def generate_data_series_label_for_varying_size(data_series: Dict[str, Union[int, list]]):
        return f"{data_series['kp instance size']} items ({data_series['kp instance identifier']} qubits)"
    
    def generate_data_series_label_for_varying_max_value(data_series: Dict[str, Union[int, list]]):
        left_bracket, right_bracket = "{", "}"
        exponent = str(int(np.log10(int(data_series['kp instance maximum value']))))
        printable_exponent = exponent if len(exponent) == 1 else left_bracket + exponent + right_bracket
        return f"Max value = 10^{printable_exponent} ({data_series['kp instance identifier']} qubits)"
    
    def save_figures_to_new_file_varying_size(sample_data: List[dict], number_of_qaoa_executions: int):
        sample_sizes = [data_series["kp instance size"] for data_series in sample_data]
        regime = "Tiny" if 5 in sample_sizes else "Small" if 20 in sample_sizes else "Medium" if 35 in sample_sizes else "Large" if 50 in sample_sizes else "Unknown"
        file_name = f"Approximation-Ratios_{regime}-Size-Regime_{number_of_qaoa_executions}-QAOA-Executions"
        plt.savefig(f"code/simulation/approximation_ratios/results/varying_size/{file_name}.png")
        plt.savefig(f"C:/Users/d92474/Documents/Uni/Master Thesis/Simulations/Approximation Ratios/Varying size/{file_name}.pdf")

    def save_figures_to_new_file_varying_max_value(sample_data: List[dict], number_of_qaoa_executions: int):
        sample_sizes = set([data_series["kp instance size"] for data_series in sample_data])
        sample_capacity_ratios = set([data_series["kp instance capacity ratio"] for data_series in sample_data])
        if len(sample_sizes) > 1 or len(sample_capacity_ratios) > 1:
            raise ValueError("There should not be different problem sizes or capacity ratios appearing in the same sample_data object; please check!")
        base_file_name = f"{list(sample_sizes)[0]}-Items_Capacity-Ratio-{list(sample_capacity_ratios)[0]}_{number_of_qaoa_executions}-QAOA-Executions"
        plt.savefig(f"code/simulation/approximation_ratios/results/varying_max_value/{base_file_name}.png")
        plt.savefig(f"C:/Users/d92474/Documents/Uni/Master Thesis/Simulations/Approximation Ratios/Varying maximum value/{'Approximation-Ratios_' + base_file_name}.pdf")
    

    def create_file_for_kp_instance_data_varying_size(kp_instance: KnapsackProblem):
        return open(
            f"C:\\Users\\d92474\\Documents\\Uni\\Master Thesis\\GitHub\\MasterThesis_Hybrid-Quantum-Classical-Branch-and-Bound\\bnb-qaoa_knapsack_christiansen\\code\\kp_instances_data\\simulation_data\\approximation_ratios\\varying_size\\{kp_instance.number_items}.txt", 
            "a"
        )
    
    def create_file_for_kp_instance_data_varying_max_value(kp_instance: KnapsackProblem):
        return open(
            f"C:\\Users\\d92474\\Documents\\Uni\\Master Thesis\\GitHub\\MasterThesis_Hybrid-Quantum-Classical-Branch-and-Bound\\bnb-qaoa_knapsack_christiansen\\code\\kp_instances_data\\simulation_data\\approximation_ratios\\varying_max_value\\{kp_instance.number_items}.txt", 
            "a"
        )
    
    def save_kp_instance_data_to_new_file(self, generated_random_kp_data: List[Dict[tuple, KnapsackProblem]], create_file):
        for kp_data_dict in generated_random_kp_data:
            for ((size, capacity_ratio, maximum_value, qubit_number), kp_instance) in kp_data_dict.items():
                file: TextIOWrapper = create_file(kp_instance)
                file.writelines([
                    f"Number of qubits: {qubit_number} \n",
                    f"Profits: {kp_instance.profits} \n",
                    f"Weights: {kp_instance.weights} \n",
                    f"Capacity: {kp_instance.capacity} \n",
                    f"Capacity ratio [capacity/sum(weights)]: {capacity_ratio} \n",
                    f"Maximum value for profits & weights: {maximum_value} \n"
                ])
                kp_instances_for_number_of_items = {key: value for (key, value) in kp_data_dict.items() if key[0] == size}
                index_in_kp_instances_of_same_item_number = list(kp_instances_for_number_of_items.keys()).index((size, capacity_ratio, maximum_value, qubit_number))
                if index_in_kp_instances_of_same_item_number != len(kp_instances_for_number_of_items) - 1:
                    file.write("-" * (2 * size + 20) + "\n")
                file.close()
    


class Visualization:

    def __init__(self, sample_depths: List[int], number_of_qaoa_executions: int, generate_labels, save_figure):
        self.sample_depths = sample_depths
        self.number_of_qaoa_executions = number_of_qaoa_executions
        self.generate_labels = generate_labels
        self.save_figure = save_figure
        

    def generate_scatter_plot(self, global_sample_data: List[List[dict]]):
        
        markers = ["o", "x", "^", "P", "d"]
        colors = ["peru", "teal", "darkorchid", "maroon", "cornflowerblue"]

        for sample_data in global_sample_data:
        
            if len(sample_data) > len(markers) or len(sample_data) > len(colors):
                raise ValueError("There need to be at least as many markers and colors as different data series to print.")
            
            #colors = cm.rainbow(np.linspace(0, 1, len(sample_data)))
            sample_depths_indices = list(range(len(self.sample_depths)))
            for data_series in sample_data:
                plt.plot(
                    sample_depths_indices, 
                    data_series["approximation ratios"], 
                    label = self.generate_labels(data_series),
                    marker = markers[sample_data.index(data_series)],
                    color = colors[sample_data.index(data_series)]
                )
                plt.legend()
                plt.xlabel("Depth")
                plt.ylabel("Approximation ratio")
                plt.xticks(sample_depths_indices, self.sample_depths)
                #plt.title(f"Grover-mixer QAOA average approximation ratios for {self.number_of_qaoa_executions} executions")
            self.save_figure(sample_data, self.number_of_qaoa_executions)
            if global_sample_data.index(sample_data) != len(global_sample_data) - 1:
                plt.figure()
        plt.show()



class ApproximationRatios:

    def __init__(self, sample_depths:List[int], number_of_qaoa_executions: int):
        self.sample_depths = sample_depths
        self.number_of_qaoa_executions = number_of_qaoa_executions


    def compute_approximation_ratios(self, kp_instance: KnapsackProblem, capacity_ratio: float, maximum_value: int, optimal_solution_value: Union[int, float]):
        approximation_ratios_for_kp_instance = []
        #optimal_solution_value = BranchAndBound(kp_instance, simulation = True).branch_and_bound_algorithm()["maximum profit"]
        qtg_output = QTG(kp_instance).quantum_tree_generator()
        for depth in self.sample_depths:
            print("New depth = ", depth)
            qaoa = QAOA(kp_instance, qtg_output, depth)
            qaoa_results_for_depth = []
            for _ in range(self.number_of_qaoa_executions):
                qaoa_results_for_depth.append(qaoa.optimize()["qaoa result"])
            average_qaoa_result = sum(qaoa_results_for_depth) / len(qaoa_results_for_depth)
            approximation_ratios_for_kp_instance.append(average_qaoa_result / optimal_solution_value) 
        return {
            "kp instance size": kp_instance.number_items,
            "kp instance capacity ratio": capacity_ratio,
            "kp instance maximum value": maximum_value,
            "kp instance identifier": AuxiliaryFunctions().calculate_number_of_qubits(kp_instance),
            "approximation ratios": approximation_ratios_for_kp_instance
        }
    
    
    def simulate_and_visualize_varying_size(self, sample_kp_data: Dict[tuple, List[int]]):
        generated_random_kp_data = []
        optimal_solutions = {}
        for (capacity_ratio, maximum_value) in sample_kp_data.keys():
            generated_random_kp_instances_per_configuration_data = {}
            for size in sample_kp_data[(capacity_ratio, maximum_value)]:
                kp_instance = GenerateKnapsackProblemInstances.generate_random_kp_instance_for_capacity_ratio(size, capacity_ratio, maximum_value)
                #print("KP instance = ", kp_instance)
                qubit_number = AuxiliaryFunctions().calculate_number_of_qubits(kp_instance)
                generated_random_kp_instances_per_configuration_data[(kp_instance.number_items, capacity_ratio, maximum_value, qubit_number)] = kp_instance
                optimal_solutions[(size, capacity_ratio, maximum_value, qubit_number)] = BranchAndBound(kp_instance, simulation = True).branch_and_bound_algorithm()["maximum profit"]
            generated_random_kp_data.append(generated_random_kp_instances_per_configuration_data)
        AuxiliaryFunctions().save_kp_instance_data_to_new_file(generated_random_kp_data, create_file = AuxiliaryFunctions.create_file_for_kp_instance_data_varying_size)
        global_sample_data = []
        for kp_data_dict in generated_random_kp_data:
            sample_data = []
            for (size, capacity_ratio, maximum_value, qubit_number) in kp_data_dict.keys():
                print("New size = ", size)
                kp_instance = kp_data_dict[(size, capacity_ratio, maximum_value, qubit_number)]
                sample_data.append(
                    self.compute_approximation_ratios(kp_instance, capacity_ratio, maximum_value, optimal_solutions[(size, capacity_ratio, maximum_value, qubit_number)])
                )
            global_sample_data.append(sample_data)
        print(global_sample_data)
        Visualization(
            self.sample_depths, 
            self.number_of_qaoa_executions,
            AuxiliaryFunctions.generate_data_series_label_for_varying_size,
            AuxiliaryFunctions.save_figures_to_new_file_varying_size
        ).generate_scatter_plot(global_sample_data)


    def simulate_and_visualize_varying_max_value(self, sample_kp_data: Dict[int, List[tuple]]):
        generated_random_kp_data = []
        optimal_solutions = {}
        for size in sample_kp_data.keys():
            generated_random_kp_instances_per_size = {}
            for (capacity_ratio, maximum_value) in sample_kp_data[size]:
                kp_instance = GenerateKnapsackProblemInstances.generate_random_kp_instance_for_capacity_ratio(size, capacity_ratio, maximum_value)
                #print("KP instance = ", kp_instance)
                qubit_number = AuxiliaryFunctions().calculate_number_of_qubits(kp_instance)
                generated_random_kp_instances_per_size[(size, capacity_ratio, maximum_value, qubit_number)] = kp_instance
                optimal_solutions[(size, capacity_ratio, maximum_value, qubit_number)] = BranchAndBound(kp_instance, simulation = True).branch_and_bound_algorithm()["maximum profit"]
            generated_random_kp_data.append(generated_random_kp_instances_per_size)
        print("Generated random kp data = ", generated_random_kp_data)
        AuxiliaryFunctions().save_kp_instance_data_to_new_file(generated_random_kp_data, create_file = AuxiliaryFunctions.create_file_for_kp_instance_data_varying_max_value)
        global_sample_data = []
        for kp_data_dict in generated_random_kp_data:
            sample_data = []
            for (size, capacity_ratio, maximum_value, qubit_number) in kp_data_dict.keys():
                print(f"New (capacity ratio, maximum value) = ({capacity_ratio}, {maximum_value})")
                kp_instance = kp_data_dict[(size, capacity_ratio, maximum_value, qubit_number)]
                sample_data.append(
                    self.compute_approximation_ratios(kp_instance, capacity_ratio, maximum_value, optimal_solutions[(size, capacity_ratio, maximum_value, qubit_number)])
                )
            global_sample_data.append(sample_data)
        print(global_sample_data)
        Visualization(
            self.sample_depths, 
            self.number_of_qaoa_executions, 
            AuxiliaryFunctions.generate_data_series_label_for_varying_max_value, 
            AuxiliaryFunctions.save_figures_to_new_file_varying_max_value
        ).generate_scatter_plot(global_sample_data)




def main():
    sample_kp_data_varying_size = {
        (0.25, 10): [5, 7, 10, 15], 
        (0.07, 1e6): [20, 23, 26, 30],
        (0.038, 1e12): [35, 38, 40, 45],
        (0.026, 1e18): [50, 54, 57, 60]
    }
    """
    sample_kp_data_varying_capacity_ratios = {
        10: [(0.25, 10), (0.5, 10), (0.75, 10), (0.9, 10)],
        20: [(0.2, 1000), (0.4, 1000), (0.6, 1000), (0.8, 1000)],
        40: [(0.1, 100000), (0.15, 100000), (0.2, 100000), (0.25, 100000)]
    }
    """
    sample_max_values = [1e3, 1e4, 1e6, 1e10, 1e18]
    sample_kp_data_variying_max_value = {
        5: [(0.25, max_value) for max_value in sample_max_values],
        20: [(0.07, max_value) for max_value in sample_max_values],
        40: [(0.033, max_value) for max_value in sample_max_values],
        60: [(0.023, max_value) for max_value in sample_max_values]
    }
    approximation_ratios = ApproximationRatios(sample_depths = list(np.arange(1, 11)), number_of_qaoa_executions = 10)
    #approximation_ratios.simulate_and_visualize_varying_size(sample_kp_data_varying_size)
    approximation_ratios.simulate_and_visualize_varying_max_value(sample_kp_data_variying_max_value)



if __name__ == "__main__":
    main()
    