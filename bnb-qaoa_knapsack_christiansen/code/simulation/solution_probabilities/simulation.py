import sys
sys.path.append("C:\\Users\\d92474\\Documents\\Uni\\Master Thesis\\GitHub\\MasterThesis_Hybrid-Quantum-Classical-Branch-and-Bound\\bnb-qaoa_knapsack_christiansen\\code")

from knapsack_problem import KnapsackProblem, GenerateKnapsackProblemInstances, exemplary_kp_instances
from quantum.classical_pretending.qaoa import QAOA, QTG
from branch_and_bound import BranchAndBound

import numpy as np 
import matplotlib.pyplot as plt
from typing import List, Dict


class AuxiliaryFunctions:

    def calculate_number_of_qubits(kp_instance: KnapsackProblem):
        return kp_instance.number_items + int(np.floor(np.log2(kp_instance.capacity)) + 1)
    

    def save_kp_instance_data_to_new_file(sample_kp_data: Dict[str, List[tuple]], generated_random_kp_instances: Dict[str, KnapsackProblem]):
        for ((size, qubit_number), kp_instance) in generated_random_kp_instances.items():
            save_kp_instance_data = open(
                f"C:\\Users\\d92474\\Documents\\Uni\\Master Thesis\\GitHub\\MasterThesis_Hybrid-Quantum-Classical-Branch-and-Bound\\bnb-qaoa_knapsack_christiansen\\code\\kp_instances_data\\simulation_data\\solution_probabilities\\{kp_instance.number_items}.txt", 
                "a"
            )
            kp_instances_for_number_of_items = {key: value for (key, value) in generated_random_kp_instances.items() if value.number_items == size}
            index_in_kp_instances_of_same_item_number = list(kp_instances_for_number_of_items.keys()).index((size, qubit_number))
            save_kp_instance_data.writelines([
                f"Number of qubits: {qubit_number} \n",
                f"Profits: {kp_instance.profits} \n",
                f"Weights: {kp_instance.weights} \n",
                f"Capacity: {kp_instance.capacity} \n",
                f"Capacity ratio [capacity/sum(weights)]: {sample_kp_data[size][index_in_kp_instances_of_same_item_number][0]} \n",
                f"Maximum value for profits & weights: {sample_kp_data[size][index_in_kp_instances_of_same_item_number][1]} \n"
            ])
            if index_in_kp_instances_of_same_item_number != len(kp_instances_for_number_of_items) - 1:
                save_kp_instance_data.write("-" * (2 * kp_instance.number_items + 20) + "\n")
            save_kp_instance_data.close()



class Visualization:

    def __init__(self, generated_random_kp_instances: Dict[tuple, KnapsackProblem], optimal_solutions: Dict[tuple, int], number_of_qaoa_executions: int, bar_width: float):
        self.generated_random_kp_instances = generated_random_kp_instances
        self.optimal_solutions = optimal_solutions
        self.number_of_qaoa_executions = number_of_qaoa_executions
        self.bar_width = bar_width
        

    def configure_bar_plot_for_depth(self, item_choices_to_display: list, data_series: dict, data_per_depth: dict, color: str):
        solution_probabilities: dict = data_per_depth["cleaned solution probabilities"]
        horizontal_bar_positions = [item_choices_to_display.index(key) + data_series["data"].index(data_per_depth) * self.bar_width for key in list(solution_probabilities.keys())]
        probabilities = [value_dict["probability"] for value_dict in solution_probabilities.values()]
        standard_deviations = [value_dict["standard deviation"] for value_dict in solution_probabilities.values()]
        plt.bar(horizontal_bar_positions, probabilities, yerr = [np.minimum(standard_deviations, probabilities), standard_deviations], label=f"$p = {data_per_depth['depth']}$", width = self.bar_width, color = color, capsize = 2)
        plt.legend()
        plt.xlabel("Item choice")
        plt.ylabel("Probability")
        #plt.title(f"Average choice probabilities for {kp_instance.number_items}-item KP instance with {self.number_of_qaoa_executions} executions")


    def style_bar_plot_for_kp_instance(self, item_choices_to_display: list, data_series: dict, optimal_solution: str):
        horizontal_ticks_positions = [item_choices_to_display.index(key) - 1/2 * self.bar_width + 1/2 * self.bar_width * len([depth_data for depth_data in data_series["data"] if key in list(depth_data["cleaned solution probabilities"].keys())]) for key in item_choices_to_display]
        fontsize_ticks = 9
        rotation_ticks = 0
        if len(horizontal_ticks_positions) > 7:
            difference = len(horizontal_ticks_positions) - 7
            if fontsize_ticks - difference < 7:
                fontsize_ticks = 7
                rotation_ticks += 10 * difference 
            else:
                fontsize_ticks -= difference
        plt.xticks(horizontal_ticks_positions, item_choices_to_display, fontsize = fontsize_ticks, rotation = min(rotation_ticks, 80))
        horizontal_ticks_labels = [label_object.get_text() for label_object in plt.gca().get_xticklabels()]
        tick_of_optimal_solution = plt.gca().get_xticklabels()[horizontal_ticks_labels.index(optimal_solution)]
        tick_of_optimal_solution.set_color("red")
        tick_of_optimal_solution.set_weight("bold")
        plt.tight_layout()


    def generate_bar_plots(self, sample_data: dict):
        colors = ["teal", "salmon", "lightblue", "purple"]
        for data_series in sample_data:
            kp_instance = self.generated_random_kp_instances[(data_series["kp instance size"], data_series["kp instance identifier"])]
            optimal_solution = self.optimal_solutions[(data_series["kp instance size"], data_series["kp instance identifier"])]
            cleaned_item_choices_sets = [data_point["cleaned solution probabilities"].keys() for data_point in data_series["data"]]
            lenghts_of_item_choices_sets = [len(item_choice_set) for item_choice_set in cleaned_item_choices_sets]
            item_choices_to_display = list(cleaned_item_choices_sets[lenghts_of_item_choices_sets.index(max(lenghts_of_item_choices_sets))])
            for data_per_depth in data_series["data"]:
                self.configure_bar_plot_for_depth(item_choices_to_display, data_series, data_per_depth, colors[data_series["data"].index(data_per_depth)])
            self.style_bar_plot_for_kp_instance(item_choices_to_display, data_series, optimal_solution)
            qubit_number = AuxiliaryFunctions.calculate_number_of_qubits(kp_instance)
            base_file_name = f"{kp_instance.number_items}-Items_{qubit_number}-Qubits_{self.number_of_qaoa_executions}-QAOA-Executions"
            plt.savefig(f"code/simulation/solution_probabilities/results/{base_file_name}.png")
            plt.savefig(f"C:/Users/d92474/Documents/Uni/Master Thesis/Simulations/Solution Probabilities/Solution-Probabilities_{base_file_name}.pdf")
            if sample_data.index(data_series) != len(sample_data) - 1:
                plt.figure()
        plt.show()



class SolutionProbabilitiesSimulation:

    def __init__(self, sample_kp_data: Dict[str, List[tuple]], sample_depths: List[int], number_of_qaoa_executions: int):
        self.sample_kp_data = sample_kp_data
        self.sample_depths = sample_depths
        self.number_of_qaoa_executions = number_of_qaoa_executions

    def compute_solution_probabilities_for_depth(self, kp_instance: KnapsackProblem, qtg_output: Dict[str, list], depth: int, relative_tolerance: float):
        qaoa = QAOA(kp_instance, qtg_output, depth)
        cleaned_solution_probabilities_for_depth = {}
        for _ in range(self.number_of_qaoa_executions):
            solution_probabilities_for_depth: dict = qaoa.optimize()["solution probabilities"]
            for (key, value) in solution_probabilities_for_depth.items():
                item_choice_key = key[:kp_instance.number_items]
                if not value < 1 / (2**AuxiliaryFunctions.calculate_number_of_qubits(kp_instance)) * relative_tolerance:
                    if item_choice_key in list(cleaned_solution_probabilities_for_depth.keys()):
                        cleaned_solution_probabilities_for_depth[item_choice_key] += [value]
                    else:
                        cleaned_solution_probabilities_for_depth[item_choice_key] = [value]
        aggregated_solution_probabilities_for_depth = {
            solution_key: {"probability": np.mean(probabilities_list), "standard deviation": np.std(probabilities_list)} for (solution_key, probabilities_list) in cleaned_solution_probabilities_for_depth.items()
        }
        max_occurring_probability = max([value_dict["probability"] for value_dict in aggregated_solution_probabilities_for_depth.values()])
        return {
            solution_key: value_dict for (solution_key, value_dict) in aggregated_solution_probabilities_for_depth.items() if value_dict["probability"] / max_occurring_probability > 0.005
        }
    
    
    def simulate_and_visualize(self, relative_tolerance: float, bar_width: float):
        generated_random_kp_instances: Dict[tuple, KnapsackProblem] = {}
        optimal_solutions: Dict[tuple, int] = {}
        for size in self.sample_kp_data.keys():
            for (capacity_ratio, maximum_value) in list(self.sample_kp_data[size]):
                kp_instance = GenerateKnapsackProblemInstances.generate_random_kp_instance_for_capacity_ratio_and_maximum_value(size, capacity_ratio, maximum_value)
                #kp_instance = exemplary_kp_instances["C"]
                qubit_number = AuxiliaryFunctions.calculate_number_of_qubits(kp_instance)
                if (kp_instance.number_items, qubit_number) in list(generated_random_kp_instances.keys()):
                    raise ValueError(f"Attention, by chance more than one KP instance identified via requiring {qubit_number} qubits at {kp_instance.number_items} items has been generated.")
                generated_random_kp_instances[(kp_instance.number_items, qubit_number)] = kp_instance
                print("KP instance = ", kp_instance)
                optimal_solutions[(kp_instance.number_items, qubit_number)] = BranchAndBound(kp_instance, simulation = True).branch_and_bound_algorithm()["optimal solution"]
        AuxiliaryFunctions.save_kp_instance_data_to_new_file(self.sample_kp_data, generated_random_kp_instances)
        sample_data = []
        for ((size, qubit_number), kp_instance) in generated_random_kp_instances.items():
            sample_data_for_kp_instance = []
            qtg_output = QTG(kp_instance).quantum_tree_generator()
            for depth in self.sample_depths:
                print("New depth = ", depth)
                cleaned_solution_probabilities_for_depth = self.compute_solution_probabilities_for_depth(kp_instance, qtg_output, depth, relative_tolerance)
                sample_data_for_kp_instance.append({"depth": depth, "cleaned solution probabilities": cleaned_solution_probabilities_for_depth})
            sample_data.append({
                "kp instance size": kp_instance.number_items,
                "kp instance identifier": qubit_number,
                "data": sample_data_for_kp_instance
            })
        print(sample_data)
        Visualization(generated_random_kp_instances, optimal_solutions, self.number_of_qaoa_executions, bar_width).generate_bar_plots(sample_data)

        # Try whether this works for cases in which the data sets for different depths do not share the same keys



def main():
    capacity_ratios = {3: 0.75, 5: 0.5, 7: 0.25}
    sample_max_values = [1e4, 1e6, 1e10, 1e18]
    SolutionProbabilitiesSimulation(
        sample_kp_data = {
            3: [(capacity_ratios[3], max_value) for max_value in sample_max_values],
            5: [(capacity_ratios[5], max_value) for max_value in sample_max_values],
            7: [(capacity_ratios[7], max_value) for max_value in sample_max_values]
        },
        sample_depths = [1, 3, 5, 10],
        number_of_qaoa_executions = 10
    ).simulate_and_visualize(relative_tolerance = 1e-05, bar_width = 0.2)


if __name__ == "__main__":
    main()