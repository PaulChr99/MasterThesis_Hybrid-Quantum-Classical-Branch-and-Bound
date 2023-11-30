import sys
sys.path.append("C:\\Users\\d92474\\Documents\\Uni\\Master Thesis\\GitHub\\MasterThesis_Hybrid-Quantum-Classical-Branch-and-Bound")

from knapsack_problem import KnapsackProblem, GenerateKnapsackProblemInstances
from branch_and_bound import BranchAndBound

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Union
from enum import Enum



class SimulationQuantity(Enum):
    number_of_explored_nodes = "number of explored nodes"
    number_of_leaves_reached = "number of leaves reached"



class AuxiliaryFunctions:

    def save_kp_instances_to_new_files(self, simulation_quantity: SimulationQuantity, generated_random_kp_instances: Dict[tuple, Dict[tuple, List[KnapsackProblem]]]):
        for (size, kp_instances_of_same_size) in generated_random_kp_instances.items():
            for ((capacity_ratio, maximum_value), equivalent_kp_instances) in kp_instances_of_same_size.items():
                for kp_instance in equivalent_kp_instances:
                    file = open(
                        f"C:\\Users\\d92474\\Documents\\Uni\\Master Thesis\\GitHub\\MasterThesis_Hybrid-Quantum-Classical-Branch-and-Bound\\\kp_instances_data\\bnb\\classical\\{simulation_quantity.name}\\{size}.txt", 
                        "a"
                    )
                    file.writelines([
                        f"Profits: {kp_instance.profits} \n",
                        f"Weights: {kp_instance.weights} \n",
                        f"Capacity: {kp_instance.capacity} \n",
                        f"Capacity ratio [capacity/sum(weights)]: {capacity_ratio} \n",
                        f"Maximum value for profits & weights: {str(maximum_value)} \n"
                    ])
                    if equivalent_kp_instances.index(kp_instance) != len(equivalent_kp_instances) - 1:
                        file.write("-" * 200 + "\n")
                if list(kp_instances_of_same_size.values()).index(equivalent_kp_instances) != len(kp_instances_of_same_size.values()) - 1:
                    file.write("-" * 300 + "\n")
                    file.write("-" * 300 + "\n")
            file.close()




class Visualization:

    def __init__(self, simulation_quantity: SimulationQuantity, number_of_bnb_repitions: int, number_of_equivalent_random_kp_instances: int):
        self.markers = ["o", "x", "^", "P", "d"]
        self.colors = ["peru", "teal", "darkorchid", "maroon", "cornflowerblue"]
        self.simulation_quantity = simulation_quantity
        self.number_of_bnb_repitions = number_of_bnb_repitions
        self.number_of_equivalent_random_kp_instances = number_of_equivalent_random_kp_instances

    
    def generate_plot(self, sample_data: Dict[int, Dict[tuple, Dict[str, Union[int, float]]]]):
        for (capacity_ratio, maximum_value) in list(sample_data.values())[0].keys():
            average_numbers_of_explored_nodes = [sample_data_for_size[(capacity_ratio, maximum_value)]["number of explored nodes"] for sample_data_for_size in sample_data.values()]
            average_standard_deviations = [sample_data_for_size[(capacity_ratio, maximum_value)]["standard deviation"] for sample_data_for_size in sample_data.values()]
            index_of_maximum_value = list(list(sample_data.values())[0].keys()).index((capacity_ratio, maximum_value))
            plt.plot(
                sample_data.keys(), 
                average_numbers_of_explored_nodes, 
                label = f"Capacity ratio = {capacity_ratio}", 
                marker = self.markers[index_of_maximum_value],
                color = self.colors[index_of_maximum_value]
            )
            plt.errorbar(
                sample_data.keys(),
                average_numbers_of_explored_nodes,
                yerr = average_standard_deviations,
                fmt = self.markers[index_of_maximum_value],
                color = self.colors[index_of_maximum_value],
                ecolor = self.colors[index_of_maximum_value],
                capsize = 2
            )
            plt.legend()
            plt.xlabel("Number of items")
            plt.ylabel(self.simulation_quantity.value.capitalize())
            #plt.title(f"Average number of nodes needed to find optimum for {bnb_repitions} B&B repitions")
        base_file_name = f"{self.number_of_bnb_repitions}-Clasical-BnB-Repitions_{self.number_of_equivalent_random_kp_instances}-Equivalent-KP-Instances"
        plt.savefig(f"code/simulation/bnb/classical/results/{self.simulation_quantity.name}/{base_file_name}.png")
        plt.savefig(f"C:/Users/d92474/Documents/Uni/Master Thesis/Simulations/B&B/Classical/{self.simulation_quantity.value.capitalize()}/{self.simulation_quantity.value.title().replace(' ', '-') + '_' + base_file_name}.pdf")
        plt.show()
        


class BnbPropertySimulation:

    def __init__(self, simulation_quantity: SimulationQuantity, sample_kp_data: Dict[int, tuple], number_of_bnb_repitions: int, number_of_equivalent_kp_instances: int):
        self.simulation_quantity = simulation_quantity
        self.sample_kp_data = sample_kp_data
        self.number_of_bnb_repitions = number_of_bnb_repitions
        self.number_of_equivalent_kp_instances = number_of_equivalent_kp_instances


    def generate_data_for_kp_instance(self, kp_instance: KnapsackProblem):
        bnb_numbers_of_explored_nodes = []
        for _ in range(self.number_of_bnb_repitions):
            print("New BnB execution")
            bnb_numbers_of_explored_nodes.append(BranchAndBound(kp_instance, simulation = True).branch_and_bound_algorithm()[self.simulation_quantity.value])
        return {"number of explored nodes": np.mean(bnb_numbers_of_explored_nodes), "standard deviation": np.std(bnb_numbers_of_explored_nodes)}
    

    """def compute_average_data_for_equivalent_kp_instances(self, sample_data_for_configuration_data: List[dict]):
        classical_bnb_averaged_numbers_of_explored_nodes = [sample_data_for_kp_instance["classical"] for sample_data_for_kp_instance in sample_data_for_configuration_data]
        classical_average = sum(classical_bnb_averaged_numbers_of_explored_nodes) / len(classical_bnb_averaged_numbers_of_explored_nodes)
        hybrid_bnb_averaged_numbers_of_explored_nodes = [sample_data_for_kp_instance["hybrid"] for sample_data_for_kp_instance in sample_data_for_configuration_data]
        hybrid_averages = {}
        for depth in self.sample_depths:
            hybrid_bnb_averaged_numbers_of_explored_nodes_for_same_depth = [hybrid_sample_data_for_kp_instance[depth] for hybrid_sample_data_for_kp_instance in hybrid_bnb_averaged_numbers_of_explored_nodes]
            hybrid_averages[depth] = sum(hybrid_bnb_averaged_numbers_of_explored_nodes_for_same_depth) / len(hybrid_bnb_averaged_numbers_of_explored_nodes_for_same_depth)
        return {
            "classical": classical_average,
            "hybrid": hybrid_averages
        }"""

    
    def generate_and_save_kp_instances(self):
        generated_random_kp_instances: Dict[int, Dict[tuple, List[KnapsackProblem]]] = {}
        for size in self.sample_kp_data.keys():
            generated_random_kp_instances_with_same_size: Dict[tuple, List[KnapsackProblem]] = {}
            for (capacity_ratio, maximum_value) in self.sample_kp_data[size]:
                generated_random_kp_instances_per_configuration_data = []
                for _ in range(self.number_of_equivalent_kp_instances):
                    generated_random_kp_instances_per_configuration_data.append(
                        GenerateKnapsackProblemInstances.generate_random_kp_instance_for_capacity_ratio_and_maximum_value(size, capacity_ratio, maximum_value)
                    )
                generated_random_kp_instances_with_same_size[(capacity_ratio, maximum_value)] = generated_random_kp_instances_per_configuration_data
            generated_random_kp_instances[size] = generated_random_kp_instances_with_same_size
        AuxiliaryFunctions().save_kp_instances_to_new_files(self.simulation_quantity, generated_random_kp_instances)
        return generated_random_kp_instances
    

    def simulate_and_visualize(self):
        generated_random_kp_instances = self.generate_and_save_kp_instances()
        sample_data: Dict[int, Dict[tuple, Union[float, int]]] = {}
        for (size, kp_instances_of_same_sizes) in generated_random_kp_instances.items():
            print("New size = ", size)
            sample_data_per_size = {}
            for ((capacity_ratio, maximum_value), equivalent_kp_instances) in kp_instances_of_same_sizes.items():
                print("New capacity ratio = ", capacity_ratio)
                sample_data_per_configuration_data = []
                for kp_instance in equivalent_kp_instances:
                    print("New attempt with equivalent instance")
                    sample_data_per_configuration_data.append(self.generate_data_for_kp_instance(kp_instance))
                sample_data_per_size[(capacity_ratio, maximum_value)] = {
                    "number of explored nodes": np.mean([data_point["number of explored nodes"] for data_point in sample_data_per_configuration_data]),
                    "standard deviation": np.mean([data_point["standard deviation"] for data_point in sample_data_per_configuration_data])
                }
            sample_data[size] = sample_data_per_size
        print(sample_data)
        Visualization(self.simulation_quantity, self.number_of_bnb_repitions, self.number_of_equivalent_kp_instances).generate_plot(sample_data)

    

def main(simulation_quantity: SimulationQuantity):
    """
    Attention: Running main() will overwrite the generated random KP instances in the folder "kp_instance_data"
        as well as the results in the folder "results" at the current position in the tree, so be careful.
        In order to avoid accidential changes in data used in the thesis, I created backup folders for both of them.
    """
    
    maximum_value = 1000
    sample_kp_data = {
        size: [(np.round(capacity_ratio, 1), maximum_value) for capacity_ratio in np.arange(0.1, 0.6, 0.1)] for size in np.arange(5, 55, 5) * 100
    }
    BnbPropertySimulation(simulation_quantity, sample_kp_data, number_of_bnb_repitions = 2, number_of_equivalent_kp_instances = 5).simulate_and_visualize()


if __name__ == "__main__":
    main(SimulationQuantity.number_of_explored_nodes)