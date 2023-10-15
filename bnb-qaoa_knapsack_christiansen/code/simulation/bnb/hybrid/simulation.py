import sys
sys.path.append("C:\\Users\\d92474\\Documents\\Uni\\Master Thesis\\GitHub\\MasterThesis_Hybrid-Quantum-Classical-Branch-and-Bound\\bnb-qaoa_knapsack_christiansen\\code")

from knapsack_problem import KnapsackProblem, GenerateKnapsackProblemInstances
from branch_and_bound import BranchAndBound

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Union
from enum import Enum



class SimulationQuantity(Enum):
    number_of_explored_nodes = "number of explored nodes"
    number_of_leafs_reached = "number of leafs reached"



class AuxiliaryFunctions:

    def calculate_number_of_qubits(self, kp_instance: KnapsackProblem):
        return kp_instance.number_items + int(np.floor(np.log2(kp_instance.capacity)) + 1)
    
    
    def save_kp_instances_to_new_files(self, simulation_quantity: SimulationQuantity, generated_random_kp_instances: Dict[tuple, List[KnapsackProblem]]):
        for ((size, capacity_ratio, maximum_value), equivalent_kp_instances) in generated_random_kp_instances.items():
            for kp_instance in equivalent_kp_instances:
                file = open(
                    f"C:\\Users\\d92474\\Documents\\Uni\\Master Thesis\\GitHub\\MasterThesis_Hybrid-Quantum-Classical-Branch-and-Bound\\bnb-qaoa_knapsack_christiansen\\code\\kp_instances_data\\simulation_data\\bnb\\hybrid\\{simulation_quantity.name}\\{size}.txt", 
                    "a"
                )
                file.writelines([
                    f"Number of qubits: {self.calculate_number_of_qubits(kp_instance)} \n",
                    f"Profits: {kp_instance.profits} \n",
                    f"Weights: {kp_instance.weights} \n",
                    f"Capacity: {kp_instance.capacity} \n",
                    f"Capacity ratio [capacity/sum(weights)]: {capacity_ratio} \n",
                    f"Maximum value for profits & weights: {str(maximum_value)} \n"
                ])
                if equivalent_kp_instances.index(kp_instance) != len(equivalent_kp_instances) - 1:
                    file.write("-" * (2 * size + 20) + "\n")
            file.close()




class Visualization:

    def __init__(self, simulation_quantity: SimulationQuantity, sample_data: Dict[int, Union[list, dict]], sample_depths: List[int], number_of_bnb_repitions: int, number_of_equivalent_random_kp_instances: int):
        self.markers = ["o", "x", "^", "P", "d"]
        self.colors = ["peru", "teal", "darkorchid", "maroon", "cornflowerblue"]
        self.simulation_quantity = simulation_quantity
        self.sample_data = sample_data
        self.sample_depths = sample_depths
        self.number_of_bnb_repitions = number_of_bnb_repitions
        self.number_of_equivalent_random_kp_instances = number_of_equivalent_random_kp_instances

    
    def configure_plot(self, number_of_explored_nodes: List[int], label: str, marker: str, color: str):
        plt.plot(self.sample_data.keys(), number_of_explored_nodes, label = label, marker = marker, color = color)
        plt.legend()
        plt.xlabel("Number of items")
        plt.ylabel(self.simulation_quantity.value.capitalize())
        #plt.title(f"Average number of nodes needed to find optimum for {bnb_repitions} B&B repitions")
    

    def generate_plot(self):
        self.configure_plot(
            number_of_explored_nodes = [averaged_sample_data_for_size["classical"] for averaged_sample_data_for_size in self.sample_data.values()],
            label = "Classical B&B",
            marker = self.markers[0],
            color = self.colors[0]
        )
        hybrid_average_numbers_of_explored_nodes = [averaged_sample_data_for_size["hybrid"] for averaged_sample_data_for_size in self.sample_data.values()]
        for depth in self.sample_depths:
            hybrid_average_number_of_explored_nodes_with_same_depth = [hybrid_sample_data_for_size[depth] for hybrid_sample_data_for_size in hybrid_average_numbers_of_explored_nodes]
            self.configure_plot(
                number_of_explored_nodes = hybrid_average_number_of_explored_nodes_with_same_depth,
                label = f"Hybrid B&B with p = {depth}",
                marker = self.markers[self.sample_depths.index(depth) + 1],
                color = self.colors[self.sample_depths.index(depth) + 1]
            )
        base_file_name = f"{self.number_of_bnb_repitions}-BnB-Repitions_{self.number_of_equivalent_random_kp_instances}-Equivalent-KP-Instances"
        plt.savefig(f"code/simulation/bnb/hybrid/results/{self.simulation_quantity.name}/{base_file_name}.png")
        plt.savefig(f"C:/Users/d92474/Documents/Uni/Master Thesis/Simulations/B&B/Hybrid/{self.simulation_quantity.value.capitalize()}/{self.simulation_quantity.value.title().replace(' ', '-') + '_' + base_file_name}.pdf")
        plt.show()
        



class NumberOfExploredNodes:

    def __init__(self, simulation_quantity: SimulationQuantity, sample_kp_data: Dict[int, tuple], sample_depths: List[int], number_of_bnb_repitions: int, number_of_equivalent_kp_instances: int):
        self.simulation_quantity = simulation_quantity
        self.sample_kp_data = sample_kp_data
        self.sample_depths = sample_depths
        self.number_of_bnb_repitions = number_of_bnb_repitions
        self.number_of_equivalent_kp_instances = number_of_equivalent_kp_instances


    def generate_data_for_kp_instance(self, kp_instance: KnapsackProblem):
        classical_bnb_numbers_of_explored_nodes = []
        hybrid_bnb_numbers_of_explored_nodes = {}
        for _ in range(self.number_of_bnb_repitions):
            print("New BnB execution")
            classical_bnb_numbers_of_explored_nodes.append(BranchAndBound(kp_instance, simulation = True).branch_and_bound_algorithm()[self.simulation_quantity.value])
            hybrid_bnb = BranchAndBound(kp_instance, simulation = True, quantum_hard = True)
            for depth in self.sample_depths:
                if depth in hybrid_bnb_numbers_of_explored_nodes.keys():
                    hybrid_bnb_numbers_of_explored_nodes[depth] += [hybrid_bnb.branch_and_bound_algorithm(hard_qaoa_depth = depth)[self.simulation_quantity.value]]
                else:
                    hybrid_bnb_numbers_of_explored_nodes[depth] = [hybrid_bnb.branch_and_bound_algorithm(hard_qaoa_depth = depth)[self.simulation_quantity.value]]
        return {
            "classical": sum(classical_bnb_numbers_of_explored_nodes) / len(classical_bnb_numbers_of_explored_nodes),
            "hybrid": {depth: sum(numbers_of_explored_nodes) / len(numbers_of_explored_nodes) for (depth, numbers_of_explored_nodes) in hybrid_bnb_numbers_of_explored_nodes.items()}
        }
    

    def compute_average_data_for_equivalent_kp_instances(self, sample_data_for_configuration_data: List[dict]):
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
        }

    
    def generate_and_save_kp_instances(self):
        generated_random_kp_instances: Dict[int, List[KnapsackProblem]] = {}
        for (size, (capacity_ratio, maximum_value)) in self.sample_kp_data.items():
            generated_random_kp_instances_per_configuration_data = []
            for _ in range(self.number_of_equivalent_kp_instances):
                generated_random_kp_instances_per_configuration_data.append(
                    GenerateKnapsackProblemInstances.generate_random_kp_instance_for_capacity_ratio_and_maximum_value(size, capacity_ratio, maximum_value)
                )   
            generated_random_kp_instances[(size, capacity_ratio, maximum_value)] = generated_random_kp_instances_per_configuration_data
        AuxiliaryFunctions().save_kp_instances_to_new_files(self.simulation_quantity, generated_random_kp_instances)
        return generated_random_kp_instances
    

    def simulate_and_visualize(self):
        generated_random_kp_instances = self.generate_and_save_kp_instances()
        sample_data = {}
        for equivalent_kp_instances in generated_random_kp_instances.values():
            print("New size = ", equivalent_kp_instances[0].number_items)
            sample_data_per_configuration_data = []
            for kp_instance in equivalent_kp_instances:
                print("New attempt for same size")
                sample_data_per_configuration_data.append(self.generate_data_for_kp_instance(kp_instance))
            averaged_sample_data = self.compute_average_data_for_equivalent_kp_instances(sample_data_per_configuration_data)
            if not all(kp_instance.number_items == equivalent_kp_instances[0].number_items for kp_instance in equivalent_kp_instances):
                raise ValueError("Check generation of equivalent KP instances, as instances that should be of the same size are not")
            sample_data[equivalent_kp_instances[0].number_items] = averaged_sample_data
        print(sample_data)
        Visualization(self.simulation_quantity, sample_data, self.sample_depths, self.number_of_bnb_repitions, self.number_of_equivalent_kp_instances).generate_plot()




def main(simulation_quantity: SimulationQuantity):
    sample_kp_data = {
        5: (0.25, 1e3), # Decrease of 0.05 in capacity ratio per size from here
        10: (0.19, 1e4),
        15: (0.135, 1e5),
        20: (0.08, 1e6), # Decrease of 0.01 in capacity ratio per size from here
        25: (0.07, 1e7),
        30: (0.06, 1e9),
        35: (0.05, 1e10),
        40: (0.04, 1e11), # Drecrease of 0.004 in capacity ratio per size from here
        45: (0.03625, 1e13), 
        50: (0.0325, 1e15),
        55: (0.02875, 1e16),
        60: (0.025, 1e18)
    }
    NumberOfExploredNodes(simulation_quantity, sample_kp_data, sample_depths = [1, 3, 5, 10], number_of_bnb_repitions = 4, number_of_equivalent_kp_instances = 10).simulate_and_visualize()


if __name__ == "__main__":
    main(SimulationQuantity.number_of_leafs_reached) # Last run, pick other enum entry to simulate the number of explored nodes