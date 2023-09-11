import sys
sys.path.append("C:\\Users\\d92474\\Documents\\Uni\\Master Thesis\\GitHub\\MasterThesis_Hybrid-Quantum-Classical-Branch-and-Bound\\common-utils")
sys.path.append("C:\\Users\\d92474\\Documents\\Uni\\Master Thesis\\GitHub\\MasterThesis_Hybrid-Quantum-Classical-Branch-and-Bound\\common-utils\\cpp-configuration")
sys.path.append("C:\\Users\\d92474\\Documents\\Uni\\Master Thesis\\GitHub\\MasterThesis_Hybrid-Quantum-Classical-Branch-and-Bound\\common-utils\\quantum-gates")

import itertools
import numpy as np
from numpy.typing import NDArray
from copy import deepcopy
from typing import List, Dict, Union
from enum import Enum

import kron_dot
from knapsack_problem import KnapsackProblem, exemplary_kp_instances
from single_qubit_gates import Projectors, BasicGates, PauliOperators, PhaseGates
from multiple_qubit_gates import ControlledGates


class ControlledOn(Enum):
    zero = 0
    one = 1



class AuxiliaryFunctions:

    num_type = np.cdouble
    
    def __init__(self, problem_instance: KnapsackProblem):
        self.problem_instance = problem_instance
        self.capacity_register_size = int(np.floor(np.log2(self.problem_instance.capacity)) + 1)
    
    def find_binary_variable_representation_for_basis_state_index(self, index_of_one: int, register_one_size: int, register_two_size: int):
        total_register_size = register_one_size + register_two_size
        binary_rep = bin(index_of_one)[2:] # No reversed order for the sake of consistency: (0,0,1,0) means including the second item but not the first, so this should be mapped to 01
        difference = total_register_size - len(binary_rep)
        full_binary_rep = "0"*int(difference) + binary_rep
        rep_to_display = full_binary_rep[:register_one_size] + "-" + full_binary_rep[register_one_size:]
        return full_binary_rep
    
    def find_basis_state_index_for_binary_variable_representation(self, binary_variable_rep: str):
        return sum([int(binary_variable_rep[-1-j]) * 2**j for j in range(len(binary_variable_rep))])

    def binary_representation_for_capacity_register(self, integer: int):
        if integer > self.problem_instance.capacity:
            raise ValueError("Integer to align with capacity register must not be larger than capacity.")
        binary_rep = list(map(int, reversed(bin(integer)[2:]))) # python starts from the most significant bit, i.e. in opposite direction to thesis
        difference = self.capacity_register_size - len(binary_rep) # may be 0
        return binary_rep + [0]*int(difference)
    
    def find_superposition_for_state(self, state: NDArray[num_type]):
        states_in_superposition = []
        for s in range(len(state)):
            if not np.isclose(state[s], 0, rtol=1e-10):
                binary_variable_representation = self.find_binary_variable_representation_for_basis_state_index(s, self.problem_instance.number_items, self.capacity_register_size)
                states_in_superposition.append(binary_variable_representation)
        return states_in_superposition


# Applying a multiple controlled gate somehow is different to the version in the common-utils folder and breaking QAOA

class ControlledGates2(Projectors):
    
    num_type = np.cdouble
    
    def apply_multiple_controlled_gate(self, target_index: int, control_dict: List[Dict[str, Union[int, ControlledOn]]], 
                                       single_qubit_gate: NDArray[num_type], state: NDArray[num_type]):
        
        if target_index in [control["control index"] for control in control_dict]:
            raise ValueError("Target qubit cannot also be a control qubit.")
        
        """ Application of the gate to be controlled """
        tmp_control = deepcopy(state)
        for control in control_dict:
            if control["controlled on"] == ControlledOn.one:
                kron_dot.kron_dot_dense(control["control index"], self.projector_one(), tmp_control)
            else:
                kron_dot.kron_dot_dense(control["control index"], self.projector_zero(), tmp_control)
        kron_dot.kron_dot_dense(target_index, single_qubit_gate, tmp_control) # Again the order of application does not matter

        """ Covering the rest of the Hilbert space, i.e. every other combination of projectors """
        control_combination = [1 if control["controlled on"] == ControlledOn.one else 0 for control in control_dict] # projector |1><1| is mapped to 1, |0><0| analogously to 0 
        all_combinations = list(map(list, itertools.product([0, 1], repeat = len(control_dict))))
        other_combinations =  [combi for combi in all_combinations if combi != control_combination]
        if len(other_combinations) != 2**len(control_dict) - 1: # Only in exactly one case the single-qubit gate should be applied
            raise ValueError("Check the computation of other combinations!")
        tmp_rest_list = []
        for combi in other_combinations:
            tmp_rest = deepcopy(state)
            for control_idx, control in enumerate(control_dict):
                corresponding_projector = self.projector_one() if combi[control_idx] == 1 else self.projector_zero()
                kron_dot.kron_dot_dense(control["control index"], corresponding_projector, tmp_rest)
            tmp_rest_list.append(tmp_rest)

        """ Adding all 2^{number of control indices} results to end up with the state how it is fully transformed under the controlled operation """
        state = tmp_control + sum(tmp_rest_list)
        return state
    


class QFT(ControlledGates2, ControlledGates, BasicGates, PhaseGates):

    def __init__(self, problem_instance: KnapsackProblem, register_start: int, register_size: int):
        self.problem_instance = problem_instance
        self.register_start = register_start
        self.register_size = register_size

    def apply_qft(self, state):
        for target_index in reversed(range(self.register_start, self.register_start + self.register_size)):
            kron_dot.kron_dot_dense(target_index, self.hadamard(), state)
            for control_index in reversed(range(self.register_start, target_index)):
                k = (target_index - control_index) + 1
                phase_gate = self.single_phase_gate_one(2 * np.pi * 1 / 2**k)
                state = self.apply_single_controlled_gate(target_index, control_index, phase_gate, state)
        return state

    def apply_inverse_qft(self, state):
        for target_index in range(self.register_start, self.register_start + self.register_size):
            for control_index in range(self.register_start, target_index):
                k = (target_index - control_index) + 1
                # As single phase gate is diagonal, only need to invert the diagonal entries to obtain the inverse matrix:
                inverse_phase_gate = self.single_phase_gate_one(- 2 * np.pi * 1 / 2**k)
                state = self.apply_single_controlled_gate(target_index, control_index, inverse_phase_gate, state)
            kron_dot.kron_dot_dense(target_index, self.hadamard(), state) # Hadamard is its own inverse
        return state
    


class Subtract(AuxiliaryFunctions, ControlledGates2, ControlledGates, PhaseGates):

    num_type = np.cdouble
    
    def __init__(self, problem_instance: KnapsackProblem, register_start: int, register_size: int):
        self.problem_instance = problem_instance
        self.register_start = register_start
        self.register_size = register_size
        self.capacity_register_size = int(np.floor(np.log2(self.problem_instance.capacity)) + 1)

    def apply_controlled_subtraction(self, control_index: int, integer_to_subtract: int, state: NDArray[num_type]):
        binary_representation = self.binary_representation_for_capacity_register(abs(integer_to_subtract))
        register_end = self.register_start + self.register_size
        for idx, bit in enumerate(reversed(binary_representation)):
            if bit == 1:
                """ As all qubits controlled by a certain classical bit are independent, we may
                first execute all phase gates controlled by the same bit before moving to the next """
                for j in range(idx + 1):
                    k = idx - j + 1
                    inverse_phase_gate = self.single_phase_gate_one(- integer_to_subtract/abs(integer_to_subtract) * 2 * np.pi * 1 / 2**k) # phase gets positive sign (= addition) in case integer < 0
                    state = self.apply_single_controlled_gate(target_index = register_end - j - 1, control_index = control_index, single_qubit_gate = inverse_phase_gate, state = state)
        return state
    


class StatePreparation(AuxiliaryFunctions, ControlledGates2, ControlledGates, PauliOperators, BasicGates):

    num_type = np.cdouble
    
    def __init__(self, problem_instance: KnapsackProblem):
        self.problem_instance = problem_instance
        self.capacity_register_start = self.problem_instance.number_items
        self.capacity_register_size = int(np.floor(np.log2(self.problem_instance.capacity)) + 1)
        self.qft = QFT(self.problem_instance, self.capacity_register_start, self.capacity_register_size)
        self.subtract = Subtract(self.problem_instance, self.capacity_register_start, self.capacity_register_size)


    def apply_digital_comparator(self, target_index: int, integer_to_compare: int, state: NDArray[num_type]):
        """ Apply Hadamard to current item register entry (interpretation: assume residual capacity >= weight) """
        kron_dot.kron_dot_dense(target_index, self.hadamard(), state)
            
        """ Apply integer comparison and controlled hadamard operator if residual capacity < weight (interpretation: reverse operation above) """
        binary_rep_comparison_integer = self.binary_representation_for_capacity_register(integer_to_compare)
        for weight_reg_idx in reversed(range(len(binary_rep_comparison_integer))):
            if binary_rep_comparison_integer[weight_reg_idx] == 1: # don't need the gate at all if the classical comparison bit is zero
                control_dict = [{"control index": self.capacity_register_start + weight_reg_idx, "controlled on": ControlledOn.zero}] # one fix control on |0> for each capacity register entry
                for control_idx in range(weight_reg_idx + 1, len(binary_rep_comparison_integer)):
                    if binary_rep_comparison_integer[control_idx] == 1:
                        control_dict.append({"control index": self.capacity_register_start + control_idx, "controlled on": ControlledOn.one})
                    else:
                        control_dict.append({"control index": self.capacity_register_start + control_idx, "controlled on": ControlledOn.zero})
                state = self.apply_multiple_controlled_gate(target_index, control_dict, self.hadamard(), state)
        
        return state
    

    def apply_state_preparation(self, state: NDArray[num_type]):
        """ To satisfy state preparation operates on state |0> = |0,...,0> all value-1 capacity register qubits need to be flipped first """
        binary_rep_capacity = self.binary_representation_for_capacity_register(self.problem_instance.capacity)
        for idx in range(len(binary_rep_capacity)):
            if binary_rep_capacity[idx] == 1:
                capacity_reg_idx = self.capacity_register_start + idx
                kron_dot.kron_dot_dense(capacity_reg_idx, self.pauli_x(), state)
        
        """ Now compare value encoded in capacity register with single weights and perform controlled subtraction """
        for item_reg_idx in range(self.problem_instance.number_items):
            
            weight = self.problem_instance.weights[item_reg_idx]
            #print("state at beg = ", state)

            """ Apply digital comparator, i.e. apply Hadamard based on whether residual capacity >= weight """
            state = self.apply_digital_comparator(target_index = item_reg_idx, integer_to_compare = weight, state = state)

            """ Apply Quantum Fourier Transform (QFT) to capacity register """
            state = self.qft.apply_qft(state)
            #print("state after qft =  ", state)

            """ Perform the controlled subtraction """
            state = self.subtract.apply_controlled_subtraction(control_index = item_reg_idx, integer_to_subtract = weight, state = state)
            #print("state after sub =  ", state)

            """ Uncompute the QFT on capacity register """
            state = self.qft.apply_inverse_qft(state)
            #print("state after iqft = ", state)

            #print("state after mix =  ", state)
        
        return state


    def apply_inverse_state_preparation(self, state: NDArray[num_type]):
        """ Reverse integer comparison and controlled subtraction """
        for item_reg_idx in reversed(range(self.problem_instance.number_items)):

            weight = self.problem_instance.weights[item_reg_idx]
            
            """ Reverse inverse QFT on capacity register, i.e. apply normal QFT """
            state = self.qft.apply_qft(state)

            """ Invert controlled subtraction by controlling the addition of the same integer (i.e. subtract negative integer) """               
            state = self.subtract.apply_controlled_subtraction(control_index = item_reg_idx, integer_to_subtract = - weight, state = state)

            """ Reverse QFT, i.e. apply inverse QFT """
            state = self.qft.apply_inverse_qft(state)

            """ Apply digital comparator as it is its own inverse """
            state = self.apply_digital_comparator(target_index = item_reg_idx, integer_to_compare = weight, state = state)

        """ Reverse flipping of capacity-register qubits """
        binary_rep_capacity = self.binary_representation_for_capacity_register(self.problem_instance.capacity)
        for idx in reversed(range(len(binary_rep_capacity))):
            if binary_rep_capacity[idx] == 1:
                capacity_reg_idx = self.capacity_register_start + idx
                kron_dot.kron_dot_dense(capacity_reg_idx, self.pauli_x(), state)

        return state



class Mixer(ControlledGates2, ControlledGates, PhaseGates):

    num_type = np.cdouble
    
    def __init__(self, problem_instance: KnapsackProblem):
        self.problem_instance = problem_instance
        self.capacity_register_size = int(np.floor(np.log2(self.problem_instance.capacity)) + 1)
        self.total_number_qubits = self.problem_instance.number_items + self.capacity_register_size
        self.preparation = StatePreparation(self.problem_instance)

    def apply_grover_mixing(self, beta: Union[float, int], state: NDArray[num_type]):
        control_dict = [{"control index": j, "controlled on": ControlledOn.zero} for j in range(self.total_number_qubits - 1)]
        state = self.apply_multiple_controlled_gate(target_index = self.total_number_qubits - 1, control_dict = control_dict, single_qubit_gate = self.single_phase_gate_zero(-beta), state = state)
        return state
    
    def apply_mixer(self, beta: Union[float, int], state: NDArray[num_type]):
        """ Apply inverse state preparation (more specifically, the adjoint operator or circuit) """
        state = self.preparation.apply_inverse_state_preparation(state)

        """ Apply Grover mixer """
        state = self.apply_grover_mixing(beta, state)

        """ Apply state preparation """
        state = self.preparation.apply_state_preparation(state)

        return state



class PhaseSeparator(BasicGates, PhaseGates):

    num_type = np.cdouble
    
    def __init__(self, problem_instance: KnapsackProblem):
        self.problem_instance = problem_instance
    
    def apply_phase_separator(self, gamma: Union[float, int], state: NDArray[num_type]):
        for j in range(self.problem_instance.number_items):
            profit = self.problem_instance.profits[j]
            phase_gate = self.single_phase_gate_one(- gamma * profit)
            kron_dot.kron_dot_dense(j, phase_gate, state)
        return state
    


class QuasiAdiabaticEvolution(AuxiliaryFunctions):

    num_type = np.cdouble
    
    def __init__(self, problem_instance: KnapsackProblem, depth: int):
        self.problem_instance = problem_instance
        self.depth = depth
        self.item_register_size = self.problem_instance.number_items
        self.capacity_register_size = int(np.floor(np.log2(self.problem_instance.capacity)) + 1)
        self.preparation = StatePreparation(self.problem_instance)
        self.phase_separator = PhaseSeparator(self.problem_instance)
        self.mixer = Mixer(self.problem_instance)


    def apply_quasiadiabatic_evolution(self, angles: List[Union[int, float]]):
        
        if len(angles) != 2 * self.depth:
            raise ValueError("Number of provided values for gamma and beta parameters need to be consistent with specified circuit depth.")
        
        gamma_values = angles[0::2]
        beta_values = angles[1::2]
        
        """ Initialize |0> = |0,...,0> state """
        state = np.zeros(2**(self.item_register_size + self.capacity_register_size), dtype = self.num_type)
        #desired_initial_binary_variables = "0"*self.problem_instance.number_items + bin(self.problem_instance.capacity)[2:][::-1] # no item selected, full capacity encoded in capacity register
        #desired_initial_index_of_one = self.find_basis_state_index_for_binary_variable_representation(desired_initial_binary_variables)
        #state[desired_initial_index_of_one] = 1 # Initial state is: |0,...,0, bin(capacity)>
        state[0] = 1
        #print("States in superposition after init  = ", self.find_superposition_for_state(state))

        """ Apply state preparation to obtain equal superposition of feasible states only """
        state = self.preparation.apply_state_preparation(state)
        #print("States in superposition after prep  = ", self.find_superposition_for_state(state))
        
        #print("state at beginning = ", state)
        """ Alternatingly apply phase separator and mixer """
        for j in range(self.depth):
            state = self.phase_separator.apply_phase_separator(gamma_values[j], state)
            #print("state after phase = ", state)
            #print("States in superposition after phase = ", self.find_superposition_for_state(state))
            state = self.mixer.apply_mixer(beta_values[j], state)
            #print("state after mixer = ", state)
            #print("States in superposition after mixer = ", self.find_superposition_for_state(state))

        return state
        
        


def main():
    num_type = np.cdouble
    problem = exemplary_kp_instances["B"]
    """
    item_reg_size = problem.number_items
    capacity_reg_size = int(np.floor(np.log2(problem.capacity)) + 1)
    state = np.zeros(2**(item_reg_size + capacity_reg_size), dtype = num_type)
    state[0] = 1 # Corresponds to state |0,0,0,0,0>
    print(AuxiliaryFunctions(problem).find_binary_variable_representation_for_basis_state_index(np.where(state == 1)[0][0], item_reg_size + capacity_reg_size))
    con_dict = [{"control index": 3, "controlled on": ControlledOn.zero}, {"control index": 4, "controlled on": ControlledOn.zero}]
    state = ControlledGates().apply_multiple_controlled_gate(target_index=2, control_dict=con_dict, single_qubit_gate=BasicGates().pauli_x(), state=state)
    print(state)
    print(AuxiliaryFunctions(problem).find_binary_variable_representation_for_basis_state_index(np.where(state == 1)[0][0], item_reg_size + capacity_reg_size))
    """
    #print(AuxiliaryFunctions(problem_instance=problem).find_binary_variable_representation_for_basis_state_index(index_of_one=3, total_register_size=5))
    #print(AuxiliaryFunctions(problem_instance=problem).find_basis_state_index_for_binary_variable_representation(binary_variable_rep="00011"))
    #np.sum(state)
    #print("state = ", state)
    #print(PhaseSeparator(problem).apply_phase_separator(1, state))
    depth = 1
    angles = [np.pi/4, np.pi/4]*depth
    state = QuasiAdiabaticEvolution(problem, depth).apply_quasiadiabatic_evolution(angles)
    print(Measurement(problem).measure_state(state))

    state1 = np.zeros(2**5, dtype=num_type)
    state1[3] = 1
    #Mixer(problem).apply_mixer(np.pi/4, state1)

    state2 = np.zeros(2**5, dtype=num_type)
    state2[16] = 1
    #Mixer(problem).apply_mixer(np.pi/4, state2)



if __name__ == "__main__":
    main()



