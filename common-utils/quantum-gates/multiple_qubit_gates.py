import sys
sys.path.append("C:\\Users\\d92474\\Documents\\Uni\\Master Thesis\\GitHub\\MasterThesis_Hybrid-Quantum-Classical-Branch-and-Bound\\common-utils\\cpp-configuration")


import numpy as np
from numpy.typing import NDArray
import itertools
from copy import deepcopy
from typing import List, Dict, Union
from enum import Enum

from single_qubit_gates import Projectors, PauliOperators, RotationOperators
import kron_dot



class ControlledOn(Enum):
    zero = 0
    one = 1



class ControlledGates(Projectors):
    
    num_type = np.cdouble
    
    def apply_single_controlled_gate(self, target_index: int, control_index: int, single_qubit_gate, state: NDArray[num_type], controlled_on_one: bool = True):
        if target_index == control_index:
            raise ValueError("Target and control qubits must not be equal.")
        
        tmp_control, tmp_rest = deepcopy(state), deepcopy(state) # We have two different operations acting on state, the controlled part and the other part

        """ Application of the gate to be controlled """
        if controlled_on_one:
            kron_dot.kron_dot_dense(control_index, self.projector_one(), tmp_control)
        else: 
            kron_dot.kron_dot_dense(control_index, self.projector_zero(), tmp_control)
        kron_dot.kron_dot_dense(target_index, single_qubit_gate, tmp_control) # As these operations commute, no distinction needed when control_index < target_index

        """ Covering the rest of the Hilbert space, i.e. here only applying the other projector """
        if controlled_on_one:
            kron_dot.kron_dot_dense(control_index, self.projector_zero(), tmp_rest)
        else:
            kron_dot.kron_dot_dense(control_index, self.projector_one(), tmp_rest)
        
        """ Adding both results to end up with the state how it is fully transformed under the controlled operation """
        state = tmp_control + tmp_rest
        return state
    
    
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
    


class TwoQubitRotations(PauliOperators, RotationOperators, ControlledGates):
    
    num_type = np.cdouble


    def apply_rotation_zz(self, index1: int, index2: int, theta: Union[float, int], state: NDArray[num_type]):
        """ Based on the concept R_zz = CNOT R_z CNOT """

        if index1 == index2:
            raise ValueError("Target and control qubits must not be equal.")
        
        """ Apply CNOT with control on first qubit """
        state = self.apply_single_controlled_gate(target_index = index2, control_index = index1, single_qubit_gate = self.pauli_x(), state = state)

        """ Apply rotation-z on second qubit """
        kron_dot.kron_dot_dense(index2, self.rotation_z(theta), state)
        
        """ Reverse CNOT using that it is its own inverse """
        state = self.apply_single_controlled_gate(target_index = index2, control_index = index1, single_qubit_gate = self.pauli_x(), state = state)

        return state
