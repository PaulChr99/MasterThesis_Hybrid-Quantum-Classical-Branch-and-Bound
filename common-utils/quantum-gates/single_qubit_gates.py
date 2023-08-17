import numpy as np
from numpy.typing import NDArray



class PauliOperators:

    num_type = np.cdouble
    
    def identity(self, register_size: int):
        return np.identity(register_size, dtype = self.num_type)
    
    def pauli_x(self):
        return np.array([[0, 1], [1, 0]], dtype = self.num_type)
    

class RotationOperators:

    num_type = np.cdouble

    def rotation_z(self, theta):
        return np.array([[np.exp(-1j * theta/2), 0], [0, np.exp(1j * theta/2)]], dtype = self.num_type)
    
    def rotation_x(self, theta):
        return np.array([[np.cos(theta/2), -1j * np.sin(theta/2)], [-1j * np.sin(theta/2), np.cos(theta/2)]], dtype = self.num_type)
    

class Projectors:

    num_type = np.cdouble

    def projector_zero(self):
        return np.array([[1, 0], [0, 0]], dtype = self.num_type)
    
    def projector_one(self):
        return np.array([[0, 0], [0, 1]], dtype = self.num_type)
    

class PhaseGates:

    num_type = np.cdouble

    def single_phase_gate_zero(self, theta):
        return np.array([[np.exp(1j * theta), 0], [0, 1]], dtype = self.num_type)
    
    def single_phase_gate_one(self, theta):
        return np.array([[1, 0], [0, np.exp(1j * theta)]], dtype = self.num_type)
    

class ToBasisStates:

    num_type = np.cdouble

    def to_state_zero(self):
        return np.array([[1, 1], [0, 0]], dtype = self.num_type)
    
    def to_state_one(self):
        return np.array([[0, 0], [1, 1]], dtype = self.num_type)
    

class BasicGates:

    num_type = np.cdouble

    def hadamard(self):
        return 1/np.sqrt(2) * np.array([[1, 1], [1, -1]], dtype = self.num_type)