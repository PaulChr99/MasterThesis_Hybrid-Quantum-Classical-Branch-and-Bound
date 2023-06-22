"""Implementations of the quantum circuits encoding the capacity constraint of the
knapsack problem via linear soft constraints.
This includes ...
- a quantum fourier transform based adding circuit,
- a feasibility oracle for the knapsack problem, and
- the QAOA circuits for the linear soft-constraint approach.
All implementations have been kept general, in the sense that they have
been defined for arbitrary instances of the knapsack problem.
"""
import sys
sys.path.append("C:\\Users\\d92474\\Documents\\Uni\\Master Thesis\\GitHub\\QuBRA\\bnb-qaoa_knapsack_christiansen\\code")

from functools import partial
from itertools import product
from fractions import Fraction
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit import Aer, transpile, execute
from qiskit.circuit import Parameter
import numpy as np
from knapsack_problem import KnapsackProblem
import math


class QFT(QuantumCircuit):
    """Compute the quantum fourier transform up to ordering of qubits."""

    def __init__(self, register):
        """Initialize the Circuit."""
        super().__init__(register, name="QFT")
        for idx, qubit in reversed(list(enumerate(register))):
            super().h(qubit)
            for c_idx, control_qubit in reversed(list(enumerate(register[:idx]))):
                k = idx - c_idx + 1
                super().cp(2 * np.pi / 2**k, qubit, control_qubit)


class Add(QuantumCircuit):
    """Circuit for adding n to intermediate state."""

    def __init__(self, register, n, control=None):
        """Initialize the Circuit."""
        self.register = register
        self.control = control
        qubits = [*register, *control] if control is not None else register
        super().__init__(qubits, name=f"Add {n}")
        binary = list(map(int, reversed(bin(n)[2:])))
        for idx, value in enumerate(binary):
            if value:
                self._add_power_of_two(idx)

    def _add_power_of_two(self, k):
        """Circuit for adding 2^k to intermediate state."""
        phase_gate = super().p
        if self.control is not None:
            phase_gate = partial(super().cp, target_qubit=self.control)
        for idx, qubit in enumerate(self.register):
            l = idx + 1
            if l > k:
                m = l - k
                phase_gate(2 * np.pi / 2**m, qubit)


class WeightCalculator(QuantumCircuit):
    """Circuit for calculating the weight of an item choice."""

    def __init__(self, choice_reg, weight_reg, problem: KnapsackProblem):
        """Initialize the circuit."""
        super().__init__(choice_reg, weight_reg, name="Calculate Weight")
        super().append(QFT(weight_reg).to_instruction(), weight_reg)
        for qubit, weight in zip(choice_reg, problem.weights):
            adder = Add(weight_reg, weight, control=[qubit]).to_instruction()
            super().append(adder, [*weight_reg, qubit])
        super().append(QFT(weight_reg).inverse().to_instruction(), weight_reg)


class FeasibilityOracle(QuantumCircuit):
    """Circuit for checking feasibility of a choice."""

    def __init__(self, choice_reg, weight_reg, flag_qubit, problem: KnapsackProblem,
                 clean_up=True):
        """Initialize the circuit."""
        c = math.floor(math.log2(problem.capacity)) + 1
        w0 = 2**c - problem.capacity - 1

        subcirc = QuantumCircuit(choice_reg, weight_reg, name="")
        qft = QFT(weight_reg)
        subcirc.append(qft.to_instruction(), weight_reg)
        for qubit, weight in zip(choice_reg, problem.weights):
            adder = Add(weight_reg, weight, control=[qubit]).to_instruction()
            subcirc.append(adder, [*weight_reg, qubit])
        adder = Add(weight_reg, w0)
        subcirc.append(adder.to_instruction(), weight_reg)
        subcirc.append(qft.inverse().to_instruction(), weight_reg)

        super().__init__(choice_reg, weight_reg, flag_qubit, name="U_v")
        super().append(subcirc.to_instruction(),
                       [*choice_reg, *weight_reg])
        super().x(weight_reg[c:])
        super().mcx(weight_reg[c:], flag_qubit)
        super().x(weight_reg[c:])
        if clean_up:
            super().append(subcirc.inverse().to_instruction(),
                           [*choice_reg, *weight_reg])


class DephaseValue(QuantumCircuit):
    """Dephase Value of an item choice."""

    def __init__(self, choice_reg, problem: KnapsackProblem):
        """Initialize the circuit."""
        self.gamma = Parameter("gamma")
        super().__init__(choice_reg, name="Dephase Value")
        for qubit, value in zip(choice_reg, problem.profits):
            super().p(- self.gamma * value, qubit)


class LinPhaseCirc(QuantumCircuit):
    """Phase seperation circuit for QAOA with linear soft constraints."""

    def __init__(self, choice_reg, weight_reg, flag_reg, problem: KnapsackProblem):
        """Initialize the circuit."""
        c = math.floor(math.log2(problem.capacity)) + 1
        self.a = Parameter("a")
        self.gamma = Parameter("gamma")
        super().__init__(choice_reg, weight_reg, flag_reg, name="UPhase")
        # initialize flag qubit
        super().x(flag_reg)
        # dephase value
        value_circ = DephaseValue(choice_reg, problem)
        super().append(value_circ.to_instruction({value_circ.gamma: self.gamma}),
                       choice_reg)
        # dephase penalty
        feasibility_oracle = FeasibilityOracle(choice_reg, weight_reg,
                                               flag_reg, problem,
                                               clean_up=False)
        super().append(feasibility_oracle.to_instruction(),
                       [*choice_reg, *weight_reg, flag_reg])
        for idx, qubit in enumerate(weight_reg):
            super().cp(2**idx * self.a * self.gamma, flag_reg, qubit)
        super().p(-2**c * self.a * self.gamma, flag_reg)
        super().append(feasibility_oracle.inverse().to_instruction(),
                       [*choice_reg, *weight_reg, flag_reg])


class DefaultMixer(QuantumCircuit):
    """Default Mixing Circuit for QAOA."""

    def __init__(self, register):
        """Initialize the circuit."""
        self.beta = Parameter("beta")
        super().__init__(register, name="UMix")
        super().rx(2 * self.beta, register)


class LinQAOACircuit(QuantumCircuit):
    """QAOA Circuit for Knapsack Problem with linear soft constraints."""

    def __init__(self, problem: KnapsackProblem, p: int):
        """Initialize the circuit."""
        self.p = p
        self.betas = [Parameter(f"beta{i}") for i in range(p)]
        self.gammas = [Parameter(f"gamma{i}") for i in range(p)]
        self.a = Parameter("a")

        n = math.floor(math.log2(problem.total_weight)) + 1
        c = math.floor(math.log2(problem.capacity)) + 1
        if c == n:
            n += 1

        choice_reg = QuantumRegister(problem.number_items, name="choices")
        weight_reg = QuantumRegister(n, name="weight")
        flag_reg = QuantumRegister(1, name="flag")

        super().__init__(choice_reg, weight_reg, flag_reg, name=f"LinQAOA {p=}")

        phase_circ = LinPhaseCirc(choice_reg, weight_reg, flag_reg, problem)
        mix_circ = DefaultMixer(choice_reg)

        # initial state
        super().h(choice_reg)

        # alternatingly apply phase seperation circuits and mixers
        for gamma, beta in zip(self.gammas, self.betas):
            # apply phase seperation circuit
            phase_params = {
                phase_circ.gamma: gamma,
                phase_circ.a: self.a,
            }
            super().append(phase_circ.to_instruction(phase_params),
                           [*choice_reg, *weight_reg, flag_reg])

            # apply mixer
            super().append(mix_circ.to_instruction({mix_circ.beta: beta}),
                           choice_reg)

        # measurement
        super().save_statevector()
        super().measure_all()

    @staticmethod
    def beta_range():
        return 0, math.pi

    @staticmethod
    def gamma_range(a):
        denominator = Fraction(a).denominator
        return 0, denominator * 2 * math.pi


def main():
    problem = KnapsackProblem(profits = [1, 2, 3], weights = [1, 2, 3],
                              capacity = 2)
    circ = LinQAOACircuit(problem, 2)
    print(circ.decompose().draw())


if __name__ == "__main__":
    main()