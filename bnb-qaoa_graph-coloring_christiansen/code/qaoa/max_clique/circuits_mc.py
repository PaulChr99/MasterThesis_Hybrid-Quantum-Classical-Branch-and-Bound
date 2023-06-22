import sys
sys.path.append("C:\\Users\\d92474\\Documents\\Uni\\Master Thesis\\GitHub\\QuBRA\\bnb-qaoa_graph-coloring_christiansen\\code")

import math
from fractions import Fraction
from qiskit import QuantumCircuit, QuantumRegister
from qiskit import Aer          # Needed to use QuantumCircuit.save_statevector() method, however only works in file qaoa_algorithm
from qiskit.circuit import Parameter

from graph import Graph, GraphRelatedFunctions, exemplary_gc_instances


class MixerCircMC(QuantumCircuit):
    """Default Mixer Circuit for QAOA."""

    def __init__(self, vertex_register: QuantumRegister, auxiliary_register: QuantumRegister):
        """Initialize the circuit."""
        self.beta = Parameter("beta")
        super().__init__(vertex_register, auxiliary_register, name="UMixer")

        """ Single-qubit x-rotations on vertex register """
        angle_x_vertex = 2 * self.beta
        for qubit in vertex_register:
            super().rx(angle_x_vertex, qubit)
        
        """ Single-qubit x-rotations on auxiliary register """
        angle_x_auxiliary = 2 * self.beta
        for qubit in auxiliary_register:
            super().rx(angle_x_auxiliary, qubit)


"""
class PhaseCircuitHelperFunctions:

    def __init__(self, graph: Graph):
        self.graph = graph
        self.max_degree = GraphRelatedFunctions(self.graph).find_max_degree()

    def find_index_in_vertex_color_register(self, vertex: int, color: int):
        if vertex > self.graph.number_vertices:
            raise ValueError("Vertex must be contained in graph.")
        if color > self.max_degree + 1:
            raise ValueError("Color must be contained in the list of valid colors.")
        return (vertex - 1) * (self.max_degree + 1) + color - 1 # Have to substract 1 at the end as list indices start at 0
"""


class PhaseCircMC(QuantumCircuit):
    """ Phase separation circuit for Graph Coloring QAOA with quadratic penalty. """

    """ Attention: Parameters a and b are switched here compared to notes and Lucas (2014). """

    def __init__(self, vertex_register: QuantumRegister, auxiliary_register: QuantumRegister, graph: Graph):
        """ Initialize the circuit """
        self.graph = graph
        self.max_degree = GraphRelatedFunctions(self.graph).find_max_degree(self.graph)
        self.d = math.floor(math.log2(self.max_degree + 1))
        self.gamma = Parameter("gamma")
        self.a = Parameter("a")
        self.b1 = Parameter("b1")
        self.b2 = Parameter("b2")
        QuantumCircuit.__init__(self, vertex_register, auxiliary_register, name = "UPhase")

        """ Single-qubit z-rotations on vertex register """
        for v in range(self.graph.number_vertices):
            """ Combining all the terms acting on the same qubit. The edge terms may appear multiple times, given by the degree of the respective vertex. """
            degree_of_vertex = GraphRelatedFunctions(self.graph).degree(self.graph.vertices[v], self.graph)
            angle_z_vertex = - 2 * self.gamma * ( ( (self.max_degree + 1 - self.graph.number_vertices) * self.b1 - self.a) + degree_of_vertex * self.b2 )
            super().rz(angle_z_vertex, vertex_register[v])

        super().barrier()

        """ Single-qubit z-rotations on auxiliary register """
        for j in range(self.d):
            angle_z_auxiliary = 2**(j + 1) * self.gamma * ( (self.max_degree + 1 - self.graph.number_vertices) * self.b1 + (2**(self.d) - 2) * self.b2 )
            super().rz(angle_z_auxiliary, auxiliary_register[j])

        super().barrier()

        """ Single-qubit z-rotation on last entry of auxiliary register """
        angle_z_auxiliary_end = 2 * self.gamma * ((self.max_degree + 1) - (2**(self.d) - 1)) * (self.max_degree * self.b1 + (2**(self.d + 1) - 4) * self.b2)
        super().rz(angle_z_auxiliary_end, auxiliary_register[-1])

        super().barrier()

        """ Two-qubit z-rotations on vertex register """
        for u in range(self.graph.number_vertices - 1):
            for v in range(u + 1, self.graph.number_vertices):
                """ Combining all terms acting on the same pairs of qubits. Both whether (u,v) or (v,u) are edges have to be checked. """
                angle_zz_vertex = 2 * self.gamma * self.b1
                if (u,v) in self.graph.edges or (v,u) in self.graph.edges:
                    angle_zz_vertex += - 2 * self.gamma * self.b2
                super().rzz(angle_zz_vertex, vertex_register[u], vertex_register[v])
        
        super().barrier()

        """ Two-qubit z-rotations on auxiliary register """
        for k in range(self.d - 1):
            for j in range(k + 1, self.d):
                angle_zz_auxiliary = 2**(k + j + 1) * self.gamma * (self.b1 + self.b2)
                super().rzz(angle_zz_auxiliary, auxiliary_register[k], auxiliary_register[j])

        super().barrier()

        """ Two-qubit z-rotations on auxiliary register with one qubit fixed to last entry """
        for j in range(self.d):
            angle_zz_auxiliary_end = 2**(j + 1) * self.gamma * ((self.max_degree + 1) - (2**(self.d) - 1)) * (self.b1 + 2 * self.b2)
            super().rzz(angle_zz_auxiliary_end, auxiliary_register[j], auxiliary_register[-1])

        super().barrier()

        """ Two-qubit z-rotations on vertex and auxiliary register mixed """
        for v in range(self.graph.number_vertices):
            for j in range(self.d):
                angle_zz_mixed = - 2**(j + 1) * self.gamma * self.b1
                super().rzz(angle_zz_mixed, vertex_register[v], auxiliary_register[j])
        
        super().barrier()

        """ Two-qubit z-rotations on vertex and last entry of auxiliary register """
        for v in range(self.graph.number_vertices):
            angle_zz_mixed_end = - 2 * self.gamma * ((self.max_degree + 1) - (2**(self.d) - 1)) * self.b1
            super().rzz(angle_zz_mixed_end, vertex_register[v], auxiliary_register[-1])



class QAOACircuitMC(QuantumCircuit):
    """ QAOA circuit for Graph Coloring with quadratic constraints. """

    def __init__(self, graph: Graph, depth: int):
        """ Initialize the circuit """

        self.p = depth
        self.betas = [Parameter(f"beta{i}") for i in range(self.p)]
        self.gammas = [Parameter(f"gamma{i}") for i in range(self.p)]
        self.a = Parameter("a")
        self.b1 = Parameter("b1")
        self.b2 = Parameter("b2")

        max_degree = GraphRelatedFunctions(graph).find_max_degree(graph)
        d = math.floor(math.log2(max_degree + 1))
        vertex_register = QuantumRegister(graph.number_vertices, name = "vertex")
        auxiliary_register = QuantumRegister(d + 1, name = "auxiliary")
        super().__init__(vertex_register, auxiliary_register, name = f"QAOA for MC with depth {self.p}")

        mixer_circ = MixerCircMC(vertex_register, auxiliary_register)
        phase_circ = PhaseCircMC(vertex_register, auxiliary_register, graph)

        """ Prepare intial state """
        super().h([*vertex_register, *auxiliary_register])

        """ Alternatingly apply phase separation and mixer for depth p """
        for gamma, beta in zip(self.gammas, self.betas):
            """ Apply phase separation circuit """
            phase_params = {
                phase_circ.gamma: gamma,
                phase_circ.a: self.a,
                phase_circ.b1: self.b1,
                phase_circ.b2: self.b2
            }
            super().append(phase_circ.to_instruction(phase_params),
                           [*vertex_register, *auxiliary_register])
            
            """ Apply mixer circuit """
            super().append(mixer_circ.to_instruction({mixer_circ.beta: beta}),
                           [*vertex_register, *auxiliary_register])

        """ Measurement """
        super().save_statevector()
        super().measure_all()


    @staticmethod
    def beta_range():
        """Return range of values for beta."""
        return 0, math.pi

    @staticmethod
    def gamma_range(a, b1, b2):
        """Return range of values for gamma."""
        gamma_min = 0
        frac_a = Fraction(a)
        frac_b1 = Fraction(b1)
        frac_b2 = Fraction(b2)
        a_num = frac_a.numerator
        a_denom = frac_a.denominator
        b1_num = frac_b1.numerator
        b1_denom = frac_b1.denominator
        b2_num = frac_b2.numerator
        b2_denom = frac_b2.denominator
        # lowest common multiple lcm(a_denom, b1_denom, b2_denom)
        lcm = abs(a_denom * b1_denom * b2_denom) / math.gcd(a_denom, b1_denom, b2_denom)
        # greatest common divisor
        gcd = math.gcd(a_num, b1_num, b2_num)
        gamma_max = lcm / gcd * 2 * math.pi
        return gamma_min, gamma_max



"""
graph = exemplary_gc_instances["B"]
max_degree = GraphRelatedFunctions(graph).find_max_degree()
d = math.floor(math.log2(max_degree + 1))
vertex_reg = QuantumRegister(graph.number_vertices, name = "vertex")
auxiliary_reg = QuantumRegister(d + 1, name = "auxiliary")
circ = PhaseCircMC(vertex_reg, auxiliary_reg, graph)
print(circ.decompose().draw())
"""
