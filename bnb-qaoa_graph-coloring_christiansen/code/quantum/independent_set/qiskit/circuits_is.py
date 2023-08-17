import sys
sys.path.append("C:\\Users\\d92474\\Documents\\Uni\\Master Thesis\\GitHub\\QuBRA\\bnb-qaoa_graph-coloring_christiansen\\code")

import math
from fractions import Fraction
from qiskit import QuantumCircuit, QuantumRegister
from qiskit import Aer          # Needed to use QuantumCircuit.save_statevector() method
from qiskit.circuit import Parameter

from graph import Graph, GraphRelatedFunctions, exemplary_gc_instances


class MixerCircIS(QuantumCircuit):
    """Default Mixer Circuit for QAOA."""

    def __init__(self, vertex_register: QuantumRegister):
        """Initialize the circuit."""
        self.beta = Parameter("beta")
        super().__init__(vertex_register, name="UMixer")

        angle_x = 2 * self.beta
        for qubit in vertex_register:
            super().rx(angle_x, qubit)

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


class PhaseCircIS(QuantumCircuit):
    """ Phase separation circuit for Graph Coloring QAOA with quadratic penalty. """

    """ Attention: Parameters a and b are switched here compared to notes and Lucas (2014). """

    def __init__(self, vertex_register: QuantumRegister, graph: Graph):
        """ Initialize the circuit """
        self.graph = graph
        self.gamma = Parameter("gamma")
        self.a = Parameter("a")
        self.b = Parameter("b")
        QuantumCircuit.__init__(self, vertex_register, name = "UPhase")

        """ Single-qubit z-rotations """
        for v in range(self.graph.number_vertices):
            degree_of_vertex = GraphRelatedFunctions(self.graph).degree(self.graph.vertices[v], self.graph)
            angle_z = - 2 * self.gamma * (self.a - degree_of_vertex * self.b)
            super().rz(angle_z, vertex_register[v]) # Vertices are labeled from 1,...,|V| whereas list indices start at 0

        """ Two-qubit z-rotations """
        angle_zz = 2 * self.gamma * self.b
        for edge in self.graph.edges:
            index_left = GraphRelatedFunctions(self.graph).find_left_index_for_edge(edge)
            index_right = GraphRelatedFunctions(self.graph).find_right_index_for_edge(edge)
            super().rzz(angle_zz, vertex_register[index_left], vertex_register[index_right])


class QAOACircuitIS(QuantumCircuit):
    """ QAOA circuit for Graph Coloring with quadratic constraints. """

    def __init__(self, graph: Graph, depth: int):
        """ Initialize the circuit """

        self.p = depth
        self.betas = [Parameter(f"beta{i}") for i in range(self.p)]
        self.gammas = [Parameter(f"gamma{i}") for i in range(self.p)]
        self.a = Parameter("a")
        self.b = Parameter("b")

        vertex_register = QuantumRegister(graph.number_vertices, name = "vertex")
        super().__init__(vertex_register, name = f"QAOA for IS with depth {self.p}")

        mixer_circ = MixerCircIS(vertex_register)
        phase_circ = PhaseCircIS(vertex_register, graph)

        """ Prepare intial state """
        super().h([*vertex_register])

        """ Alternatingly apply phase separation and mixer for depth p """
        for gamma, beta in zip(self.gammas, self.betas):
            """ Apply phase separation circuit """
            phase_params = {
                phase_circ.gamma: gamma,
                phase_circ.a: self.a,
                phase_circ.b: self.b,
            }
            super().append(phase_circ.to_instruction(phase_params),
                           [*vertex_register])
            
            """ Apply mixer circuit """
            super().append(mixer_circ.to_instruction({mixer_circ.beta: beta}),
                           [*vertex_register])

        """ Measurement """
        super().save_statevector()
        super().measure_all()


    @staticmethod
    def beta_range():
        """Return range of values for beta."""
        return 0, math.pi

    @staticmethod
    def gamma_range(a, b):
        """Return range of values for gamma."""
        gamma_min = 0
        frac_a = Fraction(a)
        frac_b = Fraction(b)
        a1 = frac_a.numerator
        a2 = frac_a.denominator
        b1 = frac_b.numerator
        b2 = frac_b.denominator
        # lowest common multiple lcm(a2, b2)
        lcm = abs(a2 * b2) / math.gcd(a2, b2)
        # greatest common divisor
        gcd = math.gcd(a1, b1)
        gamma_max = lcm / gcd * 2 * math.pi
        return gamma_min, gamma_max



"""
graph = exemplary_gc_instances["B"]
vertex_reg = QuantumRegister(graph.number_vertices, name = "vertex")
circ = PhaseCircIS(vertex_reg, graph)
print(circ.decompose().draw())
"""
