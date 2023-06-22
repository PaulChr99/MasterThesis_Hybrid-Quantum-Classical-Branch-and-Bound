import sys
sys.path.append("C:\\Users\\d92474\\Documents\\Uni\\Master Thesis\\GitHub\\QuBRA\\bnb-qaoa_graph-coloring_christiansen\\code")

import math
from fractions import Fraction
from qiskit import QuantumCircuit, QuantumRegister
from qiskit import Aer          # Needed to use QuantumCircuit.save_statevector() method
from qiskit.circuit import Parameter

from graph import Graph, GraphRelatedFunctions, exemplary_gc_instances


class MixerCircGC(QuantumCircuit):
    """Default Mixer Circuit for QAOA."""

    def __init__(self, color_register: QuantumRegister, vertex_color_register: QuantumRegister):
        """Initialize the circuit."""
        self.beta = Parameter("beta")
        super().__init__(color_register, vertex_color_register, name="UMixer")

        """ Single-qubit x-rotations on color register """
        angle_x_color = 2 * self.beta
        for qubit in color_register:
            super().rx(angle_x_color, qubit)
        
        """ Single-qubit x-rotations on register combining vertices and colors """
        angle_x_vertex_color = 2 * self.beta
        for qubit in vertex_color_register:
            super().rx(angle_x_vertex_color, qubit)


class PhaseCircuitHelperFunctions:

    def __init__(self, graph: Graph):
        self.graph = graph
        self.max_degree = GraphRelatedFunctions(self.graph).find_max_degree(self.graph)

    def find_index_in_vertex_color_register(self, vertex: int, color: int):
        if vertex not in self.graph.vertices:
            raise ValueError("Vertex must be contained in graph.")
        if color > self.max_degree + 1:
            raise ValueError("Color must be contained in the list of valid colors.")
        vertex_index = self.graph.vertices.index(vertex)
        return (vertex_index - 1) * (self.max_degree + 1) + color - 1 # Have to substract 1 at the end as list indices start at 0


class PhaseCircGC(QuantumCircuit, PhaseCircuitHelperFunctions):
    """ Phase separation circuit for Graph Coloring QAOA with quadratic penalty. """

    """ Attention: Parameters a and b are switched here compared to notes and Lucas (2014). """

    def __init__(self, color_register: QuantumRegister, vertex_color_register: QuantumRegister, graph: Graph):
        """ Initialize the circuit """
        self.graph = graph
        self.gamma = Parameter("gamma")
        self.a = Parameter("a")
        self.b = Parameter("b")
        QuantumCircuit.__init__(self, color_register, vertex_color_register, name = "UPhase")
        PhaseCircuitHelperFunctions.__init__(self, self.graph)

        """ Single-qubit z-rotations on color register """
        angle_z_color = 2 * self.gamma * (self.a - self.graph.number_vertices * self.b)
        for qubit in color_register:
            super().rz(angle_z_color, qubit)

        super().barrier()

        """ Single-qubit z-rotations on register combining vertices and colors """
        for vertex in self.graph.vertices:
            """ Combining all the terms acting on the same qubit. The edge terms may appear multiple times, given by the degree of the respective vertex. """
            degree_of_vertex = GraphRelatedFunctions(self.graph).degree(vertex, self.graph)
            angle_z_vertex_color = 2 * self.gamma * self.b * (2 * self.max_degree - 1 + degree_of_vertex)
            for color in range(1, self.max_degree + 1 + 1):
                corresponding_index = self.find_index_in_vertex_color_register(vertex, color)
                qubit = vertex_color_register[corresponding_index]
                super().rz(angle_z_vertex_color, qubit)

        super().barrier()

        """ Two-qubit z-rotations on color register and register combining vertices and colors mixed """
        angle_zz_mixed = - 2 * self.gamma * self.b
        for vertex in self.graph.vertices:
            for color in range(1, self.max_degree + 1 + 1):
                corresponding_index = self.find_index_in_vertex_color_register(vertex, color)
                super().rzz(angle_zz_mixed, color_register[color - 1], vertex_color_register[corresponding_index])
        
        super().barrier()

        """ Two-qubit z-rotations on register combining vertices and colors iterating over vertices """
        angle_zz_vertex_color_vertices = 4 * self.gamma * self.b
        for vertex in self.graph.vertices:
            for color_1 in range(1, self.max_degree + 1 + 1):
                index_with_color_1 = self.find_index_in_vertex_color_register(vertex, color_1)
                qubit_1 = vertex_color_register[index_with_color_1]
                for color_2 in range(color_1 + 1, self.max_degree + 1 + 1):
                    index_with_color_2 = self.find_index_in_vertex_color_register(vertex, color_2)
                    qubit_2 = vertex_color_register[index_with_color_2]
                    super().rzz(angle_zz_vertex_color_vertices, qubit_1, qubit_2)

        super().barrier()

        """ Two-qubit z-rotations on register combining vertices and colors iterating over edges """
        angle_zz_vertex_color_edges = 2 * self.gamma * self.b
        for edge in self.graph.edges:
            for color in range(1, self.max_degree + 1 + 1):
                index_left_vertex = self.find_index_in_vertex_color_register(edge[0], color)
                index_right_vertex = self.find_index_in_vertex_color_register(edge[1], color)
                qubit_left_vertex = vertex_color_register[index_left_vertex]
                qubit_right_vertex = vertex_color_register[index_right_vertex]
                super().rzz(angle_zz_vertex_color_edges, qubit_left_vertex, qubit_right_vertex)


class QAOACircuitGC(QuantumCircuit):
    """ QAOA circuit for Graph Coloring with quadratic constraints. """

    def __init__(self, graph: Graph, depth: int):
        """ Initialize the circuit """

        self.p = depth
        self.betas = [Parameter(f"beta{i}") for i in range(self.p)]
        self.gammas = [Parameter(f"gamma{i}") for i in range(self.p)]
        self.a = Parameter("a")
        self.b = Parameter("b")

        max_degree = GraphRelatedFunctions(graph).find_max_degree(graph)
        color_register = QuantumRegister(max_degree + 1, name = "color")
        vertex_color_register = QuantumRegister(graph.number_vertices * (max_degree + 1), name = "vertex and color")
        super().__init__(color_register, vertex_color_register, name = f"QAOA for GC with depth {self.p}")

        mixer_circ = MixerCircGC(color_register, vertex_color_register)
        phase_circ = PhaseCircGC(color_register, vertex_color_register, graph)

        """ Prepare intial state """
        super().h([*color_register, *vertex_color_register])

        """ Alternatingly apply phase separation and mixer for depth p """
        for gamma, beta in zip(self.gammas, self.betas):
            """ Apply phase separation circuit """
            phase_params = {
                phase_circ.gamma: gamma,
                phase_circ.a: self.a,
                phase_circ.b: self.b,
            }
            super().append(phase_circ.to_instruction(phase_params),
                           [*color_register, *vertex_color_register])
            
            """ Apply mixer circuit """
            super().append(mixer_circ.to_instruction({mixer_circ.beta: beta}),
                           [*color_register, *vertex_color_register])

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




#graph = exemplary_gc_instances["A"]
#max_degree = GraphRelatedFunctions(graph).find_max_degree()
#color_reg = QuantumRegister(max_degree + 1, name = "color")
#vertex_reg = QuantumRegister(graph.number_vertices * (max_degree + 1), name = "vertex and color")
#circ = QAOACircuitGC(graph, 1)
#print(circ.decompose().draw())

