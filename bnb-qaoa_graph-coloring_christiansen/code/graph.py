from dataclasses import dataclass


@dataclass
class Graph:
    """
    Class for construction and representation of instances of the Graph Coloring Problem

    Attributes:
    vertices (list): the vertices of the graph (labeled by integers)
    edges (list): the edges of the graph (consisting of tuples of integers indicating the start and end vertices)
    number_vertices (int): length of list vertices
    numver_edges (int): length of list edges
    """

    vertices: list
    edges: list

    def __post_init__(self):
        if any([v for v in self.vertices if type(v) != int]):
            raise TypeError("Vertices must be labeled by integers.")

        if any([e for e in self.edges if len(e) != 2]):
            raise ValueError("Edges must connect exactly to vertices.")

        if any([e for e in self.edges if type(e[0]) != int or type(e[1]) != int]):
            raise TypeError("Endpoints of edges must be labeled by integers.")
        
        if any([e for e in self.edges if e[0] > max(self.vertices) or e[1] > max(self.vertices)]):
            raise ValueError("Edges can only connect existing vertices.")

        self.number_vertices = len(self.vertices)
        self.number_edges = len(self.edges)


class GraphRelatedFunctions:

    def __init__(self, graph: Graph):
        self.graph = graph

    def degree(self, vertex: int, graph: Graph):
        return len([e for e in graph.edges if vertex in e])

    def sort_vertices_by_degree(self):
        vertex_degree_pairs = [{"vertex": v, "degree": self.degree(v, self.graph)} for v in self.graph.vertices]
        vertex_degree_pairs_sorted = sorted(vertex_degree_pairs, key = lambda pair: pair["degree"], reverse = True)
        return [p["vertex"] for p in vertex_degree_pairs_sorted]

    def find_max_degree(self, graph: Graph):
        return max([self.degree(v, graph) for v in graph.vertices])

    def find_left_index_for_edge(self, edge: tuple):
        return self.graph.vertices.index(edge[0])

    def find_right_index_for_edge(self, edge: tuple):
        return self.graph.vertices.index(edge[1])
    
    def find_neighbours(self, vertex: int, graph: Graph):
        """ Neighbours of a vertex are found by first filtering all edges incident to that vertex. """
        incident_edges = [e for e in graph.edges if vertex in e]
        return [v for e in incident_edges for v in e if v != vertex]

    def induced_subgraph(self, subgraph_vertices: list):
        subgraph_edges = [e for e in self.graph.edges if e[0] in subgraph_vertices and e[1] in subgraph_vertices]
        return Graph(vertices = subgraph_vertices, edges = subgraph_edges)

    def complement_graph(self, graph: Graph):
        complement_edges = [(u,v) for u in graph.vertices for v in graph.vertices if (u,v) not in graph.edges and (v,u) not in graph.edges and u < v]
        return Graph(vertices = graph.vertices, edges = complement_edges)


exemplary_graph_instances = {
    "A": Graph([1], []),
    "B": Graph([1,2], [(1,2)]),
    "C": Graph([1,2,3,4], [(1,2), (2,3), (3,4)]),
    "D": Graph([1,2,3,4,5,6], [(1,2), (1,4), (1,6), (2,3), (2,6), (3,4), (4,5), (5,6)])
}