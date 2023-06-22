from graph import Graph, GraphRelatedFunctions, exemplary_gc_instances


class GraphColoring(GraphRelatedFunctions):

    def objective_function(self, coloring: list):
        different_colors = self.find_different_colors(coloring)
        return len(different_colors)

    def find_different_colors(self, coloring: list):
        different_colors = []
        for c in coloring:
            if c not in different_colors:
                different_colors.append(c)
        return different_colors

    def upper_bound_by_degree(self):
        max_degree = self.find_max_degree(self.graph)
        return max_degree + 1

    def equivalent_subgraph(self, coloring: list):
        if len(coloring) == 0:
            return self.graph
        uncolored_vertices = self.graph.vertices[len(coloring):]
        uncolored_subgraph = self.induced_subgraph(uncolored_vertices)
        different_colors = self.find_different_colors(coloring)
        new_color_vertices = [self.graph.number_vertices + c for c in different_colors]
        new_color_edges = [(u, v) for u in new_color_vertices for v in new_color_vertices if u < v]
        new_mixture_edges = []
        for i in range(len(coloring)):
            current_vertex = self.graph.vertices[i]
            uncolored_plus_one = self.induced_subgraph(uncolored_vertices + [current_vertex])
            neighbours = self.find_neighbours(vertex = current_vertex, graph = uncolored_plus_one)
            corresponding_color_vertex = self.graph.number_vertices + coloring[i]
            for n in neighbours:
                candidate_mixture_edge = (corresponding_color_vertex, n)
                if candidate_mixture_edge not in new_mixture_edges:
                    new_mixture_edges.append(candidate_mixture_edge)
        new_vertices = uncolored_vertices + new_color_vertices
        new_edges = uncolored_subgraph.edges + new_color_edges + new_mixture_edges
        return Graph(new_vertices, new_edges)


"""
graph = Graph(vertices = [2,3,1,4], edges = [(1,2),(2,3),(3,4)])
equivalent_subgraph = GraphColoring(graph).equivalent_subgraph([1])
print("Equivalent subgraph = ", equivalent_subgraph)
complement_subgraph = GraphRelatedFunctions(graph).complement_graph(equivalent_subgraph)
print("Complement subgraph = ", complement_subgraph)
"""