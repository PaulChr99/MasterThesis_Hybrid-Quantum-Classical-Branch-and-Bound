import random
import numpy as np
from typing import Union, List

from graph_coloring import GraphColoring
from graph import Graph


class DSatur(GraphColoring):
    """
    Class for obtaining upper bounds for graph coloring problem instances using the DSatur algorithm.
    """
    """
    def __init__(self, graph: Graph, partial_coloring: list = []):
        self.graph = graph
        if any([c for c in partial_coloring if len(c) != 2]):
            raise ValueError("A coloring must map one vertices to exactly one color.")
        if any([c["vertex"] for c in partial_coloring].count(v) > 1 for v in [c["vertex"] for c in partial_coloring]):
            raise ValueError("A vertex can only be assigned one color.")
        self.partial_coloring = partial_coloring   
    """    

    def find_colors_used_by_neighbours(self, neighbours: list, coloring_dict: List[dict]):
        different_colors_of_neighbours = []
        for c in coloring_dict:
            if c["vertex"] in neighbours:
                if c["color"] not in different_colors_of_neighbours:
                    different_colors_of_neighbours.append(c["color"])
        return different_colors_of_neighbours

    def degree_of_saturation(self, vertex: int, coloring_dict: List[dict]):
        neighbours = self.find_neighbours(vertex, self.graph)
        different_colors_of_neighbours = self.find_colors_used_by_neighbours(neighbours, coloring_dict)
        return len(different_colors_of_neighbours) if len(coloring_dict) > 0 else 0

    def select_next_vertex(self, coloring_dict: List[dict]):
        def find_max_dsatur_vertex(vertices: list):
            vertex_dsatur_pairs = []
            for v in vertices:
                vertex_dsatur_pairs.append({"vertex": v, "dsatur": self.degree_of_saturation(v, coloring_dict)})
            max_dsatur = max([p["dsatur"] for p in vertex_dsatur_pairs])
            return [p["vertex"] for p in vertex_dsatur_pairs if p["dsatur"] == max_dsatur]

        def find_max_degree_vertex(vertices: list, graph: Graph):
            vertex_degree_pairs = []
            for v in vertices:
                vertex_degree_pairs.append({"vertex": v, "degree": self.degree(v, graph)})
            max_degree = max([p["degree"] for p in vertex_degree_pairs])
            return [p["vertex"] for p in vertex_degree_pairs if p["degree"] == max_degree]
        
        uncolored_vertices = list(set(self.graph.vertices) - set([c["vertex"] for c in coloring_dict]))
        candidates_by_dsatur = find_max_dsatur_vertex(uncolored_vertices)
        if len(candidates_by_dsatur) == 1:
            return candidates_by_dsatur[0]
        else:
            #print("Uncolored vertices = ", uncolored_vertices)
            uncolored_subgraph = self.induced_subgraph(uncolored_vertices)
            candidates_by_degree = find_max_degree_vertex(candidates_by_dsatur, uncolored_subgraph)
            return candidates_by_degree[0] if (len(candidates_by_degree) == 1) else random.choice(candidates_by_degree)

    def dsatur_upper_bound(self, coloring: list):
        coloring_dict = [{"vertex": self.graph.vertices[i], "color": coloring[i]} for i in range(len(coloring))]
        while len(coloring_dict) < self.graph.number_vertices:
            vertex_to_color = self.select_next_vertex(coloring_dict)
            neighbours = self.find_neighbours(vertex_to_color, self.graph)
            different_colors_of_neighbours = self.find_colors_used_by_neighbours(neighbours, coloring_dict)
            color_to_use = 1
            for c in different_colors_of_neighbours:
                if c == color_to_use:
                    color_to_use += 1
            coloring_dict.append({"vertex": vertex_to_color, "color": color_to_use})
        return self.objective_function([c["color"] for c in coloring_dict])


class HoffmanBound(GraphColoring):
    """ 
    Class for obtaining lower bounds for graph coloring problem instances using Hoffman's bound. 
    """

    def construct_adjacency_matrix(self, graph: Graph):
        """ The adjacency matrix will be symmetric as we only work with undirected graphs. """
        adjacency_matrix = np.zeros((graph.number_vertices, graph.number_vertices))
        for e in graph.edges:
            """ Vertices are labeled by positive integers whereas list elements start at index 0. """
            adjacency_matrix[graph.vertices.index(e[0]), graph.vertices.index(e[1])] = 1
            adjacency_matrix[graph.vertices.index(e[1]), graph.vertices.index(e[0])] = 1
        return adjacency_matrix

    def hoffman_lower_bound(self, partial_coloring: list):
        graph = self.equivalent_subgraph(partial_coloring)
        #print("Partial coloring = ", partial_coloring)
        #print("Graph for contracting = ", graph)
        adjacency_matrix = self.construct_adjacency_matrix(graph)
        #print("Adjacency matrix = ", adjacency_matrix)
        eigenvalues = np.linalg.eigvalsh(adjacency_matrix)
        max_eigenvalue = max(eigenvalues)
        min_eigenvalue = min(eigenvalues)
        return 1 - max_eigenvalue / min_eigenvalue


class FeasibilityAndPruning(HoffmanBound):

    def is_feasible(self, partial_coloring: list):
        for i in range(len(partial_coloring)):
            neighbours = self.find_neighbours(self.graph.vertices[i], self.graph)
            if any([j for j in range(len(partial_coloring)) if self.graph.vertices[j] in neighbours and partial_coloring[j] == partial_coloring[i]]):
            #if any([c2 for c2 in partial_coloring if c2["vertex"] in neighbours and c2["color"] == c["color"]]):
                return False
        return True

    def can_be_pruned(self, partial_coloring: list, lower_bound: Union[int, float], best_upper_bound: Union[int, float], is_first_solution: bool):
        if len(partial_coloring) == 0:
            raise ValueError("There is no node to investigate.")
        if is_first_solution:
            return True if (lower_bound > best_upper_bound) else False
        else:
            return True if (lower_bound >= best_upper_bound) else False


class BranchingSearchingBacktracking(GraphColoring):

    def branching(self, stack: List[list], branching_node: list = []):
        """ The number of children for a selected node will be the number of different colors used plus 1. """
        max_used_color = max(branching_node) if len(branching_node) != 0 else 0
        for color in range(1, max_used_color + 2):
            stack.append(branching_node + [color])
        return stack

    def select_most_promising_child(self, children: List[list]):
        
        """
        Children are first filtered by the least amount of different colors used.
        In case of ties, the remaining children are filtered by the smallest number of duplicates.
        In case of further ties, a random choice is made.
        """

        def find_min_different_colors_child(children: List[list]): 
            child_difcol_pairs = []
            for child in children:
                child_difcol_pairs.append({"child": child, "number of different colors": len(self.find_different_colors(child))})
            min_different_colors = min([p["number of different colors"] for p in child_difcol_pairs])
            return [p["child"] for p in child_difcol_pairs if p["number of different colors"] == min_different_colors]

        def find_minimax_color_duplicates_child(children: List[list]):
            child_max_pairs = []
            for child in children:
                child_max_pairs.append({"child": child, "max color duplicates": max([child.count(c) for c in child])})
            minimax_color_duplicates = min([p["max color duplicates"] for p in child_max_pairs])
            return [p["child"] for p in child_max_pairs if p["max color duplicates"] == minimax_color_duplicates]

        candidates_by_different_colors = find_min_different_colors_child(children)
        if len(candidates_by_different_colors) == 1:
            return candidates_by_different_colors[0]
        candidates_by_minimax_duplicates = find_minimax_color_duplicates_child(candidates_by_different_colors)
        if len(candidates_by_minimax_duplicates) == 1:
            return candidates_by_minimax_duplicates[0]
        return random.choice(candidates_by_minimax_duplicates)
    

    def node_selection(self, stack: List[list]):
        """ Selects the child corresponding to the color which is least used. """
        if len(stack) == 0:
            raise ValueError("No node to select.")
        max_child_color = stack[-1][-1]
        children_without_new_color = stack[(- max_child_color):]
        return self.select_most_promising_child(children_without_new_color)
        
    
    def backtracking(self, stack: List[list], current_node: list):
        """ Selects the child with the least max color duplicates or last element of stack. """
        if len(stack) == 0:
            raise ValueError("No further node to explore.")
        remaining_children = []
        i = 1
        while i <= len(stack) and len(stack[-i]) == len(current_node):
            remaining_children.append(stack[-i])
            i += 1
        if len(remaining_children) == 0:
            return stack[-1]
        remaining_children.reverse() # Just by convenience the remaining children are sorted according to their creation order in branching
        return self.select_most_promising_child(remaining_children)




"""
Refactoring needed: First methods of DSatur don't take a graph as input parameter but instead always use the whole
graph given by the problem instance. Hence, find_neighbours here takes self.graph as input. This can / should be
made more efficient as the remaining graph to consider shrinks to size zero in the procedure of the B&B algorithm.
The way it is implemented now, e.g. for degree of saturation, colors of neighbours and selection of next vertex in
the DSatur algorithm always the whole graph is considered.

Maybe when starting to implement the B&B algorithm, every input needs to be turned into a (partial) coloring. 
-> I think this is fine as the method dsatur_upper_bound is obtaining the coloring as input, which is the only
function that will be called from outside the class.

In B&B algorithm: Can stop exploring certain path if lower bound = upper bound, then corresponding chromatic number found.
"""
            
