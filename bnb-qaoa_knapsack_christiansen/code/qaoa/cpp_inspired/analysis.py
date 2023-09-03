import sys
sys.path.append("C:\\Users\\d92474\\Documents\\Uni\\Master Thesis\\GitHub\\MasterThesis_Hybrid-Quantum-Classical-Branch-and-Bound\\bnb-qaoa_knapsack_christiansen\\code")
sys.path.append("C:\\Users\\d92474\\Documents\\Uni\\Master Thesis\\GitHub\\MasterThesis_Hybrid-Quantum-Classical-Branch-and-Bound\\common-utils")

import numpy as np
import time

from circuit import QuasiAdiabaticEvolution
from knapsack_problem import KnapsackProblem, exemplary_kp_instances
from qaoa import QAOA, ProblemType



class ProblemRelatedFunctions:

    def __init__(self, problem_instance: KnapsackProblem):
        self.problem_instance = problem_instance

    def bitstring_to_choice(self, bitstring: str):
        bits = np.array(list(map(int, list(bitstring))))
        item_choice = np.array(bits[:self.problem_instance.number_items])
        return item_choice
    
    def objective_function(self, bitstring: str):
        item_choice = self.bitstring_to_choice(bitstring)
        value = np.dot(item_choice, self.problem_instance.profits)
        return value



class QAOAKnapsack(ProblemRelatedFunctions):

    def __init__(self, problem_instance: KnapsackProblem, depth: int):
        self.problem_instance = problem_instance
        self.depth = depth
        self.circuit = QuasiAdiabaticEvolution(self.problem_instance, self.depth)
        self.item_register_size = self.problem_instance.number_items
        self.capacity_register_size = int(np.floor(np.log2(self.problem_instance.capacity)) + 1)
        self.qaoa = QAOA(ProblemType.maximization, self.objective_function, self.circuit.apply_quasiadiabatic_evolution, depth, self.item_register_size, self.capacity_register_size)

    def execute_qaoa(self):
        return self.qaoa.execute_qaoa()




def main():
    problem = exemplary_kp_instances["C"]
    #print(ProblemRelatedFunctions(problem).objective_function("10010000000"))
    start_time = time.time()
    print("QAOA result = ", QAOAKnapsack(problem_instance = problem, depth = 1).execute_qaoa())
    print("Elapsed time = ", time.time() - start_time)


if __name__ == "__main__":
    main()