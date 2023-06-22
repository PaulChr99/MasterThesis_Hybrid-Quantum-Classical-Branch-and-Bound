import time
import numpy as np
import kron_dot

# important! declare the data type as defined in C++ file; np.cdouble for double and np.csingle for float
# otherwise, the C++ function will silently fail!
num_type = np.cdouble 

n_qubits = 25
x_matrix = np.array([[0, 1], [1, 0]], dtype=num_type) # declare all arrays with dtype=num_type
state = np.zeros(2**n_qubits, dtype=num_type)
state[0] = 1 # begin with the all zero state |0...0>
# access all elements once to allocate memory (somehow numpy doesn't actually allocate memory until it is accessed)
np.sum(state) 

gate_start = 1 # last qubit the gate acts on, counted from the left
gate_start_right = n_qubits - gate_start - 1 # first qubit the gate acts on, counted from the right
t_start = time.time()
kron_dot.kron_dot_dense(gate_start, x_matrix, state)
print("Time elapsed: %fs" % (time.time() - t_start))
# we expect |010...0> = |8388608> (8388608 = 2**23) now for n_qubits = 25:
print("Correct result:", np.where(state == 1)[0][0] == 2**gate_start_right)



