//
// Created by martin on 02.02.23.
//
#include <complex>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

// Very important: Do not forget to use the correct data type in python corresponding to the typedefs below!
// Otherwise, everything fails silently!
// This would be np.csingle for num_type = float and np.cdouble for num_type = double.
typedef long long ind_t; // type for state vector indices, a 32-bit int is not big enough for > 32 qubits!
typedef double num_type; // base number type for all calculations
typedef std::complex<num_type> ncomplex; // complex number type for matrix and state vector elements
typedef std::vector<ncomplex> cvec; // complex vector type for C++ internal calculations
template<typename T> using nparr = py::array_t<T>; // data type for numpy arrays
typedef nparr<ncomplex> cnparr; // data type for complex numpy arrays;

// calculate 2 to a given power and casts it to ind_t
ind_t pow2int(unsigned int exponent) {
    return static_cast<ind_t>(pow(2, exponent));
}

// This calculates matrix-vector product of a given kronecker product of the form I ⊗ ... ⊗ I ⊗ M ⊗ I ⊗ ... ⊗ I
// applied to a numpy array v_in.
// We pass the number of qubits n_qubits of the whole state vector, the first qubit the gate acts on (gate_start) and
// the last qubit the gate acts on (gate_end).
// The input matrix m_in and the input vector v_in must have to correct complex data type! This means that we always
// have to create matrices and arrays in numpy with dtype=... (see the explanation above the typedefs below).
// Everything is passed as a reference, so we do not have to copy any memory.
// We basically use the algorithm used in the Intel quantum simulator (https://arxiv.org/abs/1601.07195).
void
kron_dot_dense(int &gate_start, cnparr &m_in, cnparr &v_in) {
    // to access the numpy arrays with full speed, we have to create a memory view without bounds checking
    auto m = m_in.unchecked<2>(); // unchecked<2> creates a 2D memory view (a matrix)
    // mutable_unchecked creates a memory view that can also be modified
    auto v = v_in.mutable_unchecked<1>();

    int n_qubits = static_cast<int>(log2(v.size()));
    int n_gate_qubits = static_cast<int>(log2(sqrt(m.size())));
    ind_t gate_size = pow2int(n_gate_qubits);
    int gate_end = gate_start + n_gate_qubits - 1;

    // We count qubits from the left, so if we have the ket |0000> and we apply the X gate to the qubit with id 1,
    // we get |0100>. To get the correct step sizes, it is easier to count from the right, so we convert it first.
    int gate_start_right = n_qubits - gate_end - 1;
    ind_t step1 = pow2int(gate_start_right + n_gate_qubits);
    ind_t step2 = pow2int(gate_start_right);

    ind_t i, j, row, col;
    cvec tmp(gate_size); // this holds the output of the matrix-vector product on the subspace
    for (i = 0; i < v.size(); i += step1) {
        for (j = i; j < i + step2; ++j) {
            // this is the main loop body where everything happens
            // given a row index of the gate, we can now get the correct state vector index as j + row * step2
            // copy the elements of the current subspace to our temporary vector
            for (row = 0; row < gate_size; ++row)
                tmp[row] = v[j + row * step2];
            // then we calculate the matrix-vector product and directly write it into the state vector
            for (row = 0; row < gate_size; ++row) {
                v[j + row * step2] = 0;
                for (col = 0; col < gate_size; ++col)
                    v[j + row * step2] += m(row, col) * tmp[col]; // access pybind11 matrix using round brackets
            }
        }
    }
}

// This creates the function bindings where we have to define each function that we want to call from python.
void init_kron_dot(py::module_ & m) {
    m.def("kron_dot_dense", &kron_dot_dense);
}


// This function finally creates the python module that we import later, the name is given as the first argument.
// It is recommended to put this in another file for faster compilation but we leave it here for simplicity.
PYBIND11_MODULE(kron_dot, m) {
    m.doc() = "Fast matrix vector products for square matrices built using Kronecker products with identity factors I ⊗ M ⊗ I."; // optional module docstring
    init_kron_dot(m);
}
