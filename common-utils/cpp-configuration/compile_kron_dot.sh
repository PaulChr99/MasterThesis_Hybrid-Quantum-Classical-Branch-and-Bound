#!/bin/bash

c++ -O3 -Ofast -march=native -shared -std=c++20 -fPIC $(python3 -m pybind11 --includes) kron_dot.cpp -o kron_dot$(python3-config --extension-suffix)
