#include <pybind11/pybind11.h>
#include "mult.hpp"

namespace py = pybind11;

PYBIND11_MODULE(tokenizers_cpp, m) {
    m.doc() = "fast tokenizers built with c++";

    m.def("add", &add, "add two numbers");
    m.def("sub", &sub, "subtract two numbers");
    m.def("mult", &mult, "subtract two numbers");
}