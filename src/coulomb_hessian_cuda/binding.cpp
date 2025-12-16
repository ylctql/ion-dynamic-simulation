#include "coulomb_hessian.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

py::array_t<double> py_coulomb_hessian(
    py::array_t<double> py_pos,
    py::array_t<double> py_charges,
    double prefactor = 1.0)
{
    py::buffer_info pos_buf = py_pos.request();
    py::buffer_info q_buf   = py_charges.request();

    if (pos_buf.ndim != 1 || pos_buf.shape[0] % 3 != 0)
        throw std::runtime_error("positions must be 1D array with length 3*N");
    size_t N = pos_buf.shape[0] / 3;

    if (q_buf.ndim != 1 || q_buf.shape[0] != static_cast<ssize_t>(N))
        throw std::runtime_error("charges must have length N");

    auto result = py::array_t<double>({3*N, 3*N});
    double* ptr = result.mutable_data();

    compute_coulomb_hessian_cuda(
        static_cast<const double*>(pos_buf.ptr),
        static_cast<const double*>(q_buf.ptr),
        N, prefactor, ptr);

    return result;
}

PYBIND11_MODULE(coulomb_hessian, m) {
    m.doc() = "Ultra-fast Coulomb Hessian on GPU";
    m.def("hessian", &py_coulomb_hessian,
          "Compute Coulomb Hessian matrix",
          py::arg("positions"), py::arg("charges"), py::arg("prefactor")=1.0);
}