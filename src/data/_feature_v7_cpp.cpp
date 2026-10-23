// Minimal C++ extension stub — compiles and imports successfully.
// Real implementation comes in Task 2.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

// Forward declaration — filled in Task 2
py::array_t<float> spatial_spatial(
    py::array_t<int8_t> boards,
    py::array_t<int8_t> to_play,
    py::array_t<int16_t> ko,
    py::array_t<int32_t> my_hist,
    py::array_t<int32_t> op_hist,
    py::array_t<int8_t> prev_board,
    py::array_t<int8_t> prev_prev_board,
    py::array_t<bool> prev_valid,
    py::array_t<bool> prev_prev_valid,
    int rules_flags);

PYBIND11_MODULE(_feature_v7_cpp, m) {
    m.def("spatial_spatial", &spatial_spatial, "V7 spatial channels (C++ backbone)");
}
