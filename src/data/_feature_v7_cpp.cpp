// V7 spatial channels — C++ assembly wrapper.
// Pure-array ops (ch0/1/2/6/9-13/zeros) in C++, delegates to Python for
// liberties (scipy), ladders, and area.
// Full C++ port of ladder+area is a future task.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cstring>
#include <vector>

namespace py = pybind11;

static constexpr int BOARD_SIZE = 19;
static constexpr int N = BOARD_SIZE * BOARD_SIZE;
static constexpr int SPATIAL_CHANNELS = 22;

// ---------------------------------------------------------------------------
// Pure C++ channel assembly
// ---------------------------------------------------------------------------
static void fill_ch012(py::array_t<float>& out,
                       const py::array_t<int>& boards) {
    auto ob = boards.unchecked<3>();
    auto ot = out.mutable_unchecked<4>();
    const py::ssize_t B = ob.shape(0);
    for (py::ssize_t b = 0; b < B; ++b) {
        for (int i = 0; i < N; ++i) {
            ot(b, 0, i / 19, i % 19) = 1.0f;
            int v = ob(b, i / 19, i % 19);
            ot(b, 1, i / 19, i % 19) = (v > 0) ? 1.0f : 0.0f;
            ot(b, 2, i / 19, i % 19) = (v < 0) ? 1.0f : 0.0f;
        }
    }
}

static void fill_ch6(py::array_t<float>& out,
                     const py::array_t<int>& kos) {
    auto ok = kos.unchecked<1>();
    auto ot = out.mutable_unchecked<4>();
    const py::ssize_t B = ok.size();
    for (py::ssize_t b = 0; b < B; ++b) {
        int ko = ok(b);
        if (ko >= 0 && ko < N) {
            ot(b, 6, ko / 19, ko % 19) = 1.0f;
        }
    }
}

static void fill_ch9_13(py::array_t<float>& out,
                        const py::array_t<int>& my_hist,
                        const py::array_t<int>& op_hist) {
    auto omy = my_hist.unchecked<2>();
    auto oop = op_hist.unchecked<2>();
    auto ot = out.mutable_unchecked<4>();
    const py::ssize_t B = omy.shape(0);
    for (py::ssize_t b = 0; b < B; ++b) {
        // ch9=opp[0], ch10=pla[0], ch11=opp[1], ch12=pla[1], ch13=opp[2]
        int locs[5];
        locs[0] = (int)oop(b, 0);
        locs[1] = (int)omy(b, 0);
        locs[2] = (int)oop(b, 1);
        locs[3] = (int)omy(b, 1);
        locs[4] = (int)oop(b, 2);
        for (int ch = 0; ch < 5; ++ch) {
            int loc = locs[ch];
            if (loc >= 0 && loc < N) {
                ot(b, 9 + ch, loc / 19, loc % 19) = 1.0f;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Copy helpers
// ---------------------------------------------------------------------------
static void copy_from_float4(py::array_t<float>& out,
                              const py::array_t<float>& src,
                              int out_ch_start, int n_ch) {
    auto os = src.unchecked<4>();
    auto ot = out.mutable_unchecked<4>();
    const py::ssize_t B = os.shape(0);
    for (py::ssize_t b = 0; b < B; ++b) {
        for (int ch = 0; ch < n_ch; ++ch) {
            for (int i = 0; i < N; ++i) {
                ot(b, out_ch_start + ch, i / 19, i % 19) = os(b, ch, i / 19, i % 19);
            }
        }
    }
}

static void copy_area_to_float(py::array_t<float>& out,
                                const py::array_t<int>& area) {
    auto oa = area.unchecked<4>();
    auto ot = out.mutable_unchecked<4>();
    const py::ssize_t B = oa.shape(0);
    for (py::ssize_t b = 0; b < B; ++b) {
        for (int i = 0; i < N; ++i) {
            ot(b, 18, i / 19, i % 19) = (oa(b, 0, i / 19, i % 19) > 0) ? 1.0f : 0.0f;
            ot(b, 19, i / 19, i % 19) = (oa(b, 1, i / 19, i % 19) > 0) ? 1.0f : 0.0f;
        }
    }
}

// ---------------------------------------------------------------------------
// Main entry point
// ---------------------------------------------------------------------------
py::object spatial_spatial(
    py::array_t<int> boards,
    py::array_t<int> to_play,
    py::array_t<int> ko,
    py::array_t<int> my_hist,
    py::array_t<int> op_hist,
    py::array_t<int> prev_board,
    py::array_t<int> prev_prev_board,
    py::array_t<int> prev_valid,
    py::array_t<int> prev_prev_valid,
    int rules_flags) {

    if (boards.ndim() != 3 || boards.shape(1) != BOARD_SIZE || boards.shape(2) != BOARD_SIZE) {
        throw std::invalid_argument("boards must be (B,19,19) int");
    }

    std::vector<py::ssize_t> shape = {(py::ssize_t)boards.shape(0), SPATIAL_CHANNELS,
                                       BOARD_SIZE, BOARD_SIZE};
    py::array_t<float> out(shape);
    std::memset(out.mutable_data(), 0, out.nbytes());

    // Pure C++ assembly
    fill_ch012(out, boards);
    fill_ch6(out, ko);
    fill_ch9_13(out, my_hist, op_hist);

    // Delegate to Python (static to avoid re-import on each call)
    static py::object fv = py::module_::import("src.data.feature_v7");

    // ch3/4/5 = liberties_123
    py::array_t<float> lib = fv.attr("liberties_123")(boards).cast<py::array_t<float>>();
    copy_from_float4(out, lib, 3, 3);

    // ch14/15/16/17 = ladder_channels
    py::array_t<float> lc = fv.attr("ladder_channels")(
        boards, to_play, prev_board, prev_prev_board, ko
    ).cast<py::array_t<float>>();
    copy_from_float4(out, lc, 14, 4);

    // ch18/19 = calculate_area
    py::array_t<int> area = fv.attr("calculate_area")(
        boards, to_play, rules_flags
    ).cast<py::array_t<int>>();
    copy_area_to_float(out, area);

    // ch7, ch8, ch20, ch21 remain zero

    // Cast to float16
    py::module_ np = py::module_::import("numpy");
    py::object out_obj = py::reinterpret_borrow<py::object>(out.ptr());
    return out_obj.attr("astype")(np.attr("float16"));
}

PYBIND11_MODULE(_feature_v7_cpp, m) {
    m.doc() = "V7 spatial channels - C++ assembly wrapper";
    m.def("spatial_spatial", &spatial_spatial,
          "Compute 22-channel V7 spatial features",
          py::arg("boards"), py::arg("to_play"), py::arg("ko"),
          py::arg("my_hist"), py::arg("op_hist"),
          py::arg("prev_board"), py::arg("prev_prev_board"),
          py::arg("prev_valid"), py::arg("prev_prev_valid"),
          py::arg("rules_flags") = 0);

    // Fix pytest cleanup: empty capsule has no destructor, avoids GIL crash
    py::module_ sys = py::module_::import("atexit");
    sys.attr("register")(py::cpp_function([]() {}));
}
