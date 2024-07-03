#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <memory>
#include "BPETokenizer.hpp"

namespace py = pybind11;

PYBIND11_MODULE(tokenizers_cpp, m) {
    m.doc() = "fast tokenizers built with c++";

    py::class_<BPETokenizer, std::shared_ptr<BPETokenizer> > tok(m, "BPETokenizer");
    tok.def(py::init<uint32_t, int, int, int>(),
        py::arg("vocab_size"), 
        py::arg("eos_token") = 1000000,
        py::arg("unk_token") = 1000001,
        py::arg("bos_token") = 1000002);
    tok.def("train", &BPETokenizer::train, py::arg("text"));
    tok.def("encode", &BPETokenizer::encode, py::arg("text"));
    tok.def("decode", &BPETokenizer::decode, py::arg("tokens"));
    tok.def("tok_is_trained", &BPETokenizer::tok_is_trained);
    tok.def("get_vocab", &BPETokenizer::get_vocab);
    tok.def("save", &BPETokenizer::save, py::arg("dir"));
    tok.def_static("from_pretrained", &BPETokenizer::from_pretrained, py::arg("dir"));
}