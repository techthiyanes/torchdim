#include "minpybind.h"
#include <frameobject.h>
#include <opcode.h>
#include <utility>
#include <new>
#include <iostream>
#include <vector>
#include <torch/csrc/autograd/python_variable.h>
#include <ATen/ATen.h>
#include "arena.h"

// C++ API functions for objects to
// * construct the object, returning a ref-counted handle
// * The actual API, with methods that take/return C-typed values

// extend minpybind.h to include
// * typed handles so that -> can get to their raw API
// * object/handle distinction for the typed handles

// class Dim: ---------------


py::handle DimensionBindError_;
static py::handle DimensionBindError() {
    if(!DimensionBindError_.ptr()) {
        DimensionBindError_ = py::import("dim").attr("DimensionBindError");
    }
    return DimensionBindError_;
}

constexpr int MAX_LEVELS_IN_USE = 32;
static int n_levels_in_use = 0;
static py::handle levels_in_use[MAX_LEVELS_IN_USE];


PyTypeObject* DimType = nullptr;
struct Dim : public py::base<Dim> {
    int level_; // for stable comparisons in prototype
    py::object name_;
    Dim()
    : level_(n_levels_in_use++) {
        // weakref, cleared when this is destructed
        levels_in_use[level_] = ptr();
    }
    ~Dim() {
        AT_ASSERT(levels_in_use[level_].ptr() == ptr());
        levels_in_use[level_] = nullptr;
        while(n_levels_in_use > 0 && levels_in_use[n_levels_in_use - 1].ptr() == nullptr) {
            --n_levels_in_use;
        }
    }
    void init(py::object name, int64_t s = -1) {
        name_ = std::move(name);
        size_ = s;
    }
    int64_t size() const {
        if (size_ == -1) {
            py::raise_error(PyExc_ValueError, "dimension %S is unbound", name_.ptr());
        }
        return size_;
    }
    void set_size(int64_t v) {
        if (size_ == -1) {
            size_ = v;
        } else if(size_ != v) {
            py::raise_error(DimensionBindError(), "Dim '%R' previously bound to a dimension of size %lld cannot bind to a dimension of size %lld", this, this->size_, v);
        }
    }
    bool is_bound() const {
        return size_ != -1;
    }
    static py::obj<Dim> create(py::object name, int64_t s = -1) {
        if (!DimType) {
            DimType = (PyTypeObject*) py::import("dim").attr("Dim").ptr();
        }
        auto r = Dim::alloc(DimType);
        r->init(std::move(name), s);
        return r;
    }
    static PyTypeObject Type;
private:
    int64_t size_;
};


// Dim wrapper methods

static int Dim_init(py::hdl<Dim> self, PyObject *args, PyObject *kwds) {
    PY_BEGIN
    static char* kwlist[] = {"name", "size", nullptr};
    py::handle name;
    py::handle size = nullptr;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|O", kwlist, &name, &size)) {
        return -1;
    }
    self->init(py::object::borrow(name), (size.ptr() && !py::is_none(size)) ? py::to_int(size) : -1);
    return 0;
    PY_END(-1)
}

static PyObject* Dim_repr(Dim* self) {
    PY_BEGIN
    py::object name = (self->name_.ptr()) ? self->name_ : py::unicode_from_string("<uninitialized dim>");
    return name.release();
    PY_END(nullptr)
}


static PyObject* Dim_getsize(Dim* self, void*) {
    PY_BEGIN
    return py::from_int(self->size()).release();
    PY_END(nullptr)
}

int Dim_setsize(Dim* self, PyObject* size, void*) {
    PY_BEGIN
    self->set_size(py::to_int(size));
    return 0;
    PY_END(-1)
}

static PyObject* Dim_getis_bound(Dim* self, void*) {
    return PyBool_FromLong(self->is_bound());
}

static PyObject* Dim_getlevel(Dim* self, void*) {
    return PyLong_FromLong(31 + self->level_);
}


static PyGetSetDef Dim_getsetters[] = {
    {"size", (getter) Dim_getsize, (setter) Dim_setsize,
     "Dimension size", NULL},
    {"is_bound", (getter) Dim_getis_bound, NULL, "is_bound", NULL},
    {"_level", (getter) Dim_getlevel, NULL, "_level", NULL},
    {NULL}  /* Sentinel */
};

PyTypeObject Dim::Type = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "_C.Dim",               /* tp_name */
    sizeof(Dim),               /* tp_basicsize */
    0,                              /* tp_itemsize */
    Dim::dealloc_stub,      /* tp_dealloc */
    0,                              /* tp_vectorcall_offset */
    0,                              /* tp_getattr */
    0,                              /* tp_setattr */
    0,                              /* tp_as_async */
    (reprfunc)Dim_repr,           /* tp_repr */
    0,                 /* tp_as_number */
    0,                              /* tp_as_sequence */
    0,                              /* tp_as_mapping */
    0,      /* tp_hash */
    0,                              /* tp_call */
    0,                              /* tp_str */
    0,                              /* tp_getattro */
    0,                              /* tp_setattro */
    0,                              /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,  /* tp_flags */
    "Dim Object",                   /* tp_doc */
    0,                              /* tp_traverse */
    0,                              /* tp_clear */
    0,  /* tp_richcompare */
    0,                              /* tp_weaklistoffset */
    0,                              /* tp_iter */
    0,                              /* tp_iternext */
    0,                              /* tp_methods */
    0,                              /* tp_members */
    Dim_getsetters,                 /* tp_getset */
    0,                              /* tp_base */
    0,                              /* tp_dict */
    0,                              /* tp_descr_get */
    0,                              /* tp_descr_set */
    0,                              /* tp_dictoffset */
    (initproc) Dim_init,            /* tp_init */
    0,                              /* tp_alloc */
    Dim::new_stub,                      /* tp_new */
};


py::handle rich_comparison_fns[6];
const char* rich_comparison_table[] {
    "__lt__",
    "__le__",
    "__eq__",
    "__ne__",
    "__gt__",
    "__ge__"
};

// static PyObject *Dim_richcompare(Dim *self, PyObject *other, int op) {
//     PY_BEGIN
//         if (Py_TYPE(other) == &Dim::Type && (op == Py_EQ || op == Py_NE))  {
//             Py_RETURN_RICHCOMPARE( (void*)self, (void*) other, op);
//         } else {
//             return Tensor_richcompare((PyObject*)self, other, op);
//         }
//     PY_END(nullptr)
// }

// class DimList ------------

struct DimList : public py::base<DimList> {
    py::object name_;
    std::vector<py::obj<Dim>> dims_;
    static PyTypeObject Type;
    void init(py::object name) {
        name_ = std::move(name);
    }
    void set_dims(std::vector<py::obj<Dim>> dims) {
        bound_ = true;
        dims_ = std::move(dims);
    }
    bool is_bound() {
        return bound_;
    }
    void bind_len(int64_t size) {
        if (bound_) {
            int64_t b_size = dims_.size();
            if (b_size != size) {
                py::raise_error(DimensionBindError(), "Dimlist has size %lld but it is being bound to size %d", b_size, size);
            }
        } else {
            bound_ = true;
            dims_.resize(size);
            for (ssize_t i = 0; i < size; ++i) {
                dims_[i] = Dim::create(py::unicode_from_format("%S%i", name_.ptr(), (int)i));
            }
        }
    }
    int64_t size() const {
        if (!bound_) {
            py::raise_error(DimensionBindError(), "DimList not bound");
        }
        return dims_.size();
    }
    void set_bound(bool b) {
        bound_ = b;
    }
private:
    bool bound_ = false;
};


static int DimList_init(DimList *self, PyObject *args, PyObject *kwds);

static PyObject* DimList_repr(DimList* self) {
    PY_BEGIN
    if (self->is_bound()) {
        size_t size = self->dims_.size();
        py::tuple t(size);
        for(size_t i = 0; i < size; ++i) {
            t.set(i, self->dims_[i]);
        }
        return py::repr(t).release();
    } else if(!py::is_none(self->name_)) {
        return py::unicode_from_format("*%S", self->name_.ptr()).release();
    } else {
        return py::unicode_from_string("<unbound_dimlist>").release();
    }
    PY_END(nullptr)
}

static PyObject* DimList_bind(DimList *self,
                      PyObject *const *args,
                      Py_ssize_t nargs,
                      PyObject *kwnames) {
    PY_BEGIN
    py::handle sizes;
    static const char * const _keywords[] = {"sizes", nullptr};
    static _PyArg_Parser parser = {"O", _keywords, 0};
    if (!_PyArg_ParseStackAndKeywords(args, nargs, kwnames, &parser, &sizes)) {
        return nullptr;
    }
    if (!py::is_sequence(sizes)) {
        py::raise_error(PyExc_ValueError, "expected a sequence");
    }
    py::sequence_view seq = sizes;
    auto size = seq.size();
    self->bind_len(size);
    for (ssize_t i = 0; i < size; ++i) {
        self->dims_[i]->set_size(py::to_int(seq[i]));
    }
    Py_RETURN_NONE;
    PY_END(nullptr)
}

static PyObject* DimList_bind_len(DimList *self,
                      PyObject *const *args,
                      Py_ssize_t nargs,
                      PyObject *kwnames) {
    PY_BEGIN
    int size;
    static const char * const _keywords[] = {"N", nullptr};
    static _PyArg_Parser parser = {"i", _keywords, 0};
    if (!_PyArg_ParseStackAndKeywords(args, nargs, kwnames, &parser, &size)) {
        return nullptr;
    }
    self->bind_len(size);
    Py_RETURN_NONE;
    PY_END(nullptr)
}

static PyMethodDef DimList_methods[] = {
    {"bind", (PyCFunction) DimList_bind, METH_FASTCALL | METH_KEYWORDS},
    {"bind_len", (PyCFunction) DimList_bind_len, METH_FASTCALL | METH_KEYWORDS},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};


static Py_ssize_t DimList_len(DimList* self) {
    PY_BEGIN
    return self->size();
    PY_END(-1)
}

PyObject * DimList_item(DimList* self, Py_ssize_t idx) {
    PY_BEGIN
    if (!self->is_bound()) {
        py::raise_error(DimensionBindError(), "DimList not bound");
    }
    if (idx >= self->dims_.size()) {
        py::raise_error(PyExc_IndexError, "index out of bounds");
    }
    py::object r = self->dims_[idx];
    return r.release();
    PY_END(nullptr)
}

PySequenceMethods DimList_seq {
    (lenfunc) DimList_len, //lenfunc sq_length;
    0, //binaryfunc sq_concat;
    0, //ssizeargfunc sq_repeat;
    (ssizeargfunc) DimList_item, //ssizeargfunc sq_item;
    0, //void *was_sq_slice;
    0, //ssizeobjargproc sq_ass_item;
    0, //void *was_sq_ass_slice;
    0, //objobjproc sq_contains;

    0, //binaryfunc sq_inplace_concat;
    0, //ssizeargfunc sq_inplace_repeat;
};

static PyObject* DimList_getis_bound(DimList* self, void*) {
    return PyBool_FromLong(self->is_bound());
}

static PyGetSetDef DimList_getsetters[] = {
    {"is_bound", (getter) DimList_getis_bound, NULL, "is_bound", NULL},
    {NULL}  /* Sentinel */
};


static PyObject* DimList_subscript(DimList* self, py::handle idx) {
    PY_BEGIN
    if (py::is_int(idx)) {
        return DimList_item(self, py::to_int(idx));
    } else if (py::is_slice(idx)) {
        if (!self->is_bound()) {
            py::raise_error(DimensionBindError(), "DimList not bound");
        }
        py::slice_view s(idx, self->dims_.size());
        py::tuple r(s.slicelength);
        for (Py_ssize_t i = s.start, j = 0; i < s.stop; i += s.step) {
            r.set(j++,  self->dims_[i]);
        }
        return r.release();
    } else {
        py::raise_error(PyExc_ValueError, "expected an int or a slice");
        return nullptr;
    }
    PY_END(nullptr)
}

PyMappingMethods DimList_mapping = {
    0, //lenfunc mp_length;
    (binaryfunc) DimList_subscript, //binaryfunc mp_subscript;
    0, //objobjargproc mp_ass_subscript;
};



PyTypeObject DimList::Type = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "_C.DimList",               /* tp_name */
    sizeof(DimList),               /* tp_basicsize */
    0,                              /* tp_itemsize */
    DimList::dealloc_stub,      /* tp_dealloc */
    0,                              /* tp_vectorcall_offset */
    0,                              /* tp_getattr */
    0,                              /* tp_setattr */
    0,                              /* tp_as_async */
    (reprfunc)DimList_repr,           /* tp_repr */
    0,                 /* tp_as_number */
    &DimList_seq,                 /* tp_as_sequence */
    &DimList_mapping,             /* tp_as_mapping */
    0,      /* tp_hash */
    0,                              /* tp_call */
    0,                              /* tp_str */
    0,                              /* tp_getattro */
    0,                              /* tp_setattro */
    0,                              /* tp_as_buffer */
    0,                              /* tp_flags */
    "DimList Object",                   /* tp_doc */
    0,                              /* tp_traverse */
    0,                              /* tp_clear */
    0,  /* tp_richcompare */
    0,                              /* tp_weaklistoffset */
    0,                              /* tp_iter */
    0,                              /* tp_iternext */
    DimList_methods,                /* tp_methods */
    0,                              /* tp_members */
    DimList_getsetters,             /* tp_getset */
    0,                              /* tp_base */
    0,                              /* tp_dict */
    0,                              /* tp_descr_get */
    0,                              /* tp_descr_set */
    0,                              /* tp_dictoffset */
    (initproc) DimList_init,            /* tp_init */
    0,                              /* tp_alloc */
    DimList::new_stub,                      /* tp_new */
};

static int DimList_init(DimList *self, PyObject *args, PyObject *kwds) {
    static char* kwlist[] = {"len_or_dims", "name", nullptr};
    py::handle len_or_dims = nullptr;
    PyObject* name = nullptr;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OO", kwlist, &len_or_dims, &name)) {
        return -1;
    }
    self->init(py::object::borrow(name ? name : Py_None));
    if (len_or_dims.ptr()) {
        if(py::is_int(len_or_dims)) {
            self->bind_len(py::to_int(len_or_dims));
        } else if (py::is_sequence(len_or_dims)) {
            py::sequence_view s(len_or_dims);
            std::vector<py::obj<Dim>> dims;
            size_t size = s.size();
            dims.reserve(size);
            for (ssize_t i = 0; i < size; ++i) {
                auto r = s[i];
                if (py::is_int(r)) {
                    dims.emplace_back(Dim::create(py::unicode_from_format("%S%i", self->name_.ptr(), (int)i),  py::to_int(r)));
                } else {
                    dims.emplace_back(Dim::wrap(r));
                }
            }
            self->set_dims(std::move(dims));
        } else {
            PyErr_Format(PyExc_ValueError, "expected a length or a sequence of dimensions");
            return -1;
        }
        return 0;
    }
    return 0;
}

// Tensor -----------------------------

PyTypeObject* TensorType = nullptr; // the python wrapper type.

struct Tensor : public py::base<Tensor> {
public:
    at::Tensor tensor_;
    at::Tensor batchtensor_;
    py::object levels_;
    bool has_device_;

    static PyTypeObject Type;

    static py::obj<Tensor> create() {
        if (!TensorType) {
            TensorType = (PyTypeObject*) py::import("dim").attr("Tensor").ptr();
        }
        return Tensor::alloc(TensorType);
    }
};

static int Tensor_init(Tensor *self, PyObject *args, PyObject *kwargs) {
    PY_BEGIN
    #define ARGS(_) _(py::handle, tensor) _(py::handle, levels) _(int, has_device) _(py::handle, batchtensor)
    MPY_PARSE_ARGS_KWARGS("OOpO", ARGS)

    if (!THPVariable_Check(tensor.ptr())) {
        py::raise_error(PyExc_ValueError, "_tensor is not a Tensor?");
    }
    if (!THPVariable_Check(batchtensor.ptr())) {
        py::raise_error(PyExc_ValueError, "_batchtensor is not a Tensor?");
    }
    self->tensor_ = THPVariable_Unpack(tensor.ptr());
    self->levels_ = py::object::borrow(levels);
    self->has_device_ = has_device != 0;
    self->batchtensor_ = THPVariable_Unpack(batchtensor.ptr());
    return 0;
    PY_END(-1)
}

static PyGetSetDef Tensor_getsetters[] = {
   {"_has_device", (getter) [](PyObject* self, void*) -> PyObject* { return py::from_bool(((Tensor*)self)->has_device_).release(); }, NULL},
   {"_tensor", (getter) [](PyObject* self, void*) -> PyObject* { return THPVariable_Wrap(((Tensor*)self)->tensor_); }, NULL},
   {"_batchtensor", (getter) [](PyObject* self, void*) -> PyObject* { return THPVariable_Wrap(((Tensor*)self)->batchtensor_); }, NULL},
   {"_levels", (getter) [](PyObject* self, void*) -> PyObject* {
       py::object levels = ((Tensor*)self)->levels_;
       return levels.release();
   }},
    {NULL}  /* Sentinel */
};

static PyMethodDef Tensor_methods[] = {
    {NULL, NULL, 0, NULL}        /* Sentinel */
};


PyTypeObject Tensor::Type = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "_C.Tensor",               /* tp_name */
    sizeof(Tensor),               /* tp_basicsize */
    0,                              /* tp_itemsize */
    Tensor::dealloc_stub,      /* tp_dealloc */
    0,                              /* tp_vectorcall_offset */
    0,                              /* tp_getattr */
    0,                              /* tp_setattr */
    0,                              /* tp_as_async */
    0,           /* tp_repr */
    0,                 /* tp_as_number */
    0,                 /* tp_as_sequence */
    0,             /* tp_as_mapping */
    0,      /* tp_hash */
    0,                              /* tp_call */
    0,                              /* tp_str */
    0,                              /* tp_getattro */
    0,                              /* tp_setattro */
    0,                              /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE , /* tp_flags */
    "Tensor Object",                   /* tp_doc */
    0,                              /* tp_traverse */
    0,                              /* tp_clear */
    0,  /* tp_richcompare */
    0,                              /* tp_weaklistoffset */
    0,                              /* tp_iter */
    0,                              /* tp_iternext */
    Tensor_methods,                /* tp_methods */
    0,                              /* tp_members */
    Tensor_getsetters,             /* tp_getset */
    0,                              /* tp_base */
    0,                              /* tp_dict */
    0,                              /* tp_descr_get */
    0,                              /* tp_descr_set */
    0,                              /* tp_dictoffset */
    (initproc) Tensor_init,            /* tp_init */
    0,                              /* tp_alloc */
    Tensor::new_stub,                      /* tp_new */
};


// dim() --------------------

bool relevant_op(_Py_CODEUNIT c) {
    switch(_Py_OPCODE(c)) {
        case STORE_NAME:
        case STORE_GLOBAL:
        case STORE_FAST:
        case STORE_DEREF:
            return true;
        default:
            return false;
    }
}

py::object getname(PyCodeObject* code, _Py_CODEUNIT c) {
    PyObject* names = NULL;
    switch(_Py_OPCODE(c)) {
        case STORE_NAME:
        case STORE_GLOBAL:
          names = code->co_names;
          break;
        case STORE_FAST:
          names = code->co_varnames;
          break;
        case STORE_DEREF:
          names = code->co_cellvars;
          break;
        default:
            py::raise_error(PyExc_RuntimeError, "unknown bytecode");
    }
    return py::object::steal(PySequence_GetItem(names, _Py_OPARG(c)));
}

static PyObject* dims(PyObject *self,
                      PyObject *const *args,
                      Py_ssize_t nargs,
                      PyObject *kwnames) {
    PY_BEGIN
    int lists = 0;
    static const char * const _keywords[] = {"lists", nullptr};
    static _PyArg_Parser parser = {"|i", _keywords, 0};
    if (!_PyArg_ParseStackAndKeywords(args, nargs, kwnames, &parser, &lists)) {
        return nullptr;
    }
    PyThreadState* state = PyThreadState_GET();
    PyFrameObject* f = state->frame;
    auto code = (_Py_CODEUNIT*)PyBytes_AS_STRING(f->f_code->co_code);
    int first = f->f_lasti /  2 + 1;
    auto unpack = code[first];
    if (relevant_op(unpack)) {
        auto str = getname(f->f_code, unpack);
        return (lists) ? DimList::create(str).release() : Dim::create(str).release();
    }
    if (_Py_OPCODE(unpack) != UNPACK_SEQUENCE) {
        py::raise_error(PyExc_SyntaxError, "dims() must be assigned to a sequence of variable names");
    }

    auto ndims = _Py_OPARG(unpack);
    py::tuple result(ndims);
    for (int i = 0; i < ndims; ++i) {
        auto str = getname(f->f_code, code[first + 1 + i]);
        py::object item;
        if (i >= ndims - lists) {
            item = DimList::create(str);
        } else {
            item = Dim::create(str);
        }
        result.set(i, std::move(item));
    }
    return result.release();
    PY_END(nullptr)
}



int64_t dim_index(const std::vector<py::obj<Dim>>& dims, py::hdl<Dim> dim) {
    for (int64_t i = 0, N  = dims.size(); i < N; ++i) {
        if (dims[i].ptr() == dim.ptr()) {
            return i;
        }
    }
    return -1;
}


// py::obj<Tensor> dot(py::hdl<Tensor> lhs, py::hdl<Tensor> rhs, const std::vector<py::obj<Dim>>& sum) {
//     auto& lhs_dims  = lhs->dims();
//     auto& rhs_dims = rhs->dims();
//     auto lhs_ = lhs->data();
//     auto rhs_ = rhs->data();

//     at::IntArrayRef lhs_strides = lhs_.strides();
//     at::IntArrayRef rhs_strides = rhs_.strides();


//     int64_t max_size = lhs_dims.size() + rhs_dims.size();

//     int64_t lro_size = 1;
//     int64_t lro_dim = 0;

//     int64_t lo_size = 1;
//     int64_t lo_dim = 0;

//     int64_t ro_size = 1;
//     int64_t ro_dim = 0;

//     int64_t lr_size = 1;
//     int64_t lr_dim = 0;

//     std::vector<int64_t> n_lhs_sizes, n_lhs_strides, n_rhs_sizes, n_rhs_strides;
//     n_lhs_sizes.reserve(max_size);
//     n_rhs_sizes.reserve(max_size);
//     n_lhs_strides.reserve(max_size);
//     n_rhs_strides.reserve(max_size);

//     std::vector<py::obj<Dim>> o_dims;
//     std::vector<int64_t> o_size;
//     o_dims.reserve(max_size);
//     o_size.reserve(max_size);

//     auto insert = [&] (std::vector<int64_t>& arr, size_t i, int64_t v) {
//         arr.insert(arr.begin() + i, v);
//     };
//     auto insert_dim = [&] (py::hdl<Dim> d, int64_t lhs_idx, int64_t rhs_idx, bool sum) {
//         int64_t size = d->size();
//         int64_t lhs_stride = lhs_idx == -1 ? 0 : lhs_strides[lhs_idx];
//         int64_t rhs_stride = rhs_idx == -1 ? 0 : rhs_strides[rhs_idx];
//         if (sum) {
//             // lr
//             lr_size *= size;
//             int64_t l_idx = lro_dim + lo_dim + lr_dim;
//             int64_t r_idx = lro_dim + lr_dim;
//             insert(n_lhs_strides, l_idx, lhs_stride);
//             insert(n_lhs_sizes, l_idx, size);
//             insert(n_rhs_strides, r_idx, rhs_stride);
//             insert(n_rhs_sizes, r_idx, size);
//             lr_dim += 1;
//         } else {
//             if ((lhs_stride == 0) == (rhs_stride == 0)) {
//                 // lro
//                 insert(n_lhs_strides, lro_dim, lhs_stride);
//                 insert(n_lhs_sizes, lro_dim, size);
//                 insert(n_rhs_strides, lro_dim, rhs_stride);
//                 insert(n_rhs_sizes, lro_dim, size);
//                 insert(o_size, lro_dim, size);
//                 o_dims.insert(o_dims.begin() + lro_dim, py::obj<Dim>::borrow(d));
//                 lro_size *= size;
//                 lro_dim += 1;
//             } else if (lhs_stride != 0) {
//                 // lo
//                 int64_t idx = lro_dim + lo_dim;
//                 insert(n_lhs_strides, idx, lhs_stride);
//                 insert(n_lhs_sizes, idx, size);

//                 insert(o_size, idx, size);
//                 o_dims.insert(o_dims.begin() + idx, py::obj<Dim>::borrow(d));

//                 lo_size *= size;
//                 lo_dim += 1;
//             } else {
//                 AT_ASSERT(rhs_stride != 0);
//                 // ro
//                 int64_t idx = lro_dim + lr_dim + ro_dim;
//                 insert(n_rhs_strides, idx, rhs_stride);
//                 insert(n_rhs_sizes, idx, size);

//                 int64_t o_idx = lro_dim + lo_dim + ro_dim;
//                 insert(o_size,  o_idx, size);
//                 o_dims.insert(o_dims.begin() + o_idx, py::obj<Dim>::borrow(d));

//                 ro_size *= size;
//                 ro_dim += 1;
//             }
//         }
//     };

//     std::vector<bool> rhs_seen(rhs_dims.size(), false);

//     for (int64_t i = 0, N = lhs_dims.size(); i < N; ++i) {
//         py::hdl<Dim> d = lhs_dims[i];
//         auto rhs_idx = dim_index(rhs_dims, d);
//         if (rhs_idx != -1) {
//             rhs_seen[rhs_idx] = true;
//         }
//         auto s_idx = dim_index(sum, d);
//         insert_dim(d, i, rhs_idx, s_idx != -1);
//     }
//     for (int64_t i = 0, N = rhs_dims.size(); i < N; ++i) {
//         if (rhs_seen[i]) {
//             continue;
//         }
//         py::hdl<Dim> d = rhs_dims[i];
//         auto s_idx = dim_index(sum, d);
//         insert_dim(d, -1, i, s_idx != -1);
//     }

//     if (lr_dim != sum.size()) {
//         for (auto & d : sum) {
//             if (-1 == dim_index(o_dims, d)) {
//                 py::raise_error(DimensionBindError(), "summing over non-existant dimension %S", d.ptr());
//             }
//         }
//     }

//     lhs_ = lhs_.as_strided(n_lhs_sizes, n_lhs_strides, lhs_.storage_offset());
//     rhs_ = rhs_.as_strided(n_rhs_sizes, n_rhs_strides, rhs_.storage_offset());
//     lhs_ = lhs_.reshape({lro_size, lo_size, lr_size});
//     rhs_ = rhs_.reshape({lro_size, lr_size, ro_size});

//     auto result = at::bmm(lhs_, rhs_);
//     result = result.reshape(o_size);
//     return Tensor::create_subclass(std::move(o_dims), std::move(result), false);
// }

static PyObject* test_c(PyObject *self,
                      PyObject *const *args,
                      Py_ssize_t nargs,
                      PyObject *kwnames) {
    PY_BEGIN

    Arena A;
    Slice<int> s(A, 3, 4, 5);
    AT_ASSERT(s.size() == 3 && s.capacity() == 8);
    AT_ASSERT(s[0] == 3 && s[1] == 4 && s[2] == 5);
    s = s.append(A, 6);
    AT_ASSERT(s[3] == 6);
    for(int i : c10::irange(10)) {
        s = s.append(A, i);
    }
    AT_ASSERT(s[0] == 3 && s.back() == 9 && s.size() == 14 && s.capacity() == 16);

    Slice<int> s2(A, -1, -2, -3);
    AT_ASSERT(s2[1] == -2 && s[0] == 3);

    auto ss = s.slice(1,2);
    AT_ASSERT(ss.size() == 1);
    AT_ASSERT(ss[0] == 4);
    AT_ASSERT(ss.capacity() == 1);
    ss = ss.append(A, -4);
    AT_ASSERT(ss.size() == 2 && ss[1] == -4);
    ss[0] = 3;
    AT_ASSERT(s[1] == 4);

    s = s.insert(A, s.slice(1, 4), ss);
    AT_ASSERT(s[1] == 3  && s[2] == -4 && s[3] == 0);

    auto sz = s.size();
    s = s.insert(A, s.slice(1, 1), 4);
    AT_ASSERT(s[1] == 4 && sz + 1 == s.size());


    Slice<int> d(A, 0, 1, 2, 3, 4);

    Slice<int> b(A, 0, 1, 2, 3, 4);
    b = b.insert(A, b.slice(1,1), d);
    AT_ASSERT(b.size() == 10);
    AT_ASSERT(b[1] == 0);
    AT_ASSERT(b[5] == 4);
    AT_ASSERT(b.back() == 4);

    Py_RETURN_NONE;

    PY_END(nullptr);
}

static PyObject* call_torch_function(PyObject* self, PyObject* args, PyObject* kwargs) {
    PY_BEGIN
    auto orig = py::handle(PyTuple_GET_ITEM(self, 0));
    auto torch_function = py::handle(PyTuple_GET_ITEM(self, 1));
    return torch_function.call(orig, py::handle(Py_None), py::handle(args), py::handle(kwargs)).release();
    PY_END(nullptr)
}

static PyObject* _wrap_method(PyObject *self,
                      PyObject *const *args,
                      Py_ssize_t nargs,
                      PyObject *kwnames) {
    PY_BEGIN
    AT_ASSERT(nargs == 2);
    PyMethodDef* md = new PyMethodDef {"wrapper", (PyCFunction) call_torch_function, METH_VARARGS | METH_KEYWORDS }; // leaks PyMethodDef
    py::tuple s(2);
    s.set(0, py::object::borrow(args[0]));
    s.set(1, py::object::borrow(args[1]));
    auto r = py::object::checked_steal(PyCFunction_New(md, s.release()));
    return PyInstanceMethod_New(r.release());
    PY_END(nullptr);
}


static PyObject* _level_to_dim(PyObject *self,
                      PyObject *const *args,
                      Py_ssize_t nargs,
                      PyObject *kwnames) {
    PY_BEGIN
    AT_ASSERT(nargs == 1);
    int l = py::to_int(args[0]) - 31;
    AT_ASSERT(l < n_levels_in_use);
    return py::object::borrow(levels_in_use[l]).release();
    PY_END(nullptr);
}

static PyMethodDef methods[] = {
    {"dims", (PyCFunction) dims, METH_FASTCALL | METH_KEYWORDS},
    {"_test_c", (PyCFunction) test_c, METH_FASTCALL | METH_KEYWORDS},
    {"_wrap_method", (PyCFunction) _wrap_method, METH_FASTCALL | METH_KEYWORDS},
    {"_n_levels_in_use", [](PyObject*,PyObject*) -> PyObject* { return PyLong_FromLongLong(n_levels_in_use); }, METH_NOARGS},
    {"_level_to_dim", (PyCFunction) _level_to_dim, METH_FASTCALL | METH_KEYWORDS},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

static struct PyModuleDef module_def = {
    PyModuleDef_HEAD_INIT,
    "_C",   /* name of module */
    NULL, /* module documentation, may be NULL */
    -1,       /* size of per-interpreter state of the module,
                 or -1 if the module keeps state in global variables. */
    methods
};

PyMODINIT_FUNC PyInit__C(void) {
    try {
        py::object mod = py::object::checked_steal(PyModule_Create(&module_def));
        Dim::ready(mod, "Dim");
        DimList::ready(mod, "DimList");
        Tensor::ready(mod, "Tensor");
        return mod.release();
    } catch(py::exception_set& err) {
        return nullptr;
    }
}
