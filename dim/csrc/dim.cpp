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

static int g_creation_order_ = 0;


py::handle DimensionBindError_;
static py::handle DimensionBindError() {
    if(!DimensionBindError_.ptr()) {
        DimensionBindError_ = py::import("dim").attr("DimensionBindError");
    }
    return DimensionBindError_;
}


PyTypeObject* DimType = nullptr;
py::handle _alloc_level;
struct Dim : public py::base<Dim> {
    int creation_order_; // for stable comparisons in prototype
    py::object name_;
    Dim()
    : creation_order_(g_creation_order_++), size_(-1) {}
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
            _alloc_level = py::import("dim.batch_tensor").attr("_alloc_level");
        }
        auto r = Dim::alloc(DimType);
        r->init(std::move(name), s);
        _alloc_level.call(r);
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

static PyObject* Dim_getcreation_order(Dim* self, void*) {
    return PyLong_FromLong(self->creation_order_);
}


static PyGetSetDef Dim_getsetters[] = {
    {"size", (getter) Dim_getsize, (setter) Dim_setsize,
     "Dimension size", NULL},
    {"is_bound", (getter) Dim_getis_bound, NULL, "is_bound", NULL},
    {"creation_order", (getter) Dim_getcreation_order, NULL, "is_bound", NULL},
    {NULL}  /* Sentinel */
};

static PyObject *Dim_richcompare(Dim *self, PyObject *other, int op);

#define FORALL_NUMBER_FNS(_) \
    _(add) \
    _(sub) \
    _(mul)  \
    _(truediv) \
    _(floordiv) \
    _(neg) \
    _(pow) \
    _(lt) \
    _(gt) \
    _(le) \
    _(ge) \
    _(eq) \
    _(ne)

#define CREATE_NAME(x) "__" #x "__",
#define CREATE_ENUM(x) Num_##x,

const char* forward_table[] = {
    FORALL_NUMBER_FNS(CREATE_NAME)
};

enum Num {
    FORALL_NUMBER_FNS(CREATE_ENUM)
};



struct Tensor;
static py::obj<Tensor> pointwise(py::handle op, Py_ssize_t nargs, py::handle* args);
static PyObject* Tensor_richcompare(PyObject* self, PyObject* other, int op);

py::handle forward_table_fns[sizeof(forward_table) / sizeof(const char*)];



py::obj<Tensor> Tensor_forwardv(int idx, Py_ssize_t nargs, py::handle* args) {
    if (!forward_table_fns[idx].ptr()) {
        forward_table_fns[idx] = py::import("torch").attr("Tensor").attr(forward_table[idx]);
    }
    py::handle op = forward_table_fns[idx];
    return pointwise(op, nargs,  args);
}

template<int idx, typename... Args>
PyObject * Tensor_forward(Args... objs) {
    PY_BEGIN
    py::handle args[] = {py::handle(objs)...};
    return Tensor_forwardv(idx, sizeof(args) / sizeof(py::handle),  args).release();
    PY_END(nullptr)
}

PyObject* Tensor_pow(PyObject* lhs, PyObject* rhs, PyObject* mod) {
    PY_BEGIN
    // XXX - ignoring option mod argument...
    if (!py::is_none(mod)) {
        py::raise_error(PyExc_ValueError, "unsupported mod argument to pow");
    }
    return Tensor_forward<Num_pow>(lhs, rhs);
    PY_END(nullptr)
}

static PyObject* Tensor_mul(PyObject* lhs, PyObject* rhs);

PyNumberMethods Tensor_numbers = {
    Tensor_forward<Num_add, PyObject*, PyObject*>, //binaryfunc nb_add;
    Tensor_forward<Num_sub, PyObject*, PyObject*>, //binaryfunc nb_subtract;
    Tensor_mul, //binaryfunc nb_multiply;
    nullptr, //binaryfunc nb_remainder;
    nullptr, //binaryfunc nb_divmod;
    Tensor_pow, //ternaryfunc nb_power;
    Tensor_forward<Num_neg, PyObject*>, //unaryfunc nb_negative;
    nullptr, //unaryfunc nb_positive;
    nullptr, //unaryfunc nb_absolute;
    nullptr, //inquiry nb_bool;
    nullptr, //unaryfunc nb_invert;
    nullptr, //binaryfunc nb_lshift;
    nullptr, //binaryfunc nb_rshift;
    nullptr, //binaryfunc nb_and;
    nullptr, //binaryfunc nb_xor;
    nullptr, //binaryfunc nb_or;
    nullptr, //unaryfunc nb_int;
    nullptr, //void *nb_reserved;
    nullptr, //unaryfunc nb_float;

    nullptr, //binaryfunc nb_inplace_add;
    nullptr, //binaryfunc nb_inplace_subtract;
    nullptr, //binaryfunc nb_inplace_multiply;
    nullptr, //binaryfunc nb_inplace_remainder;
    nullptr, //ternaryfunc nb_inplace_power;
    nullptr, //binaryfunc nb_inplace_lshift;
    nullptr, //binaryfunc nb_inplace_rshift;
    nullptr, //binaryfunc nb_inplace_and;
    nullptr, //binaryfunc nb_inplace_xor;
    nullptr, //binaryfunc nb_inplace_or;

    nullptr, //binaryfunc nb_floor_divide;
    Tensor_forward<Num_truediv, PyObject*>, //binaryfunc nb_true_divide;
    Tensor_forward<Num_floordiv, PyObject*>, //binaryfunc nb_inplace_floor_divide;
    nullptr, //binaryfunc nb_inplace_true_divide;

    nullptr, //unaryfunc nb_index;

    nullptr, //binaryfunc nb_matrix_multiply;
    nullptr, //binaryfunc nb_inplace_matrix_multiply;
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
private:
    at::Tensor data_;
    std::vector<py::obj<Dim>> dims_;
    void realize() {
        if (mul_lhs_.ptr()) {
            py::handle args[] = {mul_lhs_, mul_rhs_};
            become(Tensor_forwardv(Num_mul, 2, args));
        }
    }
    void become(py::obj<Tensor> r) {
        data_ = std::move(r->data_);
        dims_ = std::move(r->dims_);
        mul_lhs_ = py::obj<Tensor>();
        mul_rhs_ = py::obj<Tensor>();
    }
public:
    py::obj<Tensor> mul_lhs_;
    py::obj<Tensor> mul_rhs_;
    bool is_any_dims_;
    at::Tensor& data() {
        realize();
        return data_;
    }
    std::vector<py::obj<Dim>>& dims() {
        realize(); // we will eventually have dims computed without having to realize,
                   // for now this is a shortcut so that the logic doesn't have to be repeated from
                   // pointwise while pointwise is eagerly executed
        return dims_;
    }
    static PyTypeObject Type;
    void init(std::vector<py::obj<Dim>> dims, at::Tensor data, bool is_any_dims) {
        dims_ = std::move(dims);
        data_ = std::move(data);
        is_any_dims_ = is_any_dims;
    }
    static py::obj<Tensor> create_subclass(std::vector<py::obj<Dim>> dims, at::Tensor data, bool is_any_dims) {
        if (!TensorType) {
            TensorType = (PyTypeObject*) py::import("dim").attr("Tensor").ptr();
        }
        auto r = Tensor::alloc(TensorType);
        r->init(std::move(dims), std::move(data), is_any_dims);
        return r;
    }
};

static int Tensor_init(Tensor *self, PyObject *args, PyObject *kwds) {
    PY_BEGIN
    static char* kwlist[] = {"dims", "data", "is_any_dims", nullptr};
    py::handle dims;
    py::handle data;
    int is_any_dims = 0;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OO|p", kwlist, &dims, &data, &is_any_dims)) {
        return -1;
    }
    if (!THPVariable_Check(data.ptr())) {
        py::raise_error(PyExc_ValueError, "data is not a Tensor?");
    }
    py::sequence_view s(dims);
    std::vector<py::obj<Dim>> dims_;
    auto size = s.size();
    dims_.reserve(size);
    for (size_t i = 0; i < size; ++i) {
        dims_.emplace_back(Dim::wrap(s[i]));
    }
    self->init(std::move(dims_), THPVariable_Unpack(data.ptr()), is_any_dims != 0);
    return 0;
    PY_END(-1)
}

static PyGetSetDef Tensor_getsetters[] = {
   {"is_any_dims", (getter) [](PyObject* self, void*) -> PyObject* { return py::from_bool(((Tensor*)self)->is_any_dims_).release(); }, NULL},
   {"data", (getter) [](PyObject* self, void*) -> PyObject* { return THPVariable_Wrap(((Tensor*)self)->data()); }, NULL},
   {"dims", (getter) [](PyObject* self_, void*) -> PyObject* {
        PY_BEGIN
        auto self = (Tensor*) self_;
        auto& dims = self->dims();
        auto N = dims.size();
        py::tuple r(N);
        for (size_t i = 0; i < N; ++i) {
            r.set(i, dims[i]);
        }
        return r.release();
        PY_END(nullptr)
    }, NULL},

    {NULL}  /* Sentinel */
};

static PyObject* Tensor_positional(Tensor *self,
                      PyObject *const *args,
                      Py_ssize_t nargs,
                      PyObject *kwnames) {
    PY_BEGIN
    std::vector<int64_t> permuted;
    std::vector<int64_t> new_sizes;
    permuted.reserve(nargs);
    new_sizes.reserve(nargs);
    auto& dims = self->dims();
    std::vector<bool> seen(dims.size(), false);
    auto index_of = [&](py::hdl<Dim> d) {
        for (int64_t i = 0, N = dims.size(); i < N; ++i) {
            if (dims[i].ptr() == d.ptr()) {
                seen[i] = true;
                return i;
            }
        }
        py::object actual_dims = py::handle((PyObject*)self).attr("dims");
        py::raise_error(DimensionBindError(), "Dimension %S not in Tensor with dims %S", d.ptr(), actual_dims.ptr());
    };
    for (int64_t i = 0; i < nargs; ++i) {
        if (Dim::check(args[i])) {
            py::hdl<Dim> d = Dim::unchecked_wrap(args[i]);
            permuted.push_back(index_of(d));
            new_sizes.push_back(d->size());
        } else if (DimList::check(args[i])) {
            auto dl = DimList::unchecked_wrap(args[i]);
            for(int64_t j = 0, N = dl->size(); j < N; ++j) {
                py::hdl<Dim> d = dl->dims_[j];
                permuted.push_back(index_of(d));
                new_sizes.push_back(d->size());
            }
        } else if (py::is_sequence(args[i])) {
            py::sequence_view sv(args[i]);
            int64_t total_size = 1;
            for(int64_t j = 0, N = sv.size(); j < N; ++j) {
                auto d = Dim::wrap(sv[j]);
                total_size *= d->size();
                permuted.push_back(index_of(d));
            }
            new_sizes.push_back(total_size);
        } else if (py::is_none(args[i])) {
            // pass
        } else {
            py::raise_error(PyExc_ValueError, "expected Dim, DimList, or sequence of Dims for argument %d", (int) nargs);
        }
    }
    if (permuted.size() != dims.size()) {
        py::object actual_dims = py::handle((PyObject*)self).attr("dims");
        for (int64_t i = 0, N = dims.size(); i < N; ++i) {
            if (!seen[i]) {
                py::raise_error(DimensionBindError(), "Dimension %S exists in tensor but is not in the positional list", dims[i].ptr());
            }
        }
    }
    auto data = self->data().permute(permuted);
    // if we collapsed a sequence of dims above, a reshape is needed:
    if (new_sizes.size() != permuted.size()) {
        data = data.reshape(new_sizes);
    }
    return THPVariable_Wrap(data);
    PY_END(nullptr)
}

static PyObject *Tensor_richcompare(PyObject *self, PyObject *other, int op) {
    PY_BEGIN
    if (!rich_comparison_fns[op].ptr()) {
        rich_comparison_fns[op] = py::import("torch").attr("Tensor").attr(rich_comparison_table[op]);
    }
    py::handle args[] = {self, other};
    return pointwise(rich_comparison_fns[op], 2, args).release();
    PY_END(nullptr)
}

static PyObject* Tensor_sum(py::hdl<Tensor> self,
                      PyObject *const *args,
                      Py_ssize_t nargs,
                      PyObject *kwname);

static PyMethodDef Tensor_methods[] = {
    {"positional", (PyCFunction) Tensor_positional, METH_FASTCALL | METH_KEYWORDS},
    {"sum", (PyCFunction) Tensor_sum, METH_FASTCALL | METH_KEYWORDS},

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
    &Tensor_numbers,                 /* tp_as_number */
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
    Tensor_richcompare,  /* tp_richcompare */
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

std::vector<py::obj<Dim>> empty_dims() { return {}; }

py::obj<Tensor> _lift(py::handle h) {
    if (Tensor::check(h)) {
        return py::obj<Tensor>::borrow(Tensor::wrap(h));
    } else if (py::is_int(h)) {
        auto r = py::to_int(h);
        return Tensor::create(empty_dims(), at::full({}, r, at::kInt), true);
    } else if (py::is_float(h)) {
        auto r = py::to_float(h);
        return Tensor::create(empty_dims(), at::full({}, r, at::kFloat), true);
    } else if (py::is_bool(h)) {
        auto r = py::to_bool_unsafe(h);
        return Tensor::create(empty_dims(), at::full({}, r, at::kBool), true);
    } else if (Dim::check(h)) {
        auto d = Dim::unchecked_wrap(h);
        std::vector<py::obj<Dim>> dims = { py::obj<Dim>::borrow(d) };
        return Tensor::create(std::move(dims), at::arange(d->size()),  true);
    } else {
        py::raise_error(PyExc_ValueError, "expected a Tensor but found %S", h.ptr());
    }
}

at::Tensor prepare_one(py::hdl<Tensor> t, const std::vector<py::obj<Dim>>& dims, const std::vector<int64_t>& sizes, at::Device& d) {
    at::Tensor data = t->data();
    auto& t_dims = t->dims();
    if (t->is_any_dims_ && data.device() != d) {
        data = data.to(d);
    }
    at::IntArrayRef strides = data.strides();
    std::vector<int64_t> rstrides;
    for (size_t i = 0, e = dims.size(); i < e; ++i) {
        auto it = std::find(t_dims.begin(), t_dims.end(), dims[i]);
        if (it == t_dims.end()) {
            rstrides.push_back(0);
        } else {
            rstrides.push_back(strides[(it - t_dims.begin())]);
        }
    }
    return data.as_strided(sizes, rstrides, data.storage_offset());
}

static py::obj<Tensor> pointwise(py::handle op, Py_ssize_t nargs, py::handle* args) {
    std::vector<py::obj<Tensor>> ts;
    ts.reserve(nargs);
    for (Py_ssize_t i = 0; i < nargs; ++i) {
        ts.push_back(_lift(args[i]));
    }
    at::Device d = at::kCPU;
    bool is_dim_tensor = false;
    for (auto& t : ts) {
        if (!t->is_any_dims_) {
            d = t->data().device();
            is_dim_tensor = true;
        }
    }
    std::vector<py::obj<Dim>> dims;
    std::vector<int64_t> sizes;
    for (auto & t : ts) {
        for (py::obj<Dim>& d : t->dims()) {
            if (std::find(dims.begin(), dims.end(), d) == dims.end()) {
                dims.push_back(d);
                sizes.push_back(d->size());
            }
        }
    }
    py::tuple datas(nargs);
    for (size_t i = 0; i < nargs; ++i) {
        at::Tensor data = prepare_one(ts[i], dims, sizes, d);
        datas.set(i, py::object::steal(THPVariable_Wrap(std::move(data))));
    }
    auto r = op.call_object(datas);
    at::Tensor rdata = THPVariable_Unpack(r.ptr());
    return Tensor::create_subclass(std::move(dims), std::move(rdata), !is_dim_tensor);
}

static PyObject* _pointwise(PyObject *module,
                      PyObject *const *args,
                      Py_ssize_t nargs,
                      PyObject *kwname) {
    PY_BEGIN
    if (nargs != 2 || !PyTuple_Check(args[1])) {
        py::raise_error(PyExc_ValueError, "expected op, tuple");
    }
    auto tup = (PyTupleObject*) args[1];
    return pointwise(args[0], Py_SIZE(tup), (py::handle*)tup->ob_item).release();
    PY_END(nullptr)
}

static py::obj<Tensor> reduce(py::handle op, py::hdl<Tensor> self, const std::vector<py::obj<Dim>>& dims, py::handle extra_args, py::handle extra_kwargs) {
    auto& t_dims = self->dims();
    py::object t_data = py::object::borrow(THPVariable_Wrap(self->data()));
    auto index_of = [&](py::hdl<Dim> d) {
        for (int64_t i = 0, N = t_dims.size(); i < N; ++i) {
            if (t_dims[i].ptr() == d.ptr()) {
                return i;
            }
        }
        py::object actual_dims = self.attr("dims");
        py::raise_error(DimensionBindError(), "Dimension %S not in Tensor with dims %S", d.ptr(), actual_dims.ptr());
    };
    size_t N = dims.size();
    std::vector<bool> remains(t_dims.size(), true);
    py::tuple indices(N);
    for (size_t i = 0; i < N; ++i) {
        auto idx = index_of(dims[i]);
        remains[idx] = false;
        indices.set(i, py::from_int(idx));
    }
    std::vector<py::obj<Dim>> new_dims;
    for (size_t i = 0; i < t_dims.size(); ++i) {
        if (remains[i]) {
            new_dims.push_back(t_dims[i]);
        }
    }
    at::Tensor data;
    if (py::is_none(extra_args)  && py::is_none(extra_kwargs)) {
        auto r = op.call(t_data, indices);
        data = THPVariable_Unpack(r.ptr());
    } else {
        auto ea = (PyTupleObject*) extra_args.ptr();
        py::tuple all_args(Py_SIZE(ea) + 2);
        all_args.set(0, std::move(t_data));
        all_args.set(1, indices);
        for (size_t i = 0; i < Py_SIZE(ea); ++i) {
            all_args.set(2 + i, py::object::borrow(ea->ob_item[i]));
        }
        auto r = op.call_object(all_args, extra_kwargs);
        data = THPVariable_Unpack(r.ptr());
    }
    return Tensor::create_subclass(std::move(new_dims), std::move(data), false);
}

static PyObject* Tensor_mul(PyObject* lhs, PyObject* rhs) {
    PY_BEGIN
    py::obj<Tensor> lhs_ = _lift(lhs);
    py::obj<Tensor> rhs_ = _lift(rhs);
    auto r = Tensor::create_subclass({}, at::Tensor(), lhs_->is_any_dims_ && rhs_->is_any_dims_);
    r->mul_lhs_ = std::move(lhs_);
    r->mul_rhs_ = std::move(rhs_);
    return r.release();
    PY_END(nullptr)
}

static std::vector<py::obj<Dim>> _dim_set(py::handle dim_specifier) {
    std::vector<py::obj<Dim>> dims;
    if (py::is_none(dim_specifier)) {
        // no dims
    } else if (Dim::check(dim_specifier)) {
        dims.push_back(Dim::unchecked_wrap(py::object::borrow(dim_specifier)));
    } else if (py::is_sequence(dim_specifier)) {
        py::sequence_view sv(dim_specifier);
        auto N = sv.size();
        dims.reserve(N);
        for (Py_ssize_t i = 0; i < N; ++i) {
            auto d = sv[i];
            if (!py::is_none(d)) {
                dims.emplace_back(Dim::wrap(d));
            }
        }
    }
    return dims;
}


static PyObject* _reduce(PyObject *module,
                      PyObject *const *args,
                      Py_ssize_t nargs,
                      PyObject *kwname) {
    PY_BEGIN
    if (nargs != 5 || !PyTuple_Check(args[3])) {
        py::raise_error(PyExc_ValueError, "expected op, self, dim, args, kwargs");
    }
    return reduce(args[0], Tensor::wrap(args[1]), _dim_set(args[2]) , args[3], args[4]).release();
    PY_END(nullptr)
}

static PyObject* _with_dims(PyObject *module,
                      PyObject *const *args,
                      Py_ssize_t nargs,
                      PyObject *kwnames) {
    PY_BEGIN
    at::Tensor data = THPVariable_Unpack(args[0]);
    args++;
    nargs--;
    at::IntArrayRef sizes =  data.sizes();
    at::IntArrayRef strides = data.strides();

    int64_t rank = 0;
    int64_t missing_rank_index = -1;
    bool splits_dims = false;
    for (int64_t i = 0; i < nargs; ++i) {
        if (Dim::check(args[i])) {
            rank += 1;
        } else if (DimList::check(args[i])) {
            auto dl = DimList::unchecked_wrap(args[i]);
            if (dl->is_bound()) {
                rank += dl->dims_.size();
            } else {
                if (missing_rank_index != -1) {
                    py::raise_error(DimensionBindError(), "only one DimList can be unsized in positional match");
                }
                missing_rank_index = i;
            }
        } else if (py::is_sequence(args[i])) {
            splits_dims = true;
            rank += 1; // flattened
        } else {
            py::raise_error(DimensionBindError(), "expected a Dim, DimList, or sequence of Dims");
        }
    }
    if (rank > sizes.size() || (missing_rank_index == -1 && rank < sizes.size())) {
        py::tuple tup(nargs);
        for (int64_t i = 0; i < nargs; ++i) {
            tup.set(i, py::object::borrow(args[i]));
        }
        py::raise_error(DimensionBindError(), "Tensor has %d dimensions but trying to match %d dimension values (%S) to it.", (int) sizes.size(), rank, tup.ptr());
    }
    int64_t missing_ranks = sizes.size() - rank;
    std::vector<py::obj<Dim>> dims;
    // if splitting...
    std::vector<int64_t> new_sizes;
    std::vector<int64_t> new_strides;
    dims.reserve(sizes.size());
    auto size_it = sizes.begin();
    auto stride_it = strides.begin();
    for (int64_t i = 0; i < nargs; ++i) {
        if (Dim::check(args[i])) {
            auto d = Dim::unchecked_wrap(args[i]);
            auto size = *size_it++;
            auto stride = *stride_it++;
            d->set_size(size);
            dims.emplace_back(py::obj<Dim>::borrow(d));
            if (splits_dims) {
                new_sizes.emplace_back(size);
                new_strides.emplace_back(stride);
            }
        } else if (DimList::check(args[i])) {
            auto dl = DimList::unchecked_wrap(args[i]);
            if (!dl->is_bound()) {
                dl->bind_len(missing_ranks);
            }
            for (auto& d : dl->dims_) {
                auto size = *size_it++;
                auto stride = *stride_it++;
                d->set_size(size);
                dims.emplace_back(d);
                if (splits_dims) {
                    new_sizes.emplace_back(size);
                    new_strides.emplace_back(stride);
                }
            }
        } else {
            py::sequence_view s(args[i]);
            auto size = *size_it++;
            auto stride = *stride_it++;
            int64_t missing_size_idx = -1;
            size_t N = s.size();
            size_t total_size = 1;
            for (size_t j = 0; j != N; ++j) {
                auto d = Dim::wrap(s[j]);
                if (d->is_bound()) {
                    size_t sz = d->size();
                    total_size *= sz;
                    new_sizes.push_back(sz);
                } else if (missing_size_idx == -1) {
                    missing_size_idx = j;
                    new_sizes.push_back(0); // placeholder
                } else {
                    auto other = s[missing_size_idx];
                    py::raise_error(DimensionBindError(), "list of splitting dimensions has two unbound dims %S and %S", other.ptr(), d.ptr());
                }
                new_strides.push_back(0); // placeholder
                dims.emplace_back(std::move(d));
            }
            if ( (missing_size_idx == -1 && total_size != size) || total_size > size) {
                py::raise_error(DimensionBindError(), "Dimension %d of size %s is to small to fit dimension %S of size %d", (int)i, (int)size, s.ptr(), (int)total_size);
            }
            // filling in potentiall missing size
            if (missing_size_idx != -1) {
                auto d = Dim::wrap(s[missing_size_idx]);
                if (size % total_size != 0) {
                    py::raise_error(DimensionBindError(), "inferred dimension %S of tuple %S does not fit evenly into the larger dimension of size %d because the rest of the dimensions have size %d", d.ptr(), s.ptr(), (int) size, (int) total_size);
                }
                auto sz = size / total_size;
                d->set_size(sz);
                *(new_sizes.end() - N + missing_size_idx) = sz;
            }
            // fill in the missing strides
            auto new_stride_it = new_strides.rbegin();
            auto new_sizes_it = new_sizes.rbegin();
            for (size_t j = 0; j != N; ++j) {
                *new_stride_it++ = stride;
                stride *= *new_sizes_it++;
            }
        }
    }
    if (splits_dims) {
        data = data.as_strided(new_sizes, new_strides, data.storage_offset());
    }
    return Tensor::create_subclass(std::move(dims), std::move(data), false).release();
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


py::obj<Tensor> dot(py::hdl<Tensor> lhs, py::hdl<Tensor> rhs, const std::vector<py::obj<Dim>>& sum) {
    auto& lhs_dims  = lhs->dims();
    auto& rhs_dims = rhs->dims();
    auto lhs_ = lhs->data();
    auto rhs_ = rhs->data();

    at::IntArrayRef lhs_strides = lhs_.strides();
    at::IntArrayRef rhs_strides = rhs_.strides();


    int64_t max_size = lhs_dims.size() + rhs_dims.size();

    int64_t lro_size = 1;
    int64_t lro_dim = 0;

    int64_t lo_size = 1;
    int64_t lo_dim = 0;

    int64_t ro_size = 1;
    int64_t ro_dim = 0;

    int64_t lr_size = 1;
    int64_t lr_dim = 0;

    std::vector<int64_t> n_lhs_sizes, n_lhs_strides, n_rhs_sizes, n_rhs_strides;
    n_lhs_sizes.reserve(max_size);
    n_rhs_sizes.reserve(max_size);
    n_lhs_strides.reserve(max_size);
    n_rhs_strides.reserve(max_size);

    std::vector<py::obj<Dim>> o_dims;
    std::vector<int64_t> o_size;
    o_dims.reserve(max_size);
    o_size.reserve(max_size);

    auto insert = [&] (std::vector<int64_t>& arr, size_t i, int64_t v) {
        arr.insert(arr.begin() + i, v);
    };
    auto insert_dim = [&] (py::hdl<Dim> d, int64_t lhs_idx, int64_t rhs_idx, bool sum) {
        int64_t size = d->size();
        int64_t lhs_stride = lhs_idx == -1 ? 0 : lhs_strides[lhs_idx];
        int64_t rhs_stride = rhs_idx == -1 ? 0 : rhs_strides[rhs_idx];
        if (sum) {
            // lr
            lr_size *= size;
            int64_t l_idx = lro_dim + lo_dim + lr_dim;
            int64_t r_idx = lro_dim + lr_dim;
            insert(n_lhs_strides, l_idx, lhs_stride);
            insert(n_lhs_sizes, l_idx, size);
            insert(n_rhs_strides, r_idx, rhs_stride);
            insert(n_rhs_sizes, r_idx, size);
            lr_dim += 1;
        } else {
            if ((lhs_stride == 0) == (rhs_stride == 0)) {
                // lro
                insert(n_lhs_strides, lro_dim, lhs_stride);
                insert(n_lhs_sizes, lro_dim, size);
                insert(n_rhs_strides, lro_dim, rhs_stride);
                insert(n_rhs_sizes, lro_dim, size);
                insert(o_size, lro_dim, size);
                o_dims.insert(o_dims.begin() + lro_dim, py::obj<Dim>::borrow(d));
                lro_size *= size;
                lro_dim += 1;
            } else if (lhs_stride != 0) {
                // lo
                int64_t idx = lro_dim + lo_dim;
                insert(n_lhs_strides, idx, lhs_stride);
                insert(n_lhs_sizes, idx, size);

                insert(o_size, idx, size);
                o_dims.insert(o_dims.begin() + idx, py::obj<Dim>::borrow(d));

                lo_size *= size;
                lo_dim += 1;
            } else {
                AT_ASSERT(rhs_stride != 0);
                // ro
                int64_t idx = lro_dim + lr_dim + ro_dim;
                insert(n_rhs_strides, idx, rhs_stride);
                insert(n_rhs_sizes, idx, size);

                int64_t o_idx = lro_dim + lo_dim + ro_dim;
                insert(o_size,  o_idx, size);
                o_dims.insert(o_dims.begin() + o_idx, py::obj<Dim>::borrow(d));

                ro_size *= size;
                ro_dim += 1;
            }
        }
    };

    std::vector<bool> rhs_seen(rhs_dims.size(), false);

    for (int64_t i = 0, N = lhs_dims.size(); i < N; ++i) {
        py::hdl<Dim> d = lhs_dims[i];
        auto rhs_idx = dim_index(rhs_dims, d);
        if (rhs_idx != -1) {
            rhs_seen[rhs_idx] = true;
        }
        auto s_idx = dim_index(sum, d);
        insert_dim(d, i, rhs_idx, s_idx != -1);
    }
    for (int64_t i = 0, N = rhs_dims.size(); i < N; ++i) {
        if (rhs_seen[i]) {
            continue;
        }
        py::hdl<Dim> d = rhs_dims[i];
        auto s_idx = dim_index(sum, d);
        insert_dim(d, -1, i, s_idx != -1);
    }

    if (lr_dim != sum.size()) {
        for (auto & d : sum) {
            if (-1 == dim_index(o_dims, d)) {
                py::raise_error(DimensionBindError(), "summing over non-existant dimension %S", d.ptr());
            }
        }
    }

    lhs_ = lhs_.as_strided(n_lhs_sizes, n_lhs_strides, lhs_.storage_offset());
    rhs_ = rhs_.as_strided(n_rhs_sizes, n_rhs_strides, rhs_.storage_offset());
    lhs_ = lhs_.reshape({lro_size, lo_size, lr_size});
    rhs_ = rhs_.reshape({lro_size, lr_size, ro_size});

    auto result = at::bmm(lhs_, rhs_);
    result = result.reshape(o_size);
    return Tensor::create_subclass(std::move(o_dims), std::move(result), false);



}

static py::handle torch_tensor_sum;

static PyObject* Tensor_sum(py::hdl<Tensor> self,
                      PyObject *const *args,
                      Py_ssize_t nargs,
                      PyObject *kwname) {
    PY_BEGIN
    if (nargs != 1) {
        py::raise_error(PyExc_ValueError, "expected dim arg\n");
    }
    if (self->mul_lhs_.ptr()) {
        auto ds = _dim_set(args[0]);
        if (ds.size() == 0) {
            return py::object::borrow(self).release();
        }
        return dot(self->mul_lhs_, self->mul_rhs_, std::move(ds)).release();
    }
    if (!torch_tensor_sum.ptr()) {
        torch_tensor_sum = py::import("torch").attr("Tensor").attr("sum");
    }
    return reduce(torch_tensor_sum, self, _dim_set(args[0]), Py_None, Py_None).release();

    PY_END(nullptr)
}


void broadcast_bind_dims(std::vector<at::IntArrayRef> sizes, py::hdl<DimList> dims) {
    size_t ndim = 0;
    for (size_t i = 0, N = sizes.size(); i < N; ++i) {
        size_t s = sizes[i].size();
        if (s > ndim) {
            ndim = s;
        }
    }
    dims->bind_len(ndim);
    for (int64_t i = 0; i < ndim; ++i) {
        int64_t target_size = 1;
        for (int64_t j = 0, N = sizes.size(); j < N; ++j) {
            if (i < sizes[j].size()) {
                int64_t dim_size = *(sizes[j].end() - 1 - i);
                if (dim_size != 1 && dim_size != target_size) {
                    TORCH_CHECK(
                      target_size == 1,
                        "The expanded size of the tensor (",
                        target_size,
                        ") must match the existing size (",
                        dim_size,
                        ") at non-singleton dimension ",
                        ndim - 1 - i);
                    target_size = dim_size;
                }
            }
        }
        dims->dims_[ndim - i - 1]->set_size(target_size);
    }
}

std::vector<py::obj<Tensor>> _broadcast_match(std::vector<at::Tensor> inputs, py::hdl<DimList> dims, at::IntArrayRef* extra) {
    std::vector<at::IntArrayRef> sizes;
    sizes.reserve(inputs.size());
    for (auto& t : inputs) {
        sizes.emplace_back(t.sizes());
    }
    if (extra) {
        sizes.emplace_back(*extra);
    }
    broadcast_bind_dims(sizes, dims);
    std::vector<int64_t> b_sizes;
    b_sizes.reserve(dims->size());
    for (auto& d : dims->dims_) {
        b_sizes.emplace_back(d->size());
    }
    std::vector<py::obj<Tensor>> result;
    result.reserve(inputs.size());
    for (auto& t : inputs) {
        result.emplace_back(Tensor::create_subclass(dims->dims_, (t.sizes() == b_sizes) ? std::move(t) : t.expand(b_sizes), false));
    }
    return result;
}

static PyObject* broadcast_match(PyObject *self,
                      PyObject *const *args,
                      Py_ssize_t nargs,
                      PyObject *kwnames) {
    PY_BEGIN
    static const char * const _keywords[] = {"tensors", "dims", "shape", nullptr};
    py::handle tensors, dims, shape;
    static _PyArg_Parser parser = {"OO|O", _keywords, 0};
    if (!_PyArg_ParseStackAndKeywords(args, nargs, kwnames, &parser, &tensors, &dims, &shape)) {
        return nullptr;
    }
    if (!py::is_sequence(tensors)) {
        py::raise_error(PyExc_ValueError, "expected a sequence of tensors");
    }
    auto sv = py::sequence_view::wrap(tensors);
    std::vector<at::Tensor> tensor_data;
    auto N = sv.size();
    tensor_data.reserve(N);
    for (int64_t i = 0; i < N; ++i) {
        auto elem = sv[i];
        if (THPVariable_Check(elem.ptr())) {
            tensor_data.emplace_back(THPVariable_Unpack(elem.ptr()));
        } else if (py::is_int(elem)) {
            int64_t r = py::to_int(elem);
            tensor_data.emplace_back(at::full({}, r, at::kInt));
        } else if (py::is_float(elem)) {
            int64_t r = py::to_float(elem);
            tensor_data.emplace_back(at::full({}, r, at::kFloat));
        } else if (py::is_bool(elem)) {
            bool r = py::to_bool_unsafe(elem);
            tensor_data.emplace_back(at::full({}, r, at::kBool));
        } else {
            py::raise_error(PyExc_ValueError, "expected tensor or constant");
        }
    }
    auto dim_list = DimList::wrap(dims);
    std::vector<int64_t> extra_sizes;
    at::IntArrayRef extra;
    bool has_shape = shape.ptr() && !py::is_none(shape.ptr());
    if (has_shape) {
        auto sizes = py::sequence_view::wrap(shape);
        for (size_t i = 0, N = sizes.size(); i < N; ++i) {
            auto s = sizes[i];
            if (!py::is_int(s)) {
                py::raise_error(PyExc_ValueError, "expected an int");
            }
            extra_sizes.emplace_back(py::to_int(s));
        }
        extra = extra_sizes;
    }
    auto r = _broadcast_match(std::move(tensor_data), dim_list, has_shape ? &extra : nullptr);
    py::tuple r_tuple(r.size());
    for (size_t i = 0, N = r.size(); i < N; ++i) {
        r_tuple.set(i, std::move(r[i]));
    }
    return r_tuple.release();

    PY_END(nullptr)
}

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

static PyMethodDef methods[] = {
    {"dims", (PyCFunction) dims, METH_FASTCALL | METH_KEYWORDS},
    {"broadcast_match", (PyCFunction) broadcast_match, METH_FASTCALL | METH_KEYWORDS},
    {"_pointwise", (PyCFunction) _pointwise, METH_FASTCALL | METH_KEYWORDS},
    {"_reduce", (PyCFunction) _reduce, METH_FASTCALL | METH_KEYWORDS},
    {"_with_dims", (PyCFunction) _with_dims, METH_FASTCALL | METH_KEYWORDS},
    {"_test_c", (PyCFunction) test_c, METH_FASTCALL | METH_KEYWORDS},
    {"_wrap_method", (PyCFunction) _wrap_method, METH_FASTCALL | METH_KEYWORDS},

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
