#include "minpybind.h"
#include <frameobject.h>
#include <opcode.h>
#include <utility>
#include <new>
#include <iostream>
#include <vector>
#include <torch/csrc/autograd/python_variable.h>
#include <functorch/csrc/BatchedTensorImpl.h>
#include <ATen/ATen.h>
#include "arena.h"


// C++ API functions for objects to
// * construct the object, returning a ref-counted handle
// * The actual API, with methods that take/return C-typed values

// extend minpybind.h to include
// * typed handles so that -> can get to their raw API
// * object/handle distinction for the typed handles

// class Dim: ---------------
namespace at {
namespace functorch {
    at::Tensor _add_batch_dim(const at::Tensor& self, int64_t batch_dim, int64_t level);
    void _vmap_add_layers(const std::vector<std::pair<int64_t, int64_t>>& levels);
    void _vmap_remove_layers(int N);
}
}
int whooo;

py::handle empty_dict;
py::handle torch_Tensor___mul__;
py::handle _Tensor;
py::handle DelayedMulTensor;
py::handle NamedTuple;
py::dict_view pointwise;
py::handle torch_Tensor_expand;
binaryfunc THPVariable_getitem;
py::handle no_slice;

static void maybeInitializeGlobals() {
    if (empty_dict.ptr()) {
        return;
    }
    auto torch = py::import("torch");
    auto dim = py::import("dim");
    empty_dict = PyDict_New();
    torch_Tensor___mul__ = torch.attr("Tensor").attr("__mul__");
    _Tensor = dim.attr("_Tensor");
    DelayedMulTensor = dim.attr("DelayedMulTensor");
    NamedTuple = py::import("typing").attr("NamedTuple");
    pointwise = dim.attr("pointwise");
    torch_Tensor_expand = torch.attr("Tensor").attr("expand");
    auto TensorBase = (PyTypeObject*) torch.attr("_C").attr("_TensorBase").ptr();
    THPVariable_getitem = TensorBase->tp_as_mapping->mp_subscript;
    no_slice = PySlice_New(NULL, NULL, NULL);
}

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
constexpr int LEVEL_OFFSET = 31;

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
    const at::Tensor& range() {
        if (!range_.defined()) {
            range_ = at::arange(size());
        }
        return range_;
    }
    const at::Tensor& batchtensor() {
        if (!batchtensor_.defined()) {
            batchtensor_ = at::functorch::_add_batch_dim(range(), 0, level_ + LEVEL_OFFSET);
        }
        return batchtensor_;
    }
private:
    int64_t size_;
    at::Tensor range_;
    at::Tensor batchtensor_;
};

struct DimEntry {
    // union of either a negative number indicating which dimension this is from the rhs,
    // or a pointer to a first-class dimension.
    // pointers do not have their highest bit set, so checking the number is negative tells us
    // that it is not a dim.
    bool is_positional() const {
        return data_ < 0;
    }
    bool is_none() const {
        return data_ == 0;
    }
    int64_t position() const {
        return data_;
    }
    py::hdl<Dim> dim() const {
        Dim* result;
        std::memcpy(&result, &data_, sizeof(Dim*));
        return py::hdl<Dim>(result);
    }

    DimEntry()
    : data_(0) {}

    DimEntry(int64_t pos)
    : data_(pos) {
        AT_ASSERT(pos < 0);
    }
    DimEntry(py::hdl<Dim> d) {
       std::memcpy(&data_, &d, sizeof(int64_t));
    }
    bool operator==(const DimEntry& rhs) {
        return data_ == rhs.data_;
    }
private:
    int64_t data_;
};

std::ostream& operator<<(std::ostream& ss, DimEntry entry) {
    if (entry.is_none()) {
        ss << "None";
    } else if (entry.is_positional()) {
        ss << entry.position();
    } else {
        ss << entry.dim();
    }
    return ss;
}

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
    return PyLong_FromLong(LEVEL_OFFSET + self->level_);
}

static PyObject* Dim_get_levels(Dim* self, void*) {
    py::tuple t(1);
    t.set(0, py::object::borrow(self->ptr()));
    return t.release();
}

static PyObject* Dim_get_has_device(Dim* self, void*) {
    Py_RETURN_FALSE;
}

static PyObject* Dim_get_tensor(Dim* self, void*) {
    return THPVariable_Wrap(self->range());
}

static PyObject* Dim_get_batchtensor(Dim* self, void*) {
    return THPVariable_Wrap(self->batchtensor());
}


static PyGetSetDef Dim_getsetters[] = {
    {"size", (getter) Dim_getsize, (setter) Dim_setsize,
     "Dimension size", NULL},
    {"is_bound", (getter) Dim_getis_bound, NULL, "is_bound", NULL},
    {"_level", (getter) Dim_getlevel, NULL, "_level", NULL},
    {"_levels", (getter) Dim_get_levels, NULL, "_levels", NULL},
    {"_has_device", (getter) Dim_get_has_device, NULL, "_has_device", NULL},
    {"_tensor", (getter) Dim_get_tensor, NULL, "_tensor", NULL},
    {"_batchtensor", (getter) Dim_get_batchtensor, NULL, "_tensor", NULL},

    {NULL}  /* Sentinel */
};

static PyObject *Dim_richcompare(Dim *self, PyObject *other, int op);

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
    (richcmpfunc) Dim_richcompare,  /* tp_richcompare */
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
const char* rich_comparison_table[] = {
    "__lt__",
    "__le__",
    "__eq__",
    "__ne__",
    "__gt__",
    "__ge__"
};

static PyObject *Dim_richcompare(Dim *self, PyObject *other, int op) {
    PY_BEGIN
        if (op == Py_EQ || op == Py_NE)  {
            Py_RETURN_RICHCOMPARE( (void*)self, (void*) other, op);
        } else {
            if (!rich_comparison_fns[0].ptr()) {
                for (auto i : irange(6)) {
                    auto r = py::import("dim").attr("_Tensor");
                    rich_comparison_fns[i] = r.attr(rich_comparison_table[i]);
                }
            }
            return rich_comparison_fns[op].call(py::handle(self->ptr()), py::handle(other)).release();
        }
    PY_END(nullptr)
}

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
    0,                              /* tp_richcompare */
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
    OwnedSlice<DimEntry> levels_;
    bool has_device_;

    static PyTypeObject Type;

    static py::obj<Tensor> create() {
        if (!TensorType) {
            TensorType = (PyTypeObject*) py::import("dim").attr("Tensor").ptr();
        }
        return Tensor::alloc(TensorType);
    }
};

at::Tensor _add_batch_dims(Arena& A, at::Tensor t, Slice<DimEntry> levels_) {
    auto levels = Slice<DimEntry>();
    levels.extend(A, levels_);
    while (true) {
        int min_real_index = -1;
        int min_index = -1;
        int min_value = INT_MAX;
        int i = 0;
        int r = 0;
        for (auto l : levels) {
            if (!l.is_none()) {
                if (!l.is_positional() && l.dim()->level_ < min_value) {
                    min_value = l.dim()->level_;
                    min_index = i;
                    min_real_index = r;
                }
                ++i;
            }
            ++r;
        }
        if (min_index == -1) {
            return t;
        }
        auto t2 = at::functorch::_add_batch_dim(std::move(t), min_index, min_value + LEVEL_OFFSET);
        t = std::move(t2);
        levels[min_real_index] = DimEntry();
    }
}

void free_levels_dims(Slice<DimEntry> levels) {
    for(auto e : levels) {
        if (!e.is_positional()) {
            py::object::steal(e.dim());
        }
    }
}

// version in header does a unnecessary refcount +/-
inline at::functorch::BatchedTensorImpl* maybeGetBatchedImpl(const at::Tensor& tensor) {
    if (at::functorch::isBatchedTensor(tensor)) {
        return static_cast<at::functorch::BatchedTensorImpl*>(tensor.unsafeGetTensorImpl());
    }
    return nullptr;
}

inline TensorRef unchecked_tensor_from(py::handle p) {
    auto v = (THPVariable*) p.ptr();
    return TensorRef(*v->cdata);
}

struct TensorInfo {
    TensorRef tensor;
    Slice<DimEntry> levels;
    bool has_device;
    TensorRef batchedtensor;
    int64_t ndim() const {
        int64_t r = 0;
        for (auto l : levels) {
            if (l.is_positional()) {
                ++r;
            }
        }
        return r;
    }
    operator bool() const {
        return tensor;
    }

    static TensorInfo create(Arena& A, py::handle h, bool ensure_batched=true, bool ensure_present=true) {
        if (Tensor::check(h)) {
            auto t = Tensor::unchecked_wrap(h);
            return TensorInfo {t->tensor_, t->levels_.slice(), t->has_device_, t->batchtensor_};
        } else if (THPVariable_Check(h.ptr())) {
            TensorRef t = unchecked_tensor_from(h);
            Slice<DimEntry> levels;
            for (auto i : irange(-t->dim(), 0)) {
                levels.append(A, i);
            }
            return TensorInfo {t, levels, true, t};
        } else if (Dim::check(h)) {
            auto d = Dim::unchecked_wrap(h);
            return TensorInfo {d->range(), Slice<DimEntry>(A, DimEntry(d)), false, ensure_batched ? d->batchtensor() : TensorRef()};
        } else if (py::isinstance(h, DelayedMulTensor)) {
            auto batchtensor = h.attr("_batchtensor");
            auto has_device = py::to_bool_unsafe(h.attr("_has_device"));
            return create(A, batchtensor, has_device);
        } else {
            if (ensure_present) {
                py::raise_error(PyExc_ValueError, "expected a tensor object");
            }
            return TensorInfo {};
        }
    }

    static TensorInfo create(Arena& A, TensorRef batchedtensor, bool has_device) {
        Slice<DimEntry> levels;
        for (auto i : irange(-batchedtensor->dim(), 0)) {
            levels.append(A, i);
        }
        TensorRef tensor;
        at::functorch::BatchedTensorImpl * impl = maybeGetBatchedImpl(*batchedtensor);
        while(true) {
            auto level = impl->level() - LEVEL_OFFSET;
            py::hdl<Dim> dim = (Dim*) levels_in_use[level].ptr();
            levels.insert(A, impl->bdim(), dim);
            at::functorch::BatchedTensorImpl * nimpl = maybeGetBatchedImpl(impl->value());
            if (!nimpl) {
                tensor = impl->value();
                break;
            }
            impl = nimpl;
        }
        return TensorInfo {tensor, levels, has_device, batchedtensor};
    }
};

static py::obj<Tensor> Tensor_from_batched(Arena& A, at::Tensor batched, bool has_device) {
    TensorInfo info = TensorInfo::create(A, batched, has_device != 0);
    py::obj<Tensor> self = Tensor::create();
    // grab ownership of the tensors
    self->tensor_ = *info.tensor;
    self->batchtensor_ = std::move(batched);
    self->has_device_ = info.has_device;
    // grab ownership of the dims inside levels
    for (auto l : info.levels) {
        if (!l.is_positional()) {
            py::object::borrow(l.dim()).release();
        }
    }
    self->levels_.set(info.levels, free_levels_dims);
    return self;
}

static PyObject* py_Tensor_from_batched(PyObject *self,
                      PyObject *const *args,
                      Py_ssize_t nargs,
                      PyObject *kwnames) {
    Arena A;
    PY_BEGIN
    #define ARGS(_) _(py::handle, py_batchedtensor) _(int, has_device)
    MPY_PARSE_ARGS_KWNAMES("Op", ARGS)
    #undef ARGS

    if (!THPVariable_Check(py_batchedtensor.ptr())) {
        py::raise_error(PyExc_ValueError, "_batchedtensor is not a Tensor?");
    }
    auto self = Tensor_from_batched(A, THPVariable_Unpack(py_batchedtensor.ptr()),  has_device != 0);
    return self.release();

    PY_END(nullptr)
}


static py::object Tensor_from_positional(Arena & A, at::Tensor tensor, Slice<DimEntry> levels, bool has_device) {
    size_t seen_dims = 0;
    int last = 0;
    for (auto l : levels) {
        if (l.is_positional()) {
            AT_ASSERT(last == 0 || last + 1 == l.position());
            last = l.position();
        } else {
            py::object::borrow(l.dim()).release();
            ++seen_dims;
        }
    }
    AT_ASSERT(last == 0 || last == -1);
    if (!seen_dims) {
        return py::object::borrow(THPVariable_Wrap(std::move(tensor)));
    }

    py::obj<Tensor> self = Tensor::create();
    self->tensor_ = std::move(tensor);
    AT_ASSERT(self->tensor_.dim() == levels.size());
    self->levels_.set(levels, free_levels_dims);
    self->has_device_ = has_device;
    self->batchtensor_ = _add_batch_dims(A, self->tensor_, self->levels_.slice());
    py::object r = std::move(self);
    return r;
}


static PyObject* py_Tensor_from_positional(PyObject *self,
                      PyObject *const *args,
                      Py_ssize_t nargs,
                      PyObject *kwnames) {
    Arena A;
    PY_BEGIN
    #define ARGS(_) _(py::handle, tensor) _(py::handle, py_levels) _(int, has_device)
    MPY_PARSE_ARGS_KWNAMES("OOp", ARGS)
    #undef ARGS

    if (!THPVariable_Check(tensor.ptr())) {
        py::raise_error(PyExc_ValueError, "_tensor is not a Tensor?");
    }

    Slice<DimEntry> levels;
    py::sequence_view sq(py_levels);
    for (auto i : sq.enumerate()) {
        py::object v = sq[i];
        if (py::is_int(v)) {
            auto vi = py::to_int(v);
            levels.append(A, vi);
        } else {
            auto dim = Dim::wrap(std::move(v));
            py::hdl<Dim> hdim = dim;
            levels.append(A, hdim);
        }
    }
    return Tensor_from_positional(A, THPVariable_Unpack(tensor.ptr()), levels, has_device != 0).release();
    PY_END(nullptr)
}

py::list slice_to_list(Slice<py::handle> h) {
    py::list lst(h.size());
    for (auto i : h.enumerate()) {
        lst.set(i, py::object::borrow(h[i]));
    }
    return lst;
}

py::tuple slice_to_tuple(Slice<py::handle> h) {
    py::tuple lst(h.size());
    for (auto i : h.enumerate()) {
        lst.set(i, py::object::borrow(h[i]));
    }
    return lst;
}

enum UType {
    U_ELEM,
    U_TUPLE_LIKE,
    U_DICT,
};

struct Unflatten {
    py::object operator()(Slice<py::handle>& elements) {
        py::object r;
        switch (type) {
            case U_ELEM: {
                r = py::object::borrow(elements[0]);
                elements = elements.slice(1);
            } break;
            case U_TUPLE_LIKE: {
                py::tuple tup(children.size());
                for (auto i : children.enumerate()) {
                    tup.set(i, children[i](elements));
                }
                r = obj.call(tup);
            } break;
            case U_DICT: {
                r = py::object::checked_steal(PyDict_New());
                py::dict_view rv(r);
                py::dict_view d(obj);
                Py_ssize_t pos = 0;
                py::handle k, v;
                for (int i = 0; d.next(&pos, &k, &v); ++i) {
                    rv.set(k, children[i](elements));
                }
            } break;
        }
        return r;
    }
    UType type;
    py::handle obj;
    Slice<Unflatten> children;
};

Unflatten tree_flatten(Arena& A, py::handle agg, Slice<py::handle>& flat_elements) {
    Slice<Unflatten> c;
    UType utype;
    py::handle obj;
    if (py::list_view::check(agg)) {
        obj = agg.type();
        utype = U_TUPLE_LIKE;
        py::list_view l(agg);
        for (auto i : l.enumerate()) {
            c.append(A, tree_flatten(A, l[i], flat_elements));
        }
    } else if (py::tuple_view::check(agg)) {
        obj = agg.type();
        utype = U_TUPLE_LIKE;
        // includes named tuples
        py::tuple_view l(agg);
        for (auto i : l.enumerate()) {
            c.append(A, tree_flatten(A, l[i], flat_elements));
        }
    } else if (py::dict_view::check(agg)) {
        utype = U_DICT;
        py::dict_view d(agg);
        obj = agg;
        Py_ssize_t pos = 0;
        py::handle k, v;
        while (d.next(&pos, &k, &v)) {
            c.append(A, tree_flatten(A, v, flat_elements));
        }
    } else {
        utype = U_ELEM;
        flat_elements.append(A, agg);
    }
    return Unflatten {utype, obj, c};
}

struct UnflattenArena {
    Arena A;
    Unflatten unflatten;
};

static PyObject* py_unflatten(PyObject *self,
                      PyObject *const *args,
                      Py_ssize_t nargs,
                      PyObject *kwnames) {
    PY_BEGIN
    #define ARGS(_) _(py::handle, ns)
    MPY_PARSE_ARGS_KWNAMES("O", ARGS)
    #undef ARGS
    py::sequence_view sv(ns);
    // because we do not have a autorelase pool yet...
    Arena A;
    Slice<py::handle> slice;
    py::handle Tuple = (PyObject*) &PyTuple_Type;
    auto inputs = Tuple.call(ns);
    py::tuple_view tv(inputs);
    for (auto i : tv.enumerate()) {
        slice.append(A, tv[i]);
    }
    auto AA = (UnflattenArena*) PyCapsule_GetPointer(self, "arena");
    auto r = AA->unflatten(slice).release();
    AT_ASSERT(r != nullptr);
    return r;
    PY_END(nullptr)
}

PyMethodDef py_unflatten_def = {"unflatten", (PyCFunction) py_unflatten, METH_FASTCALL | METH_KEYWORDS};

void free_unflatten_arena(PyObject * pc) {
    delete (UnflattenArena*) PyCapsule_GetPointer(pc, "arena");
}

static PyObject* py_tree_flatten(PyObject *self,
                      PyObject *const *args,
                      Py_ssize_t nargs,
                      PyObject *kwnames) {
    PY_BEGIN
    #define ARGS(_) _(py::handle, tree)
    MPY_PARSE_ARGS_KWNAMES("O", ARGS)
    #undef ARGS
    auto A = new UnflattenArena;
    Slice<py::handle> elements;
    A->unflatten = tree_flatten(A->A, tree, elements);
    auto cap = py::object::checked_steal(PyCapsule_New(A, "arena", free_unflatten_arena));
    auto unflatten = py::object::checked_steal(PyCFunction_New(&py_unflatten_def, cap.release()));
    py::tuple r(2);
    r.set(0, slice_to_list(elements));
    r.set(1, std::move(unflatten));
    return r.release();
    PY_END(nullptr)
}



py::object tree_map(Arena& A, std::function<py::handle(py::handle)> fn, py::handle agg) {
    Slice<py::handle> elements;
    auto unflatten = tree_flatten(A, agg, elements);
    for (auto i : elements.enumerate()) {
        elements[i] = fn(elements[i]);
    }
    return unflatten(elements);
}

// prereq: isinstance(h, _Tensor)
inline int64_t _Tensor_ndim(py::handle h) {
    if (Tensor::check(h)) {
        int64_t r = 0;
        for (auto l : Tensor::unchecked_wrap(h)->levels_.slice()) {
            if (l.is_positional()) {
                ++r;
            }
        }
        return r;
    }
    // Dim or DelayedMulTensor
    return 0;
}

inline py::handle handle_from_tensor(Arena& A, TensorRef t) {
    // fast case: tensor is live in python
    c10::optional<PyObject*> mb_obj =
        t->unsafeGetTensorImpl()->check_pyobj(getPyInterpreter());
    if (mb_obj.has_value() && !t->unsafeGetTensorImpl()->owns_pyobj()) {
        return *mb_obj;
    }
    return A.autorelease(py::object::checked_steal(THPVariable_Wrap(*t)));
}

struct EnableAllLayers {
    EnableAllLayers(Slice<DimEntry> levels) {
        std::vector<std::pair<int64_t, int64_t>> layers;
        layers.reserve(levels.size());
        for (auto l : levels) {
            if (!l.is_positional()) {
                auto d = l.dim();
                layers.emplace_back(d->level_ + LEVEL_OFFSET, d->size());
            }
        }
        std::sort(layers.begin(), layers.end());
        N = layers.size();
        at::functorch::_vmap_add_layers(layers);
    }
    ~EnableAllLayers() {
        at::functorch::_vmap_remove_layers(N);
    }
private:
    int64_t N;
};

TensorRef _match_levels(Arena& A, TensorRef v, Slice<DimEntry> from_levels, Slice<DimEntry> to_levels) {
    if (from_levels == to_levels) {
        return v;
    }
    at::IntArrayRef sz = v->sizes();
    at::IntArrayRef sd = v->strides();
    AT_ASSERT(from_levels.size() <= to_levels.size());
    Slice<int64_t> nsz;
    Slice<int64_t> nsd;
    for (auto l : to_levels) {
        auto oidx = from_levels.index(l);
        if (!oidx) {
            nsz.append(A, l.is_positional() ? 1 : l.dim()->size());
            nsd.append(A, 0);
        } else {
            auto idx = *oidx;
            nsz.append(A, sz[idx]);
            nsd.append(A, sd[idx]);
        }
    }
    return A.autorelease(v->as_strided(at::IntArrayRef(nsz.begin(), nsz.end()), at::IntArrayRef(nsd.begin(), nsd.end()), v->storage_offset()));
}


static py::object __torch_function__(Arena &A, py::handle orig, py::tuple_view args_, py::handle kwargs_) {
    maybeInitializeGlobals();
    bool is_pointwise = pointwise.contains(orig);

    if (orig == torch_Tensor___mul__) {
        AT_ASSERT(args_.size() == 2);
        if (py::isinstance(args_[0], _Tensor) && py::isinstance(args_[1], _Tensor) && _Tensor_ndim(args_[0]) == 0 && _Tensor_ndim(args_[1]) == 0) {
            std::cout << "__torch_function__ " << "delay" << " " << orig << "\n";

            return DelayedMulTensor.call_object(args_);
        }
    }
    std::cout << "__torch_function__ " << ((is_pointwise) ? "pointwise" : "functorch") << " " << orig << "\n";

    Slice<py::hdl<Dim>> all_dims;
    Slice<py::handle> flat_args;
    auto unflatten_args = tree_flatten(A, args_, flat_args);
    auto unflatten_kwargs = tree_flatten(A, kwargs_, flat_args);
    TensorRef device_holding_tensor;

    Slice<TensorInfo> infos;
    Slice<DimEntry> result_levels;
    for (auto f : flat_args) {
        infos.append(A, TensorInfo::create(A, f, !is_pointwise, false));
        if (infos.back()) {
            TensorInfo& info = infos.back();
            AT_ASSERT(is_pointwise || info.batchedtensor);
            if (!device_holding_tensor && info.has_device) {
                device_holding_tensor = infos.back().tensor;
            }
            for (auto l : info.levels) {
                if (!result_levels.contains(l)) {
                    result_levels.append(A, l);
                }
            }
        }
    }

    if (is_pointwise) {
        for (auto i : flat_args.enumerate()) {
            if (infos[i]) {
                TensorRef tensor = infos[i].tensor;
                if (device_holding_tensor && !infos[i].has_device) {
                    tensor = A.autorelease(tensor->to(device_holding_tensor->device()));
                }
                flat_args[i] = handle_from_tensor(A, _match_levels(A, tensor, infos[i].levels, result_levels));
            }
        }
        Slice<py::handle> flat_it = flat_args;
        py::object uargs = unflatten_args(flat_it);
        py::object ukwargs = unflatten_kwargs(flat_it);
        py::object result = orig.call_object(uargs, ukwargs);
        auto wrap = [&](py::handle h) {
            if (THPVariable_Check(h.ptr())){
                return A.autorelease(Tensor_from_positional(A, THPVariable_Unpack(h.ptr()), result_levels, device_holding_tensor));
            }
            return h;
        };
        return tree_map(A, wrap, result);
    } else {
        // std::cout << "rl: " << result_levels << "\n";
        EnableAllLayers guard(result_levels);
        for (auto i : flat_args.enumerate()) {
            if (infos[i]) {
                TensorRef batched = infos[i].batchedtensor;
                if (device_holding_tensor && !infos[i].has_device) {
                    batched = A.autorelease(batched->to(device_holding_tensor->device()));
                }
                flat_args[i] = handle_from_tensor(A, batched);
            }
        }
        Slice<py::handle> flat_it = flat_args;
        py::object uargs = unflatten_args(flat_it);
        py::object ukwargs = unflatten_kwargs(flat_it);
        // std::cout << uargs << " "  << ukwargs << "\n";
        AT_ASSERT(flat_it.size() == 0);
        py::object result = orig.call_object(uargs, ukwargs);
        auto wrap = [&](py::handle h) {
            if (THPVariable_Check(h.ptr())) {
                return A.autorelease(Tensor_from_batched(A, THPVariable_Unpack(h.ptr()), device_holding_tensor));
            }
            return h;
        };
        return tree_map(A, wrap, result);
    }
}


static PyObject* py___torch_function__(PyObject *self,
                      PyObject *const *args,
                      Py_ssize_t nargs,
                      PyObject *kwnames) {
    Arena A;
    PY_BEGIN
    #define ARGS(_) _(py::handle, cls) _(py::handle, orig) _(py::handle, cls2) _(py::tuple_view, args_) _(py::handle, kwargs_)
    MPY_PARSE_ARGS_KWNAMES("OOOO|O", ARGS)
    #undef ARGS
    if (!kwargs_.ptr()) {
        kwargs_ = empty_dict;
    }
    return __torch_function__(A, orig, args_, kwargs_).release();
    PY_END(nullptr)
}

py::object levels_to_tuple(Slice<DimEntry> slice) {
    py::tuple t(slice.size());
    for (auto i : slice.enumerate()) {
        t.set(i, slice[i].is_positional() ?  py::from_int(slice[i].position()) : py::object::borrow(slice[i].dim()));
    }
    py::object r = std::move(t);
    return r;
}

static PyGetSetDef Tensor_getsetters[] = {
   {"_has_device", (getter) [](PyObject* self, void*) -> PyObject* { return py::from_bool(((Tensor*)self)->has_device_).release(); }, NULL},
   {"_tensor", (getter) [](PyObject* self, void*) -> PyObject* { return THPVariable_Wrap(((Tensor*)self)->tensor_); }, NULL},
   {"_batchtensor", (getter) [](PyObject* self, void*) -> PyObject* { return THPVariable_Wrap(((Tensor*)self)->batchtensor_); }, NULL},
   {"_levels", (getter) [](PyObject* self, void*) -> PyObject* {
       PY_BEGIN
       return levels_to_tuple(((Tensor*)self)->levels_.slice()).release();
       PY_END(nullptr)
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
    0,            /* tp_init */
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
    s.append(A, 6);
    AT_ASSERT(s[3] == 6);
    for(int i : irange(10)) {
        s.append(A, i);
    }
    AT_ASSERT(s[0] == 3 && s.back() == 9 && s.size() == 14 && s.capacity() == 16);

    Slice<int> s2(A, -1, -2, -3);
    AT_ASSERT(s2[1] == -2 && s[0] == 3);

    auto ss = s.slice(1,2);
    AT_ASSERT(ss.size() == 1);
    AT_ASSERT(ss[0] == 4);
    AT_ASSERT(ss.capacity() == 1);
    ss.append(A, -4);
    AT_ASSERT(ss.size() == 2 && ss[1] == -4);
    ss[0] = 3;
    AT_ASSERT(s[1] == 4);

    s.insert(A, s.slice(1, 4), ss);
    AT_ASSERT(s[1] == 3  && s[2] == -4 && s[3] == 0);

    auto sz = s.size();
    s.insert(A, s.slice(1, 1), 4);
    AT_ASSERT(s[1] == 4 && sz + 1 == s.size());


    Slice<int> d(A, 0, 1, 2, 3, 4);

    Slice<int> b(A, 0, 1, 2, 3, 4);
    b.insert(A, b.slice(1,1), d);
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

PyMethodDef wrapper_method =  {"wrapper", (PyCFunction) call_torch_function, METH_VARARGS | METH_KEYWORDS };


static PyObject* _wrap_method(PyObject *self,
                      PyObject *const *args,
                      Py_ssize_t nargs,
                      PyObject *kwnames) {
    PY_BEGIN
    AT_ASSERT(nargs == 2);
    py::tuple s(2);
    s.set(0, py::object::borrow(args[0]));
    s.set(1, py::object::borrow(args[1]));
    auto r = py::object::checked_steal(PyCFunction_New(&wrapper_method, s.release()));
    return PyInstanceMethod_New(r.release());
    PY_END(nullptr);
}

static PyObject* positional(PyObject *_,
                      PyObject *const *args,
                      Py_ssize_t nargs,
                      PyObject *kwnames) {
    Arena A;
    PY_BEGIN
    AT_ASSERT(nargs-- > 0);
    auto self = Tensor::wrap(args++[0]);
    at::Tensor& data = self->tensor_;
    auto levels = Slice<DimEntry>();
    levels.extend(A, self->levels_.slice());


    at::IntArrayRef sz = data.sizes();
    at::IntArrayRef sd = data.strides();

    Slice<int64_t> view_sizes;
    Slice<DimEntry> new_levels;

    Slice<int64_t> nsz;
    Slice<int64_t> nsd;


    auto append = [&](py::hdl<Dim> d) {
        auto midx = levels.index(d);
        if (!midx) {
            py::raise_error(DimensionBindError(), "tensor of dimensions %R does not contain dim %R", levels_to_tuple(levels).ptr(), d.ptr());
        }
        auto idx = *midx;
        levels[idx] = DimEntry();
        nsz.append(A, sz[idx]);
        nsd.append(A, sd[idx]);
    };
    auto append_level = [&](int64_t sz) {
        view_sizes.append(A, sz);
        // we will recalculate the positional indices when we know how many were created
        new_levels.append(A, DimEntry());
    };

    bool needs_view = false;
    for (auto i :irange(nargs)) {
        py::handle arg  = args[i];
        if (DimList::check(arg)) {
            auto dl = DimList::unchecked_wrap(arg);
            for (py::obj<Dim> & d : dl->dims_) {
                append(d);
                append_level(d->size());
            }
        } else if (Dim::check(arg)) {
            auto d = Dim::unchecked_wrap(arg);
            append(d);
            append_level(d->size());

        } else {
            if (!py::is_sequence(arg)) {
                py::raise_error(PyExc_ValueError, "expected a Dim, List[Dim], or Sequence[Dim]");
            }
            py::sequence_view sq(arg);
            int64_t new_size = 1;
            for (auto j : sq.enumerate()) {
                py::obj<Dim> d = Dim::wrap(sq[j]);
                append(d);
                new_size *= d->size();
            }
            append_level(new_size);
            needs_view = true;
        }
    }
    for (auto i : levels.enumerate()) {
        auto & l = levels[i];
        if (l.is_none())
            continue;
        new_levels.append(A, l);
        nsz.append(A, sz[i]);
        nsd.append(A, sd[i]);
        view_sizes.append(A, sz[i]);
    }
    // recompute positional indices
    auto start = new_levels.size() - 1;
    int npositional = 0;
    for (auto i_ : new_levels.enumerate()) {
        auto i = start - i_;
        if (new_levels[i].is_positional() || new_levels[i].is_none()) {
            new_levels[i] = -(++npositional);
        }
    }
    at::Tensor ndata = data.as_strided(at::IntArrayRef(nsz.begin(), nsz.end()),
                                       at::IntArrayRef(nsd.begin(), nsd.end()), data.storage_offset());
    if (needs_view) {
        ndata = ndata.reshape(at::IntArrayRef(view_sizes.begin(), view_sizes.end()));
    }
    return Tensor_from_positional(A, std::move(ndata), new_levels, self->has_device_).release();
    PY_END(nullptr)
}

static PyObject* expand(PyObject *_,
                      PyObject *const *args,
                      Py_ssize_t nargs,
                      PyObject *kwnames) {
    Arena A;
    PY_BEGIN
    AT_ASSERT(nargs-- > 0);
    auto self = Tensor::wrap(args++[0]);
    for (auto i : irange(nargs)) {
        if (!Dim::check(args[i])) {
            maybeInitializeGlobals();
            auto newargs = slice_to_tuple(Slice<py::handle>((py::handle*)args - 1, (py::handle*)args + nargs));
            return __torch_function__(A, torch_Tensor_expand, newargs, empty_dict).release();
        }
    }
    at::Tensor& data = self->tensor_;
    auto levels = self->levels_.slice();
    Slice<DimEntry> new_levels;
    Slice<int64_t> sz;
    Slice<int64_t> sd;
    for (auto i : irange(nargs)) {
        auto d = Dim::unchecked_wrap(args[i]);
        if (levels.contains(d) || new_levels.contains(d)) {
            py::raise_error(DimensionBindError(), "expanding dimension %R already exists in tensor with dims", d.ptr());
        }
        new_levels.append(A, d);
        sz.append(A, d->size());
        sd.append(A, 0);
    }
    new_levels.extend(A, levels);
    at::IntArrayRef osz = data.sizes();
    at::IntArrayRef osd = data.strides();
    sz.extend(A, osz.begin(), osz.end());
    sd.extend(A, osd.begin(), osd.end());
    at::Tensor ndata = data.as_strided(at::IntArrayRef(sz.begin(), sz.end()), at::IntArrayRef(sd.begin(), sd.end()), data.storage_offset());
    return Tensor_from_positional(A, std::move(ndata), new_levels, self->has_device_).release();
    PY_END(nullptr)
}


void _bind_dims_to_size(Arena & A, int64_t sz, int64_t sd,
                        Slice<py::hdl<Dim>> dims, Slice<int64_t>& nsz, Slice<int64_t>& nsd) {
    int64_t rhs_prod = 1;
    for (auto i : dims.enumerate()) {
        if (!dims[i]->is_bound()) {
            for (auto j : irange(i + 1, dims.size())) {
                if (!dims[j]->is_bound()) {
                    py::raise_error(DimensionBindError(), "cannot infer the sizes of two dimensions at once %R and %R", dims[i].ptr(), dims[j].ptr());
                }
                rhs_prod *= dims[j]->size();
            }
            if (sz % rhs_prod != 0) {
                py::tuple tup(dims.size());
                for (auto j : dims.enumerate()) {
                    tup.set(j, dims[j]->is_bound() ? py::from_int(dims[j]->size()) : py::unicode_from_string("?"));
                }
                py::raise_error(DimensionBindError(), "inferred dimension does not evenly fit into larger dimension: %d vs %R", (int) sz, tup.ptr());
            }
            int64_t inferred_size = sz / rhs_prod;
            dims[i]->set_size(inferred_size);
            rhs_prod = sz;
            break;
        }
        rhs_prod *= dims[i]->size();
    }
    if (rhs_prod != sz) {
        py::tuple tup(dims.size());
        for (auto j : dims.enumerate()) {
            tup.set(j, py::object::borrow(dims[j]));
        }
        py::raise_error(DimensionBindError(), "Dimension sizes to do not match (%d != %d) when matching dimension pack %R", (int) sz, (int) rhs_prod, tup.ptr());
    }
    auto new_strides = A.allocate<int64_t>(dims.size());
    auto prev_stride = sd;
    for (auto i : dims.reversed_enumerate()) {
        new_strides[i] = prev_stride;
        prev_stride = dims[i]->size()*prev_stride;
    }
    for (auto i : dims.enumerate()) {
        nsd.append(A, new_strides[i]);
        nsz.append(A, dims[i]->size());
    }
}

static py::object __getitem__(Arena & A, py::handle self, py::handle index) {
    maybeInitializeGlobals();
    bool is_tuple = py::tuple_view::check(index);
    bool self_has_dims = py::isinstance(self, _Tensor);

    // nothing about first class dims here, fallback to getitem
    if (!self_has_dims && !is_tuple && !py::isinstance(index, _Tensor)) {
        return py::object::checked_steal(THPVariable_getitem(self.ptr(), index.ptr()));
    }

    // regularize single index vs tuple of indices

    Slice<py::handle> input;
    if (!is_tuple) {
        input.append(A, index);
    } else {
        py::tuple_view tv(index);
        for (auto i : tv.enumerate()) {
            input.append(A, tv[i]);
        }
    }

    bool can_call_original_getitem = !self_has_dims;
    int64_t dims_indexed = 0;
    int64_t expanding_object = -1;
    DimList* unbound_dim_list = nullptr;
    auto check_expanding = [&](int64_t i) {
        if (expanding_object != -1) {
            py::raise_error(DimensionBindError(), "at most one ... or unbound dimension list can exist in indexing list but found 2 at offsets %d and %d", (int) expanding_object, (int) i);
        }
        expanding_object = i;
    };
    Slice<int64_t> dimlists;

    // calculate how many dimensioned have been indexed in order to compute the size of ...
    // or expand a potentially unbound dimension list.

    bool has_dimpacks_or_none = false;
    for (auto i : input.enumerate()) {
        py::handle s = input[i];
        if (s.ptr() == Py_Ellipsis) {
            check_expanding(i);
        } else if (DimList::check(s)) {
            can_call_original_getitem = false;
            auto dl = DimList::unchecked_wrap(s);
            if (!dl->is_bound()) {
                check_expanding(i);
                unbound_dim_list = dl.ptr();
            } else {
                dims_indexed += dl->dims_.size();
            }
            dimlists.append(A, i);
        } else if (py::is_none(s)) {
            has_dimpacks_or_none = true;
        }  else {
            if (Dim::check(s) || Tensor::check(s)) {
                can_call_original_getitem = false;
            } else if (py::tuple_view::check(s)) {
                // can we avoid rechecking?
                py::tuple_view tv(s);
                if (tv.size() && Dim::check(tv[0])) {
                    can_call_original_getitem = false;
                    has_dimpacks_or_none = true;
                }
            }
            ++dims_indexed;
        }
    }

    // at this point if we haven't seen any Dim objects, we also can fallback to the original getitem.
    if (can_call_original_getitem) {
        return py::object::checked_steal(THPVariable_getitem(self.ptr(), index.ptr()));
    }

    std::cout << "__getitem__ " << self << " " << index << "\n";

    TensorInfo self_info = TensorInfo::create(A, self, false, true);
    auto ndim = self_info.ndim();
    if (dims_indexed > ndim) {
        py::raise_error(PyExc_ValueError, "at least %d indices were supplied but the tensor only has %d dimensions", (int) dims_indexed, (int) ndim);
    }
    // expand any unbound dimension list, or expand ... into individual : slices.
    auto expanding_dims = ndim - dims_indexed;
    if (expanding_object != -1) {
        if (unbound_dim_list) {
            unbound_dim_list->bind_len(expanding_dims);
        } else {
            // ...
            Slice<py::handle> no_slices;
            for (auto i : irange(expanding_dims)) {
                (void) i;
                no_slices.append(A, no_slice);
            }
            input.insert(A, input.slice(expanding_object, expanding_object + 1), no_slices);
        }
    }

    // flatten out any dimensions stored in dimlist elements directly into the inputs
    std::cout << dimlists << " <- dim lists!\n";
    for (int64_t i = dimlists.size() - 1; i >=0; --i) {
        auto idx = dimlists[i];
        // we added more elements to input because of ...
        // so we need to also adjust the index to get back to where the
        // dimlist existed
        if (!unbound_dim_list && expanding_object != -1 && idx > expanding_object) {
            idx += expanding_dims;
        }
        auto dl = DimList::unchecked_wrap(input[idx]);
        // XXX would be better if we used an OwnedSlice in DimList
        Slice<py::handle> more_dims((py::handle*) &*dl->dims_.begin(), (py::handle*) &*dl->dims_.end());
        std::cout << "INSERTING " << more_dims << " into " << input << "\n";
        input.insert(A, input.slice(idx, idx + 1), more_dims);
        std::cout << "AFTER " << input << "\n";
    }


    // At this point:
    // ..., DimList have been eliminated
    // Dim, Tensor, Tuple[Dim,...], int, slice still remain



    // we have to count how many times we see a dimension.
    // A[i,j] is a simple binding operation, but A[i, i+j] or A[i, i] requires advanced indexing.
    Slice<py::hdl<Dim>> seen_dims;
    Slice<int64_t> seen_dims_nuses;
    auto add_dim = [&](py::hdl<Dim> entry) {
        auto midx = seen_dims.index(entry);
        if (!midx) {
            seen_dims.append(A, entry);
            seen_dims_nuses.append(A, 1);
        } else {
            ++seen_dims_nuses[*midx];
        }
    };

    Slice<py::handle> input_it = input;

    Slice<py::handle> flat_inputs;
    // flat inputs will start with an empty py::handle if the
    // actual value is in the tensor-like object in the tensor info
    Slice<TensorInfo> tensor_inputs;

    auto append_flat_handle = [&](py::handle h) {
        flat_inputs.append(A, h);
        tensor_inputs.append(A, TensorInfo());
    };
    auto append_tensor_input = [&](TensorInfo ti) {
        flat_inputs.append(A, py::handle());
        tensor_inputs.append(A, ti);
    };

    Slice<int64_t> nsz;
    Slice<int64_t> nsd;
    at::IntArrayRef sz = self_info.tensor->sizes();
    at::IntArrayRef sd = self_info.tensor->strides();

    auto append_size = [&](int i) {
        if (has_dimpacks_or_none) {
            nsz.append(A, sz[i]);
            nsd.append(A, sd[i]);
        }
    };
    std::cout << "self levels: " << self_info.levels << "\n";

    auto parse_nones = [&]() {
        while (input_it.size() && py::is_none(input_it[0])) {
            append_flat_handle(no_slice);
            nsz.append(A, 1);
            nsd.append(A, 0);
            input_it = input_it.slice(1);
        }
    };

    // pair up the indexing expressions with dimension of self it indexes
    // self may have first-class dims, which do not participate the indexing.
    for (auto i : self_info.levels.enumerate()) {
        auto l = self_info.levels[i];
        if (!l.is_positional()) {
            add_dim(l.dim());
            append_flat_handle(l.dim());
            append_size(i);
            continue;
        }
        parse_nones();

        // we might have fewer indices than tensor dimensions,
        // which implicitly indexes the remaining dimensions with :
        if (!input_it.size()) {
            append_flat_handle(no_slice);
            append_size(i);
            continue;
        }

        py::handle arg = input_it[0];
        input_it = input_it.slice(1);
        if (Dim::check(arg)) {
            auto d = Dim::unchecked_wrap(arg);
            d->set_size(sz[i]);
            add_dim(d);
            append_size(i);
            append_flat_handle(arg);
        } else if (py::tuple_view::check(arg)) {
            py::tuple_view tv(arg);
            if (tv.size() && Dim::check(tv[0])) {
                // dim pack
                Slice<py::hdl<Dim>> dim_pack;
                for (auto j : tv.enumerate()) {
                    dim_pack.append(A, Dim::wrap(tv[j]));
                    add_dim(dim_pack.back());
                    append_flat_handle(dim_pack.back());
                }
                _bind_dims_to_size(A, sz[i], sd[i], dim_pack, nsz, nsd);

            } else {
                append_size(i);
                append_flat_handle(arg);
            }
        } else {
            append_size(i);
            TensorInfo info = TensorInfo::create(A, arg, false, false);
            if (info) {
                append_tensor_input(info);
                for (auto il : info.levels) {
                    if (!il.is_positional()) {
                        add_dim(il.dim());
                    }
                }
            } else {
                append_flat_handle(arg);
            }
        }
    }
    // any training Nones may have no existing dimension associated with them in self.
    parse_nones();

    // we have to restride the tensor to collapse dimension packs and introduce our none dimensions.
    if (has_dimpacks_or_none) {
        self_info.tensor = A.autorelease(self_info.tensor->as_strided(at::IntArrayRef(nsz.begin(), nsz.end()),at::IntArrayRef(nsd.begin(), nsd.end()), self_info.tensor->storage_offset()));
    }


    // figure out what the shape of the indexing tensors will be
    // and what the shape of the resulting tensor will be
    Slice<DimEntry> result_levels;
    Slice<DimEntry> index_levels;
    int64_t tensor_insert_point = -1;
    bool requires_getindex = false;
    for (auto i : flat_inputs.enumerate()) {
        auto inp = flat_inputs[i];
         if(tensor_inputs[i]) {
             requires_getindex = true;
             if (tensor_insert_point == -1) {
                 tensor_insert_point = result_levels.size();
             }
             for (auto l : tensor_inputs[i].levels) {
                 std::cout << "Consider to add " << l << "\n";
                 if (!index_levels.contains(l)) {
                     index_levels.append(A, l);
                 }
             }
        } else if (Dim::check(inp)) {
            auto d = Dim::unchecked_wrap(inp);
            // dimesions used once are just binding operations
            if (1 == seen_dims_nuses[*seen_dims.index(d)]) {
                flat_inputs[i] = no_slice;
                result_levels.append(A, d);
            } else {
                requires_getindex = true;
                flat_inputs[i] = py::handle();
                tensor_inputs[i] = TensorInfo {d->range(), Slice<DimEntry>(A, DimEntry(d)), false, TensorRef()};
            }
         } else {
            if (inp.ptr() != no_slice.ptr()) {
                requires_getindex = true;
            }
            if (!py::is_int(inp)) {
                // note: actual positional indexes are accurately computed later
                result_levels.append(A, -1);
            }
         }
    }

    // indexing dimensions appear in the tensor at the _first use of a tensor_ in the indexing. So insert
    // the indexing leveles into the result klevels at this spot
    if (tensor_insert_point != -1) {
        result_levels.insert(A, result_levels.slice(tensor_insert_point, tensor_insert_point), index_levels);
    }

    std::cout << "flat inputs: " << flat_inputs << "\n";
    std::cout << "result_levels: " << result_levels << "\n";
    std::cout << "index_levels: " << index_levels << "\n";

    // get all the tensors to be the right shape for indexing
    if (requires_getindex) {
        for (auto i : flat_inputs.enumerate()) {
            if (tensor_inputs[i]) {
                AT_ASSERT(!flat_inputs[i].ptr());
                std::cout << "tensor " << i << " " << tensor_inputs[i].levels << "\n";
                flat_inputs[i] = handle_from_tensor(A, _match_levels(A, tensor_inputs[i].tensor, tensor_inputs[i].levels, index_levels));
            }
        }
    }

    // previously we didn't know how many positional dimensions there would be so we couldn't number them right
    // so fill it in now.
    auto seen_positionals = 0;
    for (auto i : result_levels.reversed_enumerate()) {
        if (result_levels[i].is_positional()) {
            result_levels[i] = -(++seen_positionals);
        }
    }

    at::Tensor rtensor;
    if (requires_getindex) {
        auto self_hdl = handle_from_tensor(A, self_info.tensor);
        auto tup = slice_to_tuple(flat_inputs);
        std::cout << "calling original getindex " << self_hdl << " " << tup << "\n";
        auto pytensor = py::object::checked_steal(THPVariable_getitem(self_hdl.ptr(), tup.ptr()));
        rtensor = THPVariable_Unpack(pytensor.ptr());
    } else {
        std::cout << "skipping original getindex\n";
        rtensor = *self_info.tensor;
    }
    std::cout << "returning (from_positional)\n";
    return Tensor_from_positional(A, std::move(rtensor), result_levels, self_info.has_device);
}

static PyObject* py___getitem__(PyObject *_,
                      PyObject *const *args,
                      Py_ssize_t nargs,
                      PyObject *kwnames) {
    Arena A;
    PY_BEGIN
    AT_ASSERT(nargs == 2);
    return __getitem__(A, args[0], args[1]).release();
    PY_END(nullptr)
}

static PyMethodDef methods[] = {
    {"dims", (PyCFunction) dims, METH_FASTCALL | METH_KEYWORDS},
    {"_test_c", (PyCFunction) test_c, METH_FASTCALL | METH_KEYWORDS},
    {"_wrap_method", (PyCFunction) _wrap_method, METH_FASTCALL | METH_KEYWORDS},
    {"_n_levels_in_use", [](PyObject*,PyObject*) -> PyObject* { return PyLong_FromLongLong(n_levels_in_use); }, METH_NOARGS},
    {"Tensor_from_positional", (PyCFunction) py_Tensor_from_positional, METH_FASTCALL | METH_KEYWORDS},
    {"Tensor_from_batched", (PyCFunction) py_Tensor_from_batched, METH_FASTCALL | METH_KEYWORDS},
    {"__torch_function__", (PyCFunction) py___torch_function__, METH_FASTCALL | METH_KEYWORDS},
    {"tree_flatten", (PyCFunction) py_tree_flatten, METH_FASTCALL | METH_KEYWORDS},
    {"positional", (PyCFunction) positional, METH_FASTCALL | METH_KEYWORDS},
    {"expand", (PyCFunction) expand, METH_FASTCALL | METH_KEYWORDS},
    {"__getitem__", (PyCFunction) py___getitem__, METH_FASTCALL | METH_KEYWORDS},

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
        Py_INCREF(&PyInstanceMethod_Type);
        PyModule_AddObject(mod.ptr(), "_instancemethod", (PyObject *)&PyInstanceMethod_Type);
        return mod.release();
    } catch(py::exception_set& err) {
        return nullptr;
    }
}
