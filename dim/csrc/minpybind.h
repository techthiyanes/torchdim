#pragma once
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <utility>


#define PY_BEGIN try {
#define PY_END(v) } catch(py::exception_set & err) { return (v); }

namespace py {

struct exception_set {
};

struct object;

struct handle {
    handle(PyObject* ptr)
    : ptr_(ptr) {}
    handle()
    : ptr_(nullptr) {}

    PyObject* ptr() const {
        return ptr_;
    }
    object attr(const char* key);

    template<typename... Args>
    object call(Args&&... args);
    object call_object(py::handle args);
    object call_object(py::handle args, py::handle kwargs);

    bool operator==(handle rhs) {
        return ptr_ == rhs.ptr_;
    }

protected:
    PyObject * ptr_;
};

template<typename T>
struct obj;

template<typename T>
struct hdl : public handle {
    T* ptr() {
        return  (T*) handle::ptr();
    }
    T* operator->() {
        return ptr();
    }
    hdl(T* ptr)
    : hdl((PyObject*) ptr) {}
    hdl(obj<T> o)
    : hdl(o.ptr()) {}
private:
    hdl(handle h) : handle(h) {}
};

struct object : public handle {
    object() {}
    object(const object& other)
    : handle(other.ptr_) {
        Py_XINCREF(ptr_);
    }
    object(object&& other)
    : handle(other.ptr_) {
        other.ptr_ = nullptr;
    }
    object& operator=(const object& other) {
        return *this = object(other);
    }
    object& operator=(object&& other) {
        PyObject* tmp = ptr_;
        ptr_ = other.ptr_;
        other.ptr_ = tmp;
        return *this;
    }
    ~object() {
        Py_XDECREF(ptr_);
    }
    static object steal(handle o) {
        return object(o.ptr());
    }
    static object checked_steal(handle o) {
        if (!o.ptr()) {
            throw exception_set();
        }
        return steal(o);
    }
    static object borrow(handle o) {
        Py_XINCREF(o.ptr());
        return steal(o);
    }
    PyObject* release() {
        auto tmp = ptr_;
        ptr_ = nullptr;
        return tmp;
    }
protected:
    explicit object(PyObject* ptr)
    : handle(ptr) {}
};

template<typename T>
struct obj : public object {
    obj() {}
    obj(const obj& other)
    : object(other.ptr_) {
        Py_XINCREF(ptr_);
    }
    obj(obj&& other)
    : object(other.ptr_) {
        other.ptr_ = nullptr;
    }
    obj& operator=(const obj& other) {
        return *this = obj(other);
    }
    obj& operator=(obj&& other) {
        PyObject* tmp = ptr_;
        ptr_ = other.ptr_;
        other.ptr_ = tmp;
        return *this;
    }
    static obj steal(hdl<T> o) {
        return obj(o.ptr());
    }
    static obj checked_steal(hdl<T> o) {
        if (!o.ptr()) {
            throw exception_set();
        }
        return steal(o);
    }
    static obj borrow(hdl<T> o) {
        Py_XINCREF(o.ptr());
        return steal(o);
    }
    T* ptr() const {
        return (T*) object::ptr();
    }
    T* operator->() {
        return ptr();
    }
protected:
    explicit obj(T* ptr)
    : object((PyObject*)ptr) {}
};


bool isinstance(handle h, handle c) {
    return PyObject_IsInstance(h.ptr(), c.ptr());
}

[[ noreturn ]] void raise_error(handle exception, const char *format, ...) {
    va_list args;
    va_start(args, format);
    PyErr_FormatV(exception.ptr(), format, args);
    va_end(args);
    throw exception_set();
}

template<typename T>
struct base {
    PyObject_HEAD
    PyObject* ptr() const {
        return (PyObject*) this;
    }
    static obj<T> alloc(PyTypeObject* type = nullptr) {
        if (!type) {
            type = &T::Type;
        }
        auto self = (T*) type->tp_alloc(type, 0);
        if (!self) {
            throw py::exception_set();
        }
        new (self) T;
        return obj<T>::steal(self);
    }
    template<typename ... Args>
    static obj<T> create(Args ... args) {
        auto self = alloc();
        self->init(std::forward<Args>(args)...);
        return self;
    }
    static bool check(handle v) {
        return isinstance(v, (PyObject*)&T::Type);
    }

    static hdl<T> unchecked_wrap(handle self_) {
        return hdl<T>((T*)self_.ptr());
    }
    static hdl<T> wrap(handle self_) {
        if (!check(self_)) {
            raise_error(PyExc_ValueError, "not an instance of %S", &T::Type);
        }
        return unchecked_wrap(self_);
    }

    static obj<T> unchecked_wrap(object self_) {
        return obj<T>::steal(unchecked_wrap(self_.release()));
    }
    static obj<T> wrap(object self_) {
        return obj<T>::steal(wrap(self_.release()));
    }

    static PyObject* new_stub(PyTypeObject *type, PyObject *args, PyObject *kwds) {
        PY_BEGIN
        return (PyObject*) alloc(type).release();
        PY_END(nullptr)
    }
    static void dealloc_stub(PyObject *self) {
        ((T*)self)->~T();
        Py_TYPE(self)->tp_free(self);
    }
    static void ready(py::handle mod, const char* name) {
        if (PyType_Ready(&T::Type)) {
            throw exception_set();
        }
        if(PyModule_AddObject(mod.ptr(), name, (PyObject*) &T::Type) < 0) {
            throw exception_set();
        }
    }
};

inline object handle::attr(const char* key) {
    return object::checked_steal(PyObject_GetAttrString(ptr(), key));
}

inline object import(const char* module) {
    return object::checked_steal(PyImport_ImportModule(module));
}

template<typename... Args>
inline object handle::call(Args&&... args) {
    return object::checked_steal(PyObject_CallFunctionObjArgs(ptr_, args.ptr()..., nullptr));
}

inline object handle::call_object(py::handle args) {
    return object::checked_steal(PyObject_CallObject(ptr(), args.ptr()));
}


inline object handle::call_object(py::handle args, py::handle kwargs) {
    return object::checked_steal(PyObject_Call(ptr(), args.ptr(), kwargs.ptr()));
}

struct tuple : public object {
    void set(int i, object v) {
        PyTuple_SET_ITEM(ptr_, i, v.release());
    }
    tuple(int size)
    : object(checked_steal(PyTuple_New(size))) {}
};



py::object unicode_from_format(const char* format, ...) {
    va_list args;
    va_start(args, format);
    auto r = PyUnicode_FromFormatV(format, args);
    va_end(args);
    return py::object::checked_steal(r);
}
py::object unicode_from_string(const char * str) {
    return py::object::checked_steal(PyUnicode_FromString(str));
}

py::object from_int(Py_ssize_t s) {
    return py::object::checked_steal(PyLong_FromSsize_t(s));
}
py::object from_bool(bool b) {
    return py::object::borrow(b ? Py_True : Py_False);
}

bool is_sequence(handle h) {
    return PySequence_Check(h.ptr());
}


struct sequence_view : public handle {
    sequence_view(handle h)
    : handle(h) {}
    Py_ssize_t size() const {
        auto r = PySequence_Size(ptr());
        if (r == -1 && PyErr_Occurred()) {
            throw py::exception_set();
        }
        return r;
    }
    static sequence_view wrap(handle h) {
        if (!is_sequence(h)) {
            raise_error(PyExc_ValueError, "expected a sequence");
        }
        return sequence_view(h);
    }
    py::object operator[](Py_ssize_t i) const {
        return py::object::checked_steal(PySequence_GetItem(ptr(), i));
    }
};


py::object repr(handle h) {
    return py::object::checked_steal(PyObject_Repr(h.ptr()));
}

py::object str(handle h) {
    return py::object::checked_steal(PyObject_Str(h.ptr()));
}


bool is_int(handle h) {
    return PyLong_Check(h.ptr());
}

bool is_float(handle h) {
    return PyFloat_Check(h.ptr());
}

bool is_none(handle h) {
    return h.ptr() == Py_None;
}

bool is_bool(handle h) {
    return PyBool_Check(h.ptr());
}

Py_ssize_t to_int(handle h) {
    Py_ssize_t r = PyLong_AsSsize_t(h.ptr());
    if (r == -1 && PyErr_Occurred()) {
        throw py::exception_set();
    }
    return r;
}

double to_float(handle h) {
    double r = PyFloat_AsDouble(h.ptr());
    if (PyErr_Occurred()) {
        throw py::exception_set();
    }
    return r;
}

bool to_bool_unsafe(handle h) {
    return h.ptr() == Py_True;
}

struct slice_view {
    slice_view(handle h, Py_ssize_t size)  {
        if(PySlice_Unpack(h.ptr(), &start, &stop, &step) == -1) {
            throw py::exception_set();
        }
        slicelength = PySlice_AdjustIndices(size, &start, &stop, step);
    }
    Py_ssize_t start, stop, step, slicelength;
};

bool is_slice(handle h) {
    return PySlice_Check(h.ptr());
}

}

#define MPY_ARGS_NAME(typ, name) #name ,
#define MPY_ARGS_DECLARE(typ, name) typ name;
#define MPY_ARGS_POINTER(typ, name) &name ,
#define MPY_PARSE_ARGS_KWARGS(fmt, FORALL_ARGS) \
    static char* kwlist[] = { FORALL_ARGS(MPY_ARGS_NAME) nullptr}; \
    FORALL_ARGS(MPY_ARGS_DECLARE) \
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, fmt, kwlist, FORALL_ARGS(MPY_ARGS_POINTER) nullptr)) { \
        throw py::exception_set(); \
    }

#define MPY_PARSE_ARGS_KWNAMES(fmt, FORALL_ARGS) \
    static const char * const kwlist[] = { FORALL_ARGS(MPY_ARGS_NAME) nullptr}; \
    FORALL_ARGS(MPY_ARGS_DECLARE) \
    static _PyArg_Parser parser = {fmt, kwlist, 0}; \
    if (!_PyArg_ParseStackAndKeywords(args, nargs, kwnames, &parser, FORALL_ARGS(MPY_ARGS_POINTER nullptr))) { \
        throw py::exception_set(); \
    }
