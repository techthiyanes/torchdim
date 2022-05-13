#pragma once
#include <ATen/ATen.h>
#include "minpybind.h"


inline int round2(int num) {
   int nzeros = __builtin_clz(num - 1);
   return 1 << (32 - nzeros);
}

struct Arena;
template<typename T>
struct OwnedSlice;

template<typename T>
struct Slice {
    Slice()
    :  begin_(nullptr), size_(0), capacity_(0) {}

    template<typename... Args>
    Slice(Arena& arena, Args&&... args);

    T* begin() const {
        return begin_;
    }
    T* end() const {
        return begin_ + size_;
    }
    int size() const {
        return size_;
    }
    int capacity() const {
        return capacity_;
    }

    T& back(int i=-1) {
        return begin_[size_ + i];
    }

    T& operator[](int i) const {
        return begin_[i];
    }
    c10::optional<int> index(const T& value) {
        for (int i : enumerate()) {
            if (begin_[i] == value) {
                return i;
            }
        }
        return c10::nullopt;
    }

    Slice insert(Arena& arena, Slice where, Slice to_insert);
    Slice insert(Arena& arena, Slice where, T v) {
        return insert(arena, where, Slice(&v, &v + 1));
    }
    Slice insert(Arena& arena, int where, T v) {
        return insert(arena, slice(where, where), v);
    }
    Slice append(Arena& arena, T value);
    Slice extend(Arena& arena, Slice to_insert) {
        return insert(arena, slice(size_), to_insert);
    }

    Slice slice(int begin) {
        return slice(begin, size_);
    }

    Slice slice(int begin, int end) {
        if (begin < 0) {
            begin += size_;
        }
        if (end < 0) {
            end += size_;
        }
        Slice result;
        result.begin_ = begin_ + begin;
        result.size_ = end - begin;
        result.capacity_ = result.size_;
        return result;
    }

    bool inside(Slice where) {
        return begin() <= where.begin() && where.end() <= end();
    }

    c10::integer_range<int> enumerate() const {
        return c10::irange(size_);
    }

protected:
    Slice(T* begin, T* end)
    : begin_(begin), size_(end - begin), capacity_(size_) {}

    static int _length(const T& t) {
        return 1;
    }
    static int _length(Slice t) {
        return t.size_;
    }
    static T* _insert(T*& dst, T t) {
        *dst = std::move(t);
        return ++dst;
    }
    static T* _insert(T*& dst, Slice t) {
        std::memcpy(dst, t.begin_, sizeof(T)*t.size_);
        dst += t.size_;
        return dst;
    }
    T* begin_;
    int size_;
    int capacity_;
    friend struct OwnedSlice<T>;
};

template<typename T>
struct OwnedSlice : public Slice<T> {
    typedef void (*deleter_t)(Slice<T>);
    static void _no_delete(Slice<T>) {}
    OwnedSlice()
    : deleter_(_no_delete) {}
    OwnedSlice(const OwnedSlice&) = delete;
    OwnedSlice& operator=(const OwnedSlice&) = delete;
    ~OwnedSlice() {
        deleter_(slice_);
        if (slice_.size_ > 8) {
            delete [] slice_.begin_;
        }
    }
    void set(Slice<T> to_own, deleter_t deleter = _no_delete) {
        slice_.size_ = slice_.capacity_ = to_own.size();
        slice_.begin_ = (slice_.size_ > 8) ? new T[slice_.size_] : &small_buf[0];
        std::memcpy(slice_.begin_, to_own.begin(), slice_.size_ * sizeof(T));
        deleter_ = deleter;
    }
    Slice<T> slice() const {
        return slice_;
    }
private:
    Slice<T> slice_;
    deleter_t deleter_;
    T small_buf[8];
};

template<typename T>
inline std::ostream& operator<<(std::ostream& s, const Slice<T>& v) {
    s << "[";
    for (int i : v.enumerate()) {
        if (i > 0) {
            s << ", ";
        }
        s << v[i];
    }
    s << "]";
    return s;
}

struct TensorRef {
    TensorRef()
    : impl_(nullptr){}
    TensorRef(const at::Tensor& t)
    : impl_(t.unsafeGetTensorImpl()) {}
    at::Tensor& operator*() const {
        return *(at::Tensor*)this;
    }
    at::Tensor* operator->() const {
        return (at::Tensor*)this;
    }
    operator bool() const {
        return impl_ != nullptr;
    }
private:
    at::TensorImpl* impl_;
};

constexpr int ARENA_MAX_SIZE = 4096;
constexpr int ALIGNMENT = 8;
struct Arena {
    Arena()
    : allocated_(0) {}
    template<typename T>
    T* allocate(int n) {
        if (!n) {
            return nullptr;
        }
        int to_allocate = sizeof(T)*n;
        int to_allocate_rounded = ALIGNMENT * ((to_allocate - 1) / ALIGNMENT + 1);
        T* result = (T*) &buffer_[allocated_];
        allocated_ += to_allocate_rounded;
        AT_ASSERT(allocated_ <= ARENA_MAX_SIZE);
        return result;
    }
    TensorRef autorelease(at::Tensor s) {
        auto ref = TensorRef(s);
        s.unsafeReleaseTensorImpl();
        ar_tensors_ = ar_tensors_.append(*this, ref);
        return ref;
    }
    py::handle autorelease(py::object obj) {
        ar_objects_ = ar_objects_.append(*this, obj);
        obj.release();
        return ar_objects_.back();
    }
    ~Arena() {
        for(TensorRef t: ar_tensors_) {
            c10::intrusive_ptr<at::TensorImpl, at::UndefinedTensorImpl>::reclaim(t->unsafeGetTensorImpl());
        }
        for(py::handle h: ar_objects_) {
            py::object::steal(h);
        }
    }
private:
    int64_t allocated_;
    char buffer_[ARENA_MAX_SIZE];
    Slice<TensorRef> ar_tensors_;
    Slice<py::handle> ar_objects_;
};

template<typename T>
inline Slice<T> Slice<T>::insert(Arena& arena, Slice where, Slice to_insert) {
    AT_ASSERT(inside(where));
    Slice result = *this;
    /// b------sb---se-----e,  0----n
    T* body_dest = where.begin();
    if (where.size() != to_insert.size()) {
        int new_size = size() - where.size() + to_insert.size();
        T* tail_dest = where.begin() + to_insert.size();
        if (new_size >= capacity_) {
            int new_capacity = new_size ? round2(new_size) : 0;
            result.capacity_ = new_capacity;
            result.begin_ = arena.allocate<T>(new_capacity);
            body_dest = result.begin_ + (where.begin() - begin());
            tail_dest = body_dest + to_insert.size();
            std::memcpy(result.begin_, begin_, sizeof(T)*(where.begin() - begin()));
        }
        std::memmove(tail_dest, where.end(), sizeof(T)*(end() - where.end()));
        result.size_ = new_size;
    }

    std::memcpy(body_dest, to_insert.begin(), sizeof(T)*to_insert.size());
    return result;
}

template<typename T>
inline Slice<T> Slice<T>::append(Arena& arena, T value) {
    Slice result = *this;
    if (size_ == capacity_) {
        int new_size = size_ ? round2(size_)*2 : 8;
        T* n = arena.allocate<T>(new_size);
        memcpy(n, begin_, size_*sizeof(T));
        result.begin_ = n;
        result.capacity_ = new_size;
    }
    result[result.size_++] = std::move(value);
    return result;
}

template<typename T>
template<typename... Args>
Slice<T>::Slice(Arena& arena, Args&&... args) {
    int lens[] = {_length(args)...};
    size_ = 0;
    for (auto i : lens) {
        size_ += i;
    }
    capacity_ = round2(std::max(size_, 8));
    begin_ = arena.allocate<T>(capacity_);
    T* dst_ = begin_;
    T* unused[] = {_insert(dst_, args)...};
    (void) unused;
}
