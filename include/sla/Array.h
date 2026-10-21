// sla/Array.h — contêineres contíguos alinhados (sla::Array) e matriz 2D por linhas (sla::Array2D).
#pragma once

#include <cstddef>
#include <limits>
#include <new>
#include <span>
#include <stdexcept>
#include <vector>

namespace sla {

/// Alocador com alinhamento fixo (padrão 64 B: linha de cache e registradores AVX-512),
/// para que laços sobre pixels e nós vetorizem sem prólogo de alinhamento.
template <class T, std::size_t Align = 64>
class AlignedAllocator {
    static_assert(Align >= alignof(T), "AlignedAllocator: alinhamento menor que alignof(T)");
    static_assert((Align & (Align - 1)) == 0, "AlignedAllocator: alinhamento deve ser potência de 2");

public:
    using value_type = T;

    template <class U>
    struct rebind {
        using other = AlignedAllocator<U, Align>;
    };

    AlignedAllocator() noexcept = default;
    template <class U>
    constexpr AlignedAllocator(const AlignedAllocator<U, Align>&) noexcept {}

    [[nodiscard]] T* allocate(std::size_t n) {
        if (n > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
            throw std::bad_array_new_length();
        }
        return static_cast<T*>(::operator new(n * sizeof(T), std::align_val_t{Align}));
    }

    void deallocate(T* p, std::size_t) noexcept { ::operator delete(p, std::align_val_t{Align}); }

    template <class U>
    constexpr bool operator==(const AlignedAllocator<U, Align>&) const noexcept {
        return true;
    }
};

/// Vetor contíguo alinhado. Atenção: Array<bool> herda a especialização compactada de
/// std::vector<bool>; use std::uint8_t para máscaras.
template <class T>
using Array = std::vector<T, AlignedAllocator<T>>;

/// Matriz 2D densa, armazenamento por linhas: elemento (x, y) em data()[y·width + x].
/// operator() não verifica limites (laços internos); at() verifica e lança std::out_of_range.
template <class T>
class Array2D {
public:
    Array2D() = default;
    Array2D(int width, int height, const T& value = T{})
        : width_(checkedDim(width)),
          height_(checkedDim(height)),
          data_(static_cast<std::size_t>(width_) * static_cast<std::size_t>(height_), value) {}

    int width() const noexcept { return width_; }
    int height() const noexcept { return height_; }
    std::size_t size() const noexcept { return data_.size(); }
    bool empty() const noexcept { return data_.empty(); }

    bool inBounds(int x, int y) const noexcept { return x >= 0 && y >= 0 && x < width_ && y < height_; }
    std::size_t index(int x, int y) const noexcept {
        return static_cast<std::size_t>(y) * static_cast<std::size_t>(width_) + static_cast<std::size_t>(x);
    }

    T& operator()(int x, int y) noexcept { return data_[index(x, y)]; }
    const T& operator()(int x, int y) const noexcept { return data_[index(x, y)]; }

    T& at(int x, int y) {
        if (!inBounds(x, y)) throw std::out_of_range("sla::Array2D::at: coordenada fora dos limites");
        return data_[index(x, y)];
    }
    const T& at(int x, int y) const {
        if (!inBounds(x, y)) throw std::out_of_range("sla::Array2D::at: coordenada fora dos limites");
        return data_[index(x, y)];
    }

    std::span<T> row(int y) noexcept { return {data_.data() + index(0, y), static_cast<std::size_t>(width_)}; }
    std::span<const T> row(int y) const noexcept {
        return {data_.data() + index(0, y), static_cast<std::size_t>(width_)};
    }

    T* data() noexcept { return data_.data(); }
    const T* data() const noexcept { return data_.data(); }

    void fill(const T& value) {
        for (T& v : data_) v = value;
    }

private:
    static int checkedDim(int d) {
        if (d < 0) throw std::invalid_argument("sla::Array2D: dimensão negativa");
        return d;
    }

    int width_ = 0;
    int height_ = 0;
    Array<T> data_;
};

}  // namespace sla
