#pragma once

#include <functional>
#include <stdexcept>

#include "concepts.hpp"
#include "tensor.hpp"

namespace tt::inline v1 {
    template <typename T>
    class Tensor;

    template <typename T>
    constexpr auto operator+(const Tensor<T>& a, const Tensor<T>& b) requires SupportsAdd<T> {
        if (a.shape() != b.shape()) {
            throw std::runtime_error("Shapes are not the same");
        }
        Tensor<T> result(a.shape());
        std::transform(a.begin(), a.end(), b.begin(), result.begin(), std::plus<>{});
        return result;
    }

    template <typename T>
    constexpr auto operator-(const Tensor<T>& a, const Tensor<T>& b) requires SupportsSub<T> {
        if (a.shape() != b.shape()) {
            throw std::runtime_error("Shapes are not the same");
        }
        Tensor<T> result(a.shape());
        std::transform(a.begin(), a.end(), b.begin(), result.begin(), std::minus<>{});
        return result;
    }

    template <typename T>
    constexpr auto operator*(const Tensor<T>& a, const Tensor<T>& b) requires SupportsMul<T> {
        if (a.shape() != b.shape()) {
            throw std::runtime_error("Shapes are not the same");
        }
        Tensor<T> result(a.shape());
        std::transform(a.begin(), a.end(), b.begin(), result.begin(), std::multiplies<>{});
        return result;
    }

    template <typename T>
    constexpr auto operator/(const Tensor<T>& a, const Tensor<T>& b) requires SupportsDiv<T> {
        if (a.shape() != b.shape()) {
            throw std::runtime_error("Shapes are not the same");
        }
        Tensor<T> result(a.shape());
        std::transform(a.begin(), a.end(), b.begin(), result.begin(), std::divides<>{});
        return result;
    }

}  // namespace tt::inline v1