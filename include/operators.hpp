#pragma once

#include <concepts>
#include <functional>
#include <stdexcept>
#include <type_traits>

namespace tt::inline v1 {
    namespace details {
        // clang-format off
        template <typename T>
        concept add_operator_supported = requires(T& t, T& u) {
            { t + u } -> std::same_as<T>;
        };
        template <typename T>
        concept sub_operator_supported = requires(T& t, T& u) {
            { t - u } -> std::same_as<T>;
        };
        template <typename T>
        concept mul_operator_supported = requires(T& t, T& u) {
            { t * u } -> std::same_as<T>;
        };
        template <typename T>
        concept div_operator_supported = requires(T& t, T& u) {
            { t / u } -> std::same_as<T>;
        };
        // clang-format on
    }  // namespace details

    template <typename T>
    class Tensor;

    template <typename T>
    constexpr auto operator+(const Tensor<T>& a, const Tensor<T>& b) requires details::add_operator_supported<T> {
        if (a.shape() != b.shape()) {
            throw std::runtime_error("Shapes are not the same");
        }
        Tensor<T> result(a.shape());
        std::transform(a.begin(), a.end(), b.begin(), result.begin(), std::plus<>{});
        return result;
    }

    template <typename T>
    constexpr auto operator-(const Tensor<T>& a, const Tensor<T>& b) requires details::sub_operator_supported<T> {
        if (a.shape() != b.shape()) {
            throw std::runtime_error("Shapes are not the same");
        }
        Tensor<T> result(a.shape());
        std::transform(a.begin(), a.end(), b.begin(), result.begin(), std::minus<>{});
        return result;
    }

    template <typename T>
    constexpr auto operator*(const Tensor<T>& a, const Tensor<T>& b) requires details::mul_operator_supported<T> {
        if (a.shape() != b.shape()) {
            throw std::runtime_error("Shapes are not the same");
        }
        Tensor<T> result(a.shape());
        std::transform(a.begin(), a.end(), b.begin(), result.begin(), std::multiplies<>{});
        return result;
    }

    template <typename T>
    constexpr auto operator/(const Tensor<T>& a, const Tensor<T>& b) requires details::div_operator_supported<T> {
        if (a.shape() != b.shape()) {
            throw std::runtime_error("Shapes are not the same");
        }
        Tensor<T> result(a.shape());
        std::transform(a.begin(), a.end(), b.begin(), result.begin(), std::divides<>{});
        return result;
    }

}  // namespace tt::inline v1