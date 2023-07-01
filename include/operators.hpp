#pragma once

#include <concepts>
#include <functional>
#include <stdexcept>
#include <type_traits>

namespace tt::inline v1 {
    namespace details {
        // clang-format off
        template <typename T, typename U>
        concept add_operator_supported = requires(T& t, U& u) {
            { t + u } -> std::same_as<std::common_type_t<T, U>>;
        };
        template <typename T, typename U>
        concept sub_operator_supported = requires(T& t, U& u) {
            { t - u } -> std::same_as<std::common_type_t<T, U>>;
        };
        template <typename T, typename U>
        concept mul_operator_supported = requires(T& t, U& u) {
            { t * u } -> std::same_as<std::common_type_t<T, U>>;
        };
        template <typename T, typename U>
        concept div_operator_supported = requires(T& t, U& u) {
            { t / u } -> std::same_as<std::common_type_t<T, U>>;
        };
        // clang-format on
    }  // namespace details

    template <typename T>
    class Tensor;

    template <typename T, typename U>
    constexpr auto operator+(const Tensor<T>& a, const Tensor<U>& b) requires details::add_operator_supported<T, U> {
        if (a.shape() != b.shape()) {
            throw std::runtime_error("Shapes are not the same");
        }
        using otype = typename std::common_type<T, U>::type;
        Tensor<otype> result(a.shape());
        std::transform(a.begin(), a.end(), b.begin(), result.begin(), std::plus<>{});
        return result;
    }

    template <typename T, typename U>
    constexpr auto operator-(const Tensor<T>& a, const Tensor<U>& b) requires details::sub_operator_supported<T, U> {
        if (a.shape() != b.shape()) {
            throw std::runtime_error("Shapes are not the same");
        }
        using otype = typename std::common_type<T, U>::type;
        Tensor<otype> result(a.shape());
        std::transform(a.begin(), a.end(), b.begin(), result.begin(), std::minus<>{});
        return result;
    }

    template <typename T, typename U>
    constexpr auto operator*(const Tensor<T>& a, const Tensor<U>& b) requires details::mul_operator_supported<T, U> {
        if (a.shape() != b.shape()) {
            throw std::runtime_error("Shapes are not the same");
        }
        using otype = typename std::common_type<T, U>::type;
        Tensor<otype> result(a.shape());
        std::transform(a.begin(), a.end(), b.begin(), result.begin(), std::multiplies<>{});
        return result;
    }

    template <typename T, typename U>
    constexpr auto operator/(const Tensor<T>& a, const Tensor<U>& b) requires details::div_operator_supported<T, U> {
        if (a.shape() != b.shape()) {
            throw std::runtime_error("Shapes are not the same");
        }
        using otype = typename std::common_type<T, U>::type;
        Tensor<otype> result(a.shape());
        std::transform(a.begin(), a.end(), b.begin(), result.begin(), std::divides<>{});
        return result;
    }

}  // namespace tt::inline v1