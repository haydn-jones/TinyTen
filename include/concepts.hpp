#pragma once

#include <cmath>
#include <concepts>

namespace tt::inline v1 {
    template <typename ValueType>
    concept SupportsSin = requires(ValueType x) {
        { std::sin(x) } -> std::same_as<ValueType>;
    };

    template <typename ValueType>
    concept SupportsCos = requires(ValueType x) {
        { std::cos(x) } -> std::same_as<ValueType>;
    };

    template <typename ValueType>
    concept SupportsTan = requires(ValueType x) {
        { std::tan(x) } -> std::same_as<ValueType>;
    };

    template <typename ValueType>
    concept SupportsCot = requires(ValueType x) {
        { static_cast<ValueType>(1) / std::tan(x) } -> std::same_as<ValueType>;
    };

    template <typename ValueType>
    concept SupportsSec = requires(ValueType x) {
        { static_cast<ValueType>(1) / std::cos(x) } -> std::same_as<ValueType>;
    };

    template <typename ValueType>
    concept SupportsCsc = requires(ValueType x) {
        { static_cast<ValueType>(1) / std::sin(x) } -> std::same_as<ValueType>;
    };

    template <typename T>
    concept SupportsAdd = requires(T& t, T& u) {
        { t + u } -> std::same_as<T>;
    };

    template <typename T>
    concept SupportsSub = requires(T& t, T& u) {
        { t - u } -> std::same_as<T>;
    };

    template <typename T>
    concept SupportsMul = requires(T& t, T& u) {
        { t* u } -> std::same_as<T>;
    };

    template <typename T>
    concept SupportsDiv = requires(T& t, T& u) {
        { t / u } -> std::same_as<T>;
    };
}  // namespace tt::inline v1