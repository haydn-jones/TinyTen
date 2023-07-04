#pragma once

#include <cmath>

#include "tensor.hpp"

namespace tt::inline v1 {
    template <typename T>
    constexpr auto Tensor<T>::sin_() -> Tensor& requires SupportsSin<T> {
        return this->map_(std::sin);
    }

    template <typename T>
    constexpr auto Tensor<T>::sin() const -> Tensor {
        return Tensor(*this).sin_();
    }

    template <typename T>
    constexpr auto Tensor<T>::cos_() -> Tensor& requires SupportsCos<T> {
        return this->map_(std::cos);
    }

    template <typename T>
    constexpr auto Tensor<T>::cos() const -> Tensor {
        return Tensor(*this).cos_();
    }

    template <typename T>
    constexpr auto Tensor<T>::tan_() -> Tensor& requires SupportsTan<T> {
        return this->map_(std::tan);
    }

    template <typename T>
    constexpr auto Tensor<T>::tan() const -> Tensor {
        return Tensor(*this).tan_();
    }

    template <typename T>
    constexpr auto Tensor<T>::cot_() -> Tensor& requires SupportsCot<T> {
        return this->map_([](T x) constexpr { return static_cast<T>(1) / std::tan(x); });
    }

    template <typename T>
    constexpr auto Tensor<T>::cot() const -> Tensor {
        return Tensor(*this).cot_();
    }

    template <typename T>
    constexpr auto Tensor<T>::sec_() -> Tensor& requires SupportsSec<T> {
        return this->map_([](T x) constexpr { return static_cast<T>(1) / std::cos(x); });
    }

    template <typename T>
    constexpr auto Tensor<T>::sec() const -> Tensor {
        return Tensor(*this).sec_();
    }

    template <typename T>
    constexpr auto Tensor<T>::csc_() -> Tensor& requires SupportsCsc<T> {
        return this->map_([](T x) constexpr { return static_cast<T>(1) / std::sin(x); });
    }

    template <typename T>
    constexpr auto Tensor<T>::csc() const -> Tensor {
        return Tensor(*this).csc_();
    }
};  // namespace tt::inline v1