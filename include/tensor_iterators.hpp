#pragma once

#include "tensor.hpp"

namespace tt::inline v1 {
    // Iterators
    template <typename T>
    constexpr auto Tensor<T>::begin() -> typename Tensor::ContainerType::iterator {
        return data_.begin();
    }

    template <typename T>
    constexpr auto Tensor<T>::begin() const -> typename Tensor::ContainerType::const_iterator {
        return data_.cbegin();
    }

    template <typename T>
    constexpr auto Tensor<T>::end() -> typename Tensor::ContainerType::iterator {
        return data_.end();
    }

    template <typename T>
    constexpr auto Tensor<T>::end() const -> typename Tensor::ContainerType::const_iterator {
        return data_.cend();
    }

    template <typename T>
    auto Tensor<T>::shape_iter() -> ShapeIter {
        return ShapeIter(this->numel(), this->shape_);
    }
};  // namespace tt::inline v1