#pragma once

#include "tensor.hpp"

namespace tt::inline v1 {
    template <typename T>
    constexpr auto Tensor<T>::begin() -> typename Tensor::ContainerType::iterator {
        if (this->_is_contiguous()) {
            return this->data_.begin();
        } else {
            throw std::runtime_error("begin: tensor is not contiguous");
        }
    }

    template <typename T>
    constexpr auto Tensor<T>::begin() const -> typename Tensor::ContainerType::const_iterator {
        if (this->_is_contiguous()) {
            return this->data_.cbegin();
        } else {
            throw std::runtime_error("begin: tensor is not contiguous");
        }
    }

    template <typename T>
    constexpr auto Tensor<T>::end() -> typename Tensor::ContainerType::iterator {
        if (this->_is_contiguous()) {
            return this->data_.end();
        } else {
            throw std::runtime_error("end: tensor is not contiguous");
        }
    }

    template <typename T>
    constexpr auto Tensor<T>::end() const -> typename Tensor::ContainerType::const_iterator {
        if (this->_is_contiguous()) {
            return this->data_.cend();
        } else {
            throw std::runtime_error("end: tensor is not contiguous");
        }
    }

    template <typename T>
    auto Tensor<T>::shape_iter() -> ShapeIter {
        return ShapeIter(this->numel(), this->canon_strides_);
    }

    template <typename T>
    auto Tensor<T>::strided_iter() -> StridedIter<T> {
        return StridedIter<T>(&this->data_[0], this->numel(), this->strides_, this->canon_strides_);
    }
};  // namespace tt::inline v1