#pragma once

#include "tensor.hpp"

namespace tt::inline v1 {
    template <typename T>
    constexpr auto Tensor<T>::begin() -> StridedIterImpl<T> {
        return this->indexer_.begin(this->data_);
    }

    template <typename T>
    constexpr auto Tensor<T>::begin() const -> StridedIterImpl<const T> {
        return this->indexer_.begin(this->data_);
    }

    template <typename T>
    constexpr auto Tensor<T>::end() -> StridedIterImpl<T> {
        return this->indexer_.end(this->data_);
    }

    template <typename T>
    constexpr auto Tensor<T>::end() const -> StridedIterImpl<const T> {
        return this->indexer_.end(this->data_);
    }

    template <typename T>
    constexpr auto Tensor<T>::stlbegin() {
        if (this->_is_contiguous()) {
            return this->indexer_.stlbegin(this->data_);
        } else {
            throw std::runtime_error("begin: tensor is not contiguous");
        }
    }

    template <typename T>
    constexpr auto Tensor<T>::stlbegin() const {
        if (this->_is_contiguous()) {
            return this->indexer_.stlbegin(this->data_);
        } else {
            throw std::runtime_error("begin: tensor is not contiguous");
        }
    }

    template <typename T>
    constexpr auto Tensor<T>::stlend() -> typename Tensor::ContainerType::iterator {
        if (this->_is_contiguous()) {
            return this->indexer_.stlend(this->data_);
        } else {
            throw std::runtime_error("end: tensor is not contiguous");
        }
    }

    template <typename T>
    constexpr auto Tensor<T>::stlend() const -> typename Tensor::ContainerType::const_iterator {
        if (this->_is_contiguous()) {
            return this->indexer_.stlend(this->data_);
        } else {
            throw std::runtime_error("end: tensor is not contiguous");
        }
    }

    template <typename T>
    auto Tensor<T>::shape_iter() -> ShapeIter {
        return this->indexer_.shape_iter();
    }

    template <typename T>
    auto Tensor<T>::strided_iter() -> StridedIter<T> {
        return this->indexer_.strided_iter(this->data_);
    }
};  // namespace tt::inline v1