
#pragma once

#include "tensor.hpp"

namespace tt::inline v1 {
    template <typename T>
    constexpr auto Tensor<T>::flat(SizeType i) -> T& {
        if (i >= this->numel()) {
            throw std::runtime_error("flat: index out of bounds");
        }
        return this->data_[tt::ravel_index(tt::unravel_index(i, this->canon_strides_), this->strides_)];
    }

    template <typename T>
    constexpr auto Tensor<T>::flat(SizeType i) const -> const T& {
        if (i >= this->numel()) {
            throw std::runtime_error("flat: index out of bounds");
        }
        return this->data_[tt::ravel_index(tt::unravel_index(i, this->canon_strides_), this->strides_)];
    }

    // Indexing with vector
    template <typename T>
    constexpr auto Tensor<T>::operator()(const IndexType& indices) -> T& {
        assert(indices.size() == this->shape_.size());
        return this->data_[this->_ravel_index(indices)];
    }

    template <typename T>
    constexpr auto Tensor<T>::operator()(const IndexType& indices) const -> const T& {
        assert(indices.size() == this->shape_.size());
        return this->data_[this->_ravel_index(indices)];
    }

    template <typename T>
    constexpr auto Tensor<T>::ravel_index(const IndexType& indices) const -> SizeType {
        return tt::ravel_index(indices, this->canon_strides_);
    }

    template <typename T>
    constexpr auto Tensor<T>::unravel_index(SizeType index) const -> IndexType {
        return tt::unravel_index(index, this->canon_strides_);
    }

    // Internal indexing, handles strided tensors
    template <typename T>
    constexpr auto Tensor<T>::_unravel_index(SizeType flat_index) const -> IndexType {
        return tt::unravel_index(flat_index, this->strides_);
    }

    // Internal indexing, handles strided tensors
    template <typename T>
    constexpr auto Tensor<T>::_ravel_index(const IndexType& indices) const -> SizeType {
        return tt::ravel_index(indices, this->strides_);
    }
};  // namespace tt::inline v1