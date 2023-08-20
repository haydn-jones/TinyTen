#pragma once

#include "tensor.hpp"

namespace tt::inline v1 {
    template <typename T>
    constexpr auto Tensor<T>::flat(SizeType i) -> T& {
        if (i >= this->numel()) {
            throw std::runtime_error("flat: index out of bounds");
        }
        return this->data_[tt::ravel_unravel(i, this->strides_, this->canon_strides_)];
    }

    template <typename T>
    constexpr auto Tensor<T>::flat(SizeType i) const -> const T& {
        if (i >= this->numel()) {
            throw std::runtime_error("flat: index out of bounds");
        }
        return this->data_[tt::ravel_unravel(i, this->strides_, this->canon_strides_)];
    }

    // Indexing with vector
    template <typename T>
    constexpr auto Tensor<T>::operator()(const IndexType& indices) -> T& {
        assert(indices.size() == this->dim());
        return this->data_[tt::ravel_index(indices, this->strides_)];
    }

    template <typename T>
    constexpr auto Tensor<T>::operator()(const IndexType& indices) const -> const T& {
        assert(indices.size() == this->dim());
        return this->data_[tt::ravel_index(indices, this->strides_)];
    }

    template <typename T>
    constexpr auto Tensor<T>::ravel_index(const IndexType& indices) const -> SizeType {
        return tt::ravel_index(indices, this->canon_strides_);
    }

    template <typename T>
    constexpr auto Tensor<T>::unravel_index(SizeType index) const -> IndexType {
        return tt::unravel_index(index, this->canon_strides_);
    }
};  // namespace tt::inline v1