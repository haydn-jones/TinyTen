
#pragma once

#include "tensor.hpp"

namespace tt::inline v1 {
    template <typename T>
    constexpr auto Tensor<T>::flat(SizeType i) -> T& {
        if (i >= this->numel()) {
            throw std::runtime_error("flat: index out of bounds");
        }
        return this->data_[i];
    }

    template <typename T>
    constexpr auto Tensor<T>::flat(SizeType i) const -> const T& {
        if (i >= this->numel()) {
            throw std::runtime_error("flat: index out of bounds");
        }
        return this->data_[i];
    }

    // Indexing with vector
    template <typename T>
    constexpr auto Tensor<T>::operator()(const IndexType& indices) -> T& {
        assert(indices.size() == shape_.size());
        return this->data_[this->_flatten_index(indices)];
    }

    template <typename T>
    constexpr auto Tensor<T>::operator()(const IndexType& indices) const -> const T& {
        assert(indices.size() == shape_.size());
        return this->data_[this->_flatten_index(indices)];
    }

    template <typename T>
    constexpr auto Tensor<T>::flatten_index(const IndexType& indices) const -> SizeType {
        SizeType index = 0;
        SizeType stride = 1;
        for (int64_t i = this->shape_.size() - 1; i >= 0; --i) {
            index += indices[i] * stride;
            stride *= this->shape_[i];
        }
        return index;
    }

    template <typename T>
    constexpr auto Tensor<T>::unflatten_index(SizeType index) const -> IndexType {
        IndexType result;
        result.reserve(this->shape_.size());
        for (int64_t i = static_cast<int64_t>(this->shape_.size()) - 1; i >= 0; --i) {
            auto idx = index % this->shape_[i];
            index /= this->shape_[i];
            result.push_back(idx);
        }

        std::reverse(result.begin(), result.end());
        return result;
    }

    // Internal indexing, handles strided tensors
    template <typename T>
    constexpr auto Tensor<T>::_unflatten_index(SizeType flat_index) const -> IndexType {
        IndexType idx(shape_.size());
        std::transform(
            strides_.begin(), strides_.end(), idx.begin(), [&flat_index](size_t stride) constexpr {
                size_t idx = flat_index / stride;
                flat_index %= stride;
                return idx;
            });
        return idx;
    }

    // Internal indexing, handles strided tensors
    template <typename T>
    constexpr auto Tensor<T>::_flatten_index(const IndexType& indices) const -> SizeType {
        if (indices.size() != shape_.size()) {
            throw std::runtime_error("flatten_index: size mismatch");
        }
        return std::transform_reduce(indices.begin(), indices.end(), strides_.begin(), static_cast<SizeType>(0),
                                     std::plus<>(), std::multiplies<>());
    }
};  // namespace tt::inline v1