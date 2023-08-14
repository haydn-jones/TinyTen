#pragma once

#include "tensor.hpp"

namespace tt::inline v1 {
    template <typename T>
    [[nodiscard]] constexpr auto Tensor<T>::numel() const noexcept -> SizeType {
        if (this->shape_.empty() && this->data_.empty()) {
            return 0;
        } else if (this->shape_.empty()) {
            return 1;
        } else {
            return tt::cumprod(this->shape_);
        }
    }

    template <typename T>
    [[nodiscard]] constexpr auto Tensor<T>::shape() const noexcept -> const IndexType& {
        return this->shape_;
    }

    template <typename T>
    [[nodiscard]] constexpr auto Tensor<T>::shape(SizeType i) const -> SizeType {
        if (i >= this->shape_.size()) {
            throw std::runtime_error("shape: index out of bounds");
        }
        return this->shape_[i];
    }

    template <typename T>
    [[nodiscard]] constexpr auto Tensor<T>::strides() const noexcept -> const IndexType& {
        return this->strides_;
    }

    template <typename T>
    [[nodiscard]] constexpr auto Tensor<T>::stride(int i) const -> SizeType {
        return this->strides_[i];
    }

    template <typename T>
    constexpr auto Tensor<T>::_is_contiguous() const -> bool {
        return this->strides_ == this->canon_strides_;
    }

};  // namespace tt::inline v1