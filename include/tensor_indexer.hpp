#pragma once

#include <utility>

#include "types.hpp"
#include "utils/ShapeIterator.hpp"
#include "utils/TensorIterator.hpp"
#include "utils/utils.hpp"

namespace tt::inline v1 {
    template <typename T>
    class TensorIndexer {
      public:
        IndexType shape_;
        IndexType strides_;
        IndexType canon_strides_;

        TensorIndexer(IndexType shape, IndexType strides, IndexType canon_strides)
            : shape_(std::move(shape)), strides_(std::move(strides)), canon_strides_(std::move(canon_strides)) {}

        static auto contigous(const IndexType& shape) -> TensorIndexer {
            IndexType strides = tt::calc_strides(shape);
            return TensorIndexer(shape, strides, strides);
        }

        [[nodiscard]] constexpr auto numel() const noexcept -> SizeType {
            return tt::cumprod(this->shape_);
        }

        [[nodiscard]] constexpr auto shape() const noexcept -> const IndexType& {
            return this->shape_;
        }

        [[nodiscard]] constexpr auto shape(SizeType i) const -> SizeType {
            if (i >= this->dim()) {
                throw std::runtime_error("shape: index out of bounds");
            }
            return this->shape_[i];
        }

        [[nodiscard]] constexpr auto dim() const noexcept -> SizeType {
            return this->shape_.size();
        }

        [[nodiscard]] constexpr auto strides() const noexcept -> const IndexType& {
            return this->strides_;
        }

        [[nodiscard]] constexpr auto stride(int i) const -> SizeType {
            return this->strides_[i];
        }

        [[nodiscard]] constexpr auto _is_contiguous() const -> bool {
            return this->strides_ == this->canon_strides_;
        }

        constexpr auto begin(std::vector<T>& data) -> StridedIterImpl<T> {
            return {&data[0], 0, this->strides_, this->canon_strides_};
        }

        [[nodiscard]] constexpr auto begin(const std::vector<T>& data) const -> StridedIterImpl<const T> {
            return {&data[0], 0, this->strides_, this->canon_strides_};
        }

        constexpr auto end(std::vector<T>& data) -> StridedIterImpl<T> {
            return {&data[0], this->numel(), this->strides_, this->canon_strides_};
        }

        [[nodiscard]] constexpr auto end(const std::vector<T>& data) const -> StridedIterImpl<const T> {
            return {&data[0], this->numel(), this->strides_, this->canon_strides_};
        }

        constexpr auto stlbegin(std::vector<T>& data) {
            if (this->_is_contiguous()) {
                return data.begin();
            } else {
                throw std::runtime_error("begin: tensor is not contiguous");
            }
        }

        constexpr auto stlbegin(std::vector<T>& data) const {
            if (this->_is_contiguous()) {
                return data.cbegin();
            } else {
                throw std::runtime_error("begin: tensor is not contiguous");
            }
        }

        constexpr auto stlend(std::vector<T>& data) -> typename std::vector<T>::iterator {
            if (this->_is_contiguous()) {
                return data.end();
            } else {
                throw std::runtime_error("end: tensor is not contiguous");
            }
        }

        constexpr auto stlend(std::vector<T>& data) const -> typename std::vector<T>::const_iterator {
            if (this->_is_contiguous()) {
                return data.cend();
            } else {
                throw std::runtime_error("end: tensor is not contiguous");
            }
        }

        auto shape_iter() -> ShapeIter {
            return ShapeIter(this->numel(), this->canon_strides_);
        }

        auto strided_iter(std::vector<T>& data) -> StridedIter<T> {
            return StridedIter<T>(&data[0], this->numel(), this->strides_, this->canon_strides_);
        }
    };
};  // namespace tt::inline v1