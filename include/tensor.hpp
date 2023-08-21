#pragma once

#include <algorithm>
#include <cassert>
#include <execution>
#include <numeric>
#include <random>
#include <utility>
#include <vector>

#include "concepts.hpp"
#include "tensor_indexer.hpp"
#include "types.hpp"
#include "utils/ShapeIterator.hpp"
#include "utils/TensorIterator.hpp"
#include "utils/utils.hpp"

namespace tt::inline v1 {
    template <typename T>
    class Tensor {
      public:
        using ValueType = T;
        using ContainerType = std::vector<T>;

        ////////////////////////////////////////////////////////////////////
        // Constructors
        ////////////////////////////////////////////////////////////////////
        Tensor() : indexer_(TensorIndexer<T>::contigous({})) {}

        Tensor(IndexType shape) : indexer_(TensorIndexer<T>::contigous(shape)) {
            this->data_.resize(this->indexer_.numel());
        }

        Tensor(const IndexType shape, const ValueType& value) : Tensor(shape) {
            std::fill(this->data_.begin(), this->data_.end(), value);
        }

        template <typename U = ValueType>
        constexpr auto static iota(const IndexType shape, U value = {}) -> Tensor {
            Tensor tensor(shape);
            std::generate(tensor.begin(), tensor.end(), [&value] { return value++; });
            return tensor;
        }

        auto static randn(const IndexType shape) -> Tensor {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::normal_distribution<> d(static_cast<ValueType>(0), static_cast<ValueType>(1));
            Tensor tensor(shape);

            std::generate(tensor.begin(), tensor.end(), [&d, &gen] { return d(gen); });
            return tensor;
        }

        [[nodiscard]] constexpr auto numel() const noexcept -> SizeType {
            return this->indexer_.numel();
        }

        [[nodiscard]] constexpr auto shape() const noexcept -> const IndexType& {
            return this->indexer_.shape();
        }
        [[nodiscard]] constexpr auto shape(SizeType i) const -> SizeType {
            return this->indexer_.shape(i);
        }

        [[nodiscard]] constexpr auto dim() const noexcept -> SizeType {
            return this->indexer_.dim();
        }

        [[nodiscard]] constexpr auto strides() const noexcept -> const IndexType& {
            return this->indexer_.strides();
        }

        [[nodiscard]] constexpr auto stride(int i) const -> SizeType {
            return this->indexer_.stride(i);
        }

        [[nodiscard]] constexpr auto _is_contiguous() const -> bool {
            return this->indexer_._is_contiguous();
        }

        constexpr void reshape_(const IndexType& shape) {
            if (cumprod(shape) != this->numel()) {
                throw std::runtime_error("reshape: total size of new array must be unchanged");
            }
            // right now we cannot reshape a non-contiguous tensor
            if (!this->_is_contiguous()) {
                throw std::runtime_error("reshape: tensor is not contiguous");
            }
            this->indexer_ = TensorIndexer<T>(shape, tt::calc_strides(shape), tt::calc_strides(shape));
        }

        constexpr auto reshape(const IndexType& shape) const -> Tensor {
            Tensor tensor(*this);
            tensor.reshape_(shape);
            return tensor;
        }

        constexpr auto permute_(const IndexType& axes) {
            auto shape = permute_vec(this->indexer_.shape(), axes);
            auto stride = permute_vec(this->indexer_.strides(), axes);
            this->indexer_ = TensorIndexer<T>(shape, stride, tt::calc_strides(shape));
        }

        constexpr auto permute(const IndexType& axes) const -> Tensor {
            Tensor tensor(*this);
            tensor.permute_(axes);
            return tensor;
        }

        constexpr auto flat(SizeType i) -> ValueType&;
        constexpr auto flat(SizeType i) const -> const ValueType&;

        constexpr auto operator()(const IndexType& indices) -> ValueType&;
        constexpr auto operator()(const IndexType& indices) const -> const ValueType&;

        template <std::convertible_to<SizeType>... I>
        constexpr auto operator()(I... i) -> ValueType& {
            IndexType indices{static_cast<SizeType>(i)...};
            return this->data_[tt::ravel_index(indices, this->indexer_.strides())];
        }

        template <std::convertible_to<SizeType>... I>
        constexpr auto operator()(I... i) const -> const ValueType& {
            IndexType indices{static_cast<SizeType>(i)...};
            return this->data_[tt::ravel_index(indices, this->indexer_.strides())];
        }

        template <std::convertible_to<SizeType>... I>
        constexpr auto ravel_index(I... i) const -> SizeType {
            IndexType indices{static_cast<SizeType>(i)...};
            return this->ravel_index(indices);
        }

        [[nodiscard]] constexpr auto ravel_index(const IndexType& indices) const -> SizeType;

        [[nodiscard]] constexpr auto unravel_index(SizeType index) const -> IndexType;

        // Iterators
        constexpr auto stlbegin();
        constexpr auto stlbegin() const;
        constexpr auto stlend() -> typename ContainerType::iterator;
        constexpr auto stlend() const -> typename ContainerType::const_iterator;

        constexpr auto begin() -> StridedIterImpl<ValueType>;
        constexpr auto begin() const -> StridedIterImpl<const ValueType>;
        constexpr auto end() -> StridedIterImpl<ValueType>;
        constexpr auto end() const -> StridedIterImpl<const ValueType>;

        auto shape_iter() -> ShapeIter;
        auto strided_iter() -> StridedIter<ValueType>;

        ////////////////////////////////////////////////////////////////////
        // Trigonometric functions
        ////////////////////////////////////////////////////////////////////
        constexpr auto sin_() -> Tensor& requires SupportsSin<ValueType>;
        constexpr auto sin() const -> Tensor;

        constexpr auto cos_() -> Tensor& requires SupportsCos<ValueType>;
        constexpr auto cos() const -> Tensor;

        constexpr auto tan_() -> Tensor& requires SupportsTan<ValueType>;
        constexpr auto tan() const -> Tensor;

        constexpr auto cot_() -> Tensor& requires SupportsCot<ValueType>;
        constexpr auto cot() const -> Tensor;

        constexpr auto sec_() -> Tensor& requires SupportsSec<ValueType>;
        constexpr auto sec() const -> Tensor;

        constexpr auto csc_() -> Tensor& requires SupportsCsc<ValueType>;
        constexpr auto csc() const -> Tensor;

        ////////////////////////////////////////////////////////////////////
        // Misc functions
        ////////////////////////////////////////////////////////////////////
        constexpr auto static gather(Tensor& input, int64_t dim, Tensor<int64_t>& index) -> Tensor;

        template <typename U>
        constexpr auto astype() -> Tensor<U> {
            Tensor<U> res(this->indexer_.shape_, this->indexer_.strides_);
            std::transform(
                this->begin(), this->end(), res.begin(), [](T & x) constexpr { return static_cast<U>(x); });
            return res;
        }

        constexpr auto map_(ValueType (*f)(ValueType)) -> Tensor& {
            std::transform(std::execution::unseq, this->data_.begin(), this->data_.end(), this->data_.begin(), f);
            return *this;
        }

        constexpr auto map(ValueType (*f)(ValueType)) const -> Tensor {
            return Tensor(*this).map_(f);
        }

      private:
        TensorIndexer<T> indexer_;
        ContainerType data_;

        template <typename U>
        friend class Tensor;

        Tensor(const IndexType& dimensions, const IndexType& strides) : Tensor(dimensions) {
            this->indexer_.strides_ = strides;
        }
    };
};  // namespace tt::inline v1
