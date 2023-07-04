#pragma once

#include <algorithm>
#include <cassert>
#include <execution>
#include <numeric>
#include <random>
#include <vector>

#include "concepts.hpp"
#include "utils/ShapeIterator.hpp"
#include "utils/utils.hpp"

namespace tt::inline v1 {
    using SizeType = int64_t;
    using IndexType = std::vector<int64_t>;

    template <typename T>
    class Tensor {
      public:
        using ValueType = T;
        using ContainerType = std::vector<T>;

        ////////////////////////////////////////////////////////////////////
        // Constructors
        ////////////////////////////////////////////////////////////////////
        Tensor() = default;

        Tensor(const IndexType& shape) : shape_(shape) {
            this->_calc_strides();
            this->data_.resize(this->numel());
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
            if (this->shape_.empty() && this->data_.empty()) {
                return 0;
            } else if (this->shape_.empty()) {
                return 1;
            } else {
                return tt::cumprod(this->shape_);
            }
        }

        [[nodiscard]] constexpr auto shape() const noexcept -> const IndexType& {
            return this->shape_;
        }

        [[nodiscard]] constexpr auto shape(SizeType i) const -> SizeType {
            if (i >= this->shape_.size()) {
                throw std::runtime_error("shape: index out of bounds");
            }
            return this->shape_[i];
        }

        [[nodiscard]] constexpr auto strides() const noexcept -> const IndexType& {
            return this->strides_;
        }

        constexpr void reshape_(const IndexType& shape) {
            if (cumprod(shape) != this->numel()) {
                throw std::runtime_error("reshape: total size of new array must be unchanged");
            }
            this->shape_ = shape;
            this->_calc_strides();
        }

        constexpr auto reshape(const IndexType& shape) const -> Tensor {
            Tensor tensor(*this);
            tensor.reshape_(shape);
            return tensor;
        }

        constexpr auto permute_(const IndexType& axes) {
            this->shape_ = permute_vec(this->shape_, axes);
            this->strides_ = permute_vec(this->strides_, axes);
            this->canon_strides_ = tt::_calc_strides(this->shape_);
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

        template <typename... Args>
        constexpr auto operator()(Args... args) -> ValueType& {
            assert(sizeof...(args) == this->shape_.size());
            return this->data_[tt::ravel_index({static_cast<SizeType>(args)...}, this->strides_)];
        }

        template <typename... Args>
        constexpr auto operator()(Args... args) const -> const ValueType& {
            assert(sizeof...(args) == this->shape_.size());
            return this->data_[tt::ravel_index({static_cast<SizeType>(args)...}, this->strides_)];
        }

        template <typename... Args>
        constexpr auto ravel_index(Args... args) const -> SizeType {
            IndexType indices{static_cast<SizeType>(args)...};
            return this->ravel_index(indices);
        }
        [[nodiscard]] constexpr auto ravel_index(const IndexType& indices) const -> SizeType;

        [[nodiscard]] constexpr auto unravel_index(SizeType index) const -> IndexType;

        // Iterators
        constexpr auto begin() -> typename ContainerType::iterator;
        constexpr auto begin() const -> typename ContainerType::const_iterator;
        constexpr auto end() -> typename ContainerType::iterator;
        constexpr auto end() const -> typename ContainerType::const_iterator;
        auto shape_iter() -> ShapeIter;

        [[nodiscard]] constexpr auto stride(int i) const -> SizeType {
            return this->strides_[i];
        }

        constexpr auto map_(ValueType (*f)(ValueType)) -> Tensor& {
            std::transform(std::execution::unseq, this->data_.begin(), this->data_.end(), this->data_.begin(), f);
            return *this;
        }

        constexpr auto map(ValueType (*f)(ValueType)) const -> Tensor {
            return Tensor(*this).map_(f);
        }

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

        template <typename U>
        constexpr auto astype() -> Tensor<U> {
            Tensor<U> res(this->shape_, this->strides_);
            std::transform(
                this->begin(), this->end(), res.begin(), [](T & x) constexpr { return static_cast<U>(x); });
            return res;
        }

      private:
        ContainerType data_;
        IndexType shape_;

        // True strides
        IndexType strides_;
        IndexType canon_strides_;

        template <typename U>
        friend class Tensor;

        Tensor(const IndexType& dimensions, const IndexType& strides) : Tensor(dimensions) {
            this->strides_ = strides;
        }

        constexpr void _calc_strides() {
            this->strides_ = tt::_calc_strides(this->shape_);
            this->canon_strides_ = this->strides_;
        }
    };
};  // namespace tt::inline v1
