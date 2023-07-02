#pragma once

#include <ShapeIterator.hpp>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <concepts.hpp>
#include <execution>
#include <numeric>
#include <operators.hpp>
#include <random>
#include <utils.hpp>
#include <vector>

namespace tt::inline v1 {
    template <typename T>
    class Tensor {
      public:
        using ValueType = T;
        using ContainerType = std::vector<T>;
        using SizeType = typename ContainerType::size_type;
        using IndexType = std::vector<size_t>;

        ////////////////////////////////////////////////////////////////////
        // Constructors
        ////////////////////////////////////////////////////////////////////
        Tensor() = default;

        Tensor(const IndexType& dimensions) {
            size_t total_size = cumprod(dimensions);
            this->data_.resize(total_size);
            this->_set_shape(dimensions);
        }

        Tensor(const IndexType& dimensions, const ValueType& value) : Tensor(dimensions) {
            std::fill(data_.begin(), data_.end(), value);
        }

        template <typename U = ValueType>
        constexpr auto static iota(const IndexType& dimensions, U value = {}) -> Tensor {
            Tensor tensor(dimensions);
            std::generate(tensor.begin(), tensor.end(), [&value] { return value++; });
            return tensor;
        }

        auto static randn(const IndexType& dimensions) -> Tensor {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::normal_distribution<> d(static_cast<ValueType>(0), static_cast<ValueType>(1));
            Tensor tensor(dimensions);

            std::generate(tensor.begin(), tensor.end(), [&d, &gen] { return d(gen); });
            return tensor;
        }

        constexpr auto numel() const noexcept -> SizeType {
            return data_.size();
        }

        [[nodiscard]] constexpr auto shape() const noexcept -> const IndexType& {
            return dimensions_;
        }

        constexpr void reshape_(const IndexType& dimensions) {
            this->_set_shape(dimensions);
        }

        constexpr auto reshape(const IndexType& dimensions) const -> Tensor {
            Tensor tensor(*this);
            tensor.reshape_(dimensions);
            return tensor;
        }

        constexpr auto permute_(const IndexType& permutation) {
            dimensions_ = permute_vec(dimensions_, permutation);
            strides_ = permute_vec(strides_, permutation);
        }

        constexpr auto permute(const IndexType& permutation) const -> Tensor {
            Tensor tensor(*this);
            tensor.permute_(permutation);
            return tensor;
        }

        constexpr auto flat(SizeType i) -> ValueType& {
            if (i >= this->numel()) {
                throw std::runtime_error("flat: index out of bounds");
            }
            return data_[i];
        }

        constexpr auto flat(SizeType i) const -> const ValueType& {
            if (i >= this->numel()) {
                throw std::runtime_error("flat: index out of bounds");
            }
            return data_[i];
        }

        // Indexing with vector
        constexpr auto operator()(const IndexType& indices) -> ValueType& {
            assert(indices.size() == dimensions_.size());
            return data_[flatten_index(indices)];
        }

        constexpr auto operator()(const IndexType& indices) const -> const ValueType& {
            assert(indices.size() == dimensions_.size());
            return data_[flatten_index(indices)];
        }

        template <typename... Args>
        constexpr auto operator()(Args... args) -> ValueType& {
            assert(sizeof...(args) == dimensions_.size());
            return data_[flatten_index({static_cast<SizeType>(args)...})];
        }

        template <typename... Args>
        constexpr auto operator()(Args... args) const -> const ValueType& {
            assert(sizeof...(args) == dimensions_.size());
            return data_[flatten_index({args...})];
        }

        constexpr auto flatten_index(const IndexType& indices) const -> SizeType {
            if (indices.size() != dimensions_.size()) {
                throw std::runtime_error("flatten_index: size mismatch");
            }
            return std::transform_reduce(indices.begin(), indices.end(), strides_.begin(), static_cast<SizeType>(0),
                                         std::plus<>(), std::multiplies<>());
        }

        template <typename... Args>
        constexpr auto flatten_index(Args... args) const -> SizeType {
            assert(sizeof...(args) == dimensions_.size());
            return flatten_index({static_cast<SizeType>(args)...});
        }

        // Iterators
        constexpr auto begin() noexcept {
            return data_.begin();
        }
        constexpr auto begin() const noexcept {
            return data_.cbegin();
        }
        constexpr auto end() noexcept {
            return data_.end();
        }
        constexpr auto end() const noexcept {
            return data_.cend();
        }

        constexpr auto stride(int i) const -> SizeType {
            return strides_[i];
        }

        // Write a concept that ensures the function is supported
        constexpr auto apply_(ValueType (*f)(ValueType)) -> Tensor& {
            std::for_each(std::execution::unseq, this->begin(), this->end(), f);
            return *this;
        }

        constexpr auto apply(ValueType (*f)(ValueType)) const -> Tensor {
            return Tensor(*this).apply_(f);
        }

        ////////////////////////////////////////////////////////////////////
        // Trigonometric functions
        ////////////////////////////////////////////////////////////////////
        constexpr auto sin_() -> Tensor& requires SupportsSin<ValueType> {
            return this->apply_(std::sin);
        }

        constexpr auto sin() const -> Tensor {
            return Tensor(*this).sin_();
        }

        constexpr auto cos_() -> Tensor& requires SupportsCos<ValueType> {
            return this->apply_(std::cos);
        }

        constexpr auto cos() const -> Tensor {
            return Tensor(*this).cos_();
        }

        constexpr auto tan_() -> Tensor& requires SupportsTan<ValueType> {
            return this->apply_(std::tan);
        }

        constexpr auto tan() const -> Tensor {
            return Tensor(*this).tan_();
        }

        constexpr auto cot_() -> Tensor& requires SupportsCot<ValueType> {
            return this->apply_([](ValueType x) constexpr { return static_cast<ValueType>(1) / std::tan(x); });
        }

        constexpr auto cot() const -> Tensor {
            return Tensor(*this).cot_();
        }

        constexpr auto sec_() -> Tensor& requires SupportsSec<ValueType> {
            return this->apply_([](ValueType x) constexpr { return static_cast<ValueType>(1) / std::cos(x); });
        }

        constexpr auto sec() const -> Tensor {
            return Tensor(*this).sec_();
        }

        constexpr auto csc_() -> Tensor& requires SupportsCsc<ValueType> {
            return this->apply_([](ValueType x) constexpr { return static_cast<ValueType>(1) / std::sin(x); });
        }

        constexpr auto csc() const -> Tensor {
            return Tensor(*this).csc_();
        }

        ////////////////////////////////////////////////////////////////////
        // Misc functions
        ////////////////////////////////////////////////////////////////////

        template <typename U>
        constexpr auto astype() -> Tensor<U> {
            Tensor<U> res(this->dimensions_, this->strides_);
            std::transform(
                this->begin(), this->end(), res.begin(), [](T & x) constexpr { return static_cast<U>(x); });
            return res;
        }

      private:
        ContainerType data_;
        IndexType dimensions_;
        IndexType strides_;

        template <typename U>
        friend class Tensor;

        Tensor(const IndexType& dimensions, const IndexType& strides) : Tensor(dimensions) {
            this->strides_ = strides;
        }

        constexpr void _set_shape(const IndexType& dimensions) {
            if (cumprod(dimensions) != this->numel()) {
                throw std::runtime_error("Invalid dimensions");
            }

            dimensions_ = dimensions;

            // Calculate strides
            strides_.resize(dimensions_.size());
            std::fill(strides_.begin(), strides_.end(), 1);

            for (int i = dimensions_.size() - 2; i >= 0; --i) {
                strides_[i] = strides_[i + 1] * dimensions_[i + 1];
            }
        }
    };

};  // namespace tt::inline v1
