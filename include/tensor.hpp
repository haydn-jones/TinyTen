#pragma once

#include <ShapeIterator.hpp>
#include <algorithm>
#include <cassert>
#include <numeric>
#include <operators.hpp>
#include <vector>

namespace tt::inline v1 {
    template <typename T>
    constexpr auto cumprod(const std::vector<T>& v) -> T {
        return std::reduce(v.begin(), v.end(), 1, std::multiplies<T>());
    }

    template <typename T, typename U>
    constexpr auto permute_vec(const std::vector<T>& vals, const std::vector<U>& perm) -> std::vector<T> {
        assert(vals.size() == perm.size());
        std::vector<T> result(vals.size());
        for (size_t i = 0; i < perm.size(); ++i) {
            result[i] = vals[perm[i]];
        }
        return result;
    }

    template <typename T>
    class Tensor {
      public:
        using ValueType = T;
        using ContainerType = std::vector<T>;
        using SizeType = typename ContainerType::size_type;
        using IndexType = std::vector<size_t>;

        Tensor() = default;

        Tensor(const IndexType& dimensions) {
            size_t total_size = cumprod(dimensions);
            this->_set_shape(dimensions);
            this->data_.resize(total_size);
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

        constexpr auto size() const noexcept -> SizeType {
            return data_.size();
        }

        [[nodiscard]] constexpr auto shape() const noexcept -> const IndexType& {
            return dimensions_;
        }

        constexpr void reshape_(const IndexType& dimensions) {
            assert(cumprod(dimensions) == size());
            this->_set_shape(dimensions);
        }

        constexpr auto reshape(const IndexType& dimensions) const -> Tensor {
            Tensor tensor(*this);
            tensor.reshape_(dimensions);
            return tensor;
        }

        constexpr auto permute_(const IndexType& permutation) {
            assert(permutation.size() == dimensions_.size());
            dimensions_ = permute_vec(dimensions_, permutation);
            strides_ = permute_vec(strides_, permutation);
        }

        constexpr auto permute(const IndexType& permutation) const -> Tensor {
            Tensor tensor(*this);
            tensor.permute_(permutation);
            return tensor;
        }

        // Indexing with multiple indices
        constexpr auto operator()(SizeType idx) -> ValueType& {
            assert(idx < size());
            return data_[idx];
        }

        constexpr auto operator()(SizeType idx) const -> const ValueType& {
            assert(idx < size());
            return data_[idx];
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

      private:
        ContainerType data_;
        IndexType dimensions_;
        IndexType strides_;

        constexpr void _set_shape(const IndexType& dimensions) {
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
