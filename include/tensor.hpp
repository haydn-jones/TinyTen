#pragma once

#include <algorithm>
#include <cassert>
#include <numeric>
#include <operators.hpp>
#include <vector>

namespace tt::inline v1 {
    template <typename T>
    constexpr auto cumprod(const std::vector<T>& v) -> T {
        return std::accumulate(v.begin(), v.end(), 1, std::multiplies<T>());
    }

    template <typename T>
    class Tensor {
      public:
        using ValueType = T;
        using ContainerType = std::vector<T>;
        using SizeType = typename ContainerType::size_type;
        using IndexType = std::vector<size_t>;

        Tensor() = default;

        Tensor(const IndexType& dimensions) : dimensions_(dimensions) {
            size_t total_size = cumprod(dimensions);
            data_.resize(total_size);
            calculate_strides();
        }

        Tensor(const IndexType& dimensions, const ValueType& value) : Tensor(dimensions) {
            std::fill(data_.begin(), data_.end(), value);
        }

        template <typename U = ValueType>
        auto static iota(const IndexType& dimensions, U value = {}) -> Tensor {
            Tensor tensor(dimensions);
            std::generate(tensor.begin(), tensor.end(), [&value] { return value++; });
            return tensor;
        }

        auto size() const noexcept -> SizeType { return data_.size(); }

        [[nodiscard]] auto shape() const noexcept -> IndexType { return dimensions_; }

        void reshape(const IndexType& dimensions) {
            assert(cumprod(dimensions) == size());
            dimensions_ = dimensions;
            calculate_strides();
        }

        // Indexing with multiple indices
        auto operator()(SizeType idx) -> ValueType& {
            assert(idx < size());
            return data_[idx];
        }

        auto operator()(SizeType idx) const -> const ValueType& {
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
            SizeType idx = 0;
            for (int i = indices.size() - 1; i >= 0; --i) {
                idx += strides_[i] * indices[i];
            }
            return idx;
        }

        template <typename... Args>
        constexpr auto flatten_index(Args... args) const -> SizeType {
            assert(sizeof...(args) == dimensions_.size());
            return flatten_index({static_cast<SizeType>(args)...});
        }

        // Iterators
        auto begin() noexcept { return data_.begin(); }
        auto begin() const noexcept { return data_.cbegin(); }
        auto end() noexcept { return data_.end(); }
        auto end() const noexcept { return data_.cend(); }

        auto stride(int i) const -> SizeType { return strides_[i]; }

      private:
        ContainerType data_;
        IndexType dimensions_;
        IndexType strides_;

        void calculate_strides() {
            strides_.resize(dimensions_.size());
            std::fill(strides_.begin(), strides_.end(), 1);

            for (int i = dimensions_.size() - 2; i >= 0; --i) {
                strides_[i] = strides_[i + 1] * dimensions_[i + 1];
            }
        }
    };

};  // namespace tt::inline v1
