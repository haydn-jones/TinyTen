#pragma once

#include <algorithm>
#include <cassert>
#include <numeric>
#include <vector>

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

    static auto iota(const IndexType& dimensions) -> Tensor {
        Tensor tensor(dimensions);
        std::iota(tensor.begin(), tensor.end(), static_cast<ValueType>(0));
        return tensor;
    }

    auto size() const noexcept -> SizeType { return data_.size(); }

    [[nodiscard]] auto shape() const noexcept -> IndexType { return dimensions_; }

    void reshape(const IndexType& dimensions) {
        assert(cumprod(dimensions) == size());
        dimensions_ = dimensions;
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
    auto operator()(const IndexType& indices) -> ValueType& {
        assert(indices.size() == dimensions_.size());
        return data_[flatten_index(indices)];
    }

    auto operator()(const IndexType& indices) const -> const ValueType& {
        assert(indices.size() == dimensions_.size());
        return data_[flatten_index(indices)];
    }

    // Indexing with multiple indices
    template <typename... Args>
    auto operator()(Args... args) -> ValueType& {
        assert(sizeof...(args) == dimensions_.size());
        return data_[flatten_index({static_cast<SizeType>(args)...})];
    }

    template <typename... Args>
    auto operator()(Args... args) const -> const ValueType& {
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

    // Iterators
    auto begin() noexcept { return data_.begin(); }
    auto begin() const noexcept { return data_.cbegin(); }
    auto end() noexcept { return data_.end(); }
    auto end() const noexcept { return data_.cend(); }

  private:
    ContainerType data_;
    IndexType dimensions_;
    IndexType strides_;

    void calculate_strides() {
        strides_.resize(dimensions_.size(), 1);
        for (int i = dimensions_.size() - 2; i >= 0; --i) {
            strides_[i] = strides_[i + 1] * dimensions_[i + 1];
        }
    }
};
