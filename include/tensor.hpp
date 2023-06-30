#pragma once

#include <algorithm>
#include <cassert>
#include <numeric>
#include <vector>

template <typename T> class Tensor {
  public:
    using ValueType = T;
    using ContainerType = std::vector<T>;
    using SizeType = typename ContainerType::size_type;
    using IndexType = std::vector<size_t>;

    Tensor() = default;

    Tensor(const IndexType& dimensions) : dimensions_(dimensions) {
        int total_size = std::accumulate(dimensions.begin(), dimensions.end(), 1, std::multiplies<int>());
        data_.resize(total_size);
    }

    ValueType& operator[](const IndexType& indices) { return data_[index(indices)]; }

    const ValueType& operator[](const IndexType& indices) const { return data_[index(indices)]; }

    SizeType size() const noexcept { return data_.size(); }

    auto begin() noexcept { return data_.begin(); }
    auto begin() const noexcept { return data_.begin(); }
    auto end() noexcept { return data_.end(); }
    auto end() const noexcept { return data_.end(); }

  private:
    ContainerType data_;
    IndexType dimensions_;

    SizeType index(const IndexType& indices) const {
        assert(indices.size() == dimensions_.size());
        SizeType idx = 0;
        SizeType stride = 1;
        for (int i = indices.size() - 1; i >= 0; --i) {
            assert(indices[i] < dimensions_[i]);
            idx += stride * indices[i];
            stride *= dimensions_[i];
        }
        return idx;
    }
};
