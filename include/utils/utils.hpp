#pragma once

#include <algorithm>
#include <cstddef>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace tt::inline v1 {
    template <typename T>
    constexpr auto cumprod(const std::vector<T>& v) -> T {
        return std::reduce(v.begin(), v.end(), 1, std::multiplies<T>());
    }

    template <typename T, typename U>
    constexpr auto permute_vec(const std::vector<T>& vals, const std::vector<U>& perm) -> std::vector<T> {
        if (vals.size() != perm.size()) {
            throw std::runtime_error("permute_vec: size mismatch");
        }
        std::vector<T> result(vals.size());
        for (size_t i = 0; i < perm.size(); ++i) {
            result[i] = vals[perm[i]];
        }
        return result;
    }

    constexpr auto ravel_index(const std::vector<int64_t>& indices, const std::vector<int64_t>& strides) -> int64_t {
        if (indices.size() != strides.size()) {
            throw std::runtime_error("ravel_index: size mismatch");
        }
        return std::transform_reduce(indices.begin(), indices.end(), strides.begin(), static_cast<int64_t>(0),
                                     std::plus<>(), std::multiplies<>());
    }

    auto unravel_index(int64_t flat_index, const std::vector<int64_t>& strides) -> std::vector<int64_t> {
        std::vector<int64_t> idx(strides.size());
        std::transform(
            strides.begin(), strides.end(), idx.begin(), [&flat_index](int64_t stride) constexpr {
                int64_t idx = flat_index / stride;
                flat_index %= stride;
                return idx;
            });
        return idx;
    }

    constexpr auto ravel_unravel(int64_t flat_index, const std::vector<int64_t>& strides,
                                 const std::vector<int64_t>& canon_strides) -> int64_t {
        if (strides == canon_strides) {
            return flat_index;
        }
        int64_t idx = 0;
        for (size_t i = 0; i < strides.size(); ++i) {
            std::ldiv_t result = std::div(flat_index, canon_strides[i]);
            idx += result.quot * strides[i];
            flat_index = result.rem;
        }

        return idx;
    }

    auto _calc_strides(const std::vector<int64_t>& shape) -> std::vector<int64_t> {
        std::vector<int64_t> strides(shape.size(), 1);
        auto N = static_cast<int64_t>(shape.size());
        for (int64_t i = N - 2; i >= 0; --i) {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        return strides;
    }
}  // namespace tt::inline v1