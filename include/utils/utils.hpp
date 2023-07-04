#pragma once

#include <cstddef>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace tt::inline v1 {
    template <typename T>
    constexpr auto cumprod(const std::vector<T>& v) -> T {
        return std::accumulate(v.begin(), v.end(), 1, std::multiplies<T>());
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

}  // namespace tt::inline v1