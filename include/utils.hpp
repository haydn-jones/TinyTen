#pragma once

#include <vector>
#include <numeric>

namespace tt::inline v1
{
    template <typename T>
    constexpr auto cumprod(const std::vector<T>& v) -> T {
        return std::accumulate(v.begin(), v.end(), 1, std::multiplies<T>());
    }
} // namespace tt::