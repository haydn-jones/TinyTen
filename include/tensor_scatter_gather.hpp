#pragma once

#include "tensor.hpp"

namespace tt::inline v1 {
    template <typename T>
    constexpr auto static gather(Tensor<T>& input, int64_t dim, Tensor<int64_t>& index) -> Tensor<T> {
        // Input and index must have the same number of dimensions
        if (input.dim() != index.dim()) {
            throw std::runtime_error("Input and index must have the same number of dimensions");
        }
    }

};  // namespace tt::inline v1