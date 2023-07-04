#pragma once

#include <algorithm>
#include <utility>
#include <vector>

#include "utils/utils.hpp"

class ShapeIter {
    class ShapeIterImpl;

  public:
    ShapeIter(int64_t numel, std::vector<int64_t> strides) : numel(numel), strides(std::move(strides)) {}

    [[nodiscard]] auto begin() const -> ShapeIterImpl {
        return {this->numel, this->strides, true};
    }

    [[nodiscard]] auto end() const -> ShapeIterImpl {
        return {this->numel, this->strides, false};
    }

  private:
    const int64_t numel;
    const std::vector<int64_t> strides{};

    class ShapeIterImpl {
      public:
        using value_type = std::vector<int64_t>;
        using element_type = std::vector<int64_t>;
        using iterator_category = std::forward_iterator_tag;

        ShapeIterImpl(const int64_t numel, value_type strides, bool start) : numel(numel), strides(std::move(strides)) {
            if (!start) {
                this->cur_flat = numel;
            }
        }

        constexpr auto operator++() -> ShapeIterImpl& {
            this->cur_flat++;
            return *this;
        }

        constexpr auto operator!=(const ShapeIterImpl& other) const -> bool {
            return (this->cur_flat != other.cur_flat) || (this->numel != other.numel);
        }

        constexpr auto operator*() const -> const value_type {
            return tt::unravel_index(this->cur_flat, this->strides);
        }

      private:
        const value_type strides{};
        const int64_t numel;

        int64_t cur_flat = 0;
    };
};