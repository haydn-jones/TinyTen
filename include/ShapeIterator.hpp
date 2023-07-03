#pragma once

#include <algorithm>
#include <utility>
#include <vector>

class ShapeIter {
    class ShapeIterImpl;

  public:
    ShapeIter(size_t numel, std::vector<size_t> strides) : numel(numel), strides(std::move(strides)) {}

    [[nodiscard]] auto begin() const -> ShapeIterImpl {
        return {this->numel, this->strides, true};
    }

    [[nodiscard]] auto end() const -> ShapeIterImpl {
        return {this->numel, this->strides, false};
    }

  private:
    const size_t numel;
    const std::vector<size_t> strides;

    class ShapeIterImpl {
      public:
        using value_type = std::vector<size_t>;
        using element_type = std::vector<size_t>;
        using iterator_category = std::forward_iterator_tag;

        ShapeIterImpl(const size_t numel, const value_type& strides, bool start) : numel(numel), strides(strides) {
            if (!start) {
                this->cur_idx = numel;
            }
        }

        constexpr auto operator++() -> ShapeIterImpl& {
            this->cur_idx++;
            return *this;
        }

        constexpr auto operator!=(const ShapeIterImpl& other) const -> bool {
            return (this->cur_idx != other.cur_idx) || (this->numel != other.numel) || (this->strides != other.strides);
        }

        constexpr auto operator*() const -> const value_type {
            return unflatten_index(this->cur_idx);
        }

        [[nodiscard]] constexpr auto unflatten_index(size_t flat_index) const -> value_type {
            value_type idx(strides.size());
            std::transform(
                strides.begin(), strides.end(), idx.begin(), [&flat_index](size_t stride) constexpr {
                    size_t idx = flat_index / stride;
                    flat_index %= stride;
                    return idx;
                });

            return idx;
        }

      private:
        const value_type& strides;
        const size_t numel;
        size_t cur_idx = 0;
    };
};