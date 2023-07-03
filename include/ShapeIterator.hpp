#pragma once

#include <algorithm>
#include <utility>
#include <vector>

class ShapeIter {
    class ShapeIterImpl;

  public:
    ShapeIter(size_t numel, std::vector<size_t> shape) : numel(numel), shape(std::move(shape)) {}

    [[nodiscard]] auto begin() const -> ShapeIterImpl {
        return {this->numel, this->shape, true};
    }

    [[nodiscard]] auto end() const -> ShapeIterImpl {
        return {this->numel, this->shape, false};
    }

  private:
    const size_t numel;
    const std::vector<size_t> shape;

    class ShapeIterImpl {
      public:
        using value_type = std::vector<size_t>;
        using element_type = std::vector<size_t>;
        using iterator_category = std::forward_iterator_tag;

        ShapeIterImpl(const size_t numel, const value_type& shape, bool start) : numel(numel), shape(shape) {
            if (!start) {
                this->cur_idx = numel;
            }
        }

        constexpr auto operator++() -> ShapeIterImpl& {
            this->cur_idx++;
            return *this;
        }

        constexpr auto operator!=(const ShapeIterImpl& other) const -> bool {
            return (this->cur_idx != other.cur_idx) || (this->numel != other.numel) || (this->shape != other.shape);
        }

        constexpr auto operator*() const -> const value_type {
            return unflatten_index(this->cur_idx);
        }

        [[nodiscard]] constexpr auto unflatten_index(size_t index) const -> value_type {
            value_type result;
            result.reserve(shape.size());

            for (int64_t i = shape.size() - 1; i >= 0; --i) {
                result.emplace_back(index % shape[i]);
                index /= shape[i];
            }

            std::reverse(result.begin(), result.end());
            return result;
        }

      private:
        const value_type& shape;
        const size_t numel;
        size_t cur_idx = 0;
    };
};