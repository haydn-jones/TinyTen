#pragma once

#include <algorithm>
#include <utility>
#include <vector>

class ShapeIter {
    class ShapeIterImpl;

  public:
    ShapeIter(size_t numel, std::vector<int64_t> shape) : numel(numel), shape(std::move(shape)) {}

    [[nodiscard]] auto begin() const -> ShapeIterImpl {
        return {this->numel, this->shape, true};
    }

    [[nodiscard]] auto end() const -> ShapeIterImpl {
        return {this->numel, this->shape, false};
    }

  private:
    const size_t numel;
    const std::vector<int64_t> shape;

    class ShapeIterImpl {
      public:
        using value_type = std::vector<int64_t>;
        using element_type = std::vector<int64_t>;
        using iterator_category = std::forward_iterator_tag;

        ShapeIterImpl(const size_t numel, const value_type& shape, bool start)
            : numel(numel), shape(shape), cur_unflat(shape.size(), 0) {
            if (!start) {
                this->cur_flat = numel;
                this->unflatten_index();
            }
        }

        constexpr auto operator++() -> ShapeIterImpl& {
            this->cur_flat++;
            this->unflatten_index();
            return *this;
        }

        constexpr auto operator!=(const ShapeIterImpl& other) const -> bool {
            return (this->cur_flat != other.cur_flat) || (this->numel != other.numel) || (this->shape != other.shape);
        }

        constexpr auto operator*() const -> const value_type {
            return this->cur_unflat;
        }

        constexpr void unflatten_index() {
            int64_t index = this->cur_flat;
            this->cur_unflat.clear();
            for (int64_t i = static_cast<int64_t>(shape.size()) - 1; i >= 0; --i) {
                auto idx = index % shape[i];
                index /= shape[i];
                this->cur_unflat.push_back(idx);
            }

            std::reverse(this->cur_unflat.begin(), this->cur_unflat.end());
        }

      private:
        const value_type& shape;
        const int64_t numel;

        int64_t cur_flat = 0;
        value_type cur_unflat;
    };
};