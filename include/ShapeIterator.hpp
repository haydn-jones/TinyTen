#pragma once

#include <vector>
#include <iterator>
#include <span>
#include <ranges>
#include <algorithm>

class ShapeIter {
    class ShapeIterImpl;

public:
    ShapeIter(const std::vector<size_t>& shape) : shape(shape) {}

    ShapeIterImpl begin() const {
        return ShapeIterImpl(this->shape, true);
    }

    ShapeIterImpl end() const {
        return ShapeIterImpl(this->shape, false);
    }

private:
    const std::vector<size_t> shape;

    class ShapeIterImpl {
    public:
        using value_type = std::vector<size_t>;
        using element_type = std::vector<size_t>;
        using iterator_category = std::forward_iterator_tag;

        ShapeIterImpl(const value_type& shape, bool start): shape(shape), cur_vals(shape), iter_once(false) {
            if (start) {
                std::fill(this->cur_vals.begin(), this->cur_vals.end(), 0);
            }
        }

        ShapeIterImpl& operator++() {
            this->iter_once = true;
            for (int64_t i = cur_vals.size() - 1; i >= 0; --i) {
                if (++cur_vals[i] >= shape[i]) {
                    if (i == 0) {
                        cur_vals = shape;
                        break;
                    }
                    cur_vals[i] = 0;
                } else {
                    break;
                }
            }
            return *this;
        }

        bool operator!=(const ShapeIterImpl& other) const { 
            return !iter_once || (this->cur_vals != other.cur_vals) || (this->shape != other.shape);
        }

        const value_type operator*() const {
            return this->cur_vals;
        }

    private:
        const value_type& shape;
        value_type cur_vals;
        bool iter_once;
    }; 
};