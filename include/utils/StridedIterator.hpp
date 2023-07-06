#pragma once

#include <cstddef>
#include <iterator>
#include <utility>

#include "utils.hpp"

template <typename T>
class StridedIter {
    class StridedIterImpl;

  public:
    StridedIter(T* ptr_base, int64_t numel, std::vector<int64_t> strides, std::vector<int64_t> canon_strides)
        : m_ptr_base(ptr_base),
          numel(numel),
          m_strides(std::move(strides)),
          m_canon_strides(std::move(canon_strides)) {}

    [[nodiscard]] auto begin() const -> StridedIterImpl {
        return {this->m_ptr_base, 0, this->m_strides, this->m_canon_strides};
    }

    [[nodiscard]] auto end() const -> StridedIterImpl {
        return {this->m_ptr_base, this->numel, this->m_strides, this->m_canon_strides};
    }

  private:
    int64_t numel;
    T* m_ptr_base;

    const std::vector<int64_t> m_strides;
    const std::vector<int64_t> m_canon_strides;

    struct StridedIterImpl {
        using iterator_category = std::forward_iterator_tag;
        using difference_type = std::ptrdiff_t;
        using value_type = T;
        using pointer = value_type*;
        using reference = value_type&;

        StridedIterImpl(pointer ptr_base, int64_t loc, const std::vector<int64_t>& strides,
                        const std::vector<int64_t>& canon_strides)
            : m_ptr_base(ptr_base), loc(loc), m_strides(strides), m_canon_strides(canon_strides) {}

        auto operator*() const -> reference {
            return *this->_get_loc();
        }

        auto operator->() -> pointer {
            return this->_get_loc();
        }

        // Prefix increment
        auto operator++() -> StridedIterImpl& {
            this->loc++;
            return *this;
        }

        // Postfix increment
        auto operator++(int) -> StridedIterImpl {
            StridedIterImpl tmp = *this;
            ++(*this);
            return tmp;
        }

        friend auto operator==(const StridedIterImpl& a, const StridedIterImpl& b) -> bool {
            return a.loc == b.loc;
        };

        friend auto operator!=(const StridedIterImpl& a, const StridedIterImpl& b) -> bool {
            return a.loc != b.loc;
        };

      private:
        constexpr auto _get_loc() const -> pointer {
            return m_ptr_base + tt::ravel_unravel(this->loc, this->m_strides, this->m_canon_strides);
        }

        int64_t loc;
        const pointer m_ptr_base;

        const std::vector<int64_t>& m_strides;
        const std::vector<int64_t>& m_canon_strides;
    };
};
