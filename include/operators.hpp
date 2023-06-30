#pragma once

#include <type_traits>
#include <functional>

template <typename T>
class Tensor;

template<typename T, typename U, typename = std::void_t<>>
struct is_valid_operator : std::false_type {};

template<typename T, typename U>
struct is_valid_operator<T, U, std::void_t<decltype(std::declval<T>() + std::declval<U>(), 
                                                   std::declval<T>() - std::declval<U>(), 
                                                   std::declval<T>() * std::declval<U>(), 
                                                   std::declval<T>() / std::declval<U>())>> : std::true_type {};

template <typename T, typename U, typename Op>
constexpr auto operator_func(const Tensor<T>& a, const Tensor<U>& b, Op op)
    -> typename std::enable_if<is_valid_operator<T, U>::value, Tensor<typename std::common_type<T, U>::type>>::type
{
    using otype = typename std::common_type<T, U>::type;
    Tensor<otype> result(a.shape());
    std::transform(a.begin(), a.end(), b.begin(), result.begin(), op);
    return result;
}

template <typename T, typename U>
constexpr auto operator+(const Tensor<T>& a, const Tensor<U>& b) 
{
    return operator_func(a, b, std::plus<>{});
}

template <typename T, typename U>
constexpr auto operator-(const Tensor<T>& a, const Tensor<U>& b) 
{
    return operator_func(a, b, std::minus<>{});
}

template <typename T, typename U>
constexpr auto operator*(const Tensor<T>& a, const Tensor<U>& b) 
{
    return operator_func(a, b, std::multiplies<>{});
}

template <typename T, typename U>
constexpr auto operator/(const Tensor<T>& a, const Tensor<U>& b) 
{
    return operator_func(a, b, std::divides<>{});
}
