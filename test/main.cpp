#include <catch2/catch_test_macros.hpp>
#include <cstdint>

#include <tensor.hpp>

TEST_CASE("Building Tensor", "tensor" ) {
    auto ten = Tensor<float>({2, 3, 4});
    REQUIRE( ten.size() == 24 );
}