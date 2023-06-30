#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <tensor.hpp>

TEST_CASE("Tensor operations", "[Tensor]") {
    using TensorType = Tensor<int>;
    using IndexType = TensorType::IndexType;

    SECTION("default constructed tensor should be empty") {
        TensorType tensor;
        REQUIRE(tensor.size() == 0);
    }

    SECTION("constructed with dimensions should have correct size") {
        TensorType tensor({4, 4, 4});
        REQUIRE(tensor.size() == 4 * 4 * 4);
    }

    SECTION("indexing should work correctly") {
        TensorType tensor({4, 4, 4});
        IndexType idx = {1, 2, 3};
        tensor[idx] = 10;
        REQUIRE(tensor[idx] == 10);
    }

    SECTION("iteration should work correctly") {
        TensorType tensor({2, 2});
        int value = 0;
        for (auto& element : tensor) {
            element = value++;
        }

        value = 0;
        for (const auto& element : tensor) {
            REQUIRE(element == value++);
        }
    }
}
