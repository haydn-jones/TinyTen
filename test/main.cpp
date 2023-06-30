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
        tensor(idx) = 10;
        REQUIRE(tensor(idx) == 10);
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

TEST_CASE("iota", "[Tensor]") {
    using TensorType1 = Tensor<int>;

    auto tensor1 = TensorType1::iota({2, 2});
    REQUIRE(tensor1.size() == 4);
    REQUIRE(tensor1(0, 0) == 0);
    REQUIRE(tensor1(0, 1) == 1);
    REQUIRE(tensor1(1, 0) == 2);
    REQUIRE(tensor1(1, 1) == 3);

    using TensorType2 = Tensor<double>;

    auto tensor2 = TensorType1::iota({2, 2});
    REQUIRE(tensor2.size() == 4);
    REQUIRE(tensor2(0, 0) == 0.0);
    REQUIRE(tensor2(0, 1) == 1.0);
    REQUIRE(tensor2(1, 0) == 2.0);
    REQUIRE(tensor2(1, 1) == 3.0);
}

TEST_CASE("multi-dim indexing", "[Tensor]") {
    Tensor<int> ten({1, 3, 5, 10});

    REQUIRE(ten(0, 0, 0, 0) == 0);

    ten(0, 0, 0, 0) = 10;
    REQUIRE(ten(0, 0, 0, 0) == 10);
}

TEST_CASE("Index flattening", "[Tensor]") {
    Tensor<int> ten1({1, 3, 5, 10});

    REQUIRE(ten1.flatten_index({0, 0, 0, 0}) == 0);
    REQUIRE(ten1.flatten_index({0, 0, 0, 1}) == 1);
    REQUIRE(ten1.flatten_index({0, 0, 0, 2}) == 2);

    REQUIRE(ten1.flatten_index({0, 0, 1, 0}) == 10);
    REQUIRE(ten1.flatten_index({0, 0, 1, 1}) == 11);
    REQUIRE(ten1.flatten_index({0, 0, 2, 6}) == 26);

    REQUIRE(ten1.flatten_index({0, 2, 4, 9}) == 149);
}