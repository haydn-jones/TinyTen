#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <tensor.hpp>

TEST_CASE("Tensor operations", "[Tensor]") {
    using TensorType = tt::Tensor<int>;
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
    using TensorType1 = tt::Tensor<int>;

    auto tensor1 = TensorType1::iota({2, 2});
    REQUIRE(tensor1.size() == 4);
    REQUIRE(tensor1(0, 0) == 0);
    REQUIRE(tensor1(0, 1) == 1);
    REQUIRE(tensor1(1, 0) == 2);
    REQUIRE(tensor1(1, 1) == 3);

    using TensorType2 = tt::Tensor<double>;

    auto tensor2 = TensorType1::iota({2, 2});
    REQUIRE(tensor2.size() == 4);
    REQUIRE(tensor2(0, 0) == 0.0);
    REQUIRE(tensor2(0, 1) == 1.0);
    REQUIRE(tensor2(1, 0) == 2.0);
    REQUIRE(tensor2(1, 1) == 3.0);
}

TEST_CASE("multi-dim indexing", "[Tensor]") {
    tt::Tensor<int> ten({1, 3, 5, 10});

    REQUIRE(ten(0, 0, 0, 0) == 0);

    ten(0, 0, 0, 0) = 10;
    REQUIRE(ten(0, 0, 0, 0) == 10);
}

TEST_CASE("Index flattening", "[Tensor]") {
    tt::Tensor<int> ten1({1, 3, 5, 10});

    REQUIRE(ten1.flatten_index({0, 0, 0, 0}) == 0);
    REQUIRE(ten1.flatten_index({0, 0, 0, 1}) == 1);
    REQUIRE(ten1.flatten_index({0, 0, 0, 2}) == 2);
    REQUIRE(ten1.flatten_index({0, 0, 1, 0}) == 10);
    REQUIRE(ten1.flatten_index({0, 0, 1, 1}) == 11);
    REQUIRE(ten1.flatten_index({0, 0, 2, 6}) == 26);
    REQUIRE(ten1.flatten_index({0, 2, 4, 9}) == 149);

    REQUIRE(ten1.flatten_index(0, 0, 0, 0) == 0);
    REQUIRE(ten1.flatten_index(0, 0, 0, 1) == 1);
    REQUIRE(ten1.flatten_index(0, 0, 0, 2) == 2);
    REQUIRE(ten1.flatten_index(0, 0, 1, 0) == 10);
    REQUIRE(ten1.flatten_index(0, 0, 1, 1) == 11);
    REQUIRE(ten1.flatten_index(0, 0, 2, 6) == 26);
    REQUIRE(ten1.flatten_index(0, 2, 4, 9) == 149);
}

TEST_CASE("Strides", "[Tensor]") {
    tt::Tensor<int> ten1({1, 3, 5, 10});

    // ensure that the strides are correct
    REQUIRE(ten1.stride(0) == 150);
    REQUIRE(ten1.stride(1) == 50);
    REQUIRE(ten1.stride(2) == 10);
    REQUIRE(ten1.stride(3) == 1);

    // Reshape the tensor
    ten1.reshape({15, 2, 5});
    REQUIRE(ten1.stride(0) == 10);
    REQUIRE(ten1.stride(1) == 5);
    REQUIRE(ten1.stride(2) == 1);
}

TEST_CASE("Sum", "[Tensor]") {
    tt::Tensor<int> ten1({2, 3});
    tt::Tensor<int> ten2 = tt::Tensor<int>::iota({2, 3});

    tt::Tensor<int> ten3 = ten1 + ten2;

    REQUIRE(ten3(0, 0) == 0);
    REQUIRE(ten3(0, 1) == 1);
    REQUIRE(ten3(0, 2) == 2);
    REQUIRE(ten3(1, 0) == 3);
    REQUIRE(ten3(1, 1) == 4);
    REQUIRE(ten3(1, 2) == 5);

    tt::Tensor<float> ten4 = ten2 + tt::Tensor<float>::iota({2, 3});

    REQUIRE(ten4(0, 0) == 0.0f);
    REQUIRE(ten4(0, 1) == 2.0f);
    REQUIRE(ten4(0, 2) == 4.0f);
    REQUIRE(ten4(1, 0) == 6.0f);
    REQUIRE(ten4(1, 1) == 8.0f);
    REQUIRE(ten4(1, 2) == 10.0f);
}

TEST_CASE("Subtraction", "[Tensor]") {
    tt::Tensor<int> ten1({2, 3});
    tt::Tensor<int> ten2 = tt::Tensor<int>::iota({2, 3});

    tt::Tensor<int> ten3 = ten1 - ten2;

    REQUIRE(ten3(0, 0) == 0);
    REQUIRE(ten3(0, 1) == -1);
    REQUIRE(ten3(0, 2) == -2);
    REQUIRE(ten3(1, 0) == -3);
    REQUIRE(ten3(1, 1) == -4);
    REQUIRE(ten3(1, 2) == -5);

    tt::Tensor<float> ten4 = ten2 - tt::Tensor<float>::iota({2, 3});

    REQUIRE(ten4(0, 0) == 0.0f);
    REQUIRE(ten4(0, 1) == 0.0f);
    REQUIRE(ten4(0, 2) == 0.0f);
    REQUIRE(ten4(1, 0) == 0.0f);
    REQUIRE(ten4(1, 1) == 0.0f);
    REQUIRE(ten4(1, 2) == 0.0f);
}

TEST_CASE("Multiplication", "[Tensor]") {
    tt::Tensor<int> ten1({2, 3});
    tt::Tensor<int> ten2 = tt::Tensor<int>::iota({2, 3});

    tt::Tensor<int> ten3 = ten1 * ten2;

    REQUIRE(ten3(0, 0) == 0);
    REQUIRE(ten3(0, 1) == 0);
    REQUIRE(ten3(0, 2) == 0);
    REQUIRE(ten3(1, 0) == 0);
    REQUIRE(ten3(1, 1) == 0);
    REQUIRE(ten3(1, 2) == 0);

    tt::Tensor<float> ten4 = ten2 * tt::Tensor<float>::iota({2, 3});

    REQUIRE(ten4(0, 0) == 0.0f);
    REQUIRE(ten4(0, 1) == 1.0f);
    REQUIRE(ten4(0, 2) == 4.0f);
    REQUIRE(ten4(1, 0) == 9.0f);
    REQUIRE(ten4(1, 1) == 16.0f);
    REQUIRE(ten4(1, 2) == 25.0f);
}

TEST_CASE("Division", "[Tensor]") {
    tt::Tensor<int> ten1({2, 3}, 2);                          // Tensor of twos
    tt::Tensor<int> ten2 = tt::Tensor<int>::iota({2, 3}, 1);  // Tensor from 1 to 6

    tt::Tensor<int> ten3 = ten1 / ten2;

    REQUIRE(ten3(0, 0) == 2);
    REQUIRE(ten3(0, 1) == 1);
    REQUIRE(ten3(0, 2) == 0);
    REQUIRE(ten3(1, 0) == 0);
    REQUIRE(ten3(1, 1) == 0);
    REQUIRE(ten3(1, 2) == 0);

    tt::Tensor<float> ten4 = ten1 / tt::Tensor<float>::iota({2, 3}, 1.0f);  // Tensor from 1.0 to 6.0

    REQUIRE(ten4(0, 0) == 2.0f);
    REQUIRE(ten4(0, 1) == 1.0f);
    REQUIRE(ten4(0, 2) == 2.0f / 3.0f);
    REQUIRE(ten4(1, 0) == 0.5f);
    REQUIRE(ten4(1, 1) == 2.0f / 5.0f);
    REQUIRE(ten4(1, 2) == 1.0f / 3.0f);
}
