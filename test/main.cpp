#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cstdint>
#include <iostream>
#include <vector>

#include "TinyTensor.hpp"

using namespace tt;

TEST_CASE("Tensor operations", "[Tensor]") {
    using TensorType = Tensor<int>;

    SECTION("default constructed tensor should be empty") {
        TensorType tensor;
        REQUIRE(tensor.numel() == 0);
    }

    SECTION("constructed with dimensions should have correct size") {
        TensorType tensor({4, 4, 4});
        REQUIRE(tensor.numel() == 4 * 4 * 4);
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
    REQUIRE(tensor1.numel() == 4);
    REQUIRE(tensor1(0, 0) == 0);
    REQUIRE(tensor1(0, 1) == 1);
    REQUIRE(tensor1(1, 0) == 2);
    REQUIRE(tensor1(1, 1) == 3);

    using TensorType2 = Tensor<double>;

    auto tensor2 = TensorType1::iota({2, 2});
    REQUIRE(tensor2.numel() == 4);
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

    REQUIRE(ten1.ravel_index({0, 0, 0, 0}) == 0);
    REQUIRE(ten1.ravel_index({0, 0, 0, 1}) == 1);
    REQUIRE(ten1.ravel_index({0, 0, 0, 2}) == 2);
    REQUIRE(ten1.ravel_index({0, 0, 1, 0}) == 10);
    REQUIRE(ten1.ravel_index({0, 0, 1, 1}) == 11);
    REQUIRE(ten1.ravel_index({0, 0, 2, 6}) == 26);
    REQUIRE(ten1.ravel_index({0, 2, 4, 9}) == 149);

    REQUIRE(ten1.ravel_index(0, 0, 0, 0) == 0);
    REQUIRE(ten1.ravel_index(0, 0, 0, 1) == 1);
    REQUIRE(ten1.ravel_index(0, 0, 0, 2) == 2);
    REQUIRE(ten1.ravel_index(0, 0, 1, 0) == 10);
    REQUIRE(ten1.ravel_index(0, 0, 1, 1) == 11);
    REQUIRE(ten1.ravel_index(0, 0, 2, 6) == 26);
    REQUIRE(ten1.ravel_index(0, 2, 4, 9) == 149);
}

TEST_CASE("Strides", "[Tensor]") {
    Tensor<int> ten1 = Tensor<int>::iota({1, 3, 5, 10});

    // ensure that the strides are correct
    REQUIRE(ten1.stride(0) == 150);
    REQUIRE(ten1.stride(1) == 50);
    REQUIRE(ten1.stride(2) == 10);
    REQUIRE(ten1.stride(3) == 1);

    // Reshape the tensor
    ten1.reshape_({15, 2, 5});
    REQUIRE(ten1.stride(0) == 10);
    REQUIRE(ten1.stride(1) == 5);
    REQUIRE(ten1.stride(2) == 1);

    REQUIRE_THROWS_AS(ten1.reshape({1, 2, 3, 4}), std::runtime_error);
    REQUIRE_THROWS_AS(ten1.reshape_({1, 2, 3, 4}), std::runtime_error);

    // out of place reshape should copy the data
    auto ten2 = ten1.reshape({1, 10, 5, 3});
    for (SizeType i = 0; i < ten2.numel(); ++i) {
        REQUIRE(ten2.flat(i) == i);
    }
}

TEST_CASE("Sum", "[Tensor]") {
    Tensor<int> ten1({2, 3});
    Tensor<int> ten2 = Tensor<int>::iota({2, 3});

    Tensor<int> ten3 = ten1 + ten2;

    REQUIRE(ten3(0, 0) == 0);
    REQUIRE(ten3(0, 1) == 1);
    REQUIRE(ten3(0, 2) == 2);
    REQUIRE(ten3(1, 0) == 3);
    REQUIRE(ten3(1, 1) == 4);
    REQUIRE(ten3(1, 2) == 5);

    Tensor<float> ten4 = ten2.astype<float>() + Tensor<float>::iota({2, 3});

    REQUIRE(ten4(0, 0) == 0.0f);
    REQUIRE(ten4(0, 1) == 2.0f);
    REQUIRE(ten4(0, 2) == 4.0f);
    REQUIRE(ten4(1, 0) == 6.0f);
    REQUIRE(ten4(1, 1) == 8.0f);
    REQUIRE(ten4(1, 2) == 10.0f);
}

TEST_CASE("Subtraction", "[Tensor]") {
    Tensor<int> ten1({2, 3});
    Tensor<int> ten2 = Tensor<int>::iota({2, 3});

    Tensor<int> ten3 = ten1 - ten2;

    REQUIRE(ten3(0, 0) == 0);
    REQUIRE(ten3(0, 1) == -1);
    REQUIRE(ten3(0, 2) == -2);
    REQUIRE(ten3(1, 0) == -3);
    REQUIRE(ten3(1, 1) == -4);
    REQUIRE(ten3(1, 2) == -5);

    Tensor<float> ten4 = ten2.astype<float>() - Tensor<float>::iota({2, 3});

    REQUIRE(ten4(0, 0) == 0.0f);
    REQUIRE(ten4(0, 1) == 0.0f);
    REQUIRE(ten4(0, 2) == 0.0f);
    REQUIRE(ten4(1, 0) == 0.0f);
    REQUIRE(ten4(1, 1) == 0.0f);
    REQUIRE(ten4(1, 2) == 0.0f);
}

TEST_CASE("Multiplication", "[Tensor]") {
    Tensor<int> ten1({2, 3});
    Tensor<int> ten2 = Tensor<int>::iota({2, 3});

    Tensor<int> ten3 = ten1 * ten2;

    REQUIRE(ten3(0, 0) == 0);
    REQUIRE(ten3(0, 1) == 0);
    REQUIRE(ten3(0, 2) == 0);
    REQUIRE(ten3(1, 0) == 0);
    REQUIRE(ten3(1, 1) == 0);
    REQUIRE(ten3(1, 2) == 0);

    Tensor<float> ten4 = ten2.astype<float>() * Tensor<float>::iota({2, 3});

    REQUIRE(ten4(0, 0) == 0.0f);
    REQUIRE(ten4(0, 1) == 1.0f);
    REQUIRE(ten4(0, 2) == 4.0f);
    REQUIRE(ten4(1, 0) == 9.0f);
    REQUIRE(ten4(1, 1) == 16.0f);
    REQUIRE(ten4(1, 2) == 25.0f);
}

TEST_CASE("Division", "[Tensor]") {
    Tensor<int> ten1({2, 3}, 2);                      // Tensor of twos
    Tensor<int> ten2 = Tensor<int>::iota({2, 3}, 1);  // Tensor from 1 to 6

    Tensor<int> ten3 = ten1 / ten2;

    REQUIRE(ten3(0, 0) == 2);
    REQUIRE(ten3(0, 1) == 1);
    REQUIRE(ten3(0, 2) == 0);
    REQUIRE(ten3(1, 0) == 0);
    REQUIRE(ten3(1, 1) == 0);
    REQUIRE(ten3(1, 2) == 0);

    Tensor<float> ten4 = ten1.astype<float>() / Tensor<float>::iota({2, 3}, 1.0f);  // Tensor from 1.0 to 6.0

    REQUIRE(ten4(0, 0) == 2.0f);
    REQUIRE(ten4(0, 1) == 1.0f);
    REQUIRE(ten4(0, 2) == 2.0f / 3.0f);
    REQUIRE(ten4(1, 0) == 0.5f);
    REQUIRE(ten4(1, 1) == 2.0f / 5.0f);
    REQUIRE(ten4(1, 2) == 1.0f / 3.0f);
}

TEST_CASE("ShapeIter", "[Tensor]") {
    Tensor<int> ten({1, 3, 4});

    std::vector<std::vector<SizeType>> indices;
    for (auto& v : ten.shape_iter()) {
        indices.push_back(v);
    }

    REQUIRE(indices.size() == 12);
    for (SizeType i = 0; i < indices.size(); i++) {
        REQUIRE(ten.ravel_index(indices.at(i)) == i);
    }

    for (SizeType i = 0; i < indices.size(); i++) {
        ten(indices.at(i)) = i;
    }

    for (SizeType i = 0; i < indices.size(); i++) {
        REQUIRE(ten(indices.at(i)) == i);
    }

    for (auto& v : ten.shape_iter()) {
        REQUIRE(ten.unravel_index(ten.ravel_index(v)) == v);
    }
}

TEST_CASE("Permute", "[Tensor]") {
    /*
        0 1 2
        3 4 5
        ->
        0 3
        1 4
        2 5
    */
    Tensor<int> ten1 = Tensor<int>::iota({2, 3});
    Tensor<int> ten2 = ten1.permute({1, 0});

    REQUIRE(ten2(0, 0) == 0);
    REQUIRE(ten2(0, 1) == 3);
    REQUIRE(ten2(1, 0) == 1);
    REQUIRE(ten2(1, 1) == 4);
    REQUIRE(ten2(2, 0) == 2);
    REQUIRE(ten2(2, 1) == 5);

    Tensor<int> ten3 = ten2.permute({1, 0});

    REQUIRE(ten3(0, 0) == 0);
    REQUIRE(ten3(0, 1) == 1);
    REQUIRE(ten3(0, 2) == 2);
    REQUIRE(ten3(1, 0) == 3);
    REQUIRE(ten3(1, 1) == 4);
    REQUIRE(ten3(1, 2) == 5);

    // permuting with more than 2 dimensions should throw an error
    REQUIRE_THROWS(ten1.permute({0, 1, 2}));
}

TEST_CASE("TrigFunctions", "[Tensor]") {
    Tensor<float> ten = Tensor<float>::iota({2, 5}, 1.0f);

    SECTION("Cos") {
        auto cos = ten.cos();
        float sum = std::accumulate(cos.begin(), cos.end(), 0.0f);
        REQUIRE_THAT(sum, Catch::Matchers::WithinAbs(-1.4174476861953735, 1e-6f));
        REQUIRE_THAT(cos(0, 0), Catch::Matchers::WithinAbs(0.5403023362159729, 1e-6f));
    }

    SECTION("Sin") {
        auto sin = ten.sin();
        float sum = std::accumulate(sin.begin(), sin.end(), 0.0f);
        REQUIRE_THAT(sum, Catch::Matchers::WithinAbs(1.4111881256103516, 1e-6f));
        REQUIRE_THAT(sin(0, 0), Catch::Matchers::WithinAbs(0.8414709568023682f, 1e-6f));
    }

    SECTION("Tan") {
        auto tan = ten.tan();
        float sum = std::accumulate(tan.begin(), tan.end(), 0.0f);
        REQUIRE_THAT(sum, Catch::Matchers::WithinAbs(-9.016096115112305, 1e-6f));
        REQUIRE_THAT(tan(0, 0), Catch::Matchers::WithinAbs(1.5574077367782593f, 1e-6f));
    }

    SECTION("Csc") {
        auto csc = ten.csc();
        float sum = std::accumulate(csc.begin(), csc.end(), 0.0f);
        REQUIRE_THAT(sum, Catch::Matchers::WithinAbs(6.5524091720581055f, 1e-6f));
        REQUIRE_THAT(csc(0, 0), Catch::Matchers::WithinAbs(1.1883951425552368f, 1e-6f));
    }

    SECTION("Sec") {
        auto sec = ten.sec();
        float sum = std::accumulate(sec.begin(), sec.end(), 0.0f);
        REQUIRE_THAT(sum, Catch::Matchers::WithinAbs(-6.361124515533447f, 1e-6f));
        REQUIRE_THAT(sec(0, 0), Catch::Matchers::WithinAbs(1.8508152961730957f, 1e-6f));
    }
}

TEST_CASE("Strided-Indexing", "[Tensor]") {
    using TensorType = Tensor<int>;

    TensorType ten = TensorType::iota({2, 3});

    SECTION("Unstrided multidim indexing") {
        SizeType v = 0;
        for (SizeType i = 0; i < ten.shape(0); i++) {
            for (SizeType j = 0; j < ten.shape(1); j++) {
                REQUIRE(ten(i, j) == v);
                v++;
            }
        }
    }

    SECTION("Unstrided unflattening") {
        SizeType v = 0;
        for (SizeType i = 0; i < ten.shape(0); i++) {
            for (SizeType j = 0; j < ten.shape(1); j++) {
                REQUIRE(ten.unravel_index(v) == IndexType{i, j});
                v++;
            }
        }
    }

    SECTION("Unstrided flattening") {
        SizeType v = 0;
        for (SizeType i = 0; i < ten.shape(0); i++) {
            for (SizeType j = 0; j < ten.shape(1); j++) {
                REQUIRE(ten.ravel_index(IndexType{i, j}) == v);
                v++;
            }
        }
    }

    SECTION("Unstrided Unflat-Flat") {
        for (SizeType i = 0; i < ten.numel(); i++) {
            REQUIRE(ten.ravel_index(ten.unravel_index(i)) == i);
        }
    }

    SECTION("Unstrided Flat Indexing") {
        for (SizeType i = 0; i < ten.numel(); i++) {
            REQUIRE(ten.flat(i) == i);
        }
    }

    auto ten2 = ten.permute({1, 0});

    SECTION("Strided multidim indexing") {
        for (SizeType i = 0; i < ten2.shape(0); i++) {
            for (SizeType j = 0; j < ten2.shape(1); j++) {
                REQUIRE(ten2(i, j) == ten(j, i));
            }
        }
    }

    SECTION("Strided unflattening") {
        SizeType v = 0;
        for (SizeType i = 0; i < ten2.shape(0); i++) {
            for (SizeType j = 0; j < ten2.shape(1); j++) {
                REQUIRE(ten2.unravel_index(v) == IndexType{i, j});
                v++;
            }
        }
    }

    SECTION("Strided flattening") {
        SizeType v = 0;
        for (SizeType i = 0; i < ten2.shape(0); i++) {
            for (SizeType j = 0; j < ten2.shape(1); j++) {
                REQUIRE(ten2.ravel_index(IndexType{i, j}) == v);
                v++;
            }
        }
    }

    SECTION("Strided Unflat-Flat") {
        for (SizeType i = 0; i < ten2.numel(); i++) {
            REQUIRE(ten2.ravel_index(ten2.unravel_index(i)) == i);
        }
    }

    SECTION("Strided Flat Indexing") {
        std::vector<int> v{0, 3, 1, 4, 2, 5};
        for (SizeType i = 0; i < ten2.numel(); i++) {
            REQUIRE(ten2.flat(i) == v[i]);
        }
    }
}

TEST_CASE("Benchmark Iter", "[Tensor]") {
    auto ten = Tensor<int>({100, 100, 100});

    BENCHMARK("ShapeIter") {
        for (auto& v : ten.shape_iter()) {
            ten(v) = 1;
        }
    };

    BENCHMARK("Flat") {
        for (SizeType i = 0; i < ten.numel(); i++) {
            ten.flat(i) = i;
        }
    };
}