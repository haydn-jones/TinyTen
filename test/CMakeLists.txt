# ---- Dependencies ----

include(../cmake/CPM.cmake)

CPMAddPackage("gh:catchorg/Catch2@3.2.1")

add_executable(test main.cpp)
target_link_libraries(test PRIVATE TinyTen Catch2::Catch2WithMain)
target_compile_features(test PRIVATE cxx_std_17)

enable_testing()
add_test(test test)