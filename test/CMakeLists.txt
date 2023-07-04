# ---- Dependencies ----

include(../cmake/CPM.cmake)

CPMAddPackage("gh:catchorg/Catch2@3.3.2")

add_executable(test main.cpp)
target_link_libraries(test PRIVATE TinyTen Catch2::Catch2WithMain)
set_target_properties(test PROPERTIES CXX_STANDARD 20)

enable_testing()
add_test(test test)