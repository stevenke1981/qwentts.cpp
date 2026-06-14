// test-ints.cpp — unit tests for int-util.h narrow_cast / narrow / ssize
#include <catch2/catch_test_macros.hpp>

#include "int-util.h"

#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// narrow_cast — unconditional, always succeeds
// ---------------------------------------------------------------------------

TEST_CASE("narrow_cast preserves value", "[ints]") {
    // size_t → int
    size_t a = 42;
    CHECK(narrow_cast<int>(a) == 42);

    // int → size_t
    int b = -1;
    CHECK(narrow_cast<size_t>(b) == (size_t)-1);  // wraps, no check

    // truncation (no check)
    int64_t big = 0x1'0000'0000;
    CHECK(narrow_cast<int32_t>(big) == 0);
}

// ---------------------------------------------------------------------------
// narrow — checked, throws on precision loss
// ---------------------------------------------------------------------------

TEST_CASE("narrow succeeds for in-range values", "[ints]") {
    CHECK(narrow<int>(size_t{99}) == 99);
    CHECK(narrow<size_t>(int{0}) == 0);
    CHECK(narrow<short>(int{32000}) == 32000);
    CHECK(narrow<int>(int64_t{1'000'000'000}) == 1'000'000'000);
    CHECK(narrow<int32_t>(int64_t{0}) == 0);
}

TEST_CASE("narrow throws on overflow", "[ints]") {
    // size_t → int (value exceeds int max)
    size_t too_big = 1'000'000'000'000ULL;
    CHECK_THROWS_AS(narrow<int>(too_big), std::overflow_error);

    // int64_t → int32_t
    int64_t big = 5'000'000'000LL;
    CHECK_THROWS_AS(narrow<int32_t>(big), std::overflow_error);

    // signed negative → unsigned
    CHECK_THROWS_AS(narrow<size_t>(int{-1}), std::overflow_error);
}

// ---------------------------------------------------------------------------
// ssize — signed container/C-array size
// ---------------------------------------------------------------------------

TEST_CASE("ssize returns signed size for vector", "[ssize]") {
    std::vector<int> v{1, 2, 3};
    auto s = ssize(v);
    CHECK(s == 3);
    // Must be signed
    CHECK(std::is_signed_v<decltype(s)>);
}

TEST_CASE("ssize returns signed size for string", "[ssize]") {
    std::string s = "hello";
    CHECK(ssize(s) == 5);
}

TEST_CASE("ssize works on C arrays", "[ssize]") {
    int arr[] = {10, 20, 30, 40};
    CHECK(ssize(arr) == 4);
    CHECK(std::is_same_v<decltype(ssize(arr)), std::ptrdiff_t>);
}

TEST_CASE("ssize empty containers return 0", "[ssize]") {
    std::vector<int> empty;
    CHECK(ssize(empty) == 0);
}

// ---------------------------------------------------------------------------
// as_int / as_size convenience wrappers
// ---------------------------------------------------------------------------

TEST_CASE("as_int converts size_t to int", "[ints]") {
    CHECK(as_int(size_t{123}) == 123);
    CHECK_THROWS_AS(as_int((size_t)-1), std::overflow_error);
}

TEST_CASE("as_size converts int to size_t", "[ints]") {
    CHECK(as_size(456) == 456);
    CHECK_THROWS_AS(as_size(-1), std::overflow_error);
}

// ---------------------------------------------------------------------------
// Edge cases: boundary values closest to narrowing edge
// ---------------------------------------------------------------------------

TEST_CASE("narrow boundary: int max/min", "[ints]") {
    CHECK(narrow<int>(size_t{INT_MAX}) == INT_MAX);
    CHECK(narrow<int>(int64_t{INT_MIN}) == INT_MIN);
    // One beyond
    CHECK_THROWS_AS(narrow<int>(size_t(INT_MAX) + 1), std::overflow_error);
    CHECK_THROWS_AS(narrow<int>(int64_t(INT_MIN) - 1), std::overflow_error);
}
