#pragma once
// int-util.h — safe integer type helpers
//
// Three families:
//
//   narrow_cast<T>(v) — unconditionally casts; documents the narrowing
//       intent for human readers. Same codegen as static_cast<T>(v).
//
//   narrow<T>(v) — checked cast; throws std::overflow_error when the
//       source value cannot be represented as T. Use during loading /
//       deserialisation where inputs may be untrusted.
//
//   ssize(c) — returns the size of a contiguous container or C array
//       as a signed ptrdiff_t (mirrors C++20 std::ssize).  Eliminates
//       the (int)v.size() pattern that triggers MSVC /W3 sign-mismatch
//       warnings and allows direct comparison with signed loop counters.
//
// All entries live in a detail namespace to avoid ODR conflicts. The
// project has no C++20 requirement so we cannot rely on std::ssize /
// std::cmp_* yet.  Replace with standard facilities when the build
// target moves to C++20.

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>

// ---------------------------------------------------------------------------
// narrow / narrow_cast
// ---------------------------------------------------------------------------

namespace qt_detail {

template <typename T, typename U>
using qt_nc_enable_if = std::enable_if_t<std::is_arithmetic_v<T> && std::is_arithmetic_v<U>, int>;

// narrow_cast — unconditional narrowing cast, documents intent.
template <typename T, typename U, qt_nc_enable_if<T, U> = 0>
constexpr T narrow_cast(U v) noexcept {
    return static_cast<T>(v);
}

// narrow — checked cast, raises std::overflow_error on lossy conversion.
template <typename T, typename U, qt_nc_enable_if<T, U> = 0>
constexpr T narrow(U v) {
    T t = static_cast<T>(v);
    if (static_cast<U>(t) != v) {
        throw std::overflow_error("narrow: value out of range");
    }
    // For signed→signed narrowing, also verify sign is preserved.
    if constexpr (std::is_signed_v<T> != std::is_signed_v<U>) {
        if ((t < T{}) != (v < U{})) {
            throw std::overflow_error("narrow: sign mismatch");
        }
    }
    return t;
}

// Convenience — convert to int (the most common pattern in this codebase).
template <typename U, qt_nc_enable_if<int, U> = 0>
constexpr int as_int(U v) { return narrow<int>(v); }

template <typename U, qt_nc_enable_if<std::size_t, U> = 0>
constexpr std::size_t as_size(U v) { return narrow<std::size_t>(v); }

}  // namespace qt_detail

// Drag the helpers into the global namespace so they read like built-in
// functions:  narrow<int>(x),  narrow_cast<size_t>(k*n),  ssize(vec)
using qt_detail::narrow_cast;
using qt_detail::narrow;
using qt_detail::as_int;
using qt_detail::as_size;

// ---------------------------------------------------------------------------
// ssize — signed size for contiguous containers & C arrays
// ---------------------------------------------------------------------------

namespace qt_detail {

// Container overload (via .size() member).
template <typename C>
constexpr auto ssize(const C & c) noexcept -> std::ptrdiff_t {
    using S = decltype(c.size());
    // Most .size() return types fit in ptrdiff_t; narrow_cast documents
    // the narrowing.
    return narrow_cast<std::ptrdiff_t>(c.size());
}

// C array overload (decays to pointer — use with care).
template <typename T, std::size_t N>
constexpr std::ptrdiff_t ssize(const T (&)[N]) noexcept {
    return narrow_cast<std::ptrdiff_t>(N);
}

}  // namespace qt_detail

using qt_detail::ssize;
