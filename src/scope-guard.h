#pragma once
// scope-guard.h: minimal scope_exit / scope_fail helpers for RAII-style
// cleanup in functions with multiple error paths.
//
//   auto guard = scope_exit([&] { cleanup(); });
//   if (fail) return false;          // guard runs cleanup
//   guard.dismiss();                 // suppress cleanup on success
//   return true;

#include <type_traits>
#include <utility>

template <typename F>
class ScopeGuard {
    F    f_;
    bool active_;
public:
    explicit ScopeGuard(F && f) : f_(std::move(f)), active_(true) {}
    ScopeGuard(ScopeGuard && other) noexcept : f_(std::move(other.f_)), active_(other.active_) {
        other.active_ = false;
    }
    ~ScopeGuard() {
        if (active_) f_();
    }
    void dismiss() { active_ = false; }
    ScopeGuard(const ScopeGuard &) = delete;
    ScopeGuard & operator=(const ScopeGuard &) = delete;
};

template <typename F>
ScopeGuard<F> scope_exit(F && f) { return ScopeGuard<F>(std::forward<F>(f)); }
