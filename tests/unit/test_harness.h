// tests/unit/test_harness.h — harness mínimo (sem dependências) até a adoção de GoogleTest/Catch2.
//
//   CNX_TEST(nome) { CHECK(expr); CHECK_CTX(expr, contexto); CHECK_THROWS(Tipo, instrução); }
//   int main() { return cnxtest::runAll(); }
#pragma once

#include <cstdio>
#include <exception>
#include <string>
#include <utility>
#include <vector>

#if defined(_MSC_VER) && defined(_DEBUG)
#include <crtdbg.h>
#endif

namespace cnxtest {

using TestFn = void (*)();

inline std::vector<std::pair<const char*, TestFn>>& registry() {
    static std::vector<std::pair<const char*, TestFn>> tests;
    return tests;
}

inline int& failureCount() {
    static int failures = 0;
    return failures;
}

inline void report(bool ok, const char* expr, const char* file, int line, const std::string& context = {}) {
    if (ok) return;
    std::printf("  FALHOU %s:%d: %s\n", file, line, expr);
    if (!context.empty()) std::printf("         %s\n", context.c_str());
    ++failureCount();
}

struct Registrar {
    Registrar(const char* name, TestFn fn) { registry().emplace_back(name, fn); }
};

inline int runAll() {
    // Sem buffer: se um teste travar ou abortar, as linhas já impressas não se perdem.
    std::setvbuf(stdout, nullptr, _IONBF, 0);
#if defined(_MSC_VER) && defined(_DEBUG)
    // Asserções da CRT/STL vão para stderr em vez de abrir uma caixa de diálogo (execução não interativa).
    _CrtSetReportMode(_CRT_ASSERT, _CRTDBG_MODE_FILE);
    _CrtSetReportFile(_CRT_ASSERT, _CRTDBG_FILE_STDERR);
    _CrtSetReportMode(_CRT_ERROR, _CRTDBG_MODE_FILE);
    _CrtSetReportFile(_CRT_ERROR, _CRTDBG_FILE_STDERR);
#endif
    int failedTests = 0;
    for (const auto& [name, fn] : registry()) {
        const int before = failureCount();
        try {
            fn();
        } catch (const std::exception& e) {
            std::printf("  EXCEÇÃO: %s\n", e.what());
            ++failureCount();
        }
        const bool ok = failureCount() == before;
        std::printf("[%s] %s\n", ok ? " OK " : "FALHA", name);
        if (!ok) ++failedTests;
    }
    std::printf("%zu testes, %d com falha\n", registry().size(), failedTests);
    return failedTests == 0 ? 0 : 1;
}

}  // namespace cnxtest

#define CNX_TEST(name)                                                   \
    static void name();                                                  \
    static const ::cnxtest::Registrar name##_registrar(#name, &name);    \
    static void name()

#define CHECK(expr) ::cnxtest::report(static_cast<bool>(expr), #expr, __FILE__, __LINE__)
#define CHECK_CTX(expr, context) ::cnxtest::report(static_cast<bool>(expr), #expr, __FILE__, __LINE__, (context))
#define CHECK_THROWS(ExceptionType, statement)                                                  \
    do {                                                                                        \
        bool caught_ = false;                                                                   \
        try {                                                                                   \
            statement;                                                                          \
        } catch (const ExceptionType&) {                                                        \
            caught_ = true;                                                                     \
        }                                                                                       \
        ::cnxtest::report(caught_, "lança " #ExceptionType ": " #statement, __FILE__, __LINE__); \
    } while (0)
