#ifndef TEST_HARNESS_HPP
#define TEST_HARNESS_HPP

#include <iostream>
#include <vector>
#include <string>
#include <functional>
#include <cmath>

class TestHarness {
public:
    using TestFunc = std::function<void()>;

    static TestHarness& instance() {
        static TestHarness instance;
        return instance;
    }

    void register_test(const std::string& name, TestFunc func) {
        tests.push_back({name, func});
    }

    int run_all() {
        int passed = 0;
        int failed = 0;
        std::cout << "Running " << tests.size() << " tests...\n";

        for (const auto& test : tests) {
            std::cout << "[RUNNING] " << test.name << "... ";
            try {
                test.func();
                std::cout << "PASS\n";
                passed++;
            } catch (const std::exception& e) {
                std::cout << "FAIL: " << e.what() << "\n";
                failed++;
            } catch (...) {
                std::cout << "FAIL: Unknown error\n";
                failed++;
            }
        }

        std::cout << "\nResults: " << passed << " passed, " << failed << " failed.\n";
        return failed > 0 ? 1 : 0;
    }

private:
    struct Test {
        std::string name;
        TestFunc func;
    };
    std::vector<Test> tests;
};

#define TEST(name) \
    void name(); \
    struct Register##name { Register##name() { TestHarness::instance().register_test(#name, name); } } register_##name; \
    void name()

#define ASSERT(condition) \
    if (!(condition)) { \
        throw std::runtime_error("Assertion failed: " #condition); \
    }

#define ASSERT_EQ(a, b) \
    if ((a) != (b)) { \
        throw std::runtime_error("Assertion failed: " #a " == " #b " (Values: " + std::to_string(a) + " != " + std::to_string(b) + ")"); \
    }

#endif // TEST_HARNESS_HPP
