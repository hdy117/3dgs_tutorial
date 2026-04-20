/* ================================================================
 *  Chapter 13 — if constexpr & requires: C++20模板革命
 * ================================================================
 *
 * 🧠 推导：
 *   SFINAE(第9章)太复杂 —— enable_if嵌套像俄罗斯套娃。
 *   
 *   C++20给了答案:
 *     if constexpr → 编译期if，替代SFINAE
 *     concept      → 命名化约束，替代enable_if
 *     requires子句 → 前置条件声明
 *
 * 💡 Mental Model：
 *   if constexpr(条件)：false分支连语法分析都不做。两个分支可以都"不合法"给对方类型，编译器只编译一个。

 * ================================================================ */

#include <iostream>
#include <string>
#include <sstream>
#include <type_traits>
using namespace std;

// ──────────────────────────────────────────────────────────────────
// 1. if constexpr — 替代SFINAE的最简单方式
//    两个版本共存，编译器只编译适合当前类型的那个分支。
// ──────────────────────────────────────────────────────────────────
template<typename T>
string process(T val) {
    if constexpr (is_integral_v<T>) {
        // 只有整数类型的T才会编译这段代码！
        int v = static_cast<int>(val);
        int stars = min(v / 10 + 1, 20);   // 最多显示20个星号
        return string(stars, '*');
    } else if constexpr (is_floating_point_v<T>) {
        // 只有浮点类型会编译这里
        ostringstream oss;
        oss << val;
        return oss.str();
    } else if constexpr (is_same_v<remove_cvref_t<T>, string>) {
        // 只有string类型会编译这里
        return "\"" + val + "\"";
    } else {
        // 其他所有类型走这里 —— typeid().name()返回const char*，用ostringstream拼接
        ostringstream oss;
        oss << "unsupported: ";
        oss << typeid(T).name();
        return oss.str();
    }
}

// ──────────────────────────────────────────────────────────────────
// 2. requires子句 —— 给模板函数写"前置条件"
// ──────────────────────────────────────────────────────────────────

// C++14/SFINAE方式: template<typename T, typename = enable_if_t<is_integral_v<T>>> int square(T t) { return t * t; }
// C++20 requires方式（清晰得多）:
template<typename T>
    requires is_integral_v<T>       // ← "T必须是整数类型"
int square(T t) {
    return t * t;
}

// 浮点数版本 —— 两个函数共存，编译器自动选匹配的
template<typename T>
    requires is_floating_point_v<T>
double square(T t) {
    return t * t;
}

// ──────────────────────────────────────────────────────────────────
// 3. requires表达式 —— 检测"类型是否支持某个操作"
//    替代 std::declval + enable_if的复杂模式。
// ──────────────────────────────────────────────────────────────────

// 检测T是否有cout << t这种输出能力
template<typename T>
concept Printable = requires(T t) {
    cout << t;          // ← 编译器检查这段代码对类型T是否合法(不需要真的执行!)
};

// 检测T是否可以相加且结果是算术类型
template<typename T>
concept Addable = requires(T a, T b) {
    { a + b } -> is_arithmetic;   // ← 返回类型必须是算术类型
};

// ──────────────────────────────────────────────────────────────────
// 4. 组合concept —— AND/OR/NOT逻辑
// ──────────────────────────────────────────────────────────────────
template<typename T>
concept Numeric = Addable<T> && is_arithmetic_v<T>;

int main() {
    // ======================== 实验1: if constexpr — 无SFINAE之苦 ======
    
    cout << "=== if constexpr ===" << endl;
    cout << "process(42)     = " << process(42)         << endl;   // ******** (最多20个星号)
    cout << "process(3.14)   = " << process(3.14)       << endl;   // 3.14
    cout << "process(\"hi\")   = " << process(string("hi")) << endl;   // "hi"
    
    cout << "\n";
    
    // ======================== 实验2: requires子句 — 清晰的约束 =========
    
    cout << "=== requires子句 ===" << endl;
    cout << "square(5)   = " << square(5)     << endl;   // 25
    cout << "square(3.0) = " << square(3.0)   << endl;   // 9
    
    // ⚠️ square("hi") → ❌ ERROR! string不满足任何requires条件。
    // requires时代：直接报错"约束不满足"——清晰明了！
    
    cout << "\n";
    
    // ======================== 实验3: requires表达式 — 自动检测能力 ======
    
    cout << "=== requires表达式 ===" << endl;
    int x = 42;
    string s("hello");
    
    // Printable<int>::value → true (int可以cout)
    // Printable<string>::value → true (string可以cout)
    struct Foo { int x; };        // ← 没有operator<<，不可Printable
    // Printable<Foo>::value → false
    
    if constexpr (Printable<int>)
        cout << "int is Printable\n";
    
    if constexpr (Printable<string>)
        cout << "string is Printable\n";
        
    if constexpr (!Printable<Foo>)
        cout << "Foo NOT Printable ✓\n";
    
    // Addable和Numeric验证：
    if constexpr (Addable<int>)
        cout << "int is Addable\n";
    
    if constexpr (Numeric<double>)
        cout << "double is Numeric\n";
        
    if constexpr (!Printable<Foo>)
        cout << "Foo NOT Printable ✓\n";
    
    // Addable和Numeric验证：
    if constexpr (Addable<int>)
        cout << "int is Addable\n";
    
    if constexpr (Numeric<double>)
        cout << "double is Numeric\n";

    // ======================== 实验4: SFINAE vs if constexpr对比 ======
    
    cout << "\n=== C++17 if constexpr vs C++14 SFINAE ===" << endl;
    cout << "SFINAE (C++14):\n";
    cout << "  template<typename T>\n";
    cout << "  typename enable_if<cond, return_type>::type\n";
    cout << "  func(T t) { ... }\n";
    cout << "\nif constexpr (C++17+):\n";
    cout << "  template<typename T>\n";
    cout << "  auto func(T t) {\n";
    cout << "    if constexpr(cond) { return version_a; }\n";
    cout << "    else              { return version_b; }\n";
    cout << "  }\n";
    
    // ======================== 实验5: requires vs enable_if对比 ========
    
    cout << "\n=== C++20 requires vs C++14 SFINAE ===" << endl;
    cout << "enable_if (C++14):\n";
    cout << "  template<typename T, typename = enable_if_t<cond>>\n";
    cout << "  int func(T t) { ... }\n";
    cout << "\nrequires (C++20+):\n";
    cout << "  template<typename T>\n";
    cout << "    requires cond\n";
    cout << "  int func(T t) { ... }\n";

    // ======================== 实验6: Numeric concept验证 =============
    
    cout << "\n=== Numeric概念 ===" << endl;
    if constexpr (Numeric<int>)       cout << "int是Numeric ✓\n";
    if constexpr (Numeric<double>)    cout << "double是Numeric ✓\n";
    // string不满足Addable+arithmetic，所以不是Numeric

    cout << "\n";
    cout << "🎯 本章要点:\n";
    cout << "1. if constexpr: false分支连编译都不做 —— SFINAE的直观替代品\n";
    cout << "2. requires子句: template<T> requires Cond ← 清晰的约束语法\n";
    cout << "3. requires表达式: concept检测类型能力(无需手动写traits)\n";
    cout << "4. C++20 = SFINAE和TMP的'终结者' —— 更可读、更好维护\n";
    
    return 0;
}
