/* ================================================================
 *  Chapter 11 — Fold表达式: 打包解包的艺术
 * ================================================================
 *
 * 🧠 推导：
 *   variadic模板(第12章)可以接收任意数量的参数: args...
 *   但怎么"展开"这些参数？C++17之前需要递归+特化，极其复杂。
 *   
 *   C++17的fold表达式直接说：把参数包用运算符折叠！
 *   (args + ...) = arg1 + arg2 + arg3 + ...
 *
 * 💡 Mental Model:
 *   fold = 把参数包"压缩"成一个值，像归约(reduce)操作。
 *   
 *   四种fold：
 *     (expr op ...)    → left fold: ((e1 op e2) op e3) op ...
 *     (... op expr)    → right fold: ... op (e3 op (e2 op e1))
 *     (init op expr...)→ 有初始值的left fold
 *     (expr... op init)→ 有初始值的right fold
 *
 * ================================================================ */

#include <iostream>
#include <string>
using namespace std;

// ──────────────────────────────────────────────────────────────────
// 1. Left Fold: (args + ...) —— 累加所有参数
//    ⚠️ 空包折叠会出错，需要用有初始值版本。
// ──────────────────────────────────────────────────────────────────
template<typename... Args>
auto sum(Args... args) {
    return (args + ...);     // ← left fold: ((arg1+arg2)+arg3)+...
}

// 有初始值的left fold —— 处理空包情况
template<typename T, typename... Args>
auto sum_init(T init, Args... args) {
    return (init + ... + args);   // ← init+(args...) : ((init+arg1)+arg2)...
}

// ──────────────────────────────────────────────────────────────────
// 2. Right Fold: (... && args) —— 逻辑与折叠
// ──────────────────────────────────────────────────────────────────
template<typename... Args>
bool all_true(Args... args) {
    return (args && ...);    // ← arg1&&arg2&&arg3&&...
}

// ──────────────────────────────────────────────────────────────────
// 3. Print所有参数 —— C++17前需要递归，现在一行搞定
// ──────────────────────────────────────────────────────────────────
template<typename... Args>
void print_all(Args&&... args) {
    // initializer_list fold: 为每个参数输出一个值后放0到数组中
    int dummy[] = {(cout << args << " ", 0)...};   // ← 展开！
    (void)dummy;   // 抑制unused warning
}

// C++17更优雅的方式：逗号表达式fold
template<typename... Args>
void println(Args&&... args) {
    ((cout << args << " "), ...);   // 每个参数输出，最后多余一个逗号（无影响）
    cout << endl;
}

// ──────────────────────────────────────────────────────────────────
// 4. Fold表达式 vs C++14递归展开对比
// ──────────────────────────────────────────────────────────────────
// C++14版本（复杂，需要helper函数+特化终止）：
template<typename T>
T sum_recursive(T t) { return t; }

template<typename T, typename... Rest>
auto sum_recursive(T first, Rest... rest) {
    return first + sum_recursive(rest...);
}

int main() {
    // ======================== 实验1: Left Fold — 累加 ================
    
    cout << "=== (args + ...) ===" << endl;
    cout << "sum(1,2,3)      = " << sum(1, 2, 3)         << endl;   // 6
    cout << "sum(10)        = " << sum(10)             << endl;   // 10
    
    // sum() → ❌ ERROR! 空包没有单位元，fold表达式无法计算。
    // 解决：用有初始值的版本
    cout << "\n";
    
    // ======================== 实验2: 有初始值Fold — 空包安全 =========
    
    cout << "=== (init + ... + args) ===" << endl;
    cout << "sum_init(0)       = " << sum_init(0)         << endl;   // 0 (空包返回初始值)
    cout << "sum_init(1,2,3)   = " << sum_init(1, 2, 3)   << endl;   // 6
    
    // C++14递归版本也能处理空包：
    cout << "sum_recursive()   = " << sum_recursive(0)     << endl;   // 0 (需要至少一个参数)
    
    cout << "\n";
    
    // ======================== 实验3: Right Fold — 逻辑与 ==============
    
    cout << "=== (args && ...) ===" << endl;
    cout << "all_true(1,1,1)   = " << all_true(true, true, true)  << endl;   // true
    cout << "all_true(1,0,1)   = " << all_true(true, false, true) << endl;   // false
    
    cout << "\n";
    
    // ======================== 实验4: print_all — 展开cout ==========
    
    cout << "=== 展开打印 ===" << endl;
    print_all("Hello", " ", "World", "!");   // Hello World !
    println(1, " + ", 2.5, " = ", 3.5);      // 1 + 2.5 = 3.5 
    
    cout << "\n";
    
    // ======================== 实验5: Fold vs C++14递归对比 ===========
    
    cout << "=== fold的进化意义 ===" << endl;
    cout << "C++14之前:\n";
    cout << "  template<T> T sum_single(T t) { return t; }\n";
    cout << "  template<T, Rest...>\n";
    cout << "  auto sum(T t, Rest... rest) { return t + sum(rest...); }\n";
    cout << "\nC++17:   (args + ...) —— 一行，清晰，零歧义。\n";

    // ======================== 实验6: fold的实际应用 ===============
    
    cout << "\n=== fold实战: 格式化输出 ===" << endl;
    auto format = []<typename... Args>(const string& fmt, Args&&... args) {
        // ⚠️ lambda模板参数需要C++20，这里用简单方式演示fold
        println("格式值: ", args...);
    };
    
    cout << "🎯 本章要点:\n";
    cout << "1. fold = 把参数包用运算符压缩成一个值\n";
    cout << "2. (args + ...) = left fold, (... && args) = right fold\n";
    cout << "3. 空参数包需要初始值: (init op args...)\n";
    cout << "4. fold让C++17之前复杂的递归展开变成一行代码\n";
    
    return 0;
}
