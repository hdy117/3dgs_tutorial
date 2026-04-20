/* ================================================================
 *  Chapter 09 — SFINAE & enable_if: 编译期的"优雅回退"
 * ================================================================
 *
 * 🧠 第一性原理推导：
 *   问题: 你想为有特定能力的类型提供一个函数，没有该能力的类型跳过。
 *   
 *   ❌ 如果用if (has_plus<T>) → C++没有运行时类型检查
 *   ❌ 如果用特化 → 你需要为"不支持+"的所有类型写特化（不可能）
 *   
 *   ✅ SFINAE: Substitution Failure Is Not An Error
 *      "替换失败不是错误" —— 编译器不报错，只是从候选列表中移除这个选项。
 *
 * 💡 Mental Model:
 *   enable_if<条件, T>::type = 
 *     如果条件为真 → type = T（模板匹配成功）
 *     如果条件为假 → type不存在（模板"消失"，不是报错！）
 *   
 *   这就像编译期的try-catch：这个版本不行？没关系，换一个。
 *
 * ================================================================ */

#include <iostream>
#include <string>
#include <type_traits>    // enable_if, is_integral等工具
#include <vector>         // has_size检测用
using namespace std;

// ──────────────────────────────────────────────────────────────────
// 1. SFINAE基础 —— 用enable_if做编译期过滤
//    enable_if<cond, T>::type —— C++14简化版enable_if_t<cond> = void (默认)
//    cond为真时存在，cond为假时"消失"（不是报错，是SFINAE！）
// ──────────────────────────────────────────────────────────────────

// 通用版：所有类型都能调用
template<typename T>
void show(T val) {
    cout << "[general] " << val << endl;
}

// int特化版：只有int能走这条路（通过enable_if约束）
template<typename T>
typename enable_if<is_integral<T>::value, string>::type 
describe(T val) {
    if (val == 0) return "零";
    if (val > 0)  return string(val / 10 + 1, '*');   // 每10个值打一个星号，最多6个
    return "-";
}

// double特化版：只有浮点类型能走这条路  
template<typename T>
typename enable_if<is_floating_point<T>::value, string>::type
describe(T val) {
    // 格式化输出保留两位小数
    char buf[32];
    snprintf(buf, sizeof(buf), "%.2f", val);
    return string(buf);
}

// string特化（非模板，普通函数重载）
string describe(const string& val) {
    return "\"" + val + "\"";
}

// ──────────────────────────────────────────────────────────────────
// 2. C++14简化写法 —— enable_if_t
//    enable_if_t<cond> = typename enable_if<cond, void>::type = void (默认)
// ──────────────────────────────────────────────────────────────────

template<typename T, typename = enable_if_t<is_integral<T>::value>>
int multiply(T a, T b) {       // 只有整数类型能调用！
    return a * b;
}

// string重载版本（需要单独写，因为string不支持*运算符）
string multiply(string a, string b) {
    // 模拟重复：创建一个更长的字符串
    return string(a.size() + b.size(), 'x');
}

// ──────────────────────────────────────────────────────────────────
// 3. C++17 —— if constexpr (下一章详讲，这里先用它简化show)
// ──────────────────────────────────────────────────────────────────
template<typename T>
string categorize(T val) {
    (void)val;   // unused warning suppression
    
    if constexpr (is_integral_v<T>) {
        return "整数";
    } else if constexpr (is_floating_point_v<T>) {
        return "浮点数";
    } else if constexpr (is_same_v<remove_cvref_t<T>, string>) {
        return "字符串";
    } else {
        return "未知类型";
    }
}

// ──────────────────────────────────────────────────────────────────
// 4. 类型特征检测 —— has_size trait
//    用SFINAE自动检测类型是否有size()方法。
// ──────────────────────────────────────────────────────────────────

template<typename T, typename = void>
struct has_size : false_type {};          // 默认：没有size()

template<typename T>
struct has_size<T, decltype(void(declval<T>().size()))> : true_type {};   // 有size()

// ──────────────────────────────────────────────────────────────────
// 5. 基于has_size的函数重载
// ──────────────────────────────────────────────────────────────────
template<typename C, typename = enable_if_t<has_size<C>::value>>


void show_size(const C& container) {
    cout << "有size(): " << container.size() << endl;
}

// 标量版本（没有size的类型）—— 用enable_if做反向约束
template<typename T, typename = enable_if_t<!has_size<T>::value>>


void show_size(T value) {
    cout << "标量: " << value << endl;
}

int main() {
    // ======================== 实验1: SFINAE多版本共存 ================
    
    cout << "=== enable_if多版本 ===" << endl;
    describe(42);         // → int特化版 (is_integral<int>::value=true) → "***"
    describe(3.14f);      // → double特化版 (is_floating_point<float>::value=true) → "3.14"
    describe(string("hi"));// → string重载 → "\"hi\""
    
    show(42);             // → 通用版 [general] 42
    
    cout << "\n";
    
    // ======================== 实验2: enable_if过滤 ==================
    
    cout << "=== enable_if过滤 ===" << endl;
    multiply(3, 4);       // ✅ int → 12
    multiply(string("hi"), string("ho"));   // ✅ string重载 → xxxxxx
    
    // multiply(3.0, 4.0) ❌ ERROR! double不满足enable_if<is_integral>条件，
    //   模板版本消失；没有double的重载版本 → 编译失败！
    cout << "multiply(3.0, 4.0) → 编译错误（double不是整数类型）\n";
    
    cout << "\n";
    
    // ======================== 实验3: if constexpr ================
    // if constexpr是C++17特性，比SFINAE更直观：条件为false的分支根本不编译。
    
    cout << "=== if constexpr ===" << endl;
    cout << "int→   " << categorize(42)         << endl;  // 整数
    cout << "double→  " << categorize(3.14)      << endl;  // 浮点数
    cout << "string→  " << categorize(string("hi")) << endl; // 字符串
    
    // if constexpr的关键优势：两个分支的代码可以同时存在，但只有一个被编译。
    
    cout << "\n";
    
    // ======================== 实验4: has_size trait检测 =============
    
    cout << "=== 类型特征检测 ===" << endl;
    vector<int> v{1,2,3};
    cout << "vector has_size:" << has_size<vector<int>>::value << endl;   // true
    
    int x = 42;
    cout << "int has_size:   " << has_size<int>::value         << endl;   // false
    cout << "string has_size:" << has_size<string>::value      << endl;   // true (有size())
    
    show_size(v);     // → 有size(): 3
    show_size(x);     // → 标量: 42
    
    cout << "\n";
    
    // ======================== 实验5: SFINAE vs if constexpr对比 =======
    
    cout << "=== C++17 if constexpr vs C++14 SFINAE ===" << endl;
    cout << "SFINAE (C++14): template<typename T, typename = enable_if_t<cond>>\n";
    cout << "                  需要嵌套::type，错误信息天书\n";
    cout << "\nif constexpr (C++17+):\n";
    cout << "                  if constexpr(cond) { ... } else { ... }\n";
    cout << "                  像普通if一样写代码，编译器只编译一个分支\n";
    
    cout << "\n";
    cout << "🎯 本章要点:\n";
    cout << "1. SFINAE: 替换失败不是错误 —— 编译器跳过不匹配的模板\n";
    cout << "2. enable_if<cond>::type = cond为真时存在, 为假时消失\n";
    cout << "3. enable_if_t<cond> = C++14简化版(默认第二个参数=void)\n";
    cout << "4. if constexpr(C++17) = 更直观的编译期条件分支\n";
    
    return 0;
}
