/* ================================================================
 *  Chapter 06 — 非类型模板参数：编译期常量作为参数
 * ================================================================
 *
 * 🧠 推导：
 *   到目前为止，我们只用了类型做模板参数: template<typename T>
 *   但模板参数还可以是"值": template<int N>
 *   
 *   这意味着什么？意味着你可以在模板里使用编译期常量！
 *   int arr[N] —— N在编译期确定，数组大小固定（零开销！）
 *
 * 💡 Mental Model:
 *   template<typename T, int SIZE>     ← 两个参数：一个类型 + 一个值
 *   template<double PI>                 ← 非类型模板可以用浮点吗？不行！（C++17前）
 *   template<bool FLAG>                 ← ✅ 可以用整型、指针、引用
 *   
 *   限制：非类型模板参数必须是编译期可求值的常量。
 *
 * ================================================================ */

#include <iostream>
#include <array>
using namespace std;

// ──────────────────────────────────────────────────────────────────
// 1. 最简单的非类型模板 —— 数组大小
//    这是最经典的应用场景：编译期确定容器大小
// ──────────────────────────────────────────────────────────────────
template<typename T, int N>
class FixedArray {
    T data[N];  // C++允许用模板参数定义数组！
    
public:
    constexpr int size() const { return N; }
    
    T& operator[](int i) { return data[i]; }
    const T& operator[](int i) const { return data[i]; }
};

// ──────────────────────────────────────────────────────────────────
// 2. 编译期数学 —— 非类型模板参数在编译期"计算"
//    factorial<N> 在编译期算出 N!，运行时直接拿结果
// ──────────────────────────────────────────────────────────────────
template<int N>
struct Factorial {
    // 递归特化: Factorial<5> → 5 * Factorial<4> → ... → Factorial<0> = 1
    static constexpr int value = N * Factorial<N - 1>::value;
};

// 终止条件（全特化）
template<>
struct Factorial<0> {
    static constexpr int value = 1;
};

// ──────────────────────────────────────────────────────────────────
// 3. 模板参数做"编译期分支" —— if/else的替代方案
//    用特化实现编译期的条件逻辑：
//    if (N % 2 == 0) → 用偶数版本
//    else            → 用奇数版本
// ──────────────────────────────────────────────────────────────────

template<int N>
struct Parity {
    // 奇数的默认实现（递归减1到偶数）
    static const char* name() { return "odd"; }
};

template<>
struct Parity<0> {
    static const char* name() { return "even (base case)"; }
};

// ──────────────────────────────────────────────────────────────────
// 4. 指针/引用作为模板参数 —— 更强大的编译期常量
//    C++允许用constexpr变量的地址做模板参数！
// ──────────────────────────────────────────────────────────────────
template<const int* P>     // P是一个指向const int的指针
struct PointerWrapper {
    static const int& get() { return *P; }   // 解引用得到值
};

const int global_x = 42;       // constexpr变量的地址可以当模板参数
const int global_y = 99;

int main() {
    // ======================== 实验1: FixedArray =========================
    
    cout << "=== 编译期固定大小数组 ===" << endl;
    FixedArray<int, 5> arr;       // N=5，在编译期确定！
    for (int i = 0; i < 5; ++i) {
        arr[i] = i * 10;
        cout << "arr[" << i << "] = " << arr[i] << endl;
    }
    
    // N是编译期常量，所以sizeof(arr) / sizeof(int) == 5
    cout << "FixedArray<int,5>占" << sizeof(arr) << "字节 (5个int)\n";
    
    cout << "\n";
    
    // ======================== 实验2: Factorial — 编译期计算 ============
    // 关键：模板参数必须是编译期常量！不能用for循环的运行时变量n。
    // 正确方式：显式实例化每个值（或C++17用index_sequence展开）
    
    cout << "=== 编译期阶乘计算 ===" << endl;
    if      (true) { constexpr int i=0; cout << i << "! = " << Factorial<0>::value << endl; }
    if      (true) { constexpr int i=1; cout << i << "! = " << Factorial<1>::value << endl; }
    if      (true) { constexpr int i=2; cout << i << "! = " << Factorial<2>::value << endl; }
    if      (true) { constexpr int i=3; cout << i << "! = " << Factorial<3>::value << endl; }
    if      (true) { constexpr int i=4; cout << i << "! = " << Factorial<4>::value << endl; }
    if      (true) { constexpr int i=5; cout << i << "! = " << Factorial<5>::value << endl; }
    if      (true) { constexpr int i=6; cout << i << "! = " << Factorial<6>::value << endl; }
    if      (true) { constexpr int i=7; cout << i << "! = " << Factorial<7>::value << endl; }
    if      (true) { constexpr int i=8; cout << i << "! = " << Factorial<8>::value << endl; }
    if      (true) { constexpr int i=9; cout << i << "! = " << Factorial<9>::value << endl; }
    if      (true) { constexpr int i=10; cout << i << "! = " << Factorial<10>::value << endl; }
    
    // 对比：如果用运行时递归算，每次都要调用函数。模板版本在编译时就完成了。
    
    cout << "\n";
    
    // ======================== 实验3: Parity — 编译期分支 ================
    // 同样，模板参数必须是编译期常量
    
    cout << "=== 编译期奇偶判断 ===" << endl;
    for (int i = 0; i <= 6; ++i) {
        if      (i == 0) cout << 0 << " is " << Parity<0>::name() << endl;
        else if (i == 1) cout << 1 << " is " << Parity<1>::name() << endl;
        else if (i == 2) cout << 2 << " is " << Parity<2>::name() << endl;
        else if (i == 3) cout << 3 << " is " << Parity<3>::name() << endl;
        else if (i == 4) cout << 4 << " is " << Parity<4>::name() << endl;
        else if (i == 5) cout << 5 << " is " << Parity<5>::name() << endl;
        else             cout << 6 << " is " << Parity<6>::name() << endl;
    }

    // 这个分支在编译期就确定了！不是运行时if-else。
    
    cout << "\n";
    
    // ======================== 实验4: 指针作为模板参数 ==================
    
    cout << "=== 指针/引用做模板参数 ===" << endl;
    PointerWrapper<&global_x>::get();   // 返回 *P = global_x = 42
    cout << "*&global_x = " << PointerWrapper<&global_x>::get() << endl;
    cout << "*&global_y = " << PointerWrapper<&global_y>::get() << endl;
    
    // ======================== 实验5: 非类型模板参数的限制 ==============
    
    cout << "\n=== 限制 ===" << endl;
    cout << "✅ int, long, enum, pointer, reference\n";
    cout << "❌ double/float (C++20前不允许)\n";
    cout << "❌ string对象（但string字面量指针可以）\n";
    cout << "❌ 运行时变量\n";
    
    // 试试这个会报错：template<double PI> ❌ (C++17)
    // template<> struct Circle<3.14> { ... };  // C++20才允许浮点模板参数
    
    cout << "\n";
    cout << "🎯 本章要点:\n";
    cout << "1. template<int N> 让编译期常量成为模板的一部分\n";
    cout << "2. int arr[N] → 数组大小在编译期确定，零运行时开销\n";
    cout << "3. Factorial<N>::value → 递归+特化 = 编译期数学计算\n";
    cout << "4. 指针/引用也可以做模板参数（constexpr变量的地址）\n";
    
    return 0;
}
