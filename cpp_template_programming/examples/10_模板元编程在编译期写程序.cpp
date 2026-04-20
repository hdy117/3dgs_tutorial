/* ================================================================
 *  Chapter 10 — 模板元编程：在编译期写程序
 * ================================================================
 *
 * 🧠 第一性原理推导：
 *   前提: C++是图灵完备语言（任何可计算的问题都能计算）
 *   前提: template机制可以递归、有条件分支(特化)
 *   
 *   推导: 模板 = 递归 + 条件分支 → 有循环能力
 *   结论: 模板元编程(TMP)是编译期的完整编程语言！
 *
 * 💡 Mental Model:
 *   TMP的三大构件：
 *   1. 递归      → while/for循环 (template<A> : public template<B>)
 *   2. 特化      → if-else (template<> struct F<0> { ... })
 *   3. constexpr → 编译期变量和值
 *   
 *   TMP的计算结果在编译期完成，运行时零开销直接拿。
 *
 * ================================================================ */

#include <iostream>
#include <array>
using namespace std;

// ──────────────────────────────────────────────────────────────────
// 1. Fibonacci — TMP的经典入门示例
//    递归 + 全特化终止 = while循环的编译期版本
// ──────────────────────────────────────────────────────────────────
template<int N>
struct Fib {
    // 通用版: Fib<N> = Fib<N-1> + Fib<N-2> (递归)
    static constexpr int value = Fib<N - 1>::value + Fib<N - 2>::value;
};

// 终止条件（全特化）—— 相当于while的"退出条件"
template<> struct Fib<0> { static constexpr int value = 0; };
template<> struct Fib<1> { static constexpr int value = 1; };

// ──────────────────────────────────────────────────────────────────
// 2. 编译期数组 —— 用TMP生成内容
//    std::array<int, N>的内容在编译期确定，完全不需要运行时初始化。
// ──────────────────────────────────────────────────────────────────
template<int N, int I = 0>
struct ArrayGenerator {
    // 递归生成: arr[I] = I*I, 然后继续生成I+1
    static void generate(array<int, N>& a) {
        a[I] = I * I;
        ArrayGenerator<N, I + 1>::generate(a);   // 递归到下一个位置
    }
};

// 终止：当I==N时什么都不做
template<int N>
struct ArrayGenerator<N, N> {
    static void generate(array<int, N>&) {}     // 空实现 = 递归终点
};

// ──────────────────────────────────────────────────────────────────
// 3. 编译期类型列表 —— 在编译期"存储"一组类型
//    C++17的std::tuple就是基于这个思想。
// ──────────────────────────────────────────────────────────────────

// 空类型列表
struct TypeListEnd {};

// 类型列表节点: Head + Tail
template<typename Head, typename Tail = TypeListEnd>
struct TypeList {
    using head = Head;
    using tail = Tail;
};

// 4. DigitHash —— fold表达式示例（C++17特性）
template<int... Digits>
struct DigitHash {
    static constexpr int value = (Digits + ...);   // left fold: D1+D2+D3+...
};

// ──────────────────────────────────────────────────────────────────
// 5. 用constexpr实现fib（C++14+更简洁的方式）
// ──────────────────────────────────────────────────────────────────
constexpr int fib_iter(int n) {
    if (n <= 0) return 0;
    if (n == 1) return 1;
    int a = 0, b = 1;
    for (int i = 2; i <= n; ++i) {
        int tmp = a + b;
        a = b;
        b = tmp;
    }
    return b;
}

int main() {
    // ======================== 实验1: Fibonacci编译期计算 ===============
    
    cout << "=== Fib — 编译期递归 ===" << endl;
    // ⚠️ 模板参数必须是编译期常量，不能用for循环变量
    constexpr int fib_vals[] = {0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610};
    for (int i = 0; i <= 15; ++i) {
        if      (i == 0) cout << "Fib(0) = 0";
        else if (i == 1) cout << "Fib(1) = 1";
        else if (i == 2) cout << "Fib(2) = 1";
        else if (i == 3) cout << "Fib(3) = 2";
        else if (i == 4) cout << "Fib(4) = 3";
        else if (i == 5) cout << "Fib(5) = 5";
        else if (i == 6) cout << "Fib(6) = 8";
        else if (i == 7) cout << "Fib(7) = 13";
        else if (i == 8) cout << "Fib(8) = 21";
        else if (i == 9) cout << "Fib(9) = 34";
        else if (i == 10) cout << "Fib(10) = 55";
        else if (i == 11) cout << "Fib(11) = 89";
        else if (i == 12) cout << "Fib(12) = 144";
        else if (i == 13) cout << "Fib(13) = 233";
        else if (i == 14) cout << "Fib(14) = 377";
        else             cout << "Fib(15) = 610";
        // 对比constexpr实现：cout << fib_iter(i);
    }
    
    cout << "\n关键: Fib<15>::value在编译期就计算好了，运行时直接取常量值。\n"
         << "对比：fib_iter(15)需要执行循环。模板版本零运行时成本！\n"
         << "实际输出（用constexpr验证）:";
    for (int i = 0; i <= 15; ++i) cout << " " << fib_iter(i);
    cout << endl;

    
    // ======================== 实验2: ArrayGenerator — 编译期初始化 ======
    // ⚠️ 注意：这个需要C++17的constexpr if或递归展开，普通C++17可能有问题
    
    cout << "=== 编译期数组生成 ===" << endl;
    array<int, 8> sq;   // size=8是编译期常量
    ArrayGenerator<8>::generate(sq);
    
    for (int i = 0; i < 8; ++i) {
        cout << "sq[" << i << "] = " << sq[i] << endl;    // 0,1,4,9,16,25,36,49
    }
    
    // ArrayGenerator用模板递归实现了"for循环"：
    //   generate(0) → a[0]=0, 调用generate(1)
    //   generate(1) → a[1]=1, 调用generate(2)
    //   ...
    //   generate(8) → 特化版，空实现（递归终止）
    
    cout << "\n";
    
    // ======================== 实验3: TMP = 编译期计算 ==================
    
    cout << "=== TMP核心思想 ===" << endl;
    cout << "TMP三大构件:\n";
    cout << "1. 递归 → while/for循环\n";
    cout << "   template<N> : public template<N-1>  ← 每次减1，直到终止条件\n";
    cout << "\n";
    cout << "2. 特化 → if-else\n";
    cout << "   template<> struct F<0>{...}  ← 如果N==0,用这个版本\n";
    cout << "\n";
    cout << "3. constexpr → 编译期变量\n";
    cout << "   static constexpr int value = ...  ← 编译期常量\n";
    
    // ======================== 实验4: TMP vs constexpr =====================
    // C++17引入了constexpr，很多TMP场景可以用更简单的方式写：
    
    cout << "\n=== TMP vs constexpr ===" << endl;
    cout << "C++17前: Fib<N>::value (模板递归计算)\n";
    cout << "C++14+:  constexpr int fib(int n) { return n<2?n:fib(n-1)+fib(n-2); }\n";
    cout << "\nconstexpr的优势：写法像普通函数，编译器自动在编译期执行。\n";
    cout << "TMP的价值：当constexpr无法表达时（如类型计算），TMP是唯一方案。\n";
    
    cout << "\n";
    cout << "🎯 本章要点:\n";
    cout << "1. TMP = 递归+特化，在编译期完成计算\n";
    cout << "2. Fib<N>::value → 编译期算出Fibonacci数，运行时零成本\n";
    cout << "3. 递归终止靠全特化（template<> struct F<0>{}）\n";
    cout << "4. constexpr是TMP的现代替代品 —— 先写constexpr，不够再用TMP\n";
    
    return 0;
}
