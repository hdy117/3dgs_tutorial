/* ================================================================
 *  Chapter 03 — 多模板参数：一行变多行
 * ================================================================
 *
 * 🧠 推导：
 *   单个template<typename T>已经很强大，但现实世界需要更多维度。
 *   
 *   比如 pair<T, U> —— 两个不同类型的值绑在一起。
 *   template后面可以有任意多个参数，用逗号分隔。
 *
 * 💡 关键洞察：每个模板参数独立推导（除非你强制绑定）
 *
 * ================================================================ */

#include <iostream>
#include <string>
using namespace std;

// ──────────────────────────────────────────────────────────────────
// 1. 两个类型参数的简单函数
//    T和U各自独立推断
// ──────────────────────────────────────────────────────────────────
template<typename T, typename U>
auto add(T a, U b) -> decltype(a + b) {
    // decltype(a + b) 让返回类型等于a+b的实际结果类型
    return a + b;
}

// ──────────────────────────────────────────────────────────────────
// 2. 三个类型参数 —— 创建三元组
// ──────────────────────────────────────────────────────────────────
template<typename T, typename U, typename V>
struct Triplet {
    T first;
    U second;
    V third;
    
    // 打印所有成员
    void print() const {
        cout << "(" << first << ", " << second << ", " << third << ")" << endl;
    }
};

// ──────────────────────────────────────────────────────────────────
// 3. 混合推导：有些参数自动推断，有些需要手动指定
// ──────────────────────────────────────────────────────────────────
template<typename T, typename U>
void print_pair(T a, U b) {
    cout << "[" << typeid(a).name() << ", " << typeid(b).name() << "] = (";
    cout << a << ", " << b << ")" << endl;
}

int main() {
    // ======================== 实验1: 两个类型各自推断 ====================
    
    cout << "=== 两个类型参数 ===" << endl;
    cout << "add(3, 4.5)      = " << add(3, 4.5)          << endl;   // int + double = double
    cout << "add(\"hi\", 2)    = " << (string("hi") + string(2, '!'))  << endl;   // 字符串拼接演示
    
    cout << "\n";
    
    // ======================== 实验2: decltype自动推导返回类型 ===========
    // 如果没有 -> decltype(a + b)，编译器不知道返回什么类型。
    // C++14可以简化为 auto add(T a, U b) { return a + b; }  (auto的返回类型会被推导)
    
    cout << "=== 三元组 ===" << endl;
    Triplet<int, string, double> t{42, "hello", 3.14};
    t.print();   // (42, hello, 3.14)
    
    cout << "\n";
    
    // ======================== 实验3: N个类型参数的模式 ==================
    // template<typename T1, typename T2, ..., typename TN> 
    // 这个模式是std::tuple、std::pair等的基础。
    
    struct Quad { int a; char b; double c; float d; };
    Quad q{1, 'x', 2.0f, 3.0f};
    cout << "Quad: {" << q.a << ", '" << q.b << "', " << q.c << ", " << q.d << "}" << endl;
    
    // ======================== 实验4: 类型推导的边界 =====================
    
    cout << "\n=== 推导的边界 ===" << endl;
    print_pair(10, 20);         // T=int, U=int → [int, int] = (10, 20)
    print_pair(10, "hello");    // T=int, U=const char* → [int, const char*] = (10, hello)
    
    cout << "\n";
    cout << "🎯 本章要点:\n";
    cout << "1. template<typename T, typename U> 每个参数独立推断\n";
    cout << "2. add(3, 4.5): T=int从第一个参数来，U=double从第二个参数来\n";
    cout << "3. decltype(a+b)让返回类型自动匹配运算结果\n";
    cout << "4. C++14: auto函数返回值可以省略decltype（编译器推导）\n";
    
    return 0;
}
