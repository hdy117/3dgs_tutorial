/* ================================================================
 *  Chapter 04 — 默认模板参数：让类型可选
 * ================================================================
 *
 * 🧠 推导：
 *   函数可以有默认参数：void f(int x = 10)
 *   模板也能有默认参数：template<typename T = int>
 *   
 *   这是C++向"易用性"迈出的重要一步。
 *   pair<T, U> 中 U的默认值是allocator<T>。
 *
 * 💡 Mental Model:
 *   template<typename T = int, typename Allocator = vector<int>::allocator_type>
 *   class MyContainer { ... };
 *   
 *   使用方式：
 *     MyContainer<>       → T=int, Allocator=default
 *     MyContainer<double> → T=double, Allocator=default
 *     MyContainer<..., MyAlloc> → both specified
 *
 * ================================================================ */

#include <iostream>
#include <vector>
#include <string>
using namespace std;

// ──────────────────────────────────────────────────────────────────
// 1. 最简单的默认模板参数 —— 像默认函数参数一样自然
// ──────────────────────────────────────────────────────────────────
template<typename T = int>
struct Wrapper {
    T value;
    
    Wrapper(T v) : value(v) {}
    
    void show() const {
        cout << "Wrapper<" << typeid(T).name() << ">(" << value << ")" << endl;
    }
};

// ──────────────────────────────────────────────────────────────────
// 2. 多个默认参数 —— 从左到右可以省略
//    （注意：不能跳过中间的！template<A, , C> ❌）
// ──────────────────────────────────────────────────────────────────
template<typename T = int, typename U = double, typename V = string>
struct Trio {
    T t;
    U u;
    V v;
    
    Trio(T tt, U uu, V vv) : t(tt), u(uu), v(vv) {}  // 显式构造函数解决aggregate初始化歧义
    
    void print() const {
        cout << "Trio: (" << t << ", " << u << ", \"" << v << "\")" << endl;
    }
};

// ──────────────────────────────────────────────────────────────────
// 3. 非类型默认参数 —— 编译期常量也可以有默认值
// ──────────────────────────────────────────────────────────────────
template<int SIZE = 10>
struct FixedBuffer {
    int data[SIZE];       // C++允许用模板参数定义数组大小！
    
    void init() {
        for (int i = 0; i < SIZE; ++i) data[i] = i * i;
    }
    
    void show() const {
        cout << "Buffer<" << SIZE << ">:" << endl;
        for (int i = 0; i < SIZE && i < 5; ++i) {
            cout << "  [" << i << "]=" << data[i] << " ";
        }
        if (SIZE > 5) cout << "...";
        cout << endl;
    }
};

int main() {
    // ======================== 实验1: 完全使用默认值 =====================
    
    cout << "=== 默认模板参数 ===" << endl;
    Wrapper<> w1(42);           // T=int (默认)
    w1.show();                  // Wrapper<int>(42)
    
    Wrapper<double> w2(3.14);   // T=double (显式指定)
    w2.show();                  // Wrapper<double>(3.14)
    
    cout << "\n";
    
    // ======================== 实验2: Trio的默认参数组合 ==============
    
    Trio<> t1{42, 3.14, string("hello")};     // T=int, U=double, V=string (所有默认)
    t1.print();
    
    cout << "\n";
    
    // ======================== 实验3: 非类型默认参数 =====================
    // FixedBuffer<10> vs FixedBuffer<> → 两者等价！
    
    FixedBuffer<> buf1;         // SIZE=10 (默认)
    buf1.init();
    buf1.show();                // Buffer<10>: [0]=0 [1]=1 [2]=4 ...
    
    FixedBuffer<5> buf2;        // SIZE=5 (显式指定)
    buf2.init();
    buf2.show();                // Buffer<5>: [0]=0 [1]=1 [2]=4 [3]=9 [4]=16
    
    cout << "\n";
    
    // ======================== 实验4: 实际应用场景 =======================
    // std::vector<T, Allocator = allocator<T>> 就是这个模式的经典例子。
    // 99%的情况你只写 vector<int>，allocator自动选默认的。
    
    cout << "🎯 本章要点:\n";
    cout << "1. template<typename T = int> → 不指定T时默认为int\n";
    cout << "2. 多个默认参数从左到右省略（不能跳过中间的）\n";
    cout << "3. 非类型参数也能有默认值: template<int N = 10>\n";
    cout << "4. std::vector<T, Allocator>就是最常见的例子\n";
    
    return 0;
}
