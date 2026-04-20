/* ================================================================
 *  Chapter 12 — Variadic Templates: 参数包展开
 * ================================================================
 *
 * 🧠 第一性原理：
 *   template<typename T>    → 一个类型参数
 *   template<typename... Ts> → 零个或多个类型参数 (Ts是一个"包")
 *   
 *   "..."是核心语法。它出现在三个位置有不同含义:
 *     1. 声明处: typename... Ts  ← 定义一个包(收集多个)
 *     2. 使用处: Ts...           ← 展开这个包(释放多个)
 *     3. 调用处: func(args...)   ← 打包/解包参数
 *
 * 💡 Mental Model:
 *   variadic = "可以容纳任意数量参数的模板"。
 *   std::make_tuple(1, "hi", 3.0) → tuple<int, string, double>
 *   
 *   展开方式：递归（C++17前）或 fold表达式（C++17+）

 * ================================================================ */

#include <iostream>
#include <string>
using namespace std;

// ──────────────────────────────────────────────────────────────────
// 1. 基础: 接收任意数量的整数并求和 (递归展开)
//    C++17 fold表达式更简单(见第11章)，但理解递归展开很重要。
// ──────────────────────────────────────────────────────────────────

// 终止条件：没有参数时返回0
inline int sum_recursive() { return 0; }

// 递归步骤：取出第一个，加上剩余的和
template<typename T, typename... Rest>
auto sum_recursive(T first, Rest... rest) {
    return first + sum_recursive(rest...);   // ← rest... = "解开包"传给下一层
}

// ──────────────────────────────────────────────────────────────────
// 2. 接收任意类型 —— tuple的基础
// ──────────────────────────────────────────────────────────────────

template<typename... Types>
class Tuple;

// 空tuple（终止）
template<>
class Tuple<> {};

// 有头有尾的tuple节点：head + tail = rest...
template<typename Head, typename... Tail>
class Tuple<Head, Tail...> {
    Head head;
    Tuple<Tail...> tail;   // ← 递归！Tail...继续展开
    
public:
    Tuple(Head h, Tail... t) : head(h), tail(t...) {}  // 参数包转发到tail的构造函数
    
    // 打印所有成员
    void print() const {
        cout << "{" << head;
        if constexpr (!is_same_v<Tuple<Tail...>, Tuple<>>) {
            tail.print();
        }
        cout << "}";
    }
};

// ──────────────────────────────────────────────────────────────────
// 3. sizeof...(pack) —— 获取参数包的大小(编译期常量！)
// ──────────────────────────────────────────────────────────────────
template<typename... Ts>
constexpr size_t count_types() {
    return sizeof...(Ts);   // ← sizeof...是专用于参数包的运算符
}

// ──────────────────────────────────────────────────────────────────
// 4. 递归打印任意类型 —— 用sizeof...(args)做循环替代
// ──────────────────────────────────────────────────────────────────
template<typename T>
void print_one(const T& val) {
    cout << "[" << typeid(T).name() << ":" << val << "]";
}

template<typename Head, typename... Rest>
void print_all(Head head, Rest... rest) {
    print_one(head);
    if constexpr (sizeof...(rest) > 0) {   // C++17: 编译期if做递归终止判断
        cout << " ";
        print_all(rest...);   // ← 继续展开
    }
}

int main() {
    // ======================== 实验1: 递归展开求和 =====================
    
    cout << "=== sum_recursive — 递归展开 ===" << endl;
    cout << "sum(1,2)     = " << sum_recursive(1, 2)       << endl;   // 3
    cout << "sum(1,2,3)   = " << sum_recursive(1, 2, 3)    << endl;   // 6
    cout << "sum(1,2,3,4) = " << sum_recursive(1, 2, 3, 4) << endl;   // 10
    cout << "sum()        = " << sum_recursive()           << endl;   // 0 (终止条件)
    
    // 展开过程可视化：
    // sum(1,2,3) 
    // → 1 + sum(2,3)                    ← 取出head=1, rest=(2,3)
    // → 1 + (2 + sum(3))                ← 取出head=2, rest=(3)
    // → 1 + (2 + (3 + sum()))           ← 取出head=3, rest=()
    // → 1 + (2 + (3 + 0))               ← sum() = 0 (终止条件)
    // → 6
    
    cout << "\n";
    
    // ======================== 实验2: Tuple —— 自定义tuple实现 ==========
    
    cout << "=== 自定义Tuple ===" << endl;
    Tuple<int, string, double> t(42, string("hello"), 3.14);
    t.print();   // {42{hello}{3.14}}
    
    // 展开过程:
    // Tuple<int, string, double>(42, "hello", 3.14)
    // → head=int=42, tail=Tuple<string, double>("hello", 3.14)
    //   → head=string="hello", tail=Tuple<double>(3.14)
    //     → head=double=3.14, tail=Tuple<> (空tuple)
    
    cout << "\n";
    
    // ======================== 实验3: sizeof...(pack) ===================
    
    cout << "=== sizeof...(pack) ===" << endl;
    cout << "count_types<int>()     = " << count_types<int>()       << endl;   // 1
    cout << "count_types<int,double>() = " << count_types<int, double>() << endl; // 2
    cout << "count_types<>()        = " << count_types<>()          << endl;   // 0
    
    // sizeof...和sizeof一样是编译期运算符，结果是constexpr。
    
    cout << "\n";
    
    // ======================== 实验4: 递归打印 + if constexpr终止 ======
    
    cout << "=== 泛型打印 ===" << endl;
    print_all(1, string("hello"), 3.14, 'c');
    cout << endl;
    // [int:1] [St5basic_stringIc...:hello] [double:3.14] [char:c]
    
    cout << "\n";
    
    // ======================== 实验5: pack展开的三种方式 ===============
    
    cout << "=== 参数包展开方式 ===" << endl;
    cout << "1. 递归分解 (C++98-): head + tail...\n";
    cout << "   template<T, Rest...> f(T h, Rest... r) { ... f(r...) }\n";
    cout << "\n2. Fold表达式 (C++17+): (args op ...)\n";
    cout << "   return (args + ...);  ← 一行搞定\n";
    cout << "\n3. Initializer list (C++11+):\n";
    cout << "   int dummy[]{(f(args), 0)...};  ← 每个参数调用f()\n";
    
    // initializer_list方式: 
    auto print_init = [](auto&&... args) {
        using expand = int[];   // 创建一个int数组(内容不重要)
        (void)expand{0, (cout << args << " ", 0)...};   // ← 展开！
        cout << endl;
    };
    
    print_init("Hello", "from", "initializer_list!");
    
    cout << "\n";
    cout << "🎯 本章要点:\n";
    cout << "1. typename... Ts = 零个或多个类型, sizeof...(Ts) = 数量\n";
    cout << "2. 展开方式: 递归分解(C++11-), fold(C++17+), initializer_list(C++11+)\n";
    cout << "3. Tuple<Ts...>的递归实现展示了[包如何在类中传递]\n";
    cout << "4. if constexpr + sizeof...(rest) > 0 = 优雅的递归终止\n";
    
    return 0;
}
