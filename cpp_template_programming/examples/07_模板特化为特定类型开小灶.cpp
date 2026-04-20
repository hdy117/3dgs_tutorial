/* ================================================================
 *  Chapter 07 — 模板特化：为特定类型开小灶
 * ================================================================
 *
 * 🧠 第一性原理：
 *   模板的逻辑是"通用的"，但某些类型需要特殊处理。
 *   
 *   类比函数重载：
 *     void f(int)    → int版本
 *     void f(double) → double版本
 *   
 *   特化就是"类/函数的重载"：
 *     template<typename T> struct S { ... }  ← 通用版
 *     template<>       struct S<int>   { ... } ← int的全特化版
 *
 * 💡 Mental Model:
 *   模板特化 = "如果T是int，用这段代码；否则用通用那段"。
 *   
 *   编译器选择特化的规则：
 *   1. 如果有完全匹配的特化 → 用它（最精确）
 *   2. 如果没有 → 用通用版本
 *   
 *   这不是if-else，这是编译期的"代码路径选择"！
 *
 * ================================================================ */

#include <iostream>
#include <string>
using namespace std;

// ──────────────────────────────────────────────────────────────────
// 1. 函数模板特化 —— 为int提供专门实现
//    ⚠️ 注意：函数模板特化的签名必须匹配（参数和返回类型都要一致）
//    所以要"不同返回值"，我们用非模板重载代替。
// ──────────────────────────────────────────────────────────────────

namespace mytmpl {

template<typename T>
T describe(T value) {
    // 通用版：简单返回值本身
    return value;
}

// int的非模板重载（不是特化！）—— 因为需要返回string而非int
// 函数调用优先级: 非模板 > 模板特化 > 通用模板
string describe(int value) {
    if (value == 0) return "零";
    if (value > 0)  return string(value, '*');
    return "-";
}

} // namespace mytmpl

// ──────────────────────────────────────────────────────────────────
// 2. 类模板的全特化 —— 为特定类型提供完全不同的实现
//    特化版的成员可以完全不同于通用版！
// ──────────────────────────────────────────────────────────────────
template<typename T>
class Processor {
    T data;
public:
    explicit Processor(T d) : data(d) {}
    
    void process() const {
        cout << "通用处理: 处理" << typeid(T).name() << "类型, 值=" << data << endl;
    }
};

// int的特化 —— 完全不同的内部实现
template<>
class Processor<int> {
    int count;
public:
    explicit Processor(int d) : count(d) {}
    
    void process() const {
        cout << "int特化版: 计数" << count << "次，每次打星号";
        for (int i = 0; i < count && i < 10; ++i) cout << "*";
        if (count > 10) cout << "...(共" << count << ")";
        cout << endl;
    }
};

// char的特化
template<>
class Processor<char> {
    char c;
public:
    explicit Processor(char d) : c(d) {}
    
    void process() const {
        if (c >= 'a' && c <= 'z')
            cout << "char特化版: 小写字母 '" << c << "'\n";
        else if (c >= 'A' && c <= 'Z')
            cout << "char特化版: 大写字母 '" << c << "'\n";
        else
            cout << "char特化版: 其他字符 ASCII=" << (int)c << "\n";
    }
};

// ──────────────────────────────────────────────────────────────────
// 3. 函数模板重载 vs 特化 —— 最容易混淆的点！
//    也放在namespace里避免和std冲突
// ──────────────────────────────────────────────────────────────────

template<typename T>
T describe2(T value) {
    return value;
}

string describe2(int value) {   // 非模板重载（最高优先级）
    if (value == 0) return "零";
    if (value > 0)  return string(value, '*');
    return "-";
}

// ──────────────────────────────────────────────────────────────────
// hello: 展示普通重载 vs 模板特化 vs 通用模板的优先级
// ──────────────────────────────────────────────────────────────────

void greet(const string& s) {           // 普通函数（最优先）
    cout << "你好, " << s << "!" << endl;
}

template<typename T>
void greet(T t) {                       // 通用模板（最后优先级）
    cout << "Hello, [" << t << "]!" << endl;
}

template<>
void greet<int>(int i) {                // int的特化（中间优先级）
    cout << "Number: " << i << endl;
}

// ──────────────────────────────────────────────────────────────────
// 实验用辅助函数：展示describe的返回类型差异
// ──────────────────────────────────────────────────────────────────
void show_describe_result() {
    // describe是mytmpl::namespace里的，需要指定命名空间
    cout << "mytmpl::describe(42)     = " << mytmpl::describe(42) << endl;       // "***..." (string)
    cout << "mytmpl::describe(3.14)   = " << mytmpl::describe(3.14) << endl;     // 3.14 (double, same as input)
    cout << "\n";
    
    // describe2展示另一个模式：返回类型始终T，但int有专门重载
    cout << "describe2<int>(42)      = " << describe2(42) << endl;               // 0 (int版本返回string→截断为0?) 
    // 实际上：describe2(int)返回string, print到cout会输出字符串内容
}


int main() {
    // ======================== 实验1: 函数模板特化 =====================
    
    cout << "=== 函数模板全特化 ===" << endl;
    show_describe_result();
    
    cout << "\n";
    
    // ======================== 实验2: 类模板全特化 =====================
    // 编译器选择规则：最精确匹配优先
    
    cout << "=== 类模板全特化 ===" << endl;
    Processor<int> p_int(5);     // ✅ 找到 int的特化，用Processor<int>版本
    p_int.process();             // int特化版: 计数5次...
    
    Processor<double> p_double(3.14);  // ❌ 没有double特化，用通用版
    p_double.process();              // 通用处理: 处理double类型
    
    Processor<char> p_char('A');     // ✅ 找到 char的特化
    p_char.process();                // char特化版: 大写字母 'A'
    
    cout << "\n";
    
    // ======================== 实验3: 重载 vs 特化的选择规则 ===========
    // 这是C++模板中最复杂的部分之一。规则如下（从最精确到最不精确）：
    // 1. 普通函数（非模板）最优先
    // 2. 模板特化次优先  
    // 3. 通用模板最后
    
    cout << "=== 重载 vs 特化的调用优先级 ===" << endl;
    
    greet(string("World"));   // ✅ 匹配普通函数重载 greet(const string&) → "你好, World!"
    greet(42);                 // ✅ 匹配模板特化 greet<int>(int) → "Number: 42"
    greet(3.14);               // ✅ 匹配通用模板 greet<double> → "Hello, [3.14]!"
    greet('x');                // ✅ 匹配通用模板 greet<char> → "Hello, [x]!"

    
    cout << "\n";
    cout << "🎯 本章要点:\n";
    cout << "1. template<> struct S<int> {...} = int的全特化\n";
    cout << "2. 编译器选最精确的匹配：特化 > 通用模板\n";
    cout << "3. 类模板特化的成员可以完全不同（不仅仅是修改一行）\n";
    cout << "4. 函数调用优先级: 普通重载 > 特化 > 通用模板\n";
    
    return 0;
}
