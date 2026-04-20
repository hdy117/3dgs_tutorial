/* ================================================================
 *  Chapter 02 — 函数模板：一个原型对付所有类型
 * ================================================================
 *
 * 🧠 推导：
 *   你已经知道模板是"代码替换器"了。现在看更实际的例子——
 *   max() 函数：无论比较 int、double 还是自定义类型，逻辑完全一样。
 *
 * 💡 关键概念：模板参数推断 (Template Argument Deduction)
 *   编译器会根据你传入的参数自动确定T是什么。这叫做"推导"。
 *   但推导有陷阱！看实验3。
 *
 * ================================================================ */

#include <iostream>
#include <string>
using namespace std;

// ─── 用命名空间避免和std::max冲突 ────────────────────────────────
namespace mytmpl {

// ──────────────────────────────────────────────────────────────────
// 1. 基础版：取两个值中的较大者
//    T是占位符，调用时编译器自动推断为具体类型
// ──────────────────────────────────────────────────────────────────
template<typename T>
T mymax(T a, T b) {
    return (a > b) ? a : b;
}

// ──────────────────────────────────────────────────────────────────
// 2. 带自定义比较器的版本 —— 展示模板的高阶用法
// ──────────────────────────────────────────────────────────────────
template<typename T, typename Comp>
T mymax_with(T a, T b, Comp compare) {
    return compare(a, b) ? a : b;
}

// 按字符串长度比较大小（作为Comp使用）
bool by_length(const string& a, const string& b) {
    return a.size() > b.size();
}

} // namespace mytmpl

int main() {
    using namespace mytmpl;  // 使用我们自己的模板函数
    
    // ======================== 实验1: 基本类型自动推断 ===============
    
    cout << "=== 类型自动推断 ===" << endl;
    cout << "mymax(3, 7)       = " << mymax(3, 7)          << endl;   // T=int → int版本
    cout << "mymax(3.0, 7.1)   = " << mymax(3.0, 7.1)      << endl;   // T=double → double版本
    
    cout << "\n";
    
    // ======================== 实验2: string也能工作 ====================
    // string有operator>，所以模板可以直接用！无需额外代码。
    
    cout << "=== string比较 ===" << endl;
    cout << "mymax(\"hi\", \"hello\") = " 
         << mymax(string("hi"), string("hello")) << endl;   // T=string → string版本
    
    cout << "\n";
    
    // ======================== 实验3: 推导陷阱 ⚠️ =====================
    // mymax(3, 7.0) → ❌ ERROR!
    // a是int，b是double。T只能是一个类型，编译器无法决定选哪个。
    
    cout << "=== 类型推断陷阱 ===" << endl;
    // mymax(3, 7.0);   // ← 这行会编译错误！
    cout << "mymax(3, 7.0) → ERROR: T不确定是int还是double\n";
    
    // 解决方案1：手动指定类型
    cout << "手动指定 mymax<double>(3, 7.0): ";
    cout << mymax<double>(3, 7.0) << endl;   // T=double，3被转为3.0
    
    // ======================== 实验4: 自定义比较器 ====================
    
    cout << "\n=== 自定义比较规则 ===" << endl;
    cout << "最长字符串: " 
         << mymax_with(string("hi"), string("hello"), by_length) << endl;   // hello
    
    // Lambda也可以作为Comp！这展示了模板与高阶函数的结合。
    auto by_abs = [](int a, int b) { return abs(a) > abs(b); };
    cout << "绝对值最大: " 
         << mymax_with(-5, 3, by_abs) << endl;   // -5 (绝对值更大)
    
    // ======================== 实验5: 推导失败的常见原因 =============
    
    cout << "\n=== 推导失败场景 ===" << endl;
    cout << "1. 参数类型不匹配: mymax(1, 2.0)\n";
    cout << "   → 编译器不知道该把T定为int还是double\n";
    cout << "2. const/引用差异导致两个不同的T候选\n";
    cout << "3. 模板形参不在推导位置（如返回类型是T但参数没有T）\n";
    
    // ======================== 实验6: 理解"代码生成"的本质 ===========
    
    cout << "\n=== 核心洞察 ===" << endl;
    cout << "每次调用不同T，都是实例化一个新函数！\n";
    cout << "mymax<int> 和 mymax<double> 是两个独立的编译产物。\n";
    cout << "它们共享同一段源代码，但生成的汇编代码完全不同。\n";
    cout << "这就是「零开销抽象」的含义 —— 没有运行时虚函数调用成本。\n";
    
    // ======================== 实验7: 模板不是万能胶水 ===============
    
    cout << "\n=== 模板的局限性 ===" << endl;
    struct Foo { int x; };
    // mymax(Foo{1}, Foo{2});  ← ❌ ERROR! Foo没有operator>
    cout << "Foo没有operator> → mymax(Foo{},Foo{})编译报错\n";

    
    // 这说明：模板不保证"类型兼容"，只保证"语法替换"。
    // T=int时，函数体内所有代码必须对int合法。
    
    cout << "\n";
    cout << "🎯 本章要点:\n";
    cout << "1. 编译器根据函数参数的类型推导T\n";
    cout << "2. 所有参数必须指向同一个T → mymax(3, 7.0)会报错\n";
    cout << "3. 手动指定: mymax<double>(3, 7.0)\n";
    cout << "4. 多模板参数时，每个参数独立推导\n";
    
    return 0;
}
