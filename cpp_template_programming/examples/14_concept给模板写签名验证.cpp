/* ================================================================
 *  Chapter 14 — Concepts: 给模板写"签名验证"
 * ================================================================
 *
 * 🧠 推导：
 *   template<typename T> void sort(vector<T>&) 
 *   
 *   问题：T必须满足什么？可比较？有iterator？ assignable?
 *   SFINAE能检测但太复杂，requires子句好一些但不够结构化。
 *   
 *   C++20 Concepts = "命名化的约束"
 *     concept Sortable = requires(T t) { sort(t); };  ← 给约束起名！
 *     
 *   template<Sortable T> void sort(vector<T>& v) ← 清晰、可读、错误信息好。

 * 💡 Mental Model：
 *   Concept = 类型的"接口契约"。类似Java的Interface但工作在编译期。

 * ================================================================ */

#include <iostream>
#include <string>
#include <vector>
#include <array>
using namespace std;

// ──────────────────────────────────────────────────────────────────
// 1. 基础Concept —— "这个类型支持什么操作？"
//    GCC 13对requires返回类型约束的支持有限，用简单形式。
// ──────────────────────────────────────────────────────────────────
template<typename T>
concept Printable = requires(T t) {
    cout << t;          // ← 编译器检查这段代码对类型T是否合法
};

template<typename T>
concept HasSize = requires(T t) {
    t.size();           // ← size()能调用即可，不约束返回值类型
};

// 组合concept —— 容器必须有begin/end
template<typename C>
concept Container = HasSize<C> && requires(C c) {
    c.begin();          // 有begin()
    c.end();            // 有end()
};

// ──────────────────────────────────────────────────────────────────
// 2. Concept做模板参数约束 —— 替代所有enable_if/SFINAE
// ──────────────────────────────────────────────────────────────────

template<Printable T>
void display(const T& val) {
    cout << "显示: " << val << endl;
}

template<typename T>
    requires Printable<T>       // 等价写法(上面是简写形式)
void show(const T& val) {
    cout << "展示: " << val << endl;
}

// ──────────────────────────────────────────────────────────────────
// 3. Concept做函数签名验证 —— 类似接口的概念
// ──────────────────────────────────────────────────────────────────

template<Container C>
void print_container(const C& container) {
    cout << "[";
    bool first = true;
    for (const auto& item : container) {
        if (!first) cout << ", ";
        cout << item;
        first = false;
    }
    cout << "] 共" << container.size() << "个元素\n";
}

// ──────────────────────────────────────────────────────────────────
// 4. 自定义operator<<来让struct满足Printable —— 演示concept的灵活性
// ──────────────────────────────────────────────────────────────────
struct Point {
    double x, y;
};

ostream& operator<<(ostream& os, const Point& p) {
    return os << "(" << p.x << ", " << p.y << ")";
}

// 现在Point满足Printable concept！
static_assert(Printable<Point>, "Point must be printable!");   // ← 编译期断言验证concept

struct NoStreamIO {
    int value;
};
// static_assert(Printable<NoStreamIO>, "must be printable"); → ❌ 编译错误!

// ──────────────────────────────────────────────────────────────────
// 5. 用户自定义concept —— 放在命名空间级别（不能放main函数内）
// ──────────────────────────────────────────────────────────────────
template<typename T>
concept HasToString = requires(T t) {
    to_string(t);           // ← 有to_string()即可
};

template<typename T>
concept Comparable = requires(T a, T b) {
    a < b;                  // ← 支持小于比较
};

// ──────────────────────────────────────────────────────────────────
// 6. 约束链 —— 复杂类型验证
// ──────────────────────────────────────────────────────────────────
template<Comparable T>
void sort_hint(const vector<T>& v) {
    cout << "可以排序的vector!\n";
}

int main() {
    // ======================== 实验1: Concept约束函数 ================
    
    cout << "=== Concept约束 ===" << endl;
    display(42);           // ✅ int是Printable
    display(string("hi")); // ✅ string是Printable
    display(Point{1.5, 2.5}); // ✅ Point有operator<<
    
    show(true);            // ✅ bool可以cout
    
    cout << "🔥 看下面这行会编译错误：" << endl;
    // display(NoStreamIO{99}) → ❌ NoStreamIO不满足Printable!

    // ======================== 实验2: Container Concept ===============
    
    cout << "\n=== Container Concept ===" << endl;
    vector<int> v{1, 2, 3};
    print_container(v);     // [1, 2, 3] 共3个元素
    
    vector<string> vs{"hello", "world"};
    print_container(vs);    // [hello, world] 共2个元素
    
    array<double, 4> a{0.1, 0.2, 0.3, 0.4};
    print_container(a);     // [0.1, 0.2, 0.3, 0.4] 共4个元素
    
    // string也是Container!
    print_container(string("abc"));   // [a, b, c] 共3个元素

    // ======================== 实验3: Concept组合 ==============
    
    cout << "\n=== Concept组合 ===" << endl;
    static_assert(Printable<int>, "int should be printable!");
    static_assert(Printable<string>, "string should be printable!");
    static_assert(HasSize<vector<int>>, "vector has size()");
    static_assert(Container<string>, "string is a container");

    cout << "\n=== 编译期验证 ===" << endl;
    if constexpr (Printable<Point>)  cout << "Point IS Printable ✓\n";
    else                             cout << "Point NOT printable ✗\n";
    
    if (!Printable<NoStreamIO>)       cout << "NoStreamIO NOT printable ✓\n";

    // ======================== 实验4: Concept vs SFINAE对比 ==========
    
    cout << "\n=== 错误信息对比 ===" << endl;
    cout << "SFINAE(当类型不支持时的报错):\n";
    cout << "  error: no matching function for call to 'display'\n";
    cout << "  ... (50行模板实例化堆栈) ...\n";
    cout << "\nConcept(当类型不支持时的报错):\n";
    cout << "  error: constraint not satisfied\n";
    cout << "  note: 'NoStreamIO' does not satisfy 'Printable'\n";

    // ======================== 实验5: 用户自定义concept ================
    
    cout << "\n=== HasToString概念 ===" << endl;
    if constexpr (HasToString<int>)       cout << "int has to_string ✓\n";
    else                                  cout << "int no to_string ✗\n";
    
    if constexpr (HasToString<double>)    cout << "double has to_string ✓\n";
    else                                  cout << "double no to_string ✗\n";

    // ======================== 实验6: 约束链 —— Comparable验证 ============
    
    cout << "\n=== Comparable概念 ===" << endl;
    if constexpr (Comparable<int>)       cout << "int is Comparable ✓\n";
    else                                  cout << "int NOT comparable ✗\n";
    
    vector<int> si{3, 1, 2};
    sort_hint(si);                       // ✅ int满足Comparable

    cout << "\n";
    cout << "🎯 本章要点:\n";
    cout << "1. concept = 命名化的类型约束 —— 给[T应该能做什么]取个名字\n";
    cout << "2. template<Sortable T> ← 比template<typename T> + requires更清晰\n";
    cout << "3. error message: 'X不满足Sortable' vs SFINAE的天书\n";
    cout << "4. concept可以组合: Printable && HasSize → 新的concept\n";
    cout << "5. static_assert(Printable<T>) ← 编译期验证，T必须是bool-like类型\n";
    
    return 0;
}
