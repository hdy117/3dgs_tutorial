/* ================================================================
 *  Chapter 08 — 部分特化：精确匹配的艺术
 * ================================================================
 *
 * 🧠 推导：
 *   全特化：template<> struct S<int> { } —— 所有参数都指定了
 *   部分特化：template<typename U> struct S<int*, U> { } 
 *              —— 第一个参数固定为int*，第二个仍保持泛型
 *   
 *   ⚠️ 重要限制：函数模板不支持部分特化！只有类模板可以。
 *   （但可以用重载+SFINAE模拟）
 *
 * 💡 Mental Model:
 *   部分特化 = "如果第一个参数是int*，不管第二个是什么类型都用这个版本"
 *   
 *   它比全特化更灵活：一个部分特化可以覆盖无限多种具体类型组合。
 *
 * ================================================================ */

#include <iostream>
#include <string>
using namespace std;

// ─────────────────相同类型检测 =============================
// 用部分特化实现"两个类型是否相同"的编译期判断
// ──────────────────────────────────────────────────────────────────

// 通用版：T和U不同
template<typename T, typename U>
struct IsSame {
    static constexpr bool value = false;
};

// 全特化：T和U完全相同时（T=U的特化）
template<typename T>
struct IsSame<T, T> {        // ← 这就是部分特化！两个参数都绑定到同一个T
    static constexpr bool value = true;
};

// ──────────────────────────────────────────────────────────────────
// 2. 指针类型的全特化和部分特化
// ──────────────────────────────────────────────────────────────────

template<typename T>
struct TypeTag {             // 通用版：普通类型
    static const char* name() { return "普通类型"; }
};

template<typename T>
struct TypeTag<T*> {         // ← 部分特化！T是指针类型时匹配
    static const char* name() { return "指针"; }
};

// int*的特例（全特化，优先级更高）
template<>
struct TypeTag<int*> {       // ← 全特化：最精确匹配
    static const char* name() { return "int* (最精确)"; }
};

// ──────────────────────────────────────────────────────────────────
// 3. 引用类型的部分特化 —— std::reference_wrapper的基础
// ──────────────────────────────────────────────────────────────────

template<typename T>
struct IsReference {         // 通用版
    static constexpr bool value = false;
};

template<typename T>
struct IsReference<T&> {     // ← 部分特化：T是引用时匹配
    static constexpr bool value = true;
};

template<typename T>
struct IsReference<const T&> {   // ← 另一个部分特化：const引用
    static constexpr bool value = true;
};

// ──────────────────────────────────────────────────────────────────
// 4. 排序逻辑的部分特化 —— 根据类型选择比较策略
// ──────────────────────────────────────────────────────────────────
template<typename T, typename Enable = void>   // Enable是"开关"，默认为void
struct SortStrategy {
    static string strategy() { return "通用冒泡排序"; }
};

// 特化版：当T支持operator<时使用快排（用std::true_type做开关）
template<typename T>
struct SortStrategy<T, typename enable_if<is_integral<T>::value>::type> {
    // ⚠️ 注意：enable_if的用法在第09章详细讲解，这里先用简单的模式
    static string strategy() { return "整数快速排序"; }
};

// 用更简单的方式展示部分特化：
template<typename T>
struct SortStrategy<T*> {    // ← 指针类型的部分特化
    static string strategy() { return "指针数组排序（比较指向的值）"; }
};

int main() {
    // ======================== 实验1: IsSame — 最简单的部分特化 ======
    
    cout << "=== IsSame<T, T> ===" << endl;
    cout << "IsSame<int, int>::value   = " << IsSame<int, int>::value      << endl;  // true
    cout << "IsSame<int, double>::value = " << IsSame<int, double>::value   << endl;  // false
    cout << "IsSame<string, string>::value = " << IsSame<string, string>::value << endl; // true
    
    // 核心机制：IsSame<T, T>中两个参数都是T。
    // 当调用 IsSame<int, int>时，编译器看到T=int且第二个也是int=T → 匹配部分特化！
    // 当调用 IsSame<int, double>时，U≠T → 用通用版
    
    cout << "\n";
    
    // ======================== 实验2: TypeTag — 指针检测 ===============
    
    cout << "=== TypeTag — 类型标签 ===" << endl;
    cout << "int:     " << TypeTag<int>::name()          << endl;  // 普通类型
    cout << "int*:    " << TypeTag<int*>::name()         << endl;  // int* (最精确) ← 全特化优先！
    cout << "double:* " << TypeTag<double*>::name()       << endl;  // 指针
    
    // 优先级: 全特化(TypeTag<int*>）> 部分特化(TypeTag<T*>) > 通用版(TypeTag<T>)
    
    cout << "\n";
    
    // ======================== 实验3: IsReference — 引用检测 ===========
    
    cout << "=== IsReference — 引用检测 ===" << endl;
    cout << "int:     " << IsReference<int>::value      << endl;   // false (不是引用)
    cout << "int&:    " << IsReference<int&>::value     << endl;   // true  (引用)
    cout << "const int&: " << IsReference<const int&>::value << endl; // true (const引用)
    
    cout << "\n";
    
    // ======================== 实验4: SortStrategy — 策略选择 ===========
    
    cout << "=== SortStrategy — 排序策略 ===" << endl;
    cout << "int:     " << SortStrategy<int>::strategy()        << endl;  // 通用冒泡(无enable_if特化)
    cout << "int*:   " << SortStrategy<int*>::strategy()       << endl;  // 指针数组排序
    
    // ======================== 实验5: 理解"部分"的含义 ===============
    
    cout << "\n=== 为什么叫'部分'特化 ===" << endl;
    cout << "通用版: template<typename T, typename U> struct IsSame { ... }\n";
    cout << "   → 两个参数都泛型\n";
    cout << "部分特化: template<typename T> struct IsSame<T, T> { ... }\n";
    cout << "   → 第一个参数和第二个参数绑定到同一个T（约束了一个关系）\n";
    cout << "全特化: template<> struct S<int> { ... }\n";
    cout << "   → 所有参数都指定了具体类型\n";
    
    cout << "\n";
    cout << "🎯 本章要点:\n";
    cout << "1. 部分特化 = 约束一部分模板参数的关系（如T=T, T*, T&）\n";
    cout << "2. 只有类模板支持部分特化！函数模板不支持。\n";
    cout << "3. 优先级: 全特化 > 部分特化 > 通用版\n";
    cout << "4. IsSame<T,T>是标准库is_same的基础实现\n";
    
    return 0;
}
