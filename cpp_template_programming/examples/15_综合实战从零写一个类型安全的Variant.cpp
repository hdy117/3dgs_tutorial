/* ================================================================
 *  Chapter 15 — 综合实战：从零实现类型安全的Variant
 * ================================================================
 *
 * 🧠 回顾全书知识点：
 *   - Ch01-04: 模板基础（函数/类模板、多参数、默认参数）
 *   - Ch05-08: 高级机制（特化、部分特化）
 *   - Ch09-12: SFINAE、元编程、fold表达式、variadic
 *   - Ch13-14: if constexpr、concepts、requires (C++20)
 *   
 *   目标：实现一个简化版的std::variant<Ts...> —— 可以持有Ts中任意一种类型。

 * 💡 variant<int, string, double> v;
 *    v = 42;        → 内部存储为int
 *    v = "hello";   → 重新存储为string  
 *    get<int>(v)   → 取出int值
 *    get<string>(v)→ 如果当前不是string → runtime_error!

 * ================================================================ */

#include <iostream>
#include <string>
#include <type_traits>
#include <stdexcept>
#include <new>            // placement new
using namespace std;

// ──────────────────────────────────────────────────────────────────
// Step 1: Contains —— T是否在类型包Ts...中？(C++17 if constexpr实现)
// ──────────────────────────────────────────────────────────────────
template<typename T, typename... Ts>
struct Contains : false_type {};

template<typename T, typename First, typename... Rest>
struct Contains<T, First, Rest...> 
    : conditional_t<is_same_v<T, First>, true_type, Contains<T, Rest...>> {};

// Concept形式：T是Ts之一吗？
template<typename T, typename... Ts>
concept InList = Contains<T, Ts...>::value;

// ──────────────────────────────────────────────────────────────────
// Step 2: IndexOf —— 查找类型在列表中的位置（编译期索引）
//   variant< int, string, double >
//     IndexOf<int>     → 0
//     IndexOf<string>  → 1  
//     IndexOf<double>  → 2
// ──────────────────────────────────────────────────────────────────

template<typename T, typename... Ts>
struct IndexOf;

template<typename First, typename... Rest>
struct IndexOf<First, First, Rest...> {
    static constexpr size_t value = 0;
};

template<typename T, typename First, typename... Rest>
struct IndexOf<T, First, Rest...> {
    static constexpr size_t value = 1 + IndexOf<T, Rest...>::value;
};

// ──────────────────────────────────────────────────────────────────
// Step 3: Variant核心 —— 用unsigned char[]存储任意类型
//   variant< int, string > 内部是一个缓冲区+一个type_index
// ──────────────────────────────────────────────────────────────────
class bad_variant_access : public runtime_error {
public:
    explicit bad_variant_access(const string& msg) : runtime_error(msg) {}
};

template<typename... Ts>
class MyVariant {
    unsigned char storage_[1024];     // 足够大的缓冲区(简化版：固定1024字节)
    size_t type_id_;                  // 当前存储的类型索引
    
    // helper: 用string构造错误消息(避免ostringstream在模板内的问题)
    string make_error(const string& expected_type) const {
        return "type mismatch: expected " + expected_type;
    }

public:
    MyVariant() : type_id_(-1) {}
    
    // ─────--- 构造函数 & Assignment ---------------------------
    
    template<typename T, 
             typename = enable_if_t<InList<T, Ts...>>>   // ← Ch09 SFINAE约束
    explicit MyVariant(T value) : type_id_(IndexOf<T, Ts...>::value) {
        // placement new: 在storage_上构造T类型对象
        new (storage_) T(std::move(value));
    }

    // ─────--- get<T> —— 取出指定类型的值 ----------------------
    
    template<typename T>
    T& get() & {
        static_assert(InList<T, Ts...>, "T不在variant的类型列表中!");   // ← Ch14 concept验证
        
        if (type_id_ != IndexOf<T, Ts...>::value) {
            throw bad_variant_access(make_error(typeid(T).name()));
        }
        
        return *reinterpret_cast<T*>(storage_);
    }

    template<typename T>
    const T& get() const & {
        static_assert(InList<T, Ts...>, "T不在variant的类型列表中!");
        
        if (type_id_ != IndexOf<T, Ts...>::value) {
            throw bad_variant_access(make_error(typeid(T).name()));
        }
        
        return *reinterpret_cast<const T*>(storage_);
    }

    size_t index() const { return type_id_; }
    
    // ─────--- 赋值运算符重载 —— 改变variant持有的类型 -------------
    
    template<typename T, 
             typename = enable_if_t<InList<T, Ts...>>>
    MyVariant& operator=(T value) {
        type_id_ = IndexOf<T, Ts...>::value;
        new (storage_) T(std::move(value));
        return *this;
    }
};

// ──────────────────────────────────────────────────────────────────
// Step 4: helper函数 —— make_variant (类似std::make_pair)
//   自动推导variant的类型列表，不需要手动写。
// ──────────────────────────────────────────────────────────────────

template<typename T>
MyVariant<T> make_variant(T&& value) {
    return MyVariant<remove_cvref_t<T>>(forward<T>(value));
}

int main() {
    // ======================== 实验1: 基本使用 =========================
    
    cout << "=== Variant基础 ===" << endl;
    
    MyVariant<int, string, double> v(42);       // 当前持有int=42
    cout << "v = 42 (int): index=" << v.index() 
         << ", get<int>()=" << v.get<int>()     << endl;
    
    v = string("hello");                        // 重新赋值为string
    cout << "v = \"hello\" (string): index=" << v.index() 
         << ", get<string>()=\"" << v.get<string>() << "\"" << endl;
    
    v = 3.14;                                   // 重新赋值为double
    cout << "v = 3.14 (double): index=" << v.index() 
         << ", get<double>()=" << v.get<double>() << endl;

    cout << "\n";
    
    // ======================== 实验2: 类型不匹配 → 抛异常 =============
    
    cout << "=== 错误处理 ===" << endl;
    try {
        MyVariant<int, string> w(42);           // w持有int
        cout << w.get<string>();                // ❌ 当前是int不是string!
    } catch (const bad_variant_access& e) {
        cout << "捕获异常: " << e.what() << endl; // type mismatch!
    }

    cout << "\n";
    
    // ======================== 实验3: make_variant —— 自动推导 =======
    
    cout << "=== make_variant ===" << endl;
    auto v1 = make_variant(42);                    // MyVariant<int>(42)
    auto v2 = make_variant(3.14);                  // MyVariant<double>(3.14)  
    
    cout << "v1: get<int>()=" << v1.get<int>()     << endl;
    cout << "v2: get<double>()=" << v2.get<double>() << endl;

    cout << "\n";
    
    // ======================== 实验4: Contains & IndexOf验证 =========
    
    cout << "=== 编译期类型检测 ===" << endl;
    // using MyTypes = int, string, double;  // ❌ 这不是合法语法
    // 用Contains直接测试：
    cout << "int in <int,string,double>: " 
         << Contains<int, int, string, double>::value << endl;      // true
    
    cout << "char* in <int,string,double>: "
         << Contains<char*, int, string, double>::value << endl;     // false

    cout << "\n=== IndexOf ===" << endl;
    cout << "IndexOf<int>     = " << IndexOf<int, int, string, double>::value   << endl;  // 0
    cout << "IndexOf<string>  = " << IndexOf<string, int, string, double>::value << endl;  // 1
    cout << "IndexOf<double>  = " << IndexOf<double, int, string, double>::value << endl;  // 2

    // ======================== 实验5: static_assert验证 ==============
    
    cout << "\n=== 编译期断言 ===" << endl;
    using V1 = MyVariant<int, string, double>;
    if constexpr (InList<int, int, string, double>)
        cout << "int is in <int,string,double> ✓\n";
    
    if (!InList<char*, int, string, double>)
        cout << "char* NOT in <int,string,double> ✓\n";

    // ======================== 实验6: 全书知识回顾 ===================
    
    cout << "\n=== 本章用到的知识点 ===" << endl;
    cout << "Ch07/08: IndexOf用递归+特化查找类型索引\n";
    cout << "   struct IndexOf<T, First, Rest...> : value = 1 + IndexOf<T, Rest...>::value\n";
    cout << "\n";
    
    cout << "Ch09 (SFINAE): make_variant的enable_if约束\n";
    cout << "   template<typename T, typename = enable_if_t<InList<T, Ts...>>>\n";
    cout << "\n";
    
    cout << "Ch10 (TMP): Contains用递归+conditional实现类型搜索\n";
    cout << "   struct Contains : conditional<is_same, true_type, Contains<T, Rest...>>\n";
    cout << "\n";
    
    cout << "Ch13 (if constexpr): get中的static_assert编译期验证\n";
    cout << "   static_assert(InList<T, Ts...>, \"T不在列表中!\")\n";
    cout << "\n";
    
    cout << "Ch14 (Concept): InList是named constraint\n";
    cout << "   template<typename T> concept InList = Contains<T, Ts...>::value;\n";

    cout << "\n";
    cout << "🎯 本章要点:\n";
    cout << "1. variant = void* + type_index: 最朴素的类型擦除方案\n";
    cout << "2. placement new在缓冲区上构造对象，析构时需要手动调用destructor\n";
    cout << "3. get<T>用static_assert(compile-time) + runtime index检查双重保障\n";
    cout << "4. make_variant自动推导类型列表 —— 不需要写MyVariant<int,string>(42,\"hi\")\n";
    cout << "5. 完整variant还需要: visitation, clear/reset, move semantics\n";
    
    return 0;
}
