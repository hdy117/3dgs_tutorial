/* ================================================================
 *  Chapter 05 — 类模板：不只是函数
 * ================================================================
 *
 * 🧠 推导：
 *   函数可以模板化，类为什么不能？
 *   
 *   std::vector<int>, std::map<string, int> —— 你在用的已经是类模板。
 *   
 *   关键区别：
 *   - 函数模板：调用时自动推断T
 *   - 类模板：**必须**显式指定类型（编译器不推导入参给类）
 *     → pair(1, 2.0) ❌ 不能推断，必须 std::make_pair(1, 2.0)
 *
 * 💡 Mental Model:
 *   template<typename T> class Stack { ... };
 *   Stack<int> s;              // Stack<int>是一个完整的类型
 *   Stack<double> s2;          // Stack<double>是另一个完全不同的类型
 *   
 *   它们之间没有继承关系，就像int和double没有关系一样。
 *
 * ================================================================ */

#include <iostream>
#include <string>
using namespace std;

// ──────────────────────────────────────────────────────────────────
// 1. 最基础的类模板：一个类型安全的Box
// ──────────────────────────────────────────────────────────────────
template<typename T>
class Box {
    T contents;
    
public:
    // 构造函数 —— 注意T在这里是类模板参数
    explicit Box(T value) : contents(value) {}
    
    // getter
    T get() const { return contents; }
    
    // setter
    void set(T value) { contents = value; }
};

// ──────────────────────────────────────────────────────────────────
// 2. 类模板的友元函数 —— 两种写法
//    写法A: 每个实例化的Box<T>都有一个对应类型的friend
//    写法B: friend本身也是模板（所有T共享一个friend）
// ──────────────────────────────────────────────────────────────────

template<typename T>
class Counter {
private:
    int count;
    
public:
    Counter(int c = 0) : count(c) {}
    
    // 写法A: friend也带模板参数 —— 每个Counter<T>有自己的friend函数
    template<typename U>
    friend void compare(const Counter<U>& a, const Counter<U>& b);
};

template<typename T>
void compare(const Counter<T>& a, const Counter<T>& b) {
    if (a.count > b.count) cout << "左边更大" << endl;
    else if (a.count < b.count) cout << "右边更大" << endl;
    else cout << "相等" << endl;
}

// ──────────────────────────────────────────────────────────────────
// 3. 类模板的成员函数也可以是模板！
//    外层模板参数是类的T，内层是函数的U。
// ──────────────────────────────────────────────────────────────────
template<typename T>
class Converter {
public:
    // 成员函数也是模板 —— 这创造了"任意类型转换"的能力
    template<typename U>
    U convert() const {
        return static_cast<U>(value);
    }
    
    void set(T v) { value = v; }
    
private:
    T value;
};

// ──────────────────────────────────────────────────────────────────
// 4. 类模板 + 默认参数 —— 结合第04章的知识
// ──────────────────────────────────────────────────────────────────
template<typename T, int MAX_SIZE = 100>
class SimpleArray {
    T arr[MAX_SIZE];
    int size;
    
public:
    SimpleArray() : size(0) {}
    
    void push(const T& val) {
        if (size < MAX_SIZE) arr[size++] = val;
    }
    
    T at(int i) const { return arr[i]; }
    int count() const { return size; }
};

int main() {
    // ======================== 实验1: Box类模板 =========================
    
    cout << "=== Box类模板 ===" << endl;
    Box<int> box_int(42);          // 必须显式指定类型！
    Box<double> box_double(3.14);
    Box<string> box_string("hello");
    
    cout << "box_int.get()   = " << box_int.get() << endl;
    cout << "box_double.get()= " << box_double.get() << endl;
    cout << "box_string.get()= " << box_string.get() << endl;
    
    // 注意：Box<int>和Box<double>是完全不同的类型！
    // 不能直接赋值: box_int = box_double ❌ (类型不匹配)
    
    cout << "\n";
    
    // ======================== 实验2: 类模板必须显式指定类型 ============
    // 这是类和函数模板的最大区别：
    //   max(1, 2.0) → ✅ 编译器推断T=double (从两个参数推导)
    //   Stack(1, 2) → ❌ 编译器不推导入参给类模板
    
    cout << "=== 为什么类必须显式指定类型 ===" << endl;
    cout << "函数: max(1, 2.0) ✅ T=double (从参数推断)\n";
    cout << "类:   Stack<int>(3) ❌ 不能写成 Stack(3)\n";
    cout << "     make_pair(1, 2.0) → 工厂函数帮助推导\n";
    
    cout << "\n";
    
    // ======================== 实验3: 成员模板函数 ======================
    
    cout << "=== 成员也是模板 ===" << endl;
    Converter<double> conv;
    conv.set(42.7);
    
    // convert<U>() —— U是成员函数的模板参数，和类的T无关！
    cout << "double→int:   " << conv.convert<int>()     << endl;  // 42 (截断)
    cout << "double→string: 需要自定义\n";          // static_cast不能转string
    
    cout << "\n";
    
    // ======================== 实验4: SimpleArray实战 ===================
    
    cout << "=== SimpleArray<T, MAX_SIZE> ===" << endl;
    SimpleArray<int, 5> small;     // MAX_SIZE=5 (显式)
    small.push(10);
    small.push(20);
    small.push(30);
    cout << "small: size=" << small.count() << ", at(1)=" << small.at(1) << endl;
    
    SimpleArray<double> big;       // MAX_SIZE=100 (默认)
    for (int i = 0; i < 5; ++i) big.push(i * 1.5);
    cout << "big: size=" << big.count() << ", at(4)=" << big.at(4) << endl;
    
    // push(40)到small会静默失败（size已满）—— 这是SimpleArray的设计选择
    
    cout << "\n";
    cout << "🎯 本章要点:\n";
    cout << "1. 类模板: template<typename T> class X { ... } 使用时必须指定X<int>\n";
    cout << "2. 类模板不推导入参 —— 必须显式写类型\n";
    cout << "3. 成员函数可以是独立模板: template<U> U convert()\n";
    cout << "4. 默认参数+非类型参数=灵活且安全的容器设计基础\n";
    
    return 0;
}
