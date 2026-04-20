cmake_minimum_required(VERSION 3.14)
project(CppTemplateProgramming LANGUAGES CXX)

# C++20 required for concepts and requires expressions (Ch13-14)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_compile_options(-Wall -Wextra -pedantic)

# Helper: add an example with a simple numeric target name
function(add_ex num src_file)
    set(target "ex${num}")
    add_executable("${target}" "${src_file}")
    set_target_properties("${target}" PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/${target}
    )
endfunction()

# Phase 1: Foundation (Ch01-05)
add_ex(01 "examples/01_什么是模板不是泛型是代码生成器.cpp")
add_ex(02 "examples/02_函数模板一个原型对付所有类型.cpp")
add_ex(03 "examples/03_多模板参数一行变多行.cpp")
add_ex(04 "examples/04_默认模板参数让类型可选.cpp")
add_ex(05 "examples/05_类模板不只是函数.cpp")

# Phase 2: Advanced Mechanics (Ch06-09)
add_ex(06 "examples/06_非类型模板参数编译期常量作为参数.cpp")
add_ex(07 "examples/07_模板特化为特定类型开小灶.cpp")
add_ex(08 "examples/08_部分特化精确匹配的艺术.cpp")
add_ex(09 "examples/09_SFINAE与enable_if编译期的if-else.cpp")

# Phase 3: Metaprogramming (Ch10-12)
add_ex(10 "examples/10_模板元编程在编译期写程序.cpp")
add_ex(11 "examples/11_fold表达式打包解包的艺术.cpp")
add_ex(12 "examples/12_variadic模板参数包展开.cpp")

# Phase 4: Modern C++ (Ch13-15)
add_ex(13 "examples/13_if_constexpr与requires_C++20的约束.cpp")
add_ex(14 "examples/14_concept给模板写签名验证.cpp")
add_ex(15 "examples/15_综合实战从零写一个类型安全的Variant.cpp")
