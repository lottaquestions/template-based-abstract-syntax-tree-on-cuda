# template-based-abstract-syntax-tree-on-cuda
An implementation of GPU Gems Jade Edition Chapter 32. Operations on arrays such as addition and assignment are simplified so that users of the library can use simple binary operators such as `+`, or `=` on whole arrays. This is done by creating an AST using C++ template metaprogramming and operator overloading.
