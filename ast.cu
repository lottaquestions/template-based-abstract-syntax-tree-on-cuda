// cuda includes
#include "cuda_runtime.h"
#include "device_launch_parameters.h"



template<typename OP, typename PARAM>
struct OpWithParam
{
    PARAM _param;

    OpWithParam(const PARAM &param) : _param(param) {}

    __device__ float exec(int i){
        return OP::exec(i, _param);
    }
};

// Leaf node in the AST
template<typename PARAM>
struct LeafOp{
    __device__ static float exec(int i, const PARAM &param){
        return param.value(i);
    }
};

//
// Potential parameter types of the leaf node
struct  IdentityParam
{
    IdentityParam() = default;

    __device__ float value(int i){
        return float(i);
    }
};

struct ConstantParam{
    float _value;
    ConstantParam(float inVal) : _value(inVal){}

    __device__ float value([[maybe_unused]] int i) const {
        return _value;
    }
};

struct ArrayLookupParam;


// End of potential parameter types of the leaf node

struct DeviceArray1D{
    int _size;
    float *_ptr; // Device pointer
    DeviceArray1D(int size) : _size(size), _ptr(nullptr){
        cudaMalloc((void**)&_ptr, _size * sizeof(float));
    }

    ~DeviceArray1D(){
        cudaFree(_ptr);
    }

    OpWithParam<LeafOp<ArrayLookupParam>, ArrayLookupParam> operator[](int shift);

    template<typename OP, typename PARAM>
    DeviceArray1D& operator=(const OpWithParam<OP, PARAM> functor);
};

struct ArrayLookupParam
{
    const float* _ptr;
    int _shift;
    ArrayLookupParam(const DeviceArray1D &array1D, int shift) : _shift(shift), _ptr(array1D._ptr) {}

    __device__ float value(int i) const {
        return _ptr[(_shift + i)];
    }
};



// Exposing a constant in the AST as a leaf node function object(functor)
OpWithParam<LeafOp<ConstantParam>, ConstantParam> constant(float value){
    return OpWithParam<LeafOp<ConstantParam>, ConstantParam>(ConstantParam(value));
}

// Exposing the identity operatorin the AST as a leaf node functor
OpWithParam<LeafOp<IdentityParam>, IdentityParam> identity(){
    return OpWithParam<LeafOp<IdentityParam>, IdentityParam>(IdentityParam());
}

// Exposing DeviceArray1D::operator[] in the AST as a leaf node functor
OpWithParam<LeafOp<ArrayLookupParam>, ArrayLookupParam> DeviceArray1D::operator[](int shift){
    return OpWithParam<LeafOp<ArrayLookupParam>, ArrayLookupParam>(ArrayLookupParam(*this, shift));
}

// Kernel that does assignment of leaf nodes. Uses exec interface of leaf node function object
// the leaf node is 
template <typename T>
__global__ void kernel_assign(T functor, float *result, int size){
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < size){
        result[i] = functor.exec(i);
    }
}

template<typename OP, typename PARAM>
DeviceArray1D& DeviceArray1D::operator=(const OpWithParam<OP, PARAM> functor){
    constexpr int threadsPerBlock = 256;
    kernel_assign<<<(_size+threadsPerBlock-1)/threadsPerBlock, threadsPerBlock>>>(functor, _ptr, _size);
    return *this;
}


