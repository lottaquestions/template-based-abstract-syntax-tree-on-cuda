#include "ast.cu"
#include <iostream>
#include <cassert>
#include <cmath>
 
// Use (void) to silence unused warnings.
#define assertm(exp, msg) assert((void(msg), exp))

int main(){

    static constexpr int arraySize = 100;
    constexpr float testVal = 5.0f;
    DeviceArray1D A(arraySize), B(arraySize);

    B = constant(testVal); // Assign 5.0 to all entires in B
    A = B[0]; // Copy B to A with no offset
    float *A_h = (float *)malloc(arraySize * sizeof(float));
    cudaMemcpy(A_h, A._ptr, arraySize*sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i< arraySize; ++i){
        assertm( fabsf(A_h[i] - testVal)  < 0.0001f, "Success");
    }
    std::cout << "Success. Value found in Array: "<< A_h[0] << std::endl;

    return 0;
}