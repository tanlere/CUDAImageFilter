#include <cuda_runtime.h>
#include <iostream>

__global__ void helloFromGPU() {
    printf("Hello from GPU! Block %d, Thread %d\n", blockIdx.x, threadIdx.x);
}

int main() {

    // launch kernel with 1 block of 10 threads
    helloFromGPU<<<1, 10>>>();

    // wait for kernel to finish
    cudaDeviceSynchronize();

    return 0;
}