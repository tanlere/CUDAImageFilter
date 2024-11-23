#include <cuda_runtime.h>
#include <iostream>

__global__ void processArray(int *d_array) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    d_array[idx] = idx; // simple operation: setting array value to thread index
}

int main() {
    const int ARRAY_SIZE = 10;
    const int ARRAY_BYTES = ARRAY_SIZE * sizeof(int);

    int h_array[ARRAY_SIZE] = {0}; // host array

    // allocate memory on the device (GPU)
    int *d_array;
    cudaMalloc((void**)&d_array, ARRAY_BYTES);

    // copy data from host to device
    cudaMemcpy(d_array, h_array, ARRAY_BYTES, cudaMemcpyHostToDevice);

    // launch kernel with 1 block of 10 threads
    processArray<<<1, ARRAY_SIZE>>>(d_array);

    // copy data from device to host
    cudaMemcpy(h_array, d_array, ARRAY_BYTES, cudaMemcpyDeviceToHost);

    // print result
    for (int i = 0; i < ARRAY_SIZE; i++) {
        std::cout << "Array[" << i << "] = " << h_array[i] << std::endl;
    }

    // free memory on the device
    cudaFree(d_array);

    return 0;
}
