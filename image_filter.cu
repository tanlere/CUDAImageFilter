#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <iostream>

// CUDA kernel: Apply grayscale filter
__global__ void grayscaleFilter(unsigned char *d_input, unsigned char *d_output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int idx = (y * width + x) * 3;  // 3 channels for RGB
        unsigned char r = d_input[idx];
        unsigned char g = d_input[idx + 1];
        unsigned char b = d_input[idx + 2];
        
        unsigned char gray = 0.299f * r + 0.587f * g + 0.114f * b;
        
        d_output[idx] = gray;
        d_output[idx + 1] = gray;
        d_output[idx + 2] = gray;
    }
}

int main() {
    // Load image using OpenCV
    cv::Mat h_input = cv::imread("input_image.jpg");  // replace with your image path
    if (h_input.empty()) {
        std::cerr << "Could not open or find the image!" << std::endl;
        return -1;
    }

    int width = h_input.cols;
    int height = h_input.rows;
    int imageSize = width * height * 3;

    // Allocate memory on host (CPU)
    unsigned char *h_output = new unsigned char[imageSize];

    // Allocate memory on device (GPU)
    unsigned char *d_input, *d_output;
    cudaMalloc(&d_input, imageSize);
    cudaMalloc(&d_output, imageSize);

    // Copy image data from host to device
    cudaMemcpy(d_input, h_input.data, imageSize, cudaMemcpyHostToDevice);

    // Launch kernel (grayscale filter)
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    grayscaleFilter<<<gridSize, blockSize>>>(d_input, d_output, width, height);

    // Copy the output back from device to host
    cudaMemcpy(h_output, d_output, imageSize, cudaMemcpyDeviceToHost);

    // Save the output image
    cv::Mat h_output_image(height, width, CV_8UC3, h_output);  // Convert to OpenCV Mat
    cv::imwrite("output_image.jpg", h_output_image);

    // Free memory
    cudaFree(d_input);
    cudaFree(d_output);
    delete[] h_output;

    std::cout << "Grayscale filter applied successfully!" << std::endl;
    return 0;
}
