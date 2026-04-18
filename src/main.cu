#include <iostream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>

#define THREADS 256

// ===================== VECTOR ADD =====================
__global__ void vectorAddGPU(float *A, float *B, float *C, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) C[idx] = A[idx] + B[idx];
}

void vectorAddCPU(std::vector<float>& A, std::vector<float>& B, std::vector<float>& C) {
    for (size_t i = 0; i < A.size(); i++)
        C[i] = A[i] + B[i];
}

// ===================== SAXPY =====================
__global__ void saxpyGPU(float a, float *x, float *y, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) y[idx] = a * x[idx] + y[idx];
}

void saxpyCPU(float a, std::vector<float>& x, std::vector<float>& y) {
    for (size_t i = 0; i < x.size(); i++)
        y[i] = a * x[i] + y[i];
}

// ===================== REDUCTION =====================
__global__ void reduceSumGPU(float *input, float *output, int n) {
    __shared__ float sdata[THREADS];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (gid < n) ? input[gid] : 0;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0)
        output[blockIdx.x] = sdata[0];
}

float reduceCPU(std::vector<float>& data) {
    float total = 0;
    for (size_t i = 0; i < data.size(); i++) total += data[i];
    return total;
}

// ===================== MATRIX MULT =====================
__global__ void matMulGPU(float *A, float *B, float *C, int N) {
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;

    if (r < N && c < N) {
        float acc = 0;
        for (int k = 0; k < N; k++)
            acc += A[r * N + k] * B[k * N + c];
        C[r * N + c] = acc;
    }
}

void matMulCPU(std::vector<float>& A, std::vector<float>& B, std::vector<float>& C, int N) {
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            float sum = 0;
            for (int k = 0; k < N; k++)
                sum += A[i * N + k] * B[k * N + j];
            C[i * N + j] = sum;
        }
}

// ===================== BENCHMARK HELPERS =====================
void benchmarkVector(int n) {
    size_t size = n * sizeof(float);

    std::vector<float> A(n, 1.0f), B(n, 2.0f), C(n);
    float *d_A, *d_B, *d_C;

    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, A.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data(), size, cudaMemcpyHostToDevice);

    // warmup
    vectorAddGPU<<<(n+THREADS-1)/THREADS, THREADS>>>(d_A,d_B,d_C,n);
    cudaDeviceSynchronize();

    auto cpu_start = std::chrono::high_resolution_clock::now();
    vectorAddCPU(A,B,C);
    auto cpu_end = std::chrono::high_resolution_clock::now();

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);

    cudaEventRecord(t0);
    vectorAddGPU<<<(n+THREADS-1)/THREADS, THREADS>>>(d_A,d_B,d_C,n);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);

    float gpu_time;
    cudaEventElapsedTime(&gpu_time,t0,t1);

    std::cout << "Size: " << n
              << " | CPU: " << std::chrono::duration<double, std::milli>(cpu_end-cpu_start).count()
              << " ms | GPU: " << gpu_time << " ms\n";

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
}

void benchmarkSAXPY(int n) {
    size_t size = n * sizeof(float);

    std::vector<float> x(n,1.0f), y(n,2.0f);
    float *d_x,*d_y;

    cudaMalloc(&d_x,size);
    cudaMalloc(&d_y,size);

    cudaMemcpy(d_x,x.data(),size,cudaMemcpyHostToDevice);
    cudaMemcpy(d_y,y.data(),size,cudaMemcpyHostToDevice);

    float a = 2.5f;

    auto cpu_start = std::chrono::high_resolution_clock::now();
    saxpyCPU(a,x,y);
    auto cpu_end = std::chrono::high_resolution_clock::now();

    cudaEvent_t t0,t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);

    cudaEventRecord(t0);
    saxpyGPU<<<(n+THREADS-1)/THREADS,THREADS>>>(a,d_x,d_y,n);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);

    float gpu_time;
    cudaEventElapsedTime(&gpu_time,t0,t1);

    std::cout << "Size: " << n
              << " | CPU: " << std::chrono::duration<double, std::milli>(cpu_end-cpu_start).count()
              << " ms | GPU: " << gpu_time << " ms\n";

    cudaFree(d_x); cudaFree(d_y);
}

void benchmarkReduction(int n) {
    size_t size = n * sizeof(float);

    std::vector<float> data(n,1.0f);
    float *d_in,*d_out;

    cudaMalloc(&d_in,size);
    cudaMalloc(&d_out,size);

    cudaMemcpy(d_in,data.data(),size,cudaMemcpyHostToDevice);

    auto cpu_start = std::chrono::high_resolution_clock::now();
    reduceCPU(data);
    auto cpu_end = std::chrono::high_resolution_clock::now();

    cudaEvent_t t0,t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);

    cudaEventRecord(t0);
    reduceSumGPU<<<(n+THREADS-1)/THREADS,THREADS>>>(d_in,d_out,n);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);

    float gpu_time;
    cudaEventElapsedTime(&gpu_time,t0,t1);

    std::cout << "Size: " << n
              << " | CPU: " << std::chrono::duration<double, std::milli>(cpu_end-cpu_start).count()
              << " ms | GPU: " << gpu_time << " ms\n";

    cudaFree(d_in); cudaFree(d_out);
}

void benchmarkMatrix(int N) {
    size_t size = N*N*sizeof(float);

    std::vector<float> A(N*N,1.0f),B(N*N,1.0f),C(N*N);
    float *d_A,*d_B,*d_C;

    cudaMalloc(&d_A,size);
    cudaMalloc(&d_B,size);
    cudaMalloc(&d_C,size);

    cudaMemcpy(d_A,A.data(),size,cudaMemcpyHostToDevice);
    cudaMemcpy(d_B,B.data(),size,cudaMemcpyHostToDevice);

    auto cpu_start = std::chrono::high_resolution_clock::now();
    matMulCPU(A,B,C,N);
    auto cpu_end = std::chrono::high_resolution_clock::now();

    dim3 threads(16,16);
    dim3 blocks((N+15)/16,(N+15)/16);

    cudaEvent_t t0,t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);

    cudaEventRecord(t0);
    matMulGPU<<<blocks,threads>>>(d_A,d_B,d_C,N);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);

    float gpu_time;
    cudaEventElapsedTime(&gpu_time,t0,t1);

    std::cout << "Size: " << N << "x" << N
              << " | CPU: " << std::chrono::duration<double, std::milli>(cpu_end-cpu_start).count()
              << " ms | GPU: " << gpu_time << " ms\n";

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
}

// ===================== MAIN =====================
int main() {

    std::cout << "===== GPU vs CPU Benchmark (Scaling) =====\n";

    int sizes[] = {1<<10, 1<<20, 1<<24};

    std::cout << "\n[Vector Add]\n";
    for (int s : sizes) benchmarkVector(s);

    std::cout << "\n[SAXPY]\n";
    for (int s : sizes) benchmarkSAXPY(s);

    std::cout << "\n[Reduction]\n";
    for (int s : sizes) benchmarkReduction(s);

    std::cout << "\n[Matrix Multiplication]\n";
    int mat_sizes[] = {64,128,256};
    for (int s : mat_sizes) benchmarkMatrix(s);

    return 0;
}