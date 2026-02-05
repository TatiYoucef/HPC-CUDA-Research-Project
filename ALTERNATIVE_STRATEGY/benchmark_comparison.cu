// Benchmarking Suite: Compare Baseline vs Alternative CUDA Strategies
// This file provides utilities for systematic performance comparison

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define TILE_SIZE 16
#define WARMUP_RUNS 3
#define BENCHMARK_RUNS 10

// =============================================================================
// BASELINE KERNEL: Naive Element-per-Thread Matrix Multiplication
// =============================================================================
__global__ void mat_mult_baseline_kernel(float *A, float *B, float *C, 
                                          int A_rows, int A_cols, int B_cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < A_rows && col < B_cols) {
        float value = 0.0f;
        for (int k = 0; k < A_cols; k++) {
            value += A[row * A_cols + k] * B[k * B_cols + col];
        }
        C[row * B_cols + col] = value;
    }
}

// =============================================================================
// ALTERNATIVE KERNEL: Tiled Matrix Multiplication with Shared Memory
// =============================================================================
__global__ void mat_mult_tiled_kernel(float *A, float *B, float *C, 
                                       int A_rows, int A_cols, int B_cols) {
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;
    
    float value = 0.0f;
    int numTiles = (A_cols + TILE_SIZE - 1) / TILE_SIZE;
    
    for (int t = 0; t < numTiles; t++) {
        int aCol = t * TILE_SIZE + tx;
        if (row < A_rows && aCol < A_cols) {
            tileA[ty][tx] = A[row * A_cols + aCol];
        } else {
            tileA[ty][tx] = 0.0f;
        }
        
        int bRow = t * TILE_SIZE + ty;
        if (bRow < A_cols && col < B_cols) {
            tileB[ty][tx] = B[bRow * B_cols + col];
        } else {
            tileB[ty][tx] = 0.0f;
        }
        
        __syncthreads();
        
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            value += tileA[ty][k] * tileB[k][tx];
        }
        
        __syncthreads();
    }
    
    if (row < A_rows && col < B_cols) {
        C[row * B_cols + col] = value;
    }
}

// =============================================================================
// BENCHMARKING FUNCTIONS
// =============================================================================

typedef struct {
    float baseline_time_ms;
    float tiled_time_ms;
    float speedup;
    int A_rows, A_cols, B_cols;
} BenchmarkResult;

void run_benchmark(int A_rows, int A_cols, int B_cols, BenchmarkResult *result) {
    // Allocate host memory
    size_t sizeA = A_rows * A_cols * sizeof(float);
    size_t sizeB = A_cols * B_cols * sizeof(float);
    size_t sizeC = A_rows * B_cols * sizeof(float);
    
    float *h_A = (float*)malloc(sizeA);
    float *h_B = (float*)malloc(sizeB);
    float *h_C_baseline = (float*)malloc(sizeC);
    float *h_C_tiled = (float*)malloc(sizeC);
    
    // Initialize with random values
    for (int i = 0; i < A_rows * A_cols; i++) h_A[i] = (float)rand() / RAND_MAX;
    for (int i = 0; i < A_cols * B_cols; i++) h_B[i] = (float)rand() / RAND_MAX;
    
    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, sizeA);
    cudaMalloc(&d_B, sizeB);
    cudaMalloc(&d_C, sizeC);
    
    cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice);
    
    // Grid and block configuration
    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 numBlocks((B_cols + TILE_SIZE - 1) / TILE_SIZE,
                   (A_rows + TILE_SIZE - 1) / TILE_SIZE);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // =========================================================================
    // BENCHMARK BASELINE KERNEL
    // =========================================================================
    
    // Warmup runs
    for (int i = 0; i < WARMUP_RUNS; i++) {
        mat_mult_baseline_kernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, 
                                                                  A_rows, A_cols, B_cols);
    }
    cudaDeviceSynchronize();
    
    // Timed runs
    cudaEventRecord(start);
    for (int i = 0; i < BENCHMARK_RUNS; i++) {
        mat_mult_baseline_kernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, 
                                                                  A_rows, A_cols, B_cols);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float baseline_ms = 0;
    cudaEventElapsedTime(&baseline_ms, start, stop);
    result->baseline_time_ms = baseline_ms / BENCHMARK_RUNS;
    
    // Copy result for verification
    cudaMemcpy(h_C_baseline, d_C, sizeC, cudaMemcpyDeviceToHost);
    
    // =========================================================================
    // BENCHMARK TILED KERNEL
    // =========================================================================
    
    // Warmup runs
    for (int i = 0; i < WARMUP_RUNS; i++) {
        mat_mult_tiled_kernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, 
                                                               A_rows, A_cols, B_cols);
    }
    cudaDeviceSynchronize();
    
    // Timed runs
    cudaEventRecord(start);
    for (int i = 0; i < BENCHMARK_RUNS; i++) {
        mat_mult_tiled_kernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, 
                                                               A_rows, A_cols, B_cols);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float tiled_ms = 0;
    cudaEventElapsedTime(&tiled_ms, start, stop);
    result->tiled_time_ms = tiled_ms / BENCHMARK_RUNS;
    
    // Copy result for verification
    cudaMemcpy(h_C_tiled, d_C, sizeC, cudaMemcpyDeviceToHost);
    
    // =========================================================================
    // VERIFY CORRECTNESS
    // =========================================================================
    float max_error = 0.0f;
    for (int i = 0; i < A_rows * B_cols; i++) {
        float error = fabs(h_C_baseline[i] - h_C_tiled[i]);
        if (error > max_error) max_error = error;
    }
    
    if (max_error > 1e-4) {
        printf("WARNING: Results differ! Max error: %e\n", max_error);
    }
    
    // Calculate speedup
    result->speedup = result->baseline_time_ms / result->tiled_time_ms;
    result->A_rows = A_rows;
    result->A_cols = A_cols;
    result->B_cols = B_cols;
    
    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C_baseline);
    free(h_C_tiled);
}

int main() {
    printf("╔══════════════════════════════════════════════════════════════════╗\n");
    printf("║  CUDA Matrix Multiplication: Baseline vs Tiled Comparison        ║\n");
    printf("╠══════════════════════════════════════════════════════════════════╣\n");
    printf("║  Baseline: Naive element-per-thread approach                     ║\n");
    printf("║  Alternative: Tiled approach with shared memory                  ║\n");
    printf("╚══════════════════════════════════════════════════════════════════╝\n\n");
    
    // Print GPU info
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("Shared Memory per Block: %zu bytes\n", prop.sharedMemPerBlock);
    printf("Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
    printf("\nTile Size: %d x %d\n", TILE_SIZE, TILE_SIZE);
    printf("Warmup Runs: %d, Benchmark Runs: %d\n\n", WARMUP_RUNS, BENCHMARK_RUNS);
    
    // Test configurations (simulating neural network layer sizes)
    int test_cases[][3] = {
        // {A_rows, A_cols, B_cols} - typical neural network dimensions
        // Forward: X (batch x input) * W1 (input x hidden)
        {256, 32, 256},      // Small batch, forward pass layer 1
        {256, 256, 1},       // Small batch, forward pass layer 2
        {2560, 32, 256},     // Medium batch, forward pass layer 1
        {2560, 256, 1},      // Medium batch, forward pass layer 2
        {25600, 32, 256},    // Large batch, forward pass layer 1
        {25600, 256, 1},     // Large batch, forward pass layer 2
        
        // Backpropagation matrices
        {32, 256, 256},      // X^T * dZ1
        {256, 256, 256},     // Larger hidden layer
        
        // Scalability tests
        {512, 512, 512},     // Square matrices
        {1024, 1024, 1024},  // Larger square
        {2048, 2048, 2048},  // Even larger
    };
    
    int num_tests = sizeof(test_cases) / sizeof(test_cases[0]);
    
    printf("╔═══════════════════════════════════════════════════════════════════════════════╗\n");
    printf("║ Matrix Size         │ Baseline (ms) │ Tiled (ms) │ Speedup │ Memory Savings  ║\n");
    printf("╠═══════════════════════════════════════════════════════════════════════════════╣\n");
    
    for (int i = 0; i < num_tests; i++) {
        BenchmarkResult result;
        run_benchmark(test_cases[i][0], test_cases[i][1], test_cases[i][2], &result);
        
        // Calculate theoretical memory bandwidth reduction
        // Baseline: Each thread loads A_cols elements from A and A_cols from B
        // Tiled: Each thread loads A_cols/TILE_SIZE elements from A and B (amortized)
        float memory_reduction = (float)TILE_SIZE;
        
        printf("║ %4d x %4d x %4d   │   %10.4f  │  %9.4f │  %5.2fx  │     ~%.0fx        ║\n",
               result.A_rows, result.A_cols, result.B_cols,
               result.baseline_time_ms, result.tiled_time_ms, 
               result.speedup, memory_reduction);
    }
    
    printf("╚═══════════════════════════════════════════════════════════════════════════════╝\n");
    
    printf("\n=== Analysis Summary ===\n");
    printf("The tiled approach with shared memory provides significant speedup because:\n");
    printf("1. Reduces global memory accesses by factor of TILE_SIZE (%d)\n", TILE_SIZE);
    printf("2. Exploits data locality via shared memory (100x faster than global)\n");
    printf("3. Better coalesced memory access patterns\n");
    printf("4. Reduced memory bandwidth pressure\n");
    
    return 0;
}
