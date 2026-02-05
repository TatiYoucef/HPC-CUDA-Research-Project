// Alternative Strategy: Tiled Matrix Multiplication with Shared Memory
// This implementation uses shared memory to reduce global memory accesses
// and improve data locality for neural network training

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

// ! Network Parameters
#define INPUT_SIZE      32      // Number of input features
#define HIDDEN_SIZE     256     // Number of neurons in the hidden layer
#define OUTPUT_SIZE     1       // Number of output neurons
#define EPOCHS          100     // Number of training epochs
#define LOG_EVERY_EPOCH 1       // Log loss every n epochs
#define LEARNING_RATE   0.002f
#define BATCH_SIZE      256     // Batch size for SGD

// ! CUDA Configuration
#define TILE_SIZE       16      // Tile size for shared memory optimization
#define THREADS_PER_BLOCK TILE_SIZE

// ! Data Structures
typedef struct {
    int rows;
    int cols;
    float *data;
} Matrix;

// ! Memory Management
Matrix* allocate_matrix(int rows, int cols) {
    Matrix *m = (Matrix*)malloc(sizeof(Matrix));
    m->rows = rows;
    m->cols = cols;
    m->data = (float*)malloc(rows * cols * sizeof(float));
    return m;
}

void free_matrix(Matrix *m) {
    free(m->data);
    free(m);
}

// ! Matrix Operations
void random_init(Matrix *m) {
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            m->data[i * m->cols + j] = (float)rand() / RAND_MAX;
        }
    }
}

// =============================================================================
// ALTERNATIVE STRATEGY: Tiled Matrix Multiplication with Shared Memory
// =============================================================================
// Key differences from baseline:
// 1. Uses shared memory tiles to cache data from global memory
// 2. Threads cooperatively load tiles, reducing redundant global accesses
// 3. Synchronization ensures all threads complete loading before computation
// 4. Significantly reduces global memory bandwidth requirements
// =============================================================================

__global__ void mat_mult_tiled_kernel(float *A, float *B, float *C, 
                                       int A_rows, int A_cols, int B_cols) {
    // Shared memory tiles for A and B
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];
    
    // Thread indices
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // Output element coordinates
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;
    
    // Accumulator for dot product
    float value = 0.0f;
    
    // Number of tiles needed to cover the shared dimension
    int numTiles = (A_cols + TILE_SIZE - 1) / TILE_SIZE;
    
    // Iterate over all tiles
    for (int t = 0; t < numTiles; t++) {
        // Collaborative loading of tiles into shared memory
        // Each thread loads one element from A and one from B
        
        // Load element from A into shared memory
        int aCol = t * TILE_SIZE + tx;
        if (row < A_rows && aCol < A_cols) {
            tileA[ty][tx] = A[row * A_cols + aCol];
        } else {
            tileA[ty][tx] = 0.0f;  // Boundary padding
        }
        
        // Load element from B into shared memory
        int bRow = t * TILE_SIZE + ty;
        if (bRow < A_cols && col < B_cols) {
            tileB[ty][tx] = B[bRow * B_cols + col];
        } else {
            tileB[ty][tx] = 0.0f;  // Boundary padding
        }
        
        // Synchronize to ensure all threads have loaded their elements
        __syncthreads();
        
        // Compute partial dot product for this tile
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            value += tileA[ty][k] * tileB[k][tx];
        }
        
        // Synchronize before loading next tile to avoid race conditions
        __syncthreads();
    }
    
    // Write result to global memory
    if (row < A_rows && col < B_cols) {
        C[row * B_cols + col] = value;
    }
}

// Alternative kernel: Double-buffered tiled multiplication for hiding latency
__global__ void mat_mult_tiled_double_buffer_kernel(float *A, float *B, float *C,
                                                     int A_rows, int A_cols, int B_cols) {
    // Double buffers for prefetching
    __shared__ float tileA[2][TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[2][TILE_SIZE][TILE_SIZE];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;
    
    float value = 0.0f;
    int numTiles = (A_cols + TILE_SIZE - 1) / TILE_SIZE;
    
    // Load first tile into buffer 0
    int aCol = tx;
    int bRow = ty;
    if (row < A_rows && aCol < A_cols) {
        tileA[0][ty][tx] = A[row * A_cols + aCol];
    } else {
        tileA[0][ty][tx] = 0.0f;
    }
    if (bRow < A_cols && col < B_cols) {
        tileB[0][ty][tx] = B[bRow * B_cols + col];
    } else {
        tileB[0][ty][tx] = 0.0f;
    }
    __syncthreads();
    
    for (int t = 0; t < numTiles; t++) {
        int curr = t & 1;      // Current buffer
        int next = 1 - curr;   // Next buffer
        
        // Prefetch next tile while computing current
        if (t + 1 < numTiles) {
            int aColNext = (t + 1) * TILE_SIZE + tx;
            int bRowNext = (t + 1) * TILE_SIZE + ty;
            
            if (row < A_rows && aColNext < A_cols) {
                tileA[next][ty][tx] = A[row * A_cols + aColNext];
            } else {
                tileA[next][ty][tx] = 0.0f;
            }
            if (bRowNext < A_cols && col < B_cols) {
                tileB[next][ty][tx] = B[bRowNext * B_cols + col];
            } else {
                tileB[next][ty][tx] = 0.0f;
            }
        }
        
        // Compute using current buffer
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            value += tileA[curr][ty][k] * tileB[curr][k][tx];
        }
        
        __syncthreads();
    }
    
    if (row < A_rows && col < B_cols) {
        C[row * B_cols + col] = value;
    }
}

// Function to multiply matrices using tiled approach
Matrix* mat_mult(Matrix *A, Matrix *B) {
    if (A->cols != B->rows) {
        printf("Incompatible matrices for multiplication.\n");
        exit(1);
    }

    Matrix *C = allocate_matrix(A->rows, B->cols);

    float *d_A, *d_B, *d_C;
    size_t sizeA = A->rows * A->cols * sizeof(float);
    size_t sizeB = B->rows * B->cols * sizeof(float);
    size_t sizeC = C->rows * C->cols * sizeof(float);

    // Allocate device memory
    cudaMalloc((void **)&d_A, sizeA);
    cudaMalloc((void **)&d_B, sizeB);
    cudaMalloc((void **)&d_C, sizeC);

    // Copy data from host to device
    cudaMemcpy(d_A, A->data, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B->data, sizeB, cudaMemcpyHostToDevice);

    // Set up grid and block dimensions
    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 numBlocks((B->cols + TILE_SIZE - 1) / TILE_SIZE,
                   (A->rows + TILE_SIZE - 1) / TILE_SIZE);

    // Launch the TILED kernel (alternative strategy)
    mat_mult_tiled_kernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, 
                                                           A->rows, A->cols, B->cols);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Kernel Error: %s\n", cudaGetErrorString(err));
        exit(1);
    }

    cudaDeviceSynchronize();

    // Copy the result from device to host
    cudaMemcpy(C->data, d_C, sizeC, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return C;
}

// Matrix subtraction: C = A - B
Matrix* mat_sub(Matrix *A, Matrix *B) {
    if(A->rows != B->rows || A->cols != B->cols) {
        printf("Incompatible matrices for subtraction.\n");
        exit(1);
    }
    Matrix *C = allocate_matrix(A->rows, A->cols);
    for(int i = 0; i < A->rows; i++)
        for(int j = 0; j < A->cols; j++)
            C->data[i * A->cols + j] = A->data[i * A->cols + j] - B->data[i * A->cols + j];
    return C;
}

// Matrix scalar multiplication: A = A * scalar
void mat_scalar_mult(Matrix *A, float scalar) {
    for(int i = 0; i < A->rows; i++)
        for(int j = 0; j < A->cols; j++)
            A->data[i * A->cols + j] *= scalar;
}

// ! Activation Functions
void relu(Matrix *m) {
    for(int i = 0; i < m->rows; i++)
        for(int j = 0; j < m->cols; j++)
            m->data[i * m->cols + j] = fmaxf(0, m->data[i * m->cols + j]);
}

Matrix* relu_derivative(Matrix *m) {
    Matrix *derivative = allocate_matrix(m->rows, m->cols);
    for(int i = 0; i < m->rows; i++)
        for(int j = 0; j < m->cols; j++)
            derivative->data[i * m->cols + j] = (m->data[i * m->cols + j] > 0) ? 1 : 0;
    return derivative;
}

// ! Loss Functions
float mean_squared_error(Matrix *Y_pred, Matrix *Y_true) {
    float mse = 0.0f;
    for(int i = 0; i < Y_pred->rows; i++)
        for(int j = 0; j < Y_pred->cols; j++)
            mse += pow(Y_pred->data[i * Y_pred->cols + j] - Y_true->data[i * Y_true->cols + j], 2);
    return mse / Y_pred->rows;
}

// ! Optimization
void update_weights(Matrix *W, Matrix *grad, float learning_rate) {
    for(int i = 0; i < W->rows; i++)
        for(int j = 0; j < W->cols; j++)
            W->data[i * W->cols + j] -= learning_rate * grad->data[i * grad->cols + j];
}

void backpropagation(Matrix *X_batch, Matrix *Y_batch, Matrix *Z1, Matrix *Y_pred, 
                     Matrix *W1, Matrix *W2, int batch_size) {
    // Compute dZ2 = Y_pred - Y_batch
    Matrix *dZ2 = mat_sub(Y_pred, Y_batch);
    mat_scalar_mult(dZ2, 2.0f / batch_size);

    // Compute dW2 = Z1^T * dZ2
    Matrix *Z1_T = allocate_matrix(Z1->cols, Z1->rows);
    for(int i = 0; i < Z1->rows; i++) {
        for(int j = 0; j < Z1->cols; j++) {
            Z1_T->data[j * Z1->rows + i] = Z1->data[i * Z1->cols + j];
        }
    }
    Matrix *dW2 = mat_mult(Z1_T, dZ2);
    update_weights(W2, dW2, LEARNING_RATE);
    free_matrix(dW2);
    free_matrix(Z1_T);

    // Compute dZ1 = dZ2 * W2^T
    Matrix *W2_T = allocate_matrix(W2->cols, W2->rows);
    for(int i = 0; i < W2->rows; i++) {
        for(int j = 0; j < W2->cols; j++) {
            W2_T->data[j * W2->rows + i] = W2->data[i * W2->cols + j];
        }
    }
    Matrix *dZ1 = mat_mult(dZ2, W2_T);

    // Apply ReLU derivative
    Matrix *dZ1_derivative = relu_derivative(Z1);
    for(int i = 0; i < dZ1->rows; i++) {
        for(int j = 0; j < dZ1->cols; j++) {
            dZ1->data[i * dZ1->cols + j] *= dZ1_derivative->data[i * dZ1_derivative->cols + j];
        }
    }
    free_matrix(dZ1_derivative);
    free_matrix(W2_T);

    // Compute dW1 = X_batch^T * dZ1
    Matrix *X_batch_T = allocate_matrix(X_batch->cols, X_batch->rows);
    for(int i = 0; i < X_batch->rows; i++) {
        for(int j = 0; j < X_batch->cols; j++) {
            X_batch_T->data[j * X_batch->rows + i] = X_batch->data[i * X_batch->cols + j];
        }
    }
    Matrix *dW1 = mat_mult(X_batch_T, dZ1);
    update_weights(W1, dW1, LEARNING_RATE);
    free_matrix(dW1);
    free_matrix(X_batch_T);

    free_matrix(dZ2);
    free_matrix(dZ1);
}

// ! Batch Processing
void get_batch(Matrix *X, Matrix *Y, Matrix *X_batch, Matrix *Y_batch, 
               int batch_start, int batch_size) {
    for(int i = 0; i < batch_size; i++) {
        for(int j = 0; j < INPUT_SIZE; j++)
            X_batch->data[i * INPUT_SIZE + j] = X->data[(batch_start + i) * INPUT_SIZE + j];
        Y_batch->data[i * OUTPUT_SIZE] = Y->data[(batch_start + i) * OUTPUT_SIZE];
    }
}

// ! Data Loading
int load_csv(const char *filename, Matrix **X, Matrix **Y, int *num_samples) {
    FILE *file = fopen(filename, "r");
    if(!file) {
        printf("Failed to open file.\n");
        return -1;
    }
    char line[1024];
    int count = 0;
    while(fgets(line, sizeof(line), file)) count++;
    *num_samples = count;
    rewind(file);
    
    *X = allocate_matrix(count, INPUT_SIZE);
    *Y = allocate_matrix(count, OUTPUT_SIZE);
    int i = 0;
    while(fgets(line, sizeof(line), file)) {
        char *token = strtok(line, ",");
        int j = 0;
        while(token) {
            if(j < INPUT_SIZE) {
                (*X)->data[i * INPUT_SIZE + j] = atof(token);
            } else {
                (*Y)->data[i * OUTPUT_SIZE] = atof(token);
            }
            j++;
            token = strtok(NULL, ",");
        }
        i++;
    }
    fclose(file);
    return 0;
}

// Main function
int main(int argc, char *argv[]) {
    if(argc != 2) {
        printf("Usage: %s <data.csv>\n", argv[0]);
        return -1;
    }

    double start_time, end_time;

    Matrix *X, *Y;
    int num_samples;
    if(load_csv(argv[1], &X, &Y, &num_samples) != 0)
        return -1;

    // Allocate and initialize weights
    Matrix *W1 = allocate_matrix(INPUT_SIZE, HIDDEN_SIZE);
    Matrix *W2 = allocate_matrix(HIDDEN_SIZE, OUTPUT_SIZE);
    random_init(W1);
    random_init(W2);

    printf("=== TILED MATRIX MULTIPLICATION (Alternative Strategy) ===\n");
    printf("Using TILE_SIZE = %d\n\n", TILE_SIZE);

    // Start measuring time
    start_time = omp_get_wtime();

    // Training loop
    for(int epoch = 0; epoch < EPOCHS; epoch++) {
        for(int batch_start = 0; batch_start < num_samples; batch_start += BATCH_SIZE) {
            int batch_end = fmin(batch_start + BATCH_SIZE, num_samples);
            int batch_size = batch_end - batch_start;

            Matrix *X_batch = allocate_matrix(batch_size, INPUT_SIZE);
            Matrix *Y_batch = allocate_matrix(batch_size, OUTPUT_SIZE);
            get_batch(X, Y, X_batch, Y_batch, batch_start, batch_size);

            // Forward pass
            Matrix *Z1 = mat_mult(X_batch, W1);
            relu(Z1);
            Matrix *Y_pred = mat_mult(Z1, W2);

            // Compute loss
            float loss = mean_squared_error(Y_pred, Y_batch);
            if((batch_start == 0) && ((epoch % LOG_EVERY_EPOCH == 0 && epoch != 0) || epoch == 1 || epoch == EPOCHS - 1))
                printf("Epoch %d, MSE: %f\n", epoch, loss);

            // Backward pass
            backpropagation(X_batch, Y_batch, Z1, Y_pred, W1, W2, batch_size);

            free_matrix(Z1);
            free_matrix(Y_pred);
            free_matrix(X_batch);
            free_matrix(Y_batch);
        }
    }

    // Stop measuring time
    end_time = omp_get_wtime();

    printf("\nTraining time: %.4f seconds\n", end_time - start_time);

    // Cleanup
    free_matrix(W1);
    free_matrix(W2);
    free_matrix(X);
    free_matrix(Y);

    return 0;
}
