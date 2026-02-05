#!/bin/bash
# Run script for the Alternative Strategy (Tiled Matrix Multiplication)
# Compiles and runs the tiled CUDA implementation

# Compile the tiled implementation
nvcc -o nn_cuda_tiled nn_cuda_tiled.cu -Xcompiler -fopenmp

echo "=== Running Tiled Matrix Multiplication Strategy ==="
echo ""

# Run on different dataset sizes
echo "--- Small Dataset (256 samples) ---"
./nn_cuda_tiled ../CODE_REFERENCE/data/synthetic_convex_small.csv

echo ""
echo "--- Medium Dataset (2560 samples) ---"
./nn_cuda_tiled ../CODE_REFERENCE/data/synthetic_convex_medium.csv

echo ""
echo "--- Large Dataset (25600 samples) ---"
./nn_cuda_tiled ../CODE_REFERENCE/data/synthetic_convex_large.csv
