# HPC Project Solution Guide

## Project Summary

This project requires you to compare two CUDA parallelization strategies for training a shallow neural network and document your findings in a scientific article.

---

## Project Structure

```
HPC_Project/
├── HPC_PROJECT.md           # Original project requirements
├── REPORT_REFERENCE.md      # Reference report (baseline analysis)
├── ARTICLE_TEMPLATE.md      # Scientific article template (DELIVERABLE)
├── CODE_REFERENCE/          # Baseline implementation
│   ├── nn.c                 # Sequential implementation
│   ├── nn_pthreads.c        # Pthreads implementation
│   ├── nn_cuda.cu           # Baseline CUDA (naive approach)
│   ├── data/                # Datasets
│   └── results/             # Performance results
└── ALTERNATIVE_STRATEGY/    # Your alternative implementation
    ├── nn_cuda_tiled.cu     # Tiled matrix multiplication
    ├── benchmark_comparison.cu  # Comparison tool
    └── run_tiled.sh         # Execution script
```

---

## Solution Overview

### Baseline Strategy (Provided)
**Naive Element-per-Thread Matrix Multiplication**
- Each thread computes ONE element of the output matrix
- All memory accesses go to global memory (slow: ~400 cycles)
- No data reuse between threads
- Simple but inefficient

### Alternative Strategy (Proposed)
**Tiled Matrix Multiplication with Shared Memory**
- Threads cooperatively load data tiles into shared memory
- Data reused TILE_SIZE times (typically 16×)
- Dramatically reduces global memory traffic
- More complex but significantly faster

---

## Key Differences

| Aspect | Baseline | Alternative (Tiled) |
|--------|----------|---------------------|
| Memory Used | Global only | Global + Shared |
| Data Reuse | None | TILE_SIZE times |
| Memory Accesses | 2×M×K×N | 2×M×K×N / TILE_SIZE |
| Complexity | Simple | Moderate |
| Performance | Baseline | 1.5× - 3× faster |

---

## Step-by-Step Guide

### Step 1: Understand the Baseline
Read and understand `CODE_REFERENCE/nn_cuda.cu`:
- Focus on the `mat_mult_kernel` function
- Note how each thread computes C[row][col] independently
- Observe all memory reads from global memory

### Step 2: Compile and Run Baseline
```bash
cd CODE_REFERENCE
./run_cuda.sh
```

### Step 3: Run Alternative Strategy
```bash
cd ALTERNATIVE_STRATEGY
nvcc -o nn_cuda_tiled nn_cuda_tiled.cu -Xcompiler -fopenmp
./nn_cuda_tiled ../CODE_REFERENCE/data/synthetic_convex_large.csv
```

### Step 4: Run Benchmarks
```bash
cd ALTERNATIVE_STRATEGY
nvcc -o benchmark benchmark_comparison.cu
./benchmark
```

### Step 5: Collect Performance Data
Record execution times for:
- Small dataset (256 samples)
- Medium dataset (2560 samples)  
- Large dataset (25600 samples)

### Step 6: Write Your Article
Use `ARTICLE_TEMPLATE.md` as your starting point:
1. Fill in your team member names
2. Update experimental results with your actual measurements
3. Add analysis and observations
4. Ensure it's under 12 pages
5. Format as a proper scientific paper

---

## Expected Results

Based on the tiled optimization, you should observe:

| Dataset | Expected Speedup |
|---------|------------------|
| Small (256) | 1.3× - 1.5× |
| Medium (2560) | 1.6× - 2.0× |
| Large (25600) | 1.8× - 2.5× |

For isolated matrix multiplication kernels with square matrices:
| Matrix Size | Expected Speedup |
|-------------|------------------|
| 512×512 | 2.0× - 2.5× |
| 1024×1024 | 2.5× - 3.0× |
| 2048×2048 | 3.0× - 3.5× |

---

## Article Checklist

Your article must include:

- [ ] **Title** - Descriptive and specific
- [ ] **Abstract** - Objectives, methods, key results, conclusion (150-250 words)
- [ ] **Introduction** - Context, problem, objectives, contributions
- [ ] **Network Description** - Architecture, training algorithm, parameters
- [ ] **Baseline Analysis** - Strategy description, memory pattern, limitations
- [ ] **Alternative Strategy** - Detailed description, implementation, optimizations
- [ ] **Experimental Section** - Setup, methodology, results, analysis
- [ ] **Discussion** - Interpretation, limitations, recommendations
- [ ] **Conclusion** - Summary, future work
- [ ] **References** - APA format

---

## Common Issues & Solutions

### Compilation Errors
```bash
# Ensure CUDA toolkit is installed
nvcc --version

# Install if needed (Linux)
sudo apt install nvidia-cuda-toolkit
```

### No GPU Available
Use Google Colab with GPU runtime:
1. Go to runtime → Change runtime type → GPU
2. Upload your .cu files
3. Run with `!nvcc` command

### Low Speedup Observed
- Ensure you're timing only the kernel execution, not memory transfers
- Use CUDA events for precise timing
- Ensure matrices are large enough to benefit from tiling

---

## Files to Submit

**Only the article is required (as per project specifications):**
- `article.pdf` - Your final scientific article (max 12 pages)

**Note:** No code submission is required, but keep your implementation for verification if requested.

---

## Grading Focus

Based on the project requirements, focus on:

1. **Clear understanding** of both parallelization strategies
2. **Correct experimental methodology** with reproducible results
3. **Insightful analysis** of performance differences
4. **Professional scientific writing** following academic standards
5. **Proper citations** in APA format

---

## Timeline Recommendation

| Week | Task |
|------|------|
| 1 | Understand code, run baseline experiments |
| 2 | Implement alternative, run comparisons |
| 3 | Analyze results, create visualizations |
| 4 | Write article sections |
| 5 | Review, polish, finalize article |

**Due Date: February 5, 2026**

---

Good luck with your project!
