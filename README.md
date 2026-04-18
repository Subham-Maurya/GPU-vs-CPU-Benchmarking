# GPU vs CPU Benchmark Suite (CUDA)

## 🚀 Overview

This project benchmarks the performance difference between CPU and GPU implementations of common parallel workloads using NVIDIA CUDA. It serves as a practical study in heterogeneous computing — comparing serial CPU execution against massively parallel GPU execution across a variety of representative workloads.

The goal is to demonstrate how GPU acceleration improves performance for:
- **Compute-intensive tasks** — where arithmetic operations dominate execution time
- **Data-parallel workloads** — where the same operation is applied independently across large datasets
- **Memory-bound operations** — where throughput is limited by memory bandwidth rather than compute capacity

All benchmarks are implemented in a single CUDA C++ source file (`src/main.cu`) and compiled using `nvcc`, NVIDIA's CUDA compiler. Timing is measured using both `std::chrono` (for CPU) and CUDA Events (for GPU) to ensure high-resolution, accurate comparisons.

---

## 🧠 Benchmarks Implemented

### 1. Vector Addition
- Element-wise addition of two float arrays: `C[i] = A[i] + B[i]`
- Demonstrates basic parallelism — each thread independently computes one output element
- Serves as the "Hello World" of GPU computing

### 2. SAXPY (Single-Precision A·X Plus Y)
- A classic Level-1 BLAS operation: `y[i] = a * x[i] + y[i]`
- Scalar `a` is set to `2.5f` in the benchmark
- Shows memory-bound GPU performance — the bottleneck is reading and writing device memory, not compute throughput

### 3. Reduction (Sum)
- Computes the sum of all elements in an array using a parallel tree-based reduction
- Uses `__shared__` memory within each thread block to perform fast intra-block summation (`THREADS = 256` per block)
- Demonstrates how shared memory dramatically accelerates memory-bound reductions by minimizing redundant global memory accesses

### 4. Matrix Multiplication
- Dense square matrix multiplication: `C[i][j] = Σ A[i][k] * B[k][j]`
- Uses a 2D thread/block layout (`dim3 threads(16,16)`, `dim3 blocks(...)`)
- Compute-intensive workload with O(N³) operations — this is where GPU throughput advantages are most pronounced

---

## 📊 Performance Scaling

Each benchmark is evaluated at multiple input sizes to show how performance scales as the problem grows larger. Small inputs often favor the CPU (due to GPU kernel launch overhead), while large inputs reveal the true throughput advantage of the GPU.

### Vector Addition / SAXPY / Reduction
| Label | Elements      |
|-------|---------------|
| 1K    | 1,024         |
| 1M    | 1,048,576     |
| 16M   | 16,777,216    |

### Matrix Multiplication
| Label    | Dimensions |
|----------|------------|
| Small    | 64 × 64    |
| Medium   | 128 × 128  |
| Large    | 256 × 256  |

---

## ⚙️ GPU Implementation Details

- **Grid-stride execution**: CUDA kernels use standard `blockIdx.x * blockDim.x + threadIdx.x` indexing with bounds checking, allowing clean scaling across any grid size
- **Shared memory in reduction**: The `reduceSumGPU` kernel loads data into `__shared__ float sdata[THREADS]`, performs in-block tree reduction with `__syncthreads()` barriers, then writes each block's partial sum to global memory
- **CUDA Events for timing**: `cudaEventCreate`, `cudaEventRecord`, and `cudaEventElapsedTime` are used to time GPU kernels with sub-millisecond precision — these events operate on the GPU timeline directly, avoiding CPU-GPU synchronization artifacts
- **Warm-up kernels**: For Vector Addition, a warm-up kernel invocation is run before timing begins. This ensures that CUDA context initialization, JIT compilation, and driver overhead are excluded from measured results — giving a fair and reproducible GPU timing

---

## 📈 Sample Results

Results logged to `logs/execution_log.txt` via `run.sh`.

```
[Vector Add]
Size: 1024     | CPU: 0.008 ms  | GPU: 0.010 ms
Size: 1048576  | CPU: 8.52 ms   | GPU: 0.073 ms
Size: 16777216 | CPU: 125.6 ms  | GPU: 1.07 ms

[SAXPY]
Size: 1024     | CPU: 0.01 ms   | GPU: 0.19 ms
Size: 1048576  | CPU: 9.25 ms   | GPU: 0.19 ms
Size: 16777216 | CPU: 131 ms    | GPU: 1.18 ms

[Reduction]
Size: 1024     | CPU: 0.01 ms   | GPU: 0.17 ms
Size: 1048576  | CPU: 10.56 ms  | GPU: 0.23 ms
Size: 16777216 | CPU: 159.5 ms  | GPU: 1.42 ms

[Matrix Multiplication]
Size: 64x64    | CPU: 1.6 ms    | GPU: 0.044 ms
Size: 128x128  | CPU: 13.5 ms   | GPU: 0.017 ms
Size: 256x256  | CPU: 106 ms    | GPU: 0.23 ms
```

> **Note:** Results will vary depending on your specific GPU model, CPU, memory bandwidth, and system load at the time of execution.

---

## 🔍 Observations & Analysis

- **GPU performance improves significantly as input size increases** — the parallelism advantage only materializes once the workload is large enough to keep thousands of CUDA cores busy
- **For very small inputs (1K), CPU can outperform GPU** due to kernel launch overhead, memory transfer latency, and CUDA context setup costs that dwarf the actual computation time
- **Matrix multiplication shows the highest speedup** due to its high arithmetic intensity (O(N³) compute vs. O(N²) memory) — the GPU's thousands of cores can work in parallel on different output elements, yielding massive end-to-end gains
- **Reduction benefits from shared memory optimization** — by staging data in fast on-chip shared memory before performing the tree reduction, global memory accesses are minimized and throughput is greatly improved
- **Memory-bound operations (SAXPY) show moderate speedups** — because these kernels read/write large arrays with minimal arithmetic, performance is ultimately limited by memory bandwidth rather than pure compute, narrowing (but not eliminating) the GPU advantage

---

## 🛠 Requirements

- **NVIDIA GPU** with CUDA support (any Kepler-generation or newer card will work)
- **CUDA Toolkit** (>= 11.x recommended) — includes `nvcc`, device libraries, and headers
- **GCC / g++** — required by `nvcc` as the host compiler on Linux

### Check your installation

```bash
nvcc --version
nvidia-smi
```

---

## 🏗 Project Structure

```
GPU-Benchmarking-Project/
├── src/
│   └── main.cu          # All CUDA kernels and benchmark drivers
├── logs/
│   └── execution_log.txt  # Output from the last run (generated by run.sh)
├── Makefile             # Build rule: compiles src/main.cu → ./benchmark
└── run.sh               # Helper script: builds, runs, and logs output
```

---

## 🔧 Build & Run

### Using the Makefile (Linux / WSL)

```bash
# Build the benchmark binary
make

# Clean the compiled binary
make clean
```

### Using the run script

```bash
# Build, execute, and save output to logs/execution_log.txt
chmod +x run.sh
./run.sh
```

The `run.sh` script runs `make`, executes `./benchmark`, and pipes all output to `logs/execution_log.txt`, then prints it to the terminal.

---

## 📌 Notes

- All benchmarks use `float` (single-precision) arithmetic throughout
- Thread block size is fixed at `THREADS = 256` for 1D kernels and `16×16 = 256` for the 2D matrix kernel
- The benchmark binary is a self-contained executable with no external runtime dependencies beyond the CUDA runtime (`libcudart`)