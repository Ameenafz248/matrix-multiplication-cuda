# Matrix Multiplication Performance Analysis

Compares the performance of matrix multiplication by:
- Sequential implementation which runs exclusively on a CPU.
- Parallel implementation using CUDA or HIP for a GPU but without
using shared memory.
- Parallel implementation using CUDA or HIP for a GPU while using
shared memory effectively

## How to run?

```sh
make #compiles the CUDA program and creates the executable out
./out
```