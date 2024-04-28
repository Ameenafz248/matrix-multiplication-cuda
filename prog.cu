#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <vector>



/* loads data from to the shared memory of each block. */
template <size_t BLOCK_TILE_SIZE_X, size_t BLOCK_TILE_SIZE_Y,
          size_t BLOCK_TILE_SIZE_K, size_t NUM_THREADS, size_t BLOCK_TILE_SKEW_SIZE_X = 0U, size_t BLOCK_TILE_SKEW_SIZE_K = 0U>
__device__ void load_data_to_shared_memory(int const* A,
                                           int const* B,
                                           int A_thread_block_tile[BLOCK_TILE_SIZE_Y][BLOCK_TILE_SIZE_K + BLOCK_TILE_SKEW_SIZE_K],
                                           int B_thread_block_tile[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_X + BLOCK_TILE_SKEW_SIZE_X],
                                           size_t thread_block_tile_idx,
                                           size_t thread_linear_idx,
                                           size_t m, size_t n,
                                           size_t k)
{
#pragma unroll
    for (size_t load_idx{0U};
         load_idx <
         (BLOCK_TILE_SIZE_Y * BLOCK_TILE_SIZE_K + NUM_THREADS - 1U) /
             NUM_THREADS;
         ++load_idx)
    {
        size_t const A_thread_block_tile_row_idx{
            (thread_linear_idx + load_idx * NUM_THREADS) /
            BLOCK_TILE_SIZE_K};
        size_t const A_thread_block_tile_col_idx{
            (thread_linear_idx + load_idx * NUM_THREADS) %
            BLOCK_TILE_SIZE_K};
        size_t const A_row_idx{blockIdx.y * BLOCK_TILE_SIZE_Y +
                               A_thread_block_tile_row_idx};
        size_t const A_col_idx{thread_block_tile_idx * BLOCK_TILE_SIZE_K +
                               A_thread_block_tile_col_idx};

        int val{static_cast<int>(0)};
        if (A_row_idx < m && A_col_idx < k)
        {
            val = A[A_row_idx * k + A_col_idx];
        }
        static_assert(BLOCK_TILE_SIZE_K * BLOCK_TILE_SIZE_Y % NUM_THREADS ==
                      0U);
        A_thread_block_tile[A_thread_block_tile_row_idx]
                           [A_thread_block_tile_col_idx] = val;
    }
#pragma unroll
    for (size_t load_idx{0U};
         load_idx <
         (BLOCK_TILE_SIZE_K * BLOCK_TILE_SIZE_X + NUM_THREADS - 1U) /
             NUM_THREADS;
         ++load_idx)
    {
        size_t const B_thread_block_tile_row_idx{
            (thread_linear_idx + load_idx * NUM_THREADS) /
            BLOCK_TILE_SIZE_X};
        size_t const B_thread_block_tile_col_idx{
            (thread_linear_idx + load_idx * NUM_THREADS) %
            BLOCK_TILE_SIZE_X};
        size_t const B_row_idx{thread_block_tile_idx * BLOCK_TILE_SIZE_K +
                               B_thread_block_tile_row_idx};
        size_t const B_col_idx{blockIdx.x * BLOCK_TILE_SIZE_X +
                               B_thread_block_tile_col_idx};

        int val{static_cast<int>(0)};
        if (B_row_idx < k && B_col_idx < n)
        {
            val = B[B_row_idx * n + B_col_idx];
        }
        static_assert(BLOCK_TILE_SIZE_X * BLOCK_TILE_SIZE_K % NUM_THREADS ==
                      0U);
        B_thread_block_tile[B_thread_block_tile_row_idx]
                           [B_thread_block_tile_col_idx] = val;
    }
}


/* invokes load_data_to_shared_memory and finds out respective elements in the resultant matrix.  */
template <size_t BLOCK_TILE_SIZE_X, size_t BLOCK_TILE_SIZE_Y,
          size_t BLOCK_TILE_SIZE_K>
__global__ void gemm(size_t m, size_t n, size_t k, int const* A,
                         int const* B, int* C
                         )
{
    constexpr size_t NUM_THREADS{BLOCK_TILE_SIZE_X * BLOCK_TILE_SIZE_Y};
    size_t const thread_linear_idx{threadIdx.y * blockDim.x + threadIdx.x};

    size_t const C_col_idx{blockIdx.x * blockDim.x + threadIdx.x};
    size_t const C_row_idx{blockIdx.y * blockDim.y + threadIdx.y};

    __shared__ int A_thread_block_tile[BLOCK_TILE_SIZE_Y][BLOCK_TILE_SIZE_K];
    __shared__ int B_thread_block_tile[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_X];

    size_t const num_thread_block_tiles{(k + BLOCK_TILE_SIZE_K - 1) /
                                        BLOCK_TILE_SIZE_K};

    int sum{static_cast<int>(0)};
    for (size_t thread_block_tile_idx{0U};
         thread_block_tile_idx < num_thread_block_tiles;
         ++thread_block_tile_idx)
    {
        load_data_to_shared_memory<BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_Y,
                                   BLOCK_TILE_SIZE_K, NUM_THREADS>(
            A, B, A_thread_block_tile, B_thread_block_tile,
            thread_block_tile_idx, thread_linear_idx, m, n, k);
        __syncthreads();

#pragma unroll
        for (size_t k_i{0U}; k_i < BLOCK_TILE_SIZE_K; ++k_i)
        {
            sum += A_thread_block_tile[threadIdx.y][k_i] *
                   B_thread_block_tile[k_i][threadIdx.x];
        }
        __syncthreads();
    }
    if (C_row_idx < m && C_col_idx < n)
    {
        C[C_row_idx * n + C_col_idx] =
             sum;
    }
}


/* shared memory multiplication wrapper function with 128 threads per block. */
void gem_wrap1(size_t m, size_t n, size_t k,
                            int const* A, int const* B,
                            int* C)
                            
{
    constexpr unsigned int BLOCK_TILE_SIZE_X{16U};
    constexpr unsigned int BLOCK_TILE_SIZE_Y{8U};
    constexpr unsigned int BLOCK_TILE_SIZE_K{16U};
    constexpr unsigned int NUM_THREADS{BLOCK_TILE_SIZE_X * BLOCK_TILE_SIZE_Y};
    static_assert(BLOCK_TILE_SIZE_K * BLOCK_TILE_SIZE_Y % NUM_THREADS == 0U);
    static_assert(BLOCK_TILE_SIZE_X * BLOCK_TILE_SIZE_K % NUM_THREADS == 0U);
    dim3 const block_dim{BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_Y, 1U};
    dim3 const grid_dim{
        (static_cast<unsigned int>(n) + block_dim.x - 1U) / block_dim.x,
        (static_cast<unsigned int>(m) + block_dim.y - 1U) / block_dim.y, 1U};
    gemm<BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_Y, BLOCK_TILE_SIZE_K>
        <<<grid_dim, block_dim, 0U>>>(m, n, k, A, B,
                                               C);
}

/* shared memory multiplication wrapper function with 128 threads per block. */
void gem_wrap2(size_t m, size_t n, size_t k,
                            int const* A, int const* B,
                            int* C)
                            
{
    constexpr unsigned int BLOCK_TILE_SIZE_X{16U};
    constexpr unsigned int BLOCK_TILE_SIZE_Y{16U};
    constexpr unsigned int BLOCK_TILE_SIZE_K{16U};
    constexpr unsigned int NUM_THREADS{BLOCK_TILE_SIZE_X * BLOCK_TILE_SIZE_Y};
    static_assert(BLOCK_TILE_SIZE_K * BLOCK_TILE_SIZE_Y % NUM_THREADS == 0U);
    static_assert(BLOCK_TILE_SIZE_X * BLOCK_TILE_SIZE_K % NUM_THREADS == 0U);
    dim3 const block_dim{BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_Y, 1U};
    dim3 const grid_dim{
        (static_cast<unsigned int>(n) + block_dim.x - 1U) / block_dim.x,
        (static_cast<unsigned int>(m) + block_dim.y - 1U) / block_dim.y, 1U};
    gemm<BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_Y, BLOCK_TILE_SIZE_K>
        <<<grid_dim, block_dim, 0U>>>(m, n, k, A, B,
                                               C);
}

/* shared memory multiplication wrapper function with 128 threads per block. */
void gem_wrap3(size_t m, size_t n, size_t k,
                            int const* A, int const* B,
                            int* C)
                            
{
    constexpr unsigned int BLOCK_TILE_SIZE_X{32U};
    constexpr unsigned int BLOCK_TILE_SIZE_Y{16U};
    constexpr unsigned int BLOCK_TILE_SIZE_K{32U};
    constexpr unsigned int NUM_THREADS{BLOCK_TILE_SIZE_X * BLOCK_TILE_SIZE_Y};
    static_assert(BLOCK_TILE_SIZE_K * BLOCK_TILE_SIZE_Y % NUM_THREADS == 0U);
    static_assert(BLOCK_TILE_SIZE_X * BLOCK_TILE_SIZE_K % NUM_THREADS == 0U);
    dim3 const block_dim{BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_Y, 1U};
    dim3 const grid_dim{
        (static_cast<unsigned int>(n) + block_dim.x - 1U) / block_dim.x,
        (static_cast<unsigned int>(m) + block_dim.y - 1U) / block_dim.y, 1U};
    gemm<BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_Y, BLOCK_TILE_SIZE_K>
        <<<grid_dim, block_dim, 0U>>>(m, n, k, A, B,
                                               C);
}



void cpu_matrix_mult(int *h_a, int *h_b, int *h_result, int m, int k, int n) {
    for (int i = 0; i < m; ++i) 
    {
        for (int j = 0; j < n; ++j) 
        {
            int tmp = 0.0;
            for (int h = 0; h < k; ++h) 
            {
                tmp += h_a[i * k + h] * h_b[h * n + j];
            }
            h_result[i * n + j] = tmp;
        }
    }
}


__global__ void gpu_matrix_mult(int *a,int *b, int *c, int m, int k, int n)
{ 
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int sum = 0;
    if( col < n && row < m) 
    {
        for(int i = 0; i < k; i++) 
        {
            sum += a[row * k + i] * b[i * n + col];
        }
        c[row * n + col] = sum;
    }
} 



int main() {
        int m = 4096, k = 8192, n = 2048;
    srand(3333);

    void (*shared_functions[3])(size_t, size_t, size_t, int const*, int const*, int*) = { &gem_wrap1, &gem_wrap2, &gem_wrap3 };

    // allocate memory in host RAM, h_cc is used to store CPU result
    int *h_a, *h_b, *h_c, *h_cc;
    cudaMallocHost((void **) &h_a, sizeof(int)*m*k);
    cudaMallocHost((void **) &h_b, sizeof(int)*k*n);
    cudaMallocHost((void **) &h_c, sizeof(int)*m*n);
    cudaMallocHost((void **) &h_cc, sizeof(int)*m*n);

    // random initialize matrix A
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < k; ++j) {
            h_a[i * k + j] = rand() % 1024;
        }
    }

    // random initialize matrix B
    for (int i = 0; i < k; ++i) {
        for (int j = 0; j < n; ++j) {
            h_b[i * n + j] = rand() % 1024;
        }
    }

	float gpu_unshared_elapsed_time_ms[3], gpu_shared_elapsed_time_ms[3], cpu_elapsed_time_ms;

    // some events to count the execution time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // start the CPU version
    cudaEventRecord(start, 0);

    cpu_matrix_mult(h_a, h_b, h_cc, m, k, n);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&cpu_elapsed_time_ms, start, stop);
    printf("Time elapsed on matrix multiplication on CPU: %f ms.\n\n", cpu_elapsed_time_ms);
	

	std::vector<std::pair<int, int>> thread_dims{{16,8}, {16,16}, {32, 16}};
	int idx = 0;
    int all_ok = 1;
	printf("Time elapsed for unshared memory:\n\n");
	for (std::pair<int, int> thread_dim : thread_dims) {
		// start to count execution time of GPU version
		cudaEventRecord(start, 0);
		// Allocate memory space on the device 

		int *d_a, *d_b, *d_c;
		cudaMalloc((void **) &d_a, sizeof(int)*m*k);
		cudaMalloc((void **) &d_b, sizeof(int)*k*n);
		cudaMalloc((void **) &d_c, sizeof(int)*m*n);

		// copy matrix A and B from host to device memory
		cudaMemcpy(d_a, h_a, sizeof(int)*m*k, cudaMemcpyHostToDevice);
		cudaMemcpy(d_b, h_b, sizeof(int)*k*n, cudaMemcpyHostToDevice);

		unsigned int grid_rows = (m + thread_dim.second - 1) / thread_dim.second;
		unsigned int grid_cols = (n + thread_dim.first - 1) / thread_dim.first;
		dim3 dimGrid(grid_cols, grid_rows);
		dim3 dimBlock(thread_dim.first, thread_dim.second);
	   

		gpu_matrix_mult<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, m, k, n);    

		// Transefr results from device to host 
		cudaMemcpy(h_c, d_c, sizeof(int)*m*n, cudaMemcpyDeviceToHost);
		cudaThreadSynchronize();
		// time counting terminate
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);

		// compute time elapse on GPU computing
		cudaEventElapsedTime(&gpu_unshared_elapsed_time_ms[idx], start, stop);
		printf("\tTime elapsed on GPU with %d threads/block: %f ms.\n\n", thread_dim.first * thread_dim.second,  gpu_unshared_elapsed_time_ms[idx]);
		
		// free memory
		cudaFree(d_a);
		cudaFree(d_b);
		cudaFree(d_c);
		idx++;

        for (int i = 0; i < m; ++i)
        {
            for (int j = 0; j < n; ++j)
            {
                // printf("[%d][%d]:%d == [%d][%d]:%d, ", i, j, h_cc[i*n + j], i, j, h_c[i*n + j]);
                if(h_cc[i*n + j] != h_c[i*n + j])
                {
                    all_ok = 0;
                }
            }
            //printf("\n");
        }

	}

	printf("Time elapsed for shared memory:\n\n");
    for (auto i = 0;i < 3; ++i){
		// start to count execution time of GPU version
		cudaEventRecord(start, 0);
		// Allocate memory space on the device 

		int *d_a, *d_b, *d_c;
		cudaMalloc((void **) &d_a, sizeof(int)*m*k);
		cudaMalloc((void **) &d_b, sizeof(int)*k*n);
		cudaMalloc((void **) &d_c, sizeof(int)*m*n);

		// copy matrix A and B from host to device memory
		cudaMemcpy(d_a, h_a, sizeof(int)*m*k, cudaMemcpyHostToDevice);
		cudaMemcpy(d_b, h_b, sizeof(int)*k*n, cudaMemcpyHostToDevice);

	   

        shared_functions[i]( m, n, k, d_a, d_b, d_c);

		cudaMemcpy(h_c, d_c, sizeof(int)*m*n, cudaMemcpyDeviceToHost);
		cudaThreadSynchronize();
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);

		cudaEventElapsedTime(&gpu_shared_elapsed_time_ms[i], start, stop);
		printf("\tTime elapsed on GPU with %d threads/block: %f ms.\n\n", thread_dims[i].first * thread_dims[i].second,  gpu_shared_elapsed_time_ms[i]);
		
		// free memory
		cudaFree(d_a);
		cudaFree(d_b);
		cudaFree(d_c);

        for (int i = 0; i < m; ++i)
        {
            for (int j = 0; j < n; ++j)
            {
                // printf("[%d][%d]:%d == [%d][%d]:%d, ", i, j, h_cc[i*n + j], i, j, h_c[i*n + j]);
                if(h_cc[i*n + j] != h_c[i*n + j])
                {
                    all_ok = 0;
                }
            }
            //printf("\n");
        }

	}


    // roughly compute speedup

    
    if(all_ok)
    {
        printf("all results are correct!\n");
		printf("unshared speedup:\n");
		for (auto i = 0; i < 3; ++i) {
				
			printf("\t%.0f: %f\n", std::pow(2, 7 + i), cpu_elapsed_time_ms / gpu_unshared_elapsed_time_ms[i]);
		}
		printf("shared speedup:\n");
		for (auto i = 0; i < 3; ++i) {
				
			printf("\t%.0f: %f\n", std::pow(2, 7 + i), cpu_elapsed_time_ms / gpu_shared_elapsed_time_ms[i]);
		}
    }
    else
    {
        printf("incorrect results\n");
    }


    cudaFreeHost(h_a);
    cudaFreeHost(h_b);
    cudaFreeHost(h_c);
    cudaFreeHost(h_cc);
    return 0;
}