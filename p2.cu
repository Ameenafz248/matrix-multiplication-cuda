/*
 *  file name: matrix.cu
 *
 *  matrix.cu contains the code that realize some common used matrix operations in CUDA
 *  
 *  this is a toy program for learning CUDA, some functions are reusable in other project
 *  
 */
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <vector>


/*
*********************************************************************
function name: gpu_matrix_mult

description: dot product of two matrix (not only square)

parameters: 
            &a GPU device pointer to a m X n matrix (A)
            &b GPU device pointer to a n X k matrix (B)
            &c GPU device output purpose pointer to a m X k matrix (C) 
            to store the result

Note:
    grid and block should be configured as:
        dim3 dimGrid((k + BLOCK_SIZE - 1) / BLOCK_SIZE, (m + BLOCK_SIZE - 1) / BLOCK_SIZE);
        dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

    further sppedup can be obtained by using shared memory to decrease global memory access times
return: none
*********************************************************************
*/

__global__ void gpu_matrix_mult(int *a,int *b, int *c, int m, int n, int k)
{ 
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int sum = 0;
    if( col < k && row < m) 
    {
        for(int i = 0; i < n; i++) 
        {
            sum += a[row * n + i] * b[i * k + col];
        }
        c[row * k + col] = sum;
    }
} 



void cpu_matrix_mult(int *h_a, int *h_b, int *h_result, int m, int n, int k) {
    for (int i = 0; i < m; ++i) 
    {
        for (int j = 0; j < k; ++j) 
        {
            int tmp = 0.0;
            for (int h = 0; h < n; ++h) 
            {
                tmp += h_a[i * n + h] * h_b[h * k + j];
            }
            h_result[i * k + j] = tmp;
        }
    }
}



int main(int argc, char const *argv[])
{
    int m = 512, n = 1024, k = 256;
    /* Fixed seed for illustration */
    srand(3333);


    // allocate memory in host RAM, h_cc is used to store CPU result
    int *h_a, *h_b, *h_c, *h_cc;
    cudaMallocHost((void **) &h_a, sizeof(int)*m*n);
    cudaMallocHost((void **) &h_b, sizeof(int)*n*k);
    cudaMallocHost((void **) &h_c, sizeof(int)*m*k);
    cudaMallocHost((void **) &h_cc, sizeof(int)*m*k);

    // random initialize matrix A
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            h_a[i * n + j] = rand() % 1024;
        }
    }

    // random initialize matrix B
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < k; ++j) {
            h_b[i * k + j] = rand() % 1024;
        }
    }

	float gpu_unshared_elapsed_time_ms[3], gpu_shared_elapsed_time_ms[3], cpu_elapsed_time_ms;

    // some events to count the execution time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
	

	int block_size = 128;
	std::vector<std::pair<int, int>> thread_dims{{16,8}, {16,16}, {32, 16}};
	int idx = 0;
	printf("Time elapsed for unshared memory:\n\n");
	for (std::pair<int, int> thread_dim : thread_dims) {
		// start to count execution time of GPU version
		cudaEventRecord(start, 0);
		// Allocate memory space on the device 

		int *d_a, *d_b, *d_c;
		cudaMalloc((void **) &d_a, sizeof(int)*m*n);
		cudaMalloc((void **) &d_b, sizeof(int)*n*k);
		cudaMalloc((void **) &d_c, sizeof(int)*m*k);

		// copy matrix A and B from host to device memory
		cudaMemcpy(d_a, h_a, sizeof(int)*m*n, cudaMemcpyHostToDevice);
		cudaMemcpy(d_b, h_b, sizeof(int)*n*k, cudaMemcpyHostToDevice);

		unsigned int grid_rows = (m + thread_dim.second - 1) / thread_dim.second;
		unsigned int grid_cols = (k + thread_dim.first - 1) / thread_dim.first;
		dim3 dimGrid(grid_cols, grid_rows);
		dim3 dimBlock(thread_dim.first, thread_dim.second);
	   

		gpu_matrix_mult<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, m, n, k);    


		// Transefr results from device to host 
		cudaMemcpy(h_c, d_c, sizeof(int)*m*k, cudaMemcpyDeviceToHost);
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

	}


    // start the CPU version
    cudaEventRecord(start, 0);

    cpu_matrix_mult(h_a, h_b, h_cc, m, n, k);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&cpu_elapsed_time_ms, start, stop);
    printf("Time elapsed on matrix multiplication on CPU: %f ms.\n\n", cpu_elapsed_time_ms);

    // validate results computed by GPU
    int all_ok = 1;
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < k; ++j)
        {
            //printf("[%d][%d]:%d == [%d][%d]:%d, ", i, j, h_cc[i*k + j], i, j, h_c[i*k + j]);
            if(h_cc[i*k + j] != h_c[i*k + j])
            {
                all_ok = 0;
            }
        }
        //printf("\n");
    }

    // roughly compute speedup
    if(all_ok)
    {
        printf("all results are correct!\n");
		printf("unshared speedup:\n");
		for (auto i = 0; i < 3; ++i) {
				
			printf("\t%f: %f\n", std::pow(2, 7 + i), cpu_elapsed_time_ms / gpu_unshared_elapsed_time_ms[i]);
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
