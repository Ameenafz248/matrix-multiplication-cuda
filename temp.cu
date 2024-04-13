#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <random>
#include <iostream>
#include <cuda.h>

#define R1 4096
#define R2 8192
#define C1 8192
#define C2 2048



__global__ void matrix_multiply(int *A, int *B, int *C) {

        int column = blockDim.x * blockIdx.x + threadIdx.x;
        int row = blockDim.y * blockIdx.y + threadIdx.y;

        if (row < R1 && column < C2) {
            int sum = 0;
            for (auto i =0; i < R2; ++i) {
                sum += A[row * C1 + i] * B[i * C2 + column];
            }
            C[row * C2 + column] = sum;
        }
}

int main() {

    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_int_distribution<std::mt19937::result_type> distr(-999, 1000);

    int **A, **B, **C;

    /* initialize the arrays. */

    if ( (A = new int *[R1]) == nullptr) {
        std::cerr << "out of memory\n";
    }

    for (auto i = 0; i < R1; ++i) {
        if ( (A[i] = new int [C1]) == nullptr) {
            std::cerr << "out of memory\n";
        }
        for (auto j = 0; j < C1; ++j) {
            A[i][j] = distr(rng);
        }
    }

    B = new int *[R2];
    for (auto i = 0; i < R2; ++i) {
        if ( (B[i] = new int [C2]) == nullptr) {
            std::cerr << "out of memory\n";
        }
        for (auto j = 0; j < C2; ++j) {
            B[i][j] = distr(rng);
        }
    }

    if ( (C = new int *[R1]) == nullptr) {
        std::cerr << "out of memory\n";
    }
    for (auto i = 0; i < R1; ++i) {
        if ( (C[i] = new int [C2]) == nullptr) {
            std::cerr << "out of memory\n";
        }
        for (auto j = 0; j < C2; ++j) {
            C[i][j] = 0;
        }
    }


    int *dA, *dB, *dC;

    cudaMalloc((void **)&dA, R1 * C1 * sizeof(int));
    cudaMalloc((void **)&dB, R2 * C2 * sizeof(int));
    cudaMalloc((void **)&dC, R1 * C2 * sizeof(int));


    cudaMemcpy(dA, A, R1 * C1 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, R2 * C2 * sizeof(int), cudaMemcpyHostToDevice);
    


    dim3 threads_per_block( 16, 16, 1 );
    dim3 blocks_in_grid( ceil( float(C2) / threads_per_block.x ), ceil( float(R1) / threads_per_block.y ), 1 );


    matrix_multiply<<<blocks_in_grid,threads_per_block>>>(dA, dB, dC);
    cudaMemcpy(C, dC, R1 * C2 * sizeof(int), cudaMemcpyDeviceToHost);


     for (auto i = 0; i < R1; ++i) {
         for (auto j = 0; j < C2; ++j) {
            std::cout << C[i][j] << "\n"; 
         }
     }

    for (auto i = 0; i < R1; ++i) {
        delete[] A[i];
        delete[] C[i];
    }
    delete[] A;
    delete[] C;

    for (auto i = 0; i < R2; ++i) {
        delete[] B[i];
    }
    delete[] B;

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    return 0;

}
