#include <stdio.h>
#include <stdlib.h>
#include <random>

#define ARR_SIZE_T 15
#define R1 4096
#define R2 8192
#define C1 8192
#define C2 2048


void matrix_multiply(int **A, int **B, int **C) {
    for (auto i = 0; i < R1; ++i) {
        for (auto j = 0; j < C2; ++j) {
          for (auto k = 0; k < R2; ++k) {
            C[i][j] += A[i][k] * B[k][j];
          }  
        }
    }
}

int main() {

    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_int_distribution<std::mt19937::result_type> distr(-999, 1000);

    int **A, **B, **C;

    /* initialize the arrays. */

    A = (int **)malloc(R1 * sizeof(int));
    for (auto i = 0; i < R1; ++i) {
        A[i] = (int *)malloc(C1 * sizeof(int));
        for (auto j = 0; j < C1; ++j) {
            A[i][j] = distr(rng);
        }
    }

    B = (int **)malloc(R2 * sizeof(int));
    for (auto i = 0; i < R2; ++i) {
        B[i] = (int *)malloc(C2 * sizeof(int));
        for (auto j = 0; j < C2; ++j) {
            B[i][j] = distr(rng);
        }
    }

    C = (int **)malloc(R1 * sizeof(int));
    for (auto i = 0; i < R1; ++i) {
        C[i] = (int *)malloc(C2 * sizeof(int));
        for (auto j = 0; j < C2; ++j) {
            C[i][j] = 0;
        }
    }

    matrix_multiply(A, B, C);
    
    

    return 0;
}
