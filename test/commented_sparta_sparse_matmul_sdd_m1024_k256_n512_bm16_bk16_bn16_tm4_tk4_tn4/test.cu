


#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <assert.h>
#include <cstring>
#include <fstream>
#include <iostream>

using namespace std;

#define checkCudaErrors(func) {                                                              \
    cudaError_t e = (func);                                                                  \
    if (e != cudaSuccess)                                                                    \
        fprintf(stderr, "%s %d CUDA: %s\\n", __FILE__,  __LINE__, cudaGetErrorString(e));    \
}

template <typename T>
void loadArr(string filepath, T *__restrict__ arr, int size) {
    ifstream inputFile(filepath);
    inputFile.read((char*)arr, size * sizeof(T));
}

template <typename T>
int getArrLength(string filepath) {
    ifstream inputFile(filepath);
    inputFile.seekg(0, inputFile.end);
    return inputFile.tellg() / sizeof(T);
}




const int K = 256;
const int N = 512;
const int BM = 16;
const int BK = 16;
const int BN = 16;
const int TM = 4;
const int TK = 4;
const int TN = 4;

__global__ void BLOCK_SPARSE_MATMUL(
    float* input_A_val,
    int* input_A_block_ptr,
    int* input_A_block_idx,
    float* input_B,
    
    float* output_C
) {
    float * A_val = reinterpret_cast<float*>(input_A_val);
    int * A_block_ptr = reinterpret_cast<int*>(input_A_block_ptr);
    int * A_block_idx = reinterpret_cast<int*>(input_A_block_idx);
    float * B = reinterpret_cast<float*>(input_B);
    
    float * C = reinterpret_cast<float*>(output_C);

    int by = blockIdx.y;
    int bx = blockIdx.x;
    int ty = threadIdx.y;
    int tx = threadIdx.x;

    __shared__ float As[BM * BK];
    __shared__ float Bs[BN * BK];

    float accum[TN][TM] = {0};
    float a_frag[TM][TK];
    float b_frag[TN][TK];

    int A_THREAD_PER_ROW = BK / 4;
    int B_THREAD_PER_ROW = BN / 4;

    int bszy = BM / TM;
    int bszx = BN / TN;

    int THREADS_PER_BLOCK = bszy * bszx;

    int A_TILE_ROW_STRIDE = THREADS_PER_BLOCK / A_THREAD_PER_ROW;
    int B_TILE_ROW_STRIDE = THREADS_PER_BLOCK / B_THREAD_PER_ROW;

    int tid = ty * bszx + tx;

    int A_BLOCK_ROW_START = tid / A_THREAD_PER_ROW;
    int B_BLOCK_ROW_START = tid / B_THREAD_PER_ROW;

    int A_BLOCK_COL_START = tid % A_THREAD_PER_ROW * 4;
    int B_BLOCK_COL_START = tid % B_THREAD_PER_ROW * 4;

    int index_start = A_block_ptr[by], index_end = A_block_ptr[by+1];

    const int vBLOCK_SIZE_M = BM / TM;
    const int vBLOCK_SIZE_N = BN / TN;
    
    for (int tile_block_idx = index_start; tile_block_idx < index_end; tile_block_idx += 1) {
        int tile_idx = A_block_idx[tile_block_idx] * BK;

        // Store tiles of A/B into As, Bs with vectorized load of float4 into __shared__ memory (local to block)
        #pragma unroll
        for (int k = 0; k < BM; k += A_TILE_ROW_STRIDE) {
            *((float4 *)(&As[(k+A_BLOCK_ROW_START) * BK + A_BLOCK_COL_START])) =
                
                *((float4 *)(&A_val[(by*BM+k+A_BLOCK_ROW_START) * K + tile_idx+A_BLOCK_COL_START]));
                
        }

        #pragma unroll
        for (int k = 0; k < BK; k += B_TILE_ROW_STRIDE) {
            
            *((float4 *)(&Bs[(k+B_BLOCK_ROW_START) * BN + B_BLOCK_COL_START])) =
                *((float4 *)(&B[(tile_idx + B_BLOCK_ROW_START+k) * N + bx*BN + B_BLOCK_COL_START]));
            
        }

        __syncthreads();

        // Store fragments of As/Bs into thread-local array a_frag, b_frag and then run multiply-adds.
        #pragma unroll
        for (int k = 0; k < BK; k += TK) {
            #pragma unroll
            for (int i = 0; i < TK; i++) {
                #pragma unroll
                for (int j = 0; j < TM; j += 1) {
                    a_frag[j][i] = As[(ty + vBLOCK_SIZE_M * j) * BK + k + i];
                }
            }

            #pragma unroll
            for (int i = 0; i < TK; i++) {
                #pragma unroll
                for (int j = 0; j < TN; j += 1) {
                    b_frag[j][i] = Bs[(k + i) * BN + tx + vBLOCK_SIZE_N * j];
                }
            }

            #pragma unroll
            for (int i = 0; i < TN; i++) {
                #pragma unroll
                for (int j = 0; j < TM; j++) {
                    #pragma unroll
                    for (int k_in = 0; k_in < TK; k_in++) {
                        accum[i][j] += a_frag[j][k_in] * b_frag[i][k_in];
                    }
                }
            }
        }

        __syncthreads();
    }

    
    // Store values of accum into C.
    #pragma unroll
    for (int thread_x = 0; thread_x < TN; thread_x++) {
        #pragma unroll
        for (int thread_y = 0; thread_y < TM; thread_y+=1) {
            C[(BM * by + ty + thread_y * vBLOCK_SIZE_M) * N + BN * bx + tx + thread_x * vBLOCK_SIZE_N] =
                (accum[thread_x][thread_y]);
        }
    }
}

int main(int argc, char** argv) { // iteration number, correction check
    const int nWarm = argc > 1 ? atol(argv[1]) : 10;
    const int nIter = argc > 2 ? atol(argv[2]) : 10;
    const int checkResults = argc > 3 ? atol(argv[3]) : 1;


    int size_A_val = getArrLength<float>("./A_val.dat");
    int mem_size_A_val = sizeof(float) * size_A_val;
    float* h_A_val = (float*)malloc(mem_size_A_val);
    loadArr<float>("./A_val.dat", h_A_val, size_A_val);
    float* d_A_val;
    checkCudaErrors(cudaMalloc(&d_A_val, mem_size_A_val));
    checkCudaErrors(cudaMemcpy(d_A_val, h_A_val, mem_size_A_val, cudaMemcpyHostToDevice));

    int size_A_row_ptr = getArrLength<int>("./A_row_ptr.dat");
    int mem_size_A_row_ptr = sizeof(int) * size_A_row_ptr;
    int* h_A_row_ptr = (int*)malloc(mem_size_A_row_ptr);
    loadArr<int>("./A_row_ptr.dat", h_A_row_ptr, size_A_row_ptr);
    int* d_A_row_ptr;
    checkCudaErrors(cudaMalloc(&d_A_row_ptr, mem_size_A_row_ptr));
    checkCudaErrors(cudaMemcpy(d_A_row_ptr, h_A_row_ptr, mem_size_A_row_ptr, cudaMemcpyHostToDevice));

    int size_A_col_idx = getArrLength<int>("./A_col_idx.dat");
    int mem_size_A_col_idx = sizeof(int) * size_A_col_idx;
    int* h_A_col_idx = (int*)malloc(mem_size_A_col_idx);
    loadArr<int>("./A_col_idx.dat", h_A_col_idx, size_A_col_idx);
    int* d_A_col_idx;
    checkCudaErrors(cudaMalloc(&d_A_col_idx, mem_size_A_col_idx));
    checkCudaErrors(cudaMemcpy(d_A_col_idx, h_A_col_idx, mem_size_A_col_idx, cudaMemcpyHostToDevice));

    int size_B = getArrLength<float>("./B.dat");
    int mem_size_B = sizeof(float) * size_B;
    float* h_B = (float*)malloc(mem_size_B);
    loadArr<float>("./B.dat", h_B, size_B);
    float* d_B;
    checkCudaErrors(cudaMalloc(&d_B, mem_size_B));
    checkCudaErrors(cudaMemcpy(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice));



    int size_C = getArrLength<float>("./C.dat");
    int mem_size_C = sizeof(float) * size_C;
    float* h_C_tgt = (float*)malloc(mem_size_C);
    if (checkResults) {
        loadArr<float>("./C.dat", h_C_tgt, size_C);
    }
    float* d_C;
    checkCudaErrors(cudaMalloc(&d_C, mem_size_C));


    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    float msecTotal = 0;

    const dim3 dimBlock(4, 4);
    const dim3 dimGrid(32, 64);

    // warm-up
    for (int run = 0; run < nWarm; run++) {
        BLOCK_SPARSE_MATMUL<<<dimGrid, dimBlock>>>(
            d_A_val, d_A_row_ptr, d_A_col_idx, d_B,
            d_C
        );
    }

    checkCudaErrors(cudaEventRecord(start));
    for (int run = 0; run < nIter; run++) {
        BLOCK_SPARSE_MATMUL<<<dimGrid, dimBlock>>>(
            d_A_val, d_A_row_ptr, d_A_col_idx, d_B,
            d_C
        );
    }

    checkCudaErrors(cudaEventRecord(stop));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));
    float msecPerMatrixMul = msecTotal / nIter;
    fprintf(stdout, "%f", msecPerMatrixMul);

    double abs_eps = 1e-5;
    double rel_eps = 1e-4;


    float* h_C = (float*)malloc(mem_size_C);
    checkCudaErrors(cudaMemcpy(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost));
    if (checkResults) {
        for (int i = 0; i < size_C; i++) {
            float result = (float)h_C[i];
            float target = (float)h_C_tgt[i];
            float abs_err = abs(result - target);
            float rel_err = abs_err / abs(target);
            if (abs_err > abs_eps && rel_err > rel_eps) {
                fprintf(stderr, "Error on C[%d]: ", i);
                fprintf(stderr, "result=%f, target=%f, abs_err=%f, rel_err=%f\n", result, target, abs_err, rel_err);
                break;
            }
        }
    }



    cudaFree(d_A_val);
    free(h_A_val);

    cudaFree(d_A_row_ptr);
    free(h_A_row_ptr);

    cudaFree(d_A_col_idx);
    free(h_A_col_idx);

    cudaFree(d_B);
    free(h_B);



    cudaFree(d_C);
    free(h_C_tgt);
    free(h_C);

}