#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

__global__ void parallel_sum_kernel(int *array, long long *result, int size) {
    extern __shared__ long long sdata[];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;

    long long local_sum = 0;
    while (i < size) {
        local_sum += array[i];
        i += blockDim.x * gridDim.x;
    }
    sdata[tid] = local_sum;
    __syncthreads();

    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        result[blockIdx.x] = sdata[0];
    }
}

int main() {
    const char *env_size = getenv("ARRAY_SIZE");
    int size = env_size ? atoi(env_size) : 200000;

    int threads_per_block = 256;
    int blocks_per_grid = (size + threads_per_block - 1) / threads_per_block;
    if (blocks_per_grid > 65536) blocks_per_grid = 65536;

    int *d_array;
    long long *d_block_sums;
    long long *h_block_sums = (long long*)malloc(blocks_per_grid * sizeof(long long));

    cudaMalloc((void**)&d_array, size * sizeof(int));
    cudaMalloc((void**)&d_block_sums, blocks_per_grid * sizeof(long long));

    int *h_array = (int*)malloc(size * sizeof(int));
    for (int i = 0; i < size; i++) {
        h_array[i] = i + 1;
    }

    cudaMemcpy(d_array, h_array, size * sizeof(int), cudaMemcpyHostToDevice);

    clock_t start = clock();

    parallel_sum_kernel<<<blocks_per_grid, threads_per_block, threads_per_block * sizeof(long long)>>>(d_array, d_block_sums, size);

    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    cudaMemcpy(h_block_sums, d_block_sums, blocks_per_grid * sizeof(long long), cudaMemcpyDeviceToHost);

    long long final_sum = 0;
    for (int i = 0; i < blocks_per_grid; i++) {
        final_sum += h_block_sums[i];
    }

    clock_t end = clock();
    double time_taken = (double)(end - start) / CLOCKS_PER_SEC;

    printf("Сумма массива: %lld\n", final_sum);
    printf("Время выполнения (CUDA parallel): %f секунд\n", time_taken);

    cudaFree(d_array);
    cudaFree(d_block_sums);
    free(h_array);
    free(h_block_sums);

    return 0;
}
