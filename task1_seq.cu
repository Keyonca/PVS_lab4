#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

__global__ void sequential_sum_kernel(int *array, long long *result, int size) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        long long sum = 0;
        for (int i = 0; i < size; i++) {
            sum += array[i];
        }
        *result = sum;
    }
}

int main() {
    const char *env_size = getenv("ARRAY_SIZE");
    int size = env_size ? atoi(env_size) : 200000;

    int *d_array;
    long long *d_result;
    long long h_result = 0;

    cudaMalloc((void**)&d_array, size * sizeof(int));
    cudaMalloc((void**)&d_result, sizeof(long long));

    // Инициализация массива на хосте
    int *h_array = (int*)malloc(size * sizeof(int));
    for (int i = 0; i < size; i++) {
        h_array[i] = i + 1;
    }

    cudaMemcpy(d_array, h_array, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_result, &h_result, sizeof(long long), cudaMemcpyHostToDevice);

    clock_t start = clock();

    // Запуск ядра (1 блок, 1 нить)
    sequential_sum_kernel<<<1, 1>>>(d_array, d_result, size);

    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    cudaMemcpy(&h_result, d_result, sizeof(long long), cudaMemcpyDeviceToHost);

    clock_t end = clock();
    double time_taken = (double)(end - start) / CLOCKS_PER_SEC;

    printf("Сумма массива: %lld\n", h_result);
    printf("Время выполнения (CUDA sequential): %f секунд\n", time_taken);

    cudaFree(d_array);
    cudaFree(d_result);
    free(h_array);

    return 0;
}
