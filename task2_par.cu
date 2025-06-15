#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

__global__ void bitonic_sort_step(int *dev_values, int j, int k, int size) {
    unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i >= size) return;

    unsigned int ixj = i ^ j;

    if (ixj > i && ixj < size) {
        if ((i & k) == 0) {
            if (dev_values[i] > dev_values[ixj]) {
                int temp = dev_values[i];
                dev_values[i] = dev_values[ixj];
                dev_values[ixj] = temp;
            }
        } else {
            if (dev_values[i] < dev_values[ixj]) {
                int temp = dev_values[i];
                dev_values[i] = dev_values[ixj];
                dev_values[ixj] = temp;
            }
        }
    }
}

void bitonic_sort(int *host_array, int size) {
    int *dev_array;
    cudaMalloc((void**)&dev_array, size * sizeof(int));
    cudaMemcpy(dev_array, host_array, size * sizeof(int), cudaMemcpyHostToDevice);

    // Оптимальная конфигурация
    const int threads = 256;
    int blocks = (size + threads - 1) / threads;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int k = 2; k <= size; k *= 2) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            bitonic_sort_step<<<blocks, threads>>>(dev_array, j, k, size);

            // Проверка ошибок после каждого ядра
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                printf("Kernel Error: %s\n", cudaGetErrorString(err));
                break;
            }
        }
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Финализация
    cudaDeviceSynchronize();
    cudaError_t final_err = cudaGetLastError();
    if (final_err != cudaSuccess) {
        printf("Final CUDA Error: %s\n", cudaGetErrorString(final_err));
    }

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(host_array, dev_array, size * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(dev_array);

    printf("ВРЕМЯ_ВЫПОЛНЕНИЯ: %.6f сек\n", milliseconds / 1000.0f);
    printf("РЕЖИМ: Параллельный (Bitonic Sort)\n");
    printf("БЛОКОВ: %d, ПОТОКОВ: %d\n", blocks, threads);
    printf("ЭЛЕМЕНТОВ: %d\n", size);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    const char* env_size = getenv("ARRAY_SIZE");
    int N = env_size ? atoi(env_size) : 200000;
    int *arr = (int*)malloc(N * sizeof(int));

    srand(time(NULL));
    for (int i = 0; i < N; i++) {
        arr[i] = rand();
    }

    bitonic_sort(arr, N);

    free(arr);
    return 0;
}
