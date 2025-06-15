#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

// Последовательная сортировка слиянием (оптимизированная)
__global__ void sequentialMergeSort(int *dev_array, int *temp, int size) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        for (int width = 1; width < size; width *= 2) {
            for (int i = 0; i < size; i += 2 * width) {
                int left = i;
                int mid = min(i + width, size);
                int right = min(i + 2 * width, size);

                int i1 = left;
                int i2 = mid;
                int j = left;

                while (i1 < mid && i2 < right) {
                    if (dev_array[i1] <= dev_array[i2]) {
                        temp[j++] = dev_array[i1++];
                    } else {
                        temp[j++] = dev_array[i2++];
                    }
                }

                while (i1 < mid) temp[j++] = dev_array[i1++];
                while (i2 < right) temp[j++] = dev_array[i2++];

                for (int k = left; k < right; k++) {
                    dev_array[k] = temp[k];
                }
            }
        }
    }
}

int main() {
    const char* env_size = getenv("ARRAY_SIZE");
    int N = env_size ? atoi(env_size) : 200000;
    int *h_array = (int*)malloc(N * sizeof(int));
    int *d_array, *d_temp;

    // Инициализация
    srand(time(NULL));
    for (int i = 0; i < N; i++) h_array[i] = rand();

    // Выделение памяти на GPU
    cudaMalloc((void**)&d_array, N * sizeof(int));
    cudaMalloc((void**)&d_temp, N * sizeof(int));
    cudaMemcpy(d_array, h_array, N * sizeof(int), cudaMemcpyHostToDevice);

    // Конфигурация запуска
    dim3 blocks(1);
    dim3 threads(1);

    // Точный замер времени
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    sequentialMergeSort<<<blocks, threads>>>(d_array, d_temp, N);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Проверка ошибок
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Замер времени
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("ВРЕМЯ_ВЫПОЛНЕНИЯ: %.6f сек\n", milliseconds / 1000.0f);
    printf("РЕЖИМ: Последовательный (1 поток CUDA)\n");
    printf("ЭЛЕМЕНТОВ: %d\n", N);

    // Освобождение ресурсов
    cudaFree(d_array);
    cudaFree(d_temp);
    free(h_array);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
