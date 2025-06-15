#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cuda_runtime.h>

__global__ void matrix_operations_parallel(
    double *a, double *b,
    double *add, double *sub,
    double *mul, double *div,
    int size
) {
    // 2D-индексация для параллельной обработки
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < size && col < size) {
        int idx = row * size + col;
        add[idx] = a[idx] + b[idx];
        sub[idx] = a[idx] - b[idx];
        mul[idx] = a[idx] * b[idx];
        div[idx] = (b[idx] != 0.0) ? (a[idx] / b[idx]) : 0.0;
    }
}

double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        printf("Usage: %s <matrix_size>\n", argv[0]);
        return 1;
    }

    int size = atoi(argv[1]);
    int total_elements = size * size;
    double start_time = get_time();

    // Выделение памяти на хосте
    double *h_a = (double *)malloc(total_elements * sizeof(double));
    double *h_b = (double *)malloc(total_elements * sizeof(double));
    double *h_add = (double *)malloc(total_elements * sizeof(double));
    double *h_sub = (double *)malloc(total_elements * sizeof(double));
    double *h_mul = (double *)malloc(total_elements * sizeof(double));
    double *h_div = (double *)malloc(total_elements * sizeof(double));

    // Инициализация матриц
    srand(time(NULL));
    for (int i = 0; i < total_elements; i++) {
        h_a[i] = (double)rand() / RAND_MAX * 100.0;
        h_b[i] = (double)rand() / RAND_MAX * 100.0;
    }

    // Выделение памяти на устройстве
    double *d_a, *d_b, *d_add, *d_sub, *d_mul, *d_div;
    cudaMalloc(&d_a, total_elements * sizeof(double));
    cudaMalloc(&d_b, total_elements * sizeof(double));
    cudaMalloc(&d_add, total_elements * sizeof(double));
    cudaMalloc(&d_sub, total_elements * sizeof(double));
    cudaMalloc(&d_mul, total_elements * sizeof(double));
    cudaMalloc(&d_div, total_elements * sizeof(double));

    // Копирование данных на устройство
    cudaMemcpy(d_a, h_a, total_elements * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, total_elements * sizeof(double), cudaMemcpyHostToDevice);

    // Конфигурация запуска ядра
    dim3 block_size(16, 16);
    dim3 grid_size((size + block_size.x - 1) / block_size.x,
                   (size + block_size.y - 1) / block_size.y);

    // Запуск параллельного ядра
    matrix_operations_parallel<<<grid_size, block_size>>>(d_a, d_b, d_add, d_sub, d_mul, d_div, size);

    // Синхронизация и проверка ошибок
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Копирование результатов обратно
    cudaMemcpy(h_add, d_add, total_elements * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_sub, d_sub, total_elements * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_mul, d_mul, total_elements * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_div, d_div, total_elements * sizeof(double), cudaMemcpyDeviceToHost);

    // Освобождение памяти
    free(h_a); free(h_b);
    free(h_add); free(h_sub); free(h_mul); free(h_div);
    cudaFree(d_a); cudaFree(d_b);
    cudaFree(d_add); cudaFree(d_sub); cudaFree(d_mul); cudaFree(d_div);

    double end_time = get_time();

    printf("TYPE: Parallel (CUDA)\n");
    printf("MATRIX_SIZE: %d\n", size);
    printf("GRID_SIZE: %dx%d\n", grid_size.x, grid_size.y);
    printf("BLOCK_SIZE: %dx%d\n", block_size.x, block_size.y);
    printf("TOTAL_THREADS: %d\n", grid_size.x * grid_size.y * block_size.x * block_size.y);
    printf("EXECUTION_TIME: %.6f sec\n", end_time - start_time);

    return 0;
}
