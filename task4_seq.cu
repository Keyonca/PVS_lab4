#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cuda_runtime.h>

__global__ void matrix_operations_sequential(
    double *a, double *b,
    double *add, double *sub,
    double *mul, double *div,
    int total_elements
) {
    // Один поток обрабатывает все элементы
    for (int idx = 0; idx < total_elements; idx++) {
        add[idx] = a[idx] + b[idx];
        sub[idx] = a[idx] - b[idx];
        mul[idx] = a[idx] * b[idx];
        if (b[idx] != 0.0) {
            div[idx] = a[idx] / b[idx];
        } else {
            div[idx] = 0.0;
        }
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

    // Запуск ядра с одним потоком
    matrix_operations_sequential<<<1, 1>>>(d_a, d_b, d_add, d_sub, d_mul, d_div, total_elements);

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

    printf("TYPE: Sequential (CUDA)\n");
    printf("MATRIX_SIZE: %d\n", size);
    printf("EXECUTION_TIME: %.6f sec\n", end_time - start_time);

    return 0;
}
