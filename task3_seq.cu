#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void array_ops_sequential(double* a, double* b, double* sum,
                                    double* diff, double* prod, double* div, int n) {
    for(int i = 0; i < n; i++) {
        sum[i] = a[i] + b[i];
        diff[i] = a[i] - b[i];
        prod[i] = a[i] * b[i];
        div[i] = a[i] / b[i];
    }
}

int main(int argc, char* argv[]) {
    int N = 200000;  // Значение по умолчанию

    // Обработка параметров: приоритет у аргументов командной строки
    if (argc > 1) {
        N = atoi(argv[1]);
    } else {
        char *env_n = getenv("ARRAY_SIZE");
        if (env_n != NULL) {
            N = atoi(env_n);
        }
    }

    if (N < 100000) {
        fprintf(stderr, "N must be >= 100000\n");
        return 1;
    }

    double *a, *b, *sum, *diff, *prod, *div;
    double *d_a, *d_b, *d_sum, *d_diff, *d_prod, *d_div;

    // Выделение памяти на хосте
    a = (double*)malloc(N * sizeof(double));
    b = (double*)malloc(N * sizeof(double));
    sum = (double*)malloc(N * sizeof(double));
    diff = (double*)malloc(N * sizeof(double));
    prod = (double*)malloc(N * sizeof(double));
    div = (double*)malloc(N * sizeof(double));

    // Инициализация массивов
    for(int i = 0; i < N; i++) {
        a[i] = i + 1.0;
        b[i] = (i + 1.0) * 2.0;
    }

    // Выделение памяти на устройстве
    cudaMalloc(&d_a, N * sizeof(double));
    cudaMalloc(&d_b, N * sizeof(double));
    cudaMalloc(&d_sum, N * sizeof(double));
    cudaMalloc(&d_diff, N * sizeof(double));
    cudaMalloc(&d_prod, N * sizeof(double));
    cudaMalloc(&d_div, N * sizeof(double));

    // Копирование данных на устройство
    cudaMemcpy(d_a, a, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, N * sizeof(double), cudaMemcpyHostToDevice);

    // Создание событий для замера времени
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Прогревочный запуск
    array_ops_sequential<<<1, 1>>>(d_a, d_b, d_sum, d_diff, d_prod, d_div, N);
    cudaDeviceSynchronize();

    // Замер времени выполнения
    cudaEventRecord(start);
    array_ops_sequential<<<1, 1>>>(d_a, d_b, d_sum, d_diff, d_prod, d_div, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Проверка ошибок CUDA
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Расчет времени выполнения
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Вывод результатов
    printf("ВРЕМЯ_ВЫПОЛНЕНИЯ: %.6f s\n", milliseconds / 1000.0f);
    printf("ПОСЛЕДОВАТЕЛЬНАЯ_РЕАЛИЗАЦИЯ\n");
    printf("ЭЛЕМЕНТОВ: %d\n", N);

    // Освобождение памяти
    free(a); free(b); free(sum); free(diff); free(prod); free(div);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_sum);
    cudaFree(d_diff); cudaFree(d_prod); cudaFree(d_div);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
