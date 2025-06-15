#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>

// Функция для слияния двух подмассивов
void merge(int arr[], int left, int mid, int right) {
    int n1 = mid - left + 1;
    int n2 = right - mid;

    int *L = (int*)malloc(n1 * sizeof(int));
    int *R = (int*)malloc(n2 * sizeof(int));

    for (int i = 0; i < n1; i++)
        L[i] = arr[left + i];
    for (int j = 0; j < n2; j++)
        R[j] = arr[mid + 1 + j];

    int i = 0, j = 0, k = left;
    while (i < n1 && j < n2) {
        if (L[i] <= R[j]) {
            arr[k] = L[i];
            i++;
        } else {
            arr[k] = R[j];
            j++;
        }
        k++;
    }

    while (i < n1) {
        arr[k] = L[i];
        i++;
        k++;
    }

    while (j < n2) {
        arr[k] = R[j];
        j++;
        k++;
    }

    free(L);
    free(R);
}

// Рекурсивная сортировка слиянием
void mergeSort(int arr[], int left, int right) {
    if (left < right) {
        int mid = left + (right - left) / 2;

        mergeSort(arr, left, mid);
        mergeSort(arr, mid + 1, right);

        merge(arr, left, mid, right);
    }
}

// Точный замер времени
double get_wall_time() {
    struct timeval time;
    gettimeofday(&time, NULL);
    return (double)time.tv_sec + (double)time.tv_usec * 1e-6;
}

int main() {
    const char* env_size = getenv("ARRAY_SIZE");
    int N = env_size ? atoi(env_size) : 200000;
    int *arr = (int*)malloc(N * sizeof(int));

    if (!arr) {
        fprintf(stderr, "Ошибка выделения памяти\n");
        return 1;
    }

    srand(time(NULL));
    for (int i = 0; i < N; i++) {
        arr[i] = rand();
    }

    double start = get_wall_time();
    mergeSort(arr, 0, N - 1);
    double end = get_wall_time();

    printf("ВРЕМЯ_ВЫПОЛНЕНИЯ: %.6f сек\n", end - start);
    printf("ПРОЦЕССОР: CPU (1 ядро)\n");
    printf("ЭЛЕМЕНТОВ: %d\n", N);

    free(arr);
    return 0;
}
