#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Función para inicializar matrices con valores aleatorios
void initializeMatrix(float *matrix, int size) {
    for (int i = 0; i < size; i++) {
        matrix[i] = (float)(rand() % 100);  // Valores aleatorios entre 0 y 99
    }
}

// Función para multiplicar dos matrices de tamaño NxN en la CPU
void multiplyMatrices(float *A, float *B, float *C, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            C[i * n + j] = 0;
            for (int k = 0; k < n; k++) {
                C[i * n + j] += A[i * n + k] * B[k * n + j];
            }
        }
    }
}

int main() {
    // Definir diferentes tamaños de matrices
    int sizes[] = {128, 256, 512, 1024, 2048};  // Tamaños de las matrices NxN
    int numSizes = sizeof(sizes) / sizeof(sizes[0]);

    // Iterar sobre cada tamaño de matriz
    for (int idx = 0; idx < numSizes; idx++) {
        int N = sizes[idx];  // Tamaño actual de la matriz NxN
        int size = N * N * sizeof(float);  // Tamaño en bytes de las matrices

        // Asignar memoria para las matrices en el heap
        float *A = (float *)malloc(size);
        float *B = (float *)malloc(size);
        float *C = (float *)malloc(size);

        // Inicializar matrices A y B con valores aleatorios
        initializeMatrix(A, N * N);
        initializeMatrix(B, N * N);

        // Medir el tiempo de ejecución para este tamaño de matriz
        clock_t start = clock();

        // Multiplicar matrices A y B
        multiplyMatrices(A, B, C, N);

        // Calcular el tiempo transcurrido
        clock_t end = clock();
        double time_spent = (double)(end - start) / CLOCKS_PER_SEC;

        // Imprimir el tamaño de la matriz y el tiempo de ejecución
        printf("Multiplicacion de matrices %dx%d completada en %f segundos.\n", N, N, time_spent);

        // Liberar la memoria asignada
        free(A);
        free(B);
        free(C);
    }

    return 0;
}
