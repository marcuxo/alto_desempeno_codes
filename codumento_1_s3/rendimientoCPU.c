#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 1024  // Tamaño de las matrices NxN

// Función para inicializar una matriz con valores aleatorios
void initializeMatrix(float *matrix, int size) {
    for (int i = 0; i < size; i++) {
        matrix[i] = (float)(rand() % 100);  // Valores aleatorios entre 0 y 99
    }
}

// Función para multiplicar dos matrices de tamaño NxN en la CPU
void multiplyMatricesCPU(float *A, float *B, float *C, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            C[i * n + j] = 0;  // Inicializar el elemento C[i][j]
            for (int k = 0; k < n; k++) {
                C[i * n + j] += A[i * n + k] * B[k * n + j];
            }
        }
    }
}

int main() {
    int size = N * N * sizeof(float);  // Tamaño de las matrices en bytes

    // Asignar memoria para las matrices en el heap
    float *A = (float *)malloc(size);
    float *B = (float *)malloc(size);
    float *C = (float *)malloc(size);

    // Inicializar matrices A y B con valores aleatorios
    initializeMatrix(A, N * N);
    initializeMatrix(B, N * N);

    // Medir el tiempo de ejecución en la CPU
    clock_t startCPU = clock();

    // Multiplicar matrices en la CPU
    multiplyMatricesCPU(A, B, C, N);

    clock_t endCPU = clock();
    double timeCPU = (double)(endCPU - startCPU) / CLOCKS_PER_SEC;

    // Mostrar el tiempo de ejecución en la CPU
    printf("Multiplicacion de matrices en la CPU completada en %f segundos.\n", timeCPU);

    // Liberar memoria asignada
    free(A);
    free(B);
    free(C);

    return 0;
}
