#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 2048  // Tamaño de las matrices NxN

// Función para inicializar una matriz con valores aleatorios
void initializeMatrix(float *matrix, int size) {
    for (int i = 0; i < size; i++) {
        matrix[i] = (float)(rand() % 100);  // Valores aleatorios entre 0 y 99
    }
}

// Función para multiplicar dos matrices de tamaño NxN
void multiplyMatrices(float *A, float *B, float *C, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            C[i * n + j] = 0;  // Inicializar el elemento C[i][j]
            for (int k = 0; k < n; k++) {
                C[i * n + j] += A[i * n + k] * B[k * n + j];
            }
        }
    }
}

// Función para imprimir la matriz (opcional para matrices pequeñas)
void printMatrix(float *matrix, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%.0f ", matrix[i * n + j]);
        }
        printf("\n");
    }
}

int main() {
    // Tamaño de las matrices
    int size = N * N * sizeof(float);

    // Asignar memoria para las matrices en el heap
    float *A = (float *)malloc(size);
    float *B = (float *)malloc(size);
    float *C = (float *)malloc(size);

    //verificar la asignacion de memoria
    if (A == NULL || B == NULL || C == NULL) {
        fprintf(stderr, "Error al asignar memoria\n");
        return 1;
    }

    // Inicializar matrices A y B con valores aleatorios
    initializeMatrix(A, N * N);
    initializeMatrix(B, N * N);

    // Medir el tiempo de ejecución
    clock_t start = clock();

    // Multiplicar matrices A y B, almacenar el resultado en C
    multiplyMatrices(A, B, C, N);

    // Medir el tiempo de finalización
    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;

    // Mostrar el tiempo que tomó la operación
    printf("Multiplicacion de matrices completada en %f segundos.\n", time_spent);

    // (Opcional) Imprimir la matriz C para matrices pequeñas
    // printMatrix(A, N);
    // printf("\n");
    // printMatrix(B, N);
    // printf("\n");
    // printMatrix(C, N);

    // Liberar la memoria asignada
    free(A);
    free(B);
    free(C);

    return 0;
}
