
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <time.h>
#include <stdio.h>
#include <cstdlib>

#define N 4  // Definir el tamaño de la matriz NxN

// Función del kernel que ejecutará la multiplicación de matrices en la GPU
__global__ void matrixMultiplyGPU(float* A, float* B, float* C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0.0;

    if (row < n && col < n) {
        for (int i = 0; i < n; i++) {
            sum += A[row * n + i] * B[i * n + col];
        }
        C[row * n + col] = sum;
    }
}

// Función para inicializar matrices con valores aleatorios
void initializeMatrix(float* matrix, int size) {
    for (int i = 0; i < size; i++) {
        matrix[i] = (float)(rand() % 10);
    }
}

// Función para imprimir la matriz (opcional para matrices pequeñas)
void printMatrix(float* matrix, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%.2f ", matrix[i * n + j]);
        }
        printf("\n");
    }
}

int main() {
    srand(time(NULL));  // Inicializar la semilla para números aleatorios
    int size = N * N * sizeof(float);  // Tamaño en bytes de las matrices

    // Asignar memoria en el host (CPU)
    float* h_A = (float*)malloc(size);
    float* h_B = (float*)malloc(size);
    float* h_C = (float*)malloc(size);

    // Inicializar matrices en el host
    initializeMatrix(h_A, N * N);
    initializeMatrix(h_B, N * N);

    // Asignar memoria en el dispositivo (GPU)
    float* d_A, * d_B, * d_C;
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    // Copiar las matrices desde el host a la GPU
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Definir el tamaño de los bloques y la cuadrícula
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x, (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Ejecutar el kernel en la GPU
    matrixMultiplyGPU<<<blocksPerGrid, threadsPerBlock >>>(d_A, d_B, d_C, N);
    
    // Verificar si el kernel se lanzó correctamente
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error al lanzar el kernel: %s\\n", cudaGetErrorString(err));
        return 1;  // salir si hay error
    }

    // Sincronizar dispositivo y verificar errores posteriores
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Error después de sincronizar: %s\\n", cudaGetErrorString(err));
        return 1;  // salir si hay error
    }


    // Copiar el resultado de la GPU al host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Imprimir la matriz resultante (opcional)
    printMatrix(h_A, N);
    printf("\n");
    printMatrix(h_B, N);
    printf("\n");
    printMatrix(h_C, N);

    // Liberar memoria en el dispositivo
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Liberar memoria en el host
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
