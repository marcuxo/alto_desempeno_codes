#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define N 4 // Tamaño de la matriz NxN

// Kernel de multiplicación de matrices
__global__ void matrixMultiplyGPU(float *A, float *B, float *C, int n) {
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

// Inicializar matriz con valores aleatorios
void initializeMatrix(float *matrix, int size) {
    for (int i = 0; i < size; i++) {
        matrix[i] = (float)(rand() % 10);  // valores más pequeños para depuración
    }
}

// Imprimir matriz NxN
void printMatrix(float *matrix, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%0.f ", matrix[i * n + j]);
        }
        printf("\\n");
    }
}

int main() {
    int size = N * N * sizeof(float);

    // Reservar memoria en host
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);

    initializeMatrix(h_A, N * N);
    initializeMatrix(h_B, N * N);
    initializeMatrix(h_C, N * N);

    // Imprimir matrices para verificar
    printf("Matriz A antes de copiar a device:\\n");
    printMatrix(h_A, N);

    printf("Matriz B antes de copiar a device:\\n");
    printMatrix(h_B, N);

    // Reservar memoria en device
    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);

    // Copiar datos al device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Configurar el kernel para N=X
    dim3 threadsPerBlock(N, N);
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x, (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Ejecutar kernel
    matrixMultiplyGPU<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // Verificar errores inmediatamente después de lanzar el kernel
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

    // Copiar resultado a host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Imprimir matrices
    printf("Matriz A:\\n");
    printMatrix(h_A, N);
    printf("Matriz B:\\n");
    printMatrix(h_B, N);
    printf("Resultado C = A x B:\\n");
    printMatrix(h_C, N);

    // Liberar memoria
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}