#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

#define N 1024  // Tamaño de las matrices NxN

// Función para inicializar matrices con valores aleatorios
void initializeMatrix(float *matrix, int size) {
    for (int i = 0; i < size; i++) {
        matrix[i] = (float)(rand() % 100);  // Valores aleatorios entre 0 y 99
    }
}

// Función para multiplicar dos matrices de tamaño NxN en la CPU
void multiplyMatricesCPU(float *A, float *B, float *C, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            C[i * n + j] = 0;
            for (int k = 0; k < n; k++) {
                C[i * n + j] += A[i * n + k] * B[k * n + j];
            }
        }
    }
}

// Kernel CUDA para la multiplicación de matrices en la GPU
__global__ void matrixMultiplyGPU(float *A, float *B, float *C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        float sum = 0.0;
        for (int k = 0; k < n; k++) {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}

int main() {
    int size = N * N * sizeof(float);  // Tamaño de las matrices en bytes

    // Asignar memoria para las matrices en el heap
    float *A = (float *)malloc(size);
    float *B = (float *)malloc(size);
    float *C_CPU = (float *)malloc(size);
    float *C_GPU = (float *)malloc(size);

    // Inicializar matrices A y B con valores aleatorios
    initializeMatrix(A, N * N);
    initializeMatrix(B, N * N);

    // -------- Multiplicación de Matrices en la CPU --------
    clock_t startCPU = clock();

    // Multiplicar matrices A y B en la CPU
    multiplyMatricesCPU(A, B, C_CPU, N);

    clock_t endCPU = clock();
    double timeCPU = (double)(endCPU - startCPU) / CLOCKS_PER_SEC;

    // Mostrar el tiempo de ejecución en la CPU
    printf("Multiplicación de matrices en la CPU completada en %f segundos.\n", timeCPU);

    // -------- Multiplicación de Matrices en la GPU --------

    // Asignar memoria en el dispositivo (GPU)
    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);

    // Copiar matrices A y B desde el host a la GPU
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    // Definir el tamaño de los bloques y la cuadrícula
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x, (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Medir el tiempo de ejecución en la GPU
    clock_t startGPU = clock();

    // Ejecutar el kernel de multiplicación de matrices en la GPU
    matrixMultiplyGPU<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // Sincronizar la GPU
    cudaDeviceSynchronize();

    clock_t endGPU = clock();
    double timeGPU = (double)(endGPU - startGPU) / CLOCKS_PER_SEC;

    // Copiar el resultado de vuelta al host
    cudaMemcpy(C_GPU, d_C, size, cudaMemcpyDeviceToHost);

    // Mostrar el tiempo de ejecución en la GPU
    printf("Multiplicación de matrices en la GPU completada en %f segundos.\n", timeGPU);

    // Liberar memoria
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(A);
    free(B);
    free(C_CPU);
    free(C_GPU);

    return 0;
}
