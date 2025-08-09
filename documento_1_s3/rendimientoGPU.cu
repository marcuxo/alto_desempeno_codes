#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

#define N 1024  // Tamaño de las matrices NxN

#define cudaErrorCheck(ans) {cudaAssert(ans, __FILE__, __LINE__);}
inline void cudaAssert(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s:%d: %s\n", file, line, cudaGetErrorString(code));
        exit(code);
    }
}

// Función para inicializar matrices con valores aleatorios
void initializeMatrix(float *matrix, int size) {
    for (int i = 0; i < size; i++) {
        matrix[i] = (float)(rand() % 100);
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
    int size = N * N * sizeof(float);  // Tamaño en bytes de las matrices

    // Asignar memoria en el host (CPU)
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);

    // Inicializar matrices A y B en el host con valores aleatorios
    initializeMatrix(h_A, N * N);
    initializeMatrix(h_B, N * N);

    // Asignar memoria en el dispositivo (GPU)
    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);

    // Copiar matrices A y B desde el host a la GPU
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Definir el tamaño de los bloques e hilos
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x, (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Medir el tiempo de ejecución en la GPU
    clock_t startGPU = clock();

    
    // Ejecutar el kernel de multiplicación de matrices en la GPU
    matrixMultiplyGPU<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    cudaErrorCheck(cudaGetLastError());

    // Sincronizar la GPU
    cudaDeviceSynchronize();

    clock_t endGPU = clock();
    double timeGPU = (double)(endGPU - startGPU) / CLOCKS_PER_SEC;

    // Copiar el resultado de vuelta al host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Mostrar el tiempo de ejecución en la GPU
    printf("Multiplicacion de matrices en la GPU completada en %f segundos.\n", timeGPU);

    // Liberar la memoria en la GPU y el host
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
