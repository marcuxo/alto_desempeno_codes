#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char* argv[]) {
    int numtasks, rank;
    const int root = 0; // Proceso raíz
    int send_data[16];  // Datos a enviar desde el proceso raíz
    int recv_data[4];   // Buffer de recepción en cada proceso
    // Inicializar el entorno MPI
    MPI_Init(&argc, &argv);

    // Obtener el número total de procesos y el rango del proceso actual
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    // Proceso raíz inicializa los datos a enviar
    if (rank == root) {
        for (int i = 0; i < 16; i++) {
            send_data[i] = i; // Inicializa el arreglo con valores del 0 al 15
        }
        printf("Proceso raíz envía datos: ");
        for (int i = 0; i < 16; i++) {
            printf("%d ", send_data[i]);
        }
        printf("\n");
    }
    // Distribuir los datos a todos los procesos
    MPI_Scatter(send_data, 4, MPI_INT, recv_data, 4, MPI_INT, root, MPI_COMM_WORLD);
    
    // Cada proceso imprime su parte recibida
    printf("Proceso %d recibió: ", rank);
    for (int i = 0; i < 4; i++) {
        printf("%d ", recv_data[i]);
    }
    printf("\n");
    // Aquí se podría realizar algún procesamiento con los datos recibidos
    for (int i = 0; i < 4; i++) {
        recv_data[i] += 10; // Sumar 10 a cada elemento como ejemplo de procesamiento
    }
    // Imprimir los datos procesados por cada proceso
    printf("Proceso %d después del procesamiento: ", rank);
    for (int i = 0; i < 4; i++) {
        printf("%d ", recv_data[i]);
    }
    printf("\n");
    // Finalizar el entorno MPI
    MPI_Finalize();
    return 0;
}
