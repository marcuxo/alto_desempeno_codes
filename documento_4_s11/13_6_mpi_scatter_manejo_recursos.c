#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char* argv[]) {
    int numtasks, rank;
    const int root = 0; // Proceso raíz
    int work_size = 16; // Total de elementos a procesar
    int *send_data;     // Datos a enviar desde el proceso raíz
    int *recv_data;     // Buffer de recepción en cada proceso
    // Inicializar el entorno MPI
    MPI_Init(&argc, &argv);

    // Obtener el número total de procesos y el rango del proceso actual
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    // Solo el proceso raíz inicializa los datos
    if (rank == root) {
        send_data = (int*)malloc(work_size * sizeof(int)); // Reservar memoria para los datos
        if (send_data == NULL) {
            fprintf(stderr, "Error al asignar memoria en el proceso raíz.\n");
            MPI_Abort(MPI_COMM_WORLD, 1); // Finaliza el programa en caso de error
        }
        for (int i = 0; i < work_size; i++) {
            send_data[i] = i + 1; // Inicializa los datos con valores del 1 al 16
        }
    }
    // Calcular cuántos elementos recibe cada proceso
    int elements_per_process = work_size / numtasks;
    recv_data = (int*)malloc(elements_per_process * sizeof(int)); // Reservar memoria para el buffer
    if (recv_data == NULL) {
        fprintf(stderr, "Error al asignar memoria en el proceso %d.\n", rank);
        free(send_data); // Liberar memoria antes de abortar
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Distribuir los datos a todos los procesos
    MPI_Scatter(send_data, elements_per_process, MPI_INT, recv_data, elements_per_process, MPI_INT, root, MPI_COMM_WORLD);
    // Cada proceso calcula la suma de su porción de datos
    int local_sum = 0; // Suma local de cada proceso
    for (int i = 0; i < elements_per_process; i++) {
        local_sum += recv_data[i]; // Sumar los elementos locales
    }
    // Reducir las sumas locales en el proceso raíz
    int total_sum = 0; // Suma total que será calculada por el proceso raíz
    MPI_Reduce(&local_sum, &total_sum, 1, MPI_INT, MPI_SUM, root, MPI_COMM_WORLD);
    // El proceso raíz imprime la suma total
    if (rank == root) {
        printf("La suma total es: %d\n", total_sum);
        free(send_data); // Liberar memoria del proceso raíz
    }
    free(recv_data); // Liberar memoria en todos los procesos
    // Finalizar el entorno MPI
    MPI_Finalize();
    return 0;
}
