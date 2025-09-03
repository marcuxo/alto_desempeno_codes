#include <mpi.h>
#include <stdio.h>

int main(int argc, char* argv[]) {
    int numtasks, rank;

    // Paso 2: Inicializar el entorno MPI
    MPI_Init(&argc, &argv);
    // Obtener el número total de procesos
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
    // Obtener el rango del proceso actual
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    // Imprimir información del proceso actual
    printf("Número total de procesos: %d\n", numtasks);
    printf("Hola desde el proceso %d\n", rank);
    // Finalizar el entorno MPI
    MPI_Finalize();
    return 0;
}
