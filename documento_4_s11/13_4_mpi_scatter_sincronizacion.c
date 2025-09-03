#include <mpi.h>
#include <stdio.h>
#include <unistd.h> // Para la función sleep

int main(int argc, char* argv[]) {
    int numtasks, rank;

    // Paso 2: Inicializar el entorno MPI
    MPI_Init(&argc, &argv);
    // Obtener el número total de procesos
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
    // Obtener el rango del proceso actual
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    // Simular un trabajo diferente en cada proceso
    printf("Proceso %d está trabajando...\n", rank);
    sleep(rank); // Cada proceso duerme por un número de segundos igual a su rango
    // Sincronizar todos los procesos
    MPI_Barrier(MPI_COMM_WORLD);
    printf("Proceso %d ha alcanzado la barrera.\n", rank);
    // Continuar con el trabajo después de la sincronización
    printf("Proceso %d está continuando su trabajo...\n", rank);
    // Finalizar el entorno MPI
    MPI_Finalize();
    return 0;
}
