#include <mpi.h>
#include <stdio.h>

int main(int argc, char* argv[]) {
    int my_rank, comm_size; // Variables para almacenar el rango y tamaño del comunicador
    int data; // Variable para almacenar el dato a difundir

    // Paso 1: Inicializar el entorno MPI
    MPI_Init(&argc, &argv);

    // Paso 2: Obtener el rango del proceso actual
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    // Paso 3: Obtener el número total de procesos
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

    // Paso 4: Definir el proceso raíz y los datos que se van a enviar
    if (my_rank == 0) {
        data = 100; // El proceso raíz inicializa el valor
        printf("Proceso %d: enviando valor %d a todos los procesos.\n", my_rank, data);
    }

    // Paso 5: Realizar el Broadcast desde el proceso raíz a todos los demás procesos
    MPI_Bcast(&data, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Paso 6: Todos los procesos imprimen el valor recibido
    printf("Proceso %d: recibió valor %d\n", my_rank, data);

    // Paso 7: Finalizar el entorno MPI
    MPI_Finalize();
    return 0;
}
