#include <stdio.h>
#include <stdlib.h>
#include <sys/shm.h>  // Para memoria compartida
#include <sys/ipc.h>  // Para claves de IPC
#include <unistd.h>   // Para fork
#include <sys/wait.h> // Para esperar a los procesos hijos

int main() {
    // Paso 2: Crear una clave para la memoria compartida
    key_t key = ftok("shmfile", 65);

    // Paso 3: Crear o acceder a un segmento de memoria compartida
    int shmid = shmget(key, sizeof(int), 0666|IPC_CREAT);

    // Paso 4: Adjuntar el segmento de memoria compartida
    int *shared_value = (int*) shmat(shmid, NULL, 0);

    // Inicializar el valor compartido
    *shared_value = 0;

    // Paso 5: Crear el proceso hijo usando fork()
    if (fork() == 0) {
        // Proceso hijo
        *shared_value = 100;  // Modificar el valor en memoria compartida
        printf("Proceso hijo: valor en memoria compartida = %d\n", *shared_value);
        shmdt(shared_value);  // Desadjuntar el segmento de memoria compartida
        exit(0);  // Terminar el proceso hijo
    } else {
        // Proceso padre
        wait(NULL);  // Esperar a que el proceso hijo termine
        printf("Proceso padre: valor en memoria compartida = %d\n", *shared_value);

        // Paso 6: Desadjuntar y eliminar el segmento de memoria compartida
        shmdt(shared_value);  // Desadjuntar el segmento de memoria compartida
        shmctl(shmid, IPC_RMID, NULL);  // Eliminar el segmento de memoria compartida
    }

    return 0;
}
// Compilación: gcc -o multiproceso multiproceso.c -lrt