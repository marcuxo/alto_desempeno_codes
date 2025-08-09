#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <semaphore.h>
#include <unistd.h>

#define BUFFER_SIZE 5

typedef struct {
    int buffer[BUFFER_SIZE];
    int in;
    int out;
    sem_t full;   // Semáforo para contar los mensajes llenos
    sem_t empty;  // Semáforo para contar los espacios vacíos
    sem_t tasks_done; // Semáforo para contar las tareas completadas
    pthread_mutex_t mutex; // Mutex para proteger el acceso al buffer
} MessageQueue;

void initQueue(MessageQueue *queue) {
    queue->in = 0;
    queue->out = 0;
    sem_init(&queue->full, 0, 0);         // No hay mensajes llenos al inicio
    sem_init(&queue->empty, 0, BUFFER_SIZE); // Todos los espacios están vacíos
    sem_init(&queue->tasks_done, 0, 0);   // Ninguna tarea ha sido completada al inicio
    pthread_mutex_init(&queue->mutex, NULL); // Inicializamos el mutex
}

void* producer(void *arg) {
    MessageQueue *queue = (MessageQueue*)arg;

    for (int i = 0; i < 10; i++) {
        sem_wait(&queue->empty); // Espera si no hay espacio disponible en el buffer
        pthread_mutex_lock(&queue->mutex); // Bloquear el acceso al buffer

        // Produce un mensaje
        queue->buffer[queue->in] = i;
        printf("Productor ha creado el mensaje %d\n", i);
        queue->in = (queue->in + 1) % BUFFER_SIZE;

        pthread_mutex_unlock(&queue->mutex); // Desbloquear el acceso al buffer
        sem_post(&queue->full); // Indicar que hay un nuevo mensaje lleno

        sleep(1); // Simular tiempo de producción de mensajes
    }

    // Esperar a que todas las tareas sean completadas
    for (int i = 0; i < 10; i++) {
        sem_wait(&queue->tasks_done); // Espera que cada mensaje sea procesado
    }

    printf("Productor: Todos los mensajes han sido procesados.\n");

    return NULL;
}

void* consumer(void *arg) {
    MessageQueue *queue = (MessageQueue*)arg;

    for (int i = 0; i < 10; i++) {
        sem_wait(&queue->full); // Espera si no hay mensajes llenos
        pthread_mutex_lock(&queue->mutex); // Bloquear el acceso al buffer

        // Consumir un mensaje
        int message = queue->buffer[queue->out];
        printf("Consumidor ha procesado el mensaje %d\n", message);
        queue->out = (queue->out + 1) % BUFFER_SIZE;

        pthread_mutex_unlock(&queue->mutex); // Desbloquear el acceso al buffer
        sem_post(&queue->empty); // Indicar que hay un nuevo espacio vacío

        // Indicar que el mensaje ha sido procesado
        sem_post(&queue->tasks_done); // Simular task_done()

        sleep(2); // Simular tiempo de procesamiento del mensaje
    }

    return NULL;
}

int main() {
    MessageQueue queue;
    initQueue(&queue); // Inicializar la cola de mensajes

    pthread_t producerThread, consumerThread;

    // Crear e iniciar los hilos del productor y el consumidor
    pthread_create(&producerThread, NULL, producer, (void*)&queue);
    pthread_create(&consumerThread, NULL, consumer, (void*)&queue);

    // Esperar a que los hilos terminen
    pthread_join(producerThread, NULL);
    pthread_join(consumerThread, NULL);

    printf("Todos los mensajes han sido procesados.\n");

    // Destruir los semáforos y el mutex
    sem_destroy(&queue.full);
    sem_destroy(&queue.empty);
    sem_destroy(&queue.tasks_done);
    pthread_mutex_destroy(&queue.mutex);

    return 0;
}

