#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <semaphore.h>
#include <time.h>
#include <unistd.h>

#define BUFFER_SIZE 5

typedef struct {
    int buffer[BUFFER_SIZE];
    int in;
    int out;
    sem_t full;   // Semáforo que cuenta los mensajes llenos
    sem_t empty;  // Semáforo que cuenta los espacios vacíos
    pthread_mutex_t mutex; // Mutex para asegurar acceso exclusivo al buffer
} MessageQueue;

void initQueue(MessageQueue *queue) {
    queue->in = 0;
    queue->out = 0;
    sem_init(&queue->full, 0, 0);         // Inicialmente no hay mensajes llenos
    sem_init(&queue->empty, 0, BUFFER_SIZE); // Todos los espacios están vacíos
    pthread_mutex_init(&queue->mutex, NULL); // Inicializamos el mutex
}

void* consumer(void *arg) {
    MessageQueue *queue = (MessageQueue*)arg;

    for (int i = 0; i < 10; i++) {
        struct timespec ts;
        clock_gettime(CLOCK_REALTIME, &ts);
        ts.tv_sec += 2; // Esperar hasta 2 segundos

        // Intentar obtener un mensaje con un timeout de 2 segundos
        if (sem_timedwait(&queue->full, &ts) == -1) {
            printf("Consumidor: timeout al intentar obtener un mensaje.\n");
            continue;
        }

        // Bloquear el acceso al buffer
        pthread_mutex_lock(&queue->mutex);

        // Consumir el mensaje
        int message = queue->buffer[queue->out];
        printf("Consumidor ha procesado el mensaje %d\n", message);
        queue->out = (queue->out + 1) % BUFFER_SIZE;

        pthread_mutex_unlock(&queue->mutex); // Desbloquear el buffer
        sem_post(&queue->empty); // Señalar que hay un espacio vacío

        sleep(1); // Simular el tiempo de procesamiento del mensaje
    }

    return NULL;
}

void* producer(void *arg) {
    MessageQueue *queue = (MessageQueue*)arg;

    for (int i = 0; i < 10; i++) {
        sem_wait(&queue->empty); // Esperar si no hay espacio disponible
        pthread_mutex_lock(&queue->mutex); // Bloquear acceso al buffer

        // Produce un mensaje
        queue->buffer[queue->in] = i;
        printf("Productor ha creado el mensaje %d\n", i);
        queue->in = (queue->in + 1) % BUFFER_SIZE;

        pthread_mutex_unlock(&queue->mutex); // Desbloquear el buffer
        sem_post(&queue->full); // Indicar que hay un nuevo mensaje lleno

        sleep(1); // Simular tiempo de producción
    }

    return NULL;
}

int main() {
    MessageQueue queue;
    initQueue(&queue); // Inicializar la cola de mensajes

    pthread_t producerThread, consumerThread;

    // Crear e iniciar los hilos del productor y consumidor
    pthread_create(&producerThread, NULL, producer, (void*)&queue);
    pthread_create(&consumerThread, NULL, consumer, (void*)&queue);

    // Esperar a que los hilos terminen
    pthread_join(producerThread, NULL);
    pthread_join(consumerThread, NULL);

    printf("Todos los mensajes han sido procesados.\n");

    // Destruir los semáforos y el mutex
    sem_destroy(&queue.full);
    sem_destroy(&queue.empty);
    pthread_mutex_destroy(&queue.mutex);

    return 0;
}

