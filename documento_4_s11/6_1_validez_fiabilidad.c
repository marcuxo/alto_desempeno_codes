#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define RESPONDENTS 100  // Número de encuestados
#define REPEAT_SURVEY 3  // Número de veces que se realiza la encuesta

// Función para simular la satisfacción del cliente
int getCustomerSatisfaction() {
    // Genera una respuesta aleatoria entre 1 y 5
    return (rand() % 5) + 1;
}

// Función para calcular la media de las respuestas
float calculateMean(int responses[], int size) {
    int sum = 0;
    for (int i = 0; i < size; i++) {
        sum += responses[i];
    }
    return (float)sum / size;
}

// Función principal
int main() {
    srand(time(NULL)); // Inicializa la semilla para generar números aleatorios

    int responses[RESPONDENTS][REPEAT_SURVEY]; // Matriz para almacenar las respuestas
    float means[REPEAT_SURVEY]; // Para almacenar las medias de cada encuesta

    // Realizar la encuesta varias veces
    for (int survey = 0; survey < REPEAT_SURVEY; survey++) {
        printf("Encuesta %d:\n", survey + 1);
        for (int i = 0; i < RESPONDENTS; i++) {
            responses[i][survey] = getCustomerSatisfaction();
            printf("  Cliente %d: %d\n", i + 1, responses[i][survey]);
        }
        // Calcular la media de esta encuesta
        means[survey] = calculateMean(responses[survey], RESPONDENTS);
        printf("  Media de satisfacción: %.2f\n\n", means[survey]);
    }

    // Evaluar la fiabilidad de los resultados comparando las medias
    printf("Evaluando fiabilidad de las encuestas...\n");
    float overallMean = calculateMean(means, REPEAT_SURVEY);
    printf("Media general de satisfacción: %.2f\n", overallMean);

    // Calcular la variación entre las encuestas
    float variance = 0.0;
    for (int i = 0; i < REPEAT_SURVEY; i++) {
        variance += (means[i] - overallMean) * (means[i] - overallMean);
    }
    variance /= REPEAT_SURVEY;
    printf("Variación entre las encuestas: %.2f\n", variance);

    return 0;
}
