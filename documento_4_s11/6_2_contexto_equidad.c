#include <stdio.h>

#define STUDENTS 5  // Número de estudiantes

// Estructura para almacenar la información del estudiante
typedef struct {
    char name[50];
    int score;
    int socioeconomicStatus; // 1: Bajo, 2: Medio, 3: Alto
} Student;

// Función para ajustar la calificación según el contexto socioeconómico
float adjustScore(int score, int status) {
    float adjustmentFactor;

    // Definir un factor de ajuste basado en el contexto
    switch (status) {
        case 1: // Bajo
            adjustmentFactor = 1.1; // Aumenta el puntaje en un 10%
            break;
        case 2: // Medio
            adjustmentFactor = 1.0; // No hay ajuste
            break;
        case 3: // Alto
            adjustmentFactor = 0.9; // Reduce el puntaje en un 10%
            break;
        default:
            adjustmentFactor = 1.0; // Sin ajuste por defecto
            break;
    }
    return score * adjustmentFactor; // Retorna el puntaje ajustado
}

// Función principal
int main() {
    Student students[STUDENTS] = {
        {"Alice", 85, 1},   // Estudiante con bajo contexto
        {"Bob", 90, 2},     // Estudiante con contexto medio
        {"Charlie", 75, 3}, // Estudiante con alto contexto
        {"David", 92, 2},   // Estudiante con contexto medio
        {"Eva", 70, 1}      // Estudiante con bajo contexto
    };

    printf("Evaluación de estudiantes considerando contexto socioeconómico:\n");
    for (int i = 0; i < STUDENTS; i++) {
        float adjustedScore = adjustScore(students[i].score, students[i].socioeconomicStatus);
        printf("Estudiante: %s, Puntaje original: %d, Puntaje ajustado: %.2f\n",
               students[i].name, students[i].score, adjustedScore);
    }

    return 0;
}
