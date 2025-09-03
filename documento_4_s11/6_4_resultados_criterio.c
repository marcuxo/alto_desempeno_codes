#include <stdio.h>

#define STUDENTS 5 // Número de estudiantes
#define PASSING_SCORE 60 // Criterio de aprobación

// Estructura para almacenar la información del estudiante
typedef struct {
    char name[50];
    int score; // Calificación del estudiante
} Student;

// Función para evaluar los resultados
void evaluateResults(Student students[]) {
    printf("Resultados de la Evaluación:\n");
    for (int i = 0; i < STUDENTS; i++) {
        if (students[i].score >= PASSING_SCORE) {
            printf("Estudiante: %s, Calificación: %d - Resultado: Aprobado\n", students[i].name, students[i].score);
        } else {
            printf("Estudiante: %s, Calificación: %d - Resultado: Reprobado\n", students[i].name, students[i].score);
        }
    }
}

// Función principal
int main() {
    // Inicialización de estudiantes con sus calificaciones
    Student students[STUDENTS] = {
        {"Alice", 85},
        {"Bob", 90},
        {"Charlie", 55},
        {"David", 72},
        {"Eva", 48}
    };

    // Evaluar los resultados
    evaluateResults(students);

    return 0;
}
