#include <stdio.h>

#define STUDENTS 5 // Número de estudiantes

// Estructura para almacenar la información del estudiante
typedef struct {
    char name[50];
    int score; // Calificación cuantitativa
    char feedback[100]; // Comentario cualitativo
} Student;

// Función para mostrar la evaluación cuantitativa
void quantitativeEvaluation(Student students[]) {
    printf("Evaluación Cuantitativa:\n");
    for (int i = 0; i < STUDENTS; i++) {
        printf("Estudiante: %s, Calificación: %d\n", students[i].name, students[i].score);
    }
}

// Función para mostrar la evaluación cualitativa
void qualitativeEvaluation(Student students[]) {
    printf("\nEvaluación Cualitativa:\n");
    for (int i = 0; i < STUDENTS; i++) {
        printf("Estudiante: %s, Comentario: %s\n", students[i].name, students[i].feedback);
    }
}

// Función principal
int main() {
    // Inicialización de estudiantes con calificaciones y comentarios
    Student students[STUDENTS] = {
        {"Alice", 85, "Buena participación y esfuerzo."},
        {"Bob", 90, "Excelente trabajo y dedicación."},
        {"Charlie", 75, "Necesita mejorar en la asistencia."},
        {"David", 92, "Muy buen desempeño, sigue así."},
        {"Eva", 70, "Requiere atención en los temas fundamentales."}
    };

    // Evaluaciones
    quantitativeEvaluation(students);
    qualitativeEvaluation(students);

    return 0;
}
