#include <stdio.h>
#include <stdlib.h>

#define MAX_ESTUDIANTES 100
#define MAX_PRUEBAS 5

typedef struct {
    char nombre[50];
    int calificaciones[MAX_PRUEBAS];
    float promedio;
} Estudiante;
float calcularPromedio(int calificaciones[], int num_pruebas) {
    int suma = 0;
    for (int i = 0; i < num_pruebas; i++) {
        suma += calificaciones[i];
    }
    return (float)suma / num_pruebas;
}
void evaluacionFormativa(Estudiante* estudiante, int prueba) {
    printf("Retroalimentación para %s en prueba %d: ", estudiante->nombre, prueba + 1);
    if (estudiante->calificaciones[prueba] >= 60) {
        printf("¡Buen trabajo! Sigue así.\n");
    } else {
        printf("Necesitas mejorar. Estudia más para la próxima.\n");
    }
}
int main() {
    Estudiante estudiantes[MAX_ESTUDIANTES];
    int num_estudiantes, num_pruebas = MAX_PRUEBAS;

    printf("Ingrese el número de estudiantes: ");
    scanf("%d", &num_estudiantes);

    // Ingreso de datos
    for (int i = 0; i < num_estudiantes; i++) {
        printf("Ingrese el nombre del estudiante %d: ", i + 1);
        scanf("%s", estudiantes[i].nombre);

        for (int j = 0; j < num_pruebas; j++) {
            printf("Ingrese la calificación de %s en prueba %d: ", estudiantes[i].nombre, j + 1);
            scanf("%d", &estudiantes[i].calificaciones[j]);

            // Evaluación formativa
            evaluacionFormativa(&estudiantes[i], j);
        }

        // Calcular promedio
        estudiantes[i].promedio = calcularPromedio(estudiantes[i].calificaciones, num_pruebas);
        printf("El promedio de %s es: %.2f\n", estudiantes[i].nombre, estudiantes[i].promedio);
    }

    return 0;
}
