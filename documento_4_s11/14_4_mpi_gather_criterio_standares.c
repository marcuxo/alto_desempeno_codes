#include <stdio.h>
#include <stdlib.h>

#define MAX_ESTUDIANTES 100
#define MAX_PRUEBAS 5

typedef struct {
    char nombre[50];
    int calificaciones[MAX_PRUEBAS];
    float promedio;
} Estudiante;

typedef struct {
    float criterio_aprobacion;  // Criterio mínimo para aprobar
} Criterios;
float calcularPromedio(int calificaciones[], int num_pruebas) {
    int suma = 0;
    for (int i = 0; i < num_pruebas; i++) {
        suma += calificaciones[i];
    }
    return (float)suma / num_pruebas;
}
int evaluarCriterios(float promedio, Criterios criterios) {
    return promedio >= criterios.criterio_aprobacion;  // Retorna 1 si cumple, 0 si no
}
int main() {
    Estudiante estudiantes[MAX_ESTUDIANTES];
    Criterios criterios;
    int num_estudiantes;

    // Definir criterios de aprobación
    printf("Ingrese el criterio de aprobación (0-100): ");
    scanf("%f", &criterios.criterio_aprobacion);

    printf("Ingrese el número de estudiantes: ");
    scanf("%d", &num_estudiantes);

    // Ingreso de datos
    for (int i = 0; i < num_estudiantes; i++) {
        printf("Ingrese el nombre del estudiante %d: ", i + 1);
        scanf("%s", estudiantes[i].nombre);

        for (int j = 0; j < MAX_PRUEBAS; j++) {
            printf("Ingrese la calificación de %s en prueba %d: ", estudiantes[i].nombre, j + 1);
            scanf("%d", &estudiantes[i].calificaciones[j]);
        }

        // Calcular promedio
        estudiantes[i].promedio = calcularPromedio(estudiantes[i].calificaciones, MAX_PRUEBAS);
        printf("El promedio de %s es: %.2f\n", estudiantes[i].nombre, estudiantes[i].promedio);

        // Evaluar criterios
        if (evaluarCriterios(estudiantes[i].promedio, criterios)) {
            printf("%s ha aprobado con un promedio de %.2f\n", estudiantes[i].nombre, estudiantes[i].promedio);
        } else {
            printf("%s no ha aprobado con un promedio de %.2f\n", estudiantes[i].nombre, estudiantes[i].promedio);
        }
    }

    return 0;
}
