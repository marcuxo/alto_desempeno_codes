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
void encontrarEstudiantes(Estudiante estudiantes[], int num_estudiantes, Estudiante* mejor, Estudiante* peor) {
    mejor->promedio = -1;  // Inicializar con un valor bajo
    peor->promedio = 101;   // Inicializar con un valor alto

    for (int i = 0; i < num_estudiantes; i++) {
        if (estudiantes[i].promedio > mejor->promedio) {
            *mejor = estudiantes[i];  // Actualizar mejor estudiante
        }
        if (estudiantes[i].promedio < peor->promedio) {
            *peor = estudiantes[i];    // Actualizar peor estudiante
        }
    }
}

void mostrarEstadisticas(Estudiante estudiantes[], int num_estudiantes) {
    float suma_total = 0;
    for (int i = 0; i < num_estudiantes; i++) {
        suma_total += estudiantes[i].promedio;
    }
    float promedio_general = suma_total / num_estudiantes;
    printf("Promedio general de la clase: %.2f\n", promedio_general);
}
int main() {
    Estudiante estudiantes[MAX_ESTUDIANTES];
    Estudiante mejor, peor;
    int num_estudiantes;

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
    }

    // Análisis de datos
    encontrarEstudiantes(estudiantes, num_estudiantes, &mejor, &peor);
    mostrarEstadisticas(estudiantes, num_estudiantes);

    printf("El mejor estudiante es: %s con un promedio de %.2f\n", mejor.nombre, mejor.promedio);
    printf("El peor estudiante es: %s con un promedio de %.2f\n", peor.nombre, peor.promedio);

    return 0;
}
