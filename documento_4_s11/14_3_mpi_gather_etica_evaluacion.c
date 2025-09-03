	#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_ESTUDIANTES 100
#define MAX_PRUEBAS 5

typedef struct {
    char nombre[50];
    int calificaciones[MAX_PRUEBAS];
    float promedio;
    int valido;  // 1 para válido, 0 para no válido
} Estudiante;
float calcularPromedio(int calificaciones[], int num_pruebas) {
    int suma = 0;
    int conteo = 0;  // Para contar cuántas calificaciones son válidas
    for (int i = 0; i < num_pruebas; i++) {
        if (calificaciones[i] >= 0 && calificaciones[i] <= 100) {
            suma += calificaciones[i];
            conteo++;
        }
    }
    return conteo > 0 ? (float)suma / conteo : 0;  // Evitar división por cero
}
int validarCalificacion(int calificacion) {
    return calificacion >= 0 && calificacion <= 100;  // Validar que esté entre 0 y 100
}
int main() {
    Estudiante estudiantes[MAX_ESTUDIANTES];
    int num_estudiantes;

    printf("Ingrese el número de estudiantes: ");
    scanf("%d", &num_estudiantes);

    // Ingreso de datos
    for (int i = 0; i < num_estudiantes; i++) {
        printf("Ingrese el nombre del estudiante %d: ", i + 1);
        scanf("%s", estudiantes[i].nombre);
        estudiantes[i].valido = 1;  // Asumir que el estudiante es válido al inicio

        for (int j = 0; j < MAX_PRUEBAS; j++) {
            int calificacion;
            printf("Ingrese la calificación de %s en prueba %d (0-100): ", estudiantes[i].nombre, j + 1);
            scanf("%d", &calificacion);

            // Validar calificación
            if (validarCalificacion(calificacion)) {
                estudiantes[i].calificaciones[j] = calificacion;
            } else {
                printf("Calificación no válida. Se asignará 0 a esta prueba.\n");
                estudiantes[i].calificaciones[j] = 0;  // Asignar un valor predeterminado
                estudiantes[i].valido = 0;  // Marcar estudiante como no válido
            }
        }

        // Calcular promedio
        estudiantes[i].promedio = calcularPromedio(estudiantes[i].calificaciones, MAX_PRUEBAS);
        if (estudiantes[i].valido) {
            printf("El promedio de %s es: %.2f\n", estudiantes[i].nombre, estudiantes[i].promedio);
        } else {
            printf("El estudiante %s tiene calificaciones no válidas, promedio no calculable.\n", estudiantes[i].nombre);
        }
    }

    return 0;
}
