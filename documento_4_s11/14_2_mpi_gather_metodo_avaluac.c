#include <stdio.h>
#include <stdlib.h>

#define MAX_ESTUDIANTES 100
#define MAX_PRUEBAS 5
#define MAX_COMENTARIO 256

typedef struct {
    char nombre[50];
    int calificaciones[MAX_PRUEBAS];
    float promedio;
    char comentario[MAX_COMENTARIO];
} Estudiante;
float calcularPromedio(int calificaciones[], int num_pruebas) {
    int suma = 0;
    for (int i = 0; i < num_pruebas; i++) {
        suma += calificaciones[i];
    }
    return (float)suma / num_pruebas;
}
void recolectarComentario(Estudiante* estudiante) {
    printf("Ingrese un comentario sobre la experiencia de aprendizaje de %s: ", estudiante->nombre);
    getchar();  // Para limpiar el buffer
    fgets(estudiante->comentario, MAX_COMENTARIO, stdin);
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
        }

        // Calcular promedio
        estudiantes[i].promedio = calcularPromedio(estudiantes[i].calificaciones, num_pruebas);
        printf("El promedio de %s es: %.2f\n", estudiantes[i].nombre, estudiantes[i].promedio);

        // Recolectar comentario
        recolectarComentario(&estudiantes[i]);
    }

    // Mostrar comentarios
    for (int i = 0; i < num_estudiantes; i++) {
        printf("Comentario de %s: %s", estudiantes[i].nombre, estudiantes[i].comentario);
    }

    return 0;
}
