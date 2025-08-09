
#include <stdio.h>
#include <ctype.h>
#include <string.h>
// Función para validar si una cadena contiene solo dígitos
int esNumero(const char *cadena) {
    for (int i = 0; i < strlen(cadena); i++) {
        if (!isdigit(cadena[i])) {
            return 0; // No es número
        }
    }
    return 1; // Es número
}

int simulate_register_operation(int data) {
    // Esta operación representa una operación rápida en registros
    return data * 2;
}

int main() {
    char entrada[100];
    int input_data;
    
    // Pedimos al usuario que ingrese un número para simular su procesamiento
    printf("Ingrese un numero para procesar en los registros: ");
    scanf("%s", entrada);

    if (esNumero(entrada)) {
        // Conversión manual de string a int
        input_data = 0;
        for (int i = 0; entrada[i] != '\0'; i++) {
            input_data = input_data * 10 + (entrada[i] - '0');
        }
        // Llamamos a la función que simula la operación rápida en registros
        int result = simulate_register_operation(input_data);
        // Mostramos el resultado de la operación
        printf("Resultado de la operacion en registros: %d\n", result);
    } else {
        printf("Solo se permiten números.\n");
    }
    
    return 0;  // Terminamos el programa
}
