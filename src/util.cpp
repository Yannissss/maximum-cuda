#include "util.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Initialisation aleatoire d'un vecteur au chaque case suit
// une loi 1/X ou X ~ N(0; 1)
void rand_vec(float* v, size_t n, int seed) {
    size_t i;
    srand((49 * seed + time(NULL) * 2023));

    for (i = 0; i < n; i++) {
        // Box-Muller transformation
        // (https://en.wikipedia.org/wiki/Box-Muller_transform)
        float U1 = (float)(rand() + 1) / RAND_MAX;
        float U2 = (float)(rand() + 1) / RAND_MAX;
        float Z = sqrt(-2.f * log(U1)) * cos(2.f * M_PI * U2);
        // printf("%f, %f, %f \n", U1, U2, Z);

        v[i] = 10.f * (Z / 4.f + 0.5f);
    }
}

// Affiche un vecteur
void print_vec(float* v, size_t n) {
    int i;
    if (n <= 10) {
        printf("[");
        for (i = 0; i < n - 1; i++)
            printf("%6.2f ", v[i]);
        printf("%6.2f]", v[n - 1]);
    } else {
        printf("[");
        for (i = 0; i < 5; i++)
            printf("%6.2f ", v[i]);
        printf("... ");
        for (i = 4; i > 0; i--)
            printf("%6.2f ", v[n - 1 - i]);
        printf("%6.2f]", v[n - 1]);
    }
}

// Affiche une durée
void print_duration(float elapsed_ms) {
    if (elapsed_ms >= 1000.0) { // Seconds
        elapsed_ms /= 1000.0f;
        printf("%.2fs", elapsed_ms);
    } else if (elapsed_ms < 1.0) { // Microseconds
        elapsed_ms *= 1000.0f;
        printf("%.2fµs", elapsed_ms);
    } else { // Milliseconds
        printf("%.2fms", elapsed_ms);
    }
}

void print_duration_dev(float elapsed_ms, float std_dev) {
    if (elapsed_ms >= 1000.0) { // Seconds
        elapsed_ms /= 1000.0f;
        std_dev /= 1000.0f;
        printf("%6.2f ± %6.2fs", elapsed_ms, std_dev);
    } else if (elapsed_ms < 1.0) { // Microseconds
        elapsed_ms *= 1000.0f;
        std_dev *= 1000.0f;
        printf("%6.2f ± %6.2fµs", elapsed_ms, std_dev);
    } else { // Milliseconds
        printf("%6.2f ± %6.2fms", elapsed_ms, std_dev);
    }
}