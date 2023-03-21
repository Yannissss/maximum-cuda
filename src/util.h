#ifndef __UTIL_H__
#define __UTIL_H__

#include <stdlib.h>

// Quitte le programme et affiche un message s'il y a une erreur
#define EXPECT(err, msg, ...)                                                            \
    if (err != cudaSuccess) {                                                            \
        fprintf(stderr, "Cuda error at %s:%d, " msg "\n", __FILE__, __LINE__,            \
                ##__VA_ARGS__);                                                          \
        fprintf(stderr, " %d: %s \n", err, cudaGetErrorString(err));                     \
        exit(EXIT_FAILURE);                                                              \
    }

// Calcul le minimum entre deux expressions
#define MIN(A, B) (A < B ? A : B)

// Calcul le maximum entre deux expressions
#define MAX(A, B) (A > B ? A : B)

// Initialisation aleatoire d'un vecteur
void rand_vec(float* v, size_t n, int seed = 1);

// Affiche un vecteur
void print_vec(float* v, size_t n);

// Affiche une durée
void print_duration(float elapsed_ms);

// Affiche une durée avec un ecart-type
void print_duration_dev(float elapsed_ms, float std_dev);

#endif // !__UTIL_HPP__
