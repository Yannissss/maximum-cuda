#ifndef __MAXIMUM_H__
#define __MAXIMUM_H__

#include <stdlib.h>

#define MAX_THREADS_PER_BLOCK 1024

// Version CPU séquentielle afin de comparer les performances et de s'assurer de
// la correction des implémentations
float maximum_cpu(float* v, size_t d, float* elapsed_ms);

// Recherche du maximum dans un tableau avec CUDA
// Cette version fonctionne sur le modèle de l'hypercube en traitant dimension par
// dimension avec une syncronisation CPU
// d représente la dimension de l'hypercube
float maximum_dim_seq(float* v, size_t d, float* elapsed_ms);

#endif // !__MAXIMUM_H__