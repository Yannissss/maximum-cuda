#ifdef __clang__
#include <__clang_cuda_runtime_wrapper.h>
#endif

#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

#include "maximum.h"
#include "util.h"

int main(int argc, char** argv) {
    // Données
    float* v;

    // Récupération des paramètres
    size_t d = 3; // Degré de l'hypercube
    if (argc > 1) {
        d = atoi(argv[1]);
    }
    size_t len = 1 << d;

    // Remplissage du tableau
    v = (float*)malloc(sizeof(float) * len);
    rand_vec(v, len);

    // Affichage
    printf("%-26s: %ld\n", "Dimension de l'hypercube", d);
    printf("%-26s: %ld\n", "Nombre d'élmts du tableau ", len);

    // Benchmark des fonctions
    float elapsed_ms_cpu, elapsed_ms_dim_seq;
    float max_cpu, max_dim_seq;

    max_cpu = maximum_cpu(v, d, &elapsed_ms_cpu);
    max_dim_seq = maximum_dim_seq(v, d, &elapsed_ms_dim_seq);

    print_vec(v, len);
    printf("\n; max_atomic = %f \n", max_dim_seq);

    printf("=> Validation\n");
    printf("%-26s: %f \n", "Max ver. CPU", max_cpu);
    printf("%-26s: %f \n", "Max ver. GPU (DimSeq)", max_dim_seq);

    printf("\n=> Mesure des performances\n");

    printf("%-26s: ", "Temps ver. CPU");
    print_duration(elapsed_ms_cpu);
    printf("\n");

    printf("%-26s: ", "Temps ver. GPU (DimSeq)");
    print_duration(elapsed_ms_dim_seq);
    printf("\n");

    // Cleanup
    free(v);

    return EXIT_SUCCESS;
}
