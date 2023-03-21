#ifdef __clang__
#include <__clang_cuda_runtime_wrapper.h>
#endif

#include <cuda.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "maximum.h"
#include "util.h"

// Nombre de répétitions pour le benchmark
#define REPS 100

typedef struct {
    double avg;     // Moyenne empirique
    double mse;     // Erreur moyenne quadratique
    double std_dev; // Ecart type
} estimation_t;

void bench(int d, estimation_t* cpu, estimation_t* gpu_seq, estimation_t* gpu_async) {
    float* v;

    size_t len = 1 << d;

    // Remplissage du tableau
    v = (float*)malloc(sizeof(float) * len);

    float elapsed_ms_cpu, elapsed_ms_dim_seq;
    float max_cpu, max_dim_seq;

    cpu->avg = 0.;
    cpu->mse = 0.;
    cpu->std_dev = 0.;

    gpu_seq->avg = 0.;
    gpu_seq->mse = 0.;
    gpu_seq->std_dev = 0.;

    // Benchmark
    for (int i = 0; i < REPS; i++) {
        // Génération d'un problème aléatoire
        rand_vec(v, len, i + 1);

        // Recherde maximum
        max_cpu = maximum_cpu(v, d, &elapsed_ms_cpu);
        max_dim_seq = maximum_dim_seq(v, d, &elapsed_ms_dim_seq);

        // Calculs des estimateurs
        cpu->avg += elapsed_ms_cpu;
        gpu_seq->avg += elapsed_ms_dim_seq;

        gpu_seq->mse += pow(max_cpu - max_dim_seq, 2.0) / max_cpu / max_cpu;

        cpu->std_dev += pow(elapsed_ms_cpu, 2.0);
        gpu_seq->std_dev += pow(elapsed_ms_dim_seq, 2.0);
    }

    cpu->avg /= REPS;
    gpu_seq->avg /= REPS;

    gpu_seq->mse /= REPS;

    cpu->std_dev /= REPS;
    gpu_seq->std_dev /= REPS;

    cpu->std_dev -= pow(cpu->avg, 2.0);
    gpu_seq->std_dev -= pow(gpu_seq->avg, 2.0);

    cpu->std_dev = sqrt(cpu->std_dev);
    gpu_seq->std_dev = sqrt(gpu_seq->std_dev);

    // Cleanup
    free(v);
}

int main(int argc, char** argv) {
    // Récupération des paramètres
    size_t d = 3; // Degré de l'hypercube
    if (argc > 1) {
        d = atoi(argv[1]);
    }
    size_t len = 1 << d;

    size_t threadsPerBlock = MIN(MAX_THREADS_PER_BLOCK, len);
    size_t blocksPerGrid = (len + threadsPerBlock - 1) / threadsPerBlock;
    printf("Recherche d'un maximum sur GPU \n");
    printf("v1: Blocks = %ld, Threads = %ld, Mode = %s \n", blocksPerGrid,
           threadsPerBlock, len <= MAX_THREADS_PER_BLOCK ? "atomic" : "complex");
    printf("v2: \n");
    printf("bench: %d reps. \n", REPS);
    printf("\n");

    // Affichage
    printf("%-26s: %ld\n", "Dimension de l'hypercube", d);
    printf("%-26s: %ld\n", "Nombre d'élmts du tableau ", len);
    printf("\n");

    // Benchmark des fonctions
    estimation_t cpu, gpu_seq;
    bench(d, &cpu, &gpu_seq, NULL);

    printf("=> Validation (erreur moyenne quadratique)\n");
    printf("%-20s: - \n", "Max CPU");
    printf("%-20s: MSE = %.2f%% \n", "Max GPU (DimSeq)", 100 * gpu_seq.mse);

    printf("\n=> Mesure des performances (en moyenne)\n");

    printf("%-20s: ", "Temps CPU");
    print_duration_dev(cpu.avg, cpu.std_dev);
    printf("\n");

    printf("%-20s: ", "Temps GPU (DimSeq)");
    print_duration_dev(gpu_seq.avg, gpu_seq.std_dev);
    printf("\n");

    return EXIT_SUCCESS;
}
