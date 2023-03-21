#include "maximum.h"

#ifdef __clang__
#include <__clang_cuda_builtin_vars.h>
#include <__clang_cuda_runtime_wrapper.h>
#endif

#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <time.h>

#include "util.h"

float maximum_cpu(float* v, size_t d, float* elapsed_ms) {
    float max = log(0.f); // negative infinity
    size_t len = 1 << d;  // Calcul de la taille à partir de la dimension de l'hypercube

    clock_t start = clock();
    for (size_t idx = 0; idx < len; idx++)
        max = MAX(max, v[idx]);
    clock_t end = clock();

    *elapsed_ms = 1000.f * (float)(end - start) / CLOCKS_PER_SEC;

    return max;
}

__global__ void maximum_dim_seq_kernel(float* read, float* write, int i) {
    size_t idx = blockDim.x * blockIdx.x + threadIdx.x;

    size_t neightbour_mask = 1 << i;
    write[idx] = MAX(read[idx], read[idx ^ neightbour_mask]);
}

// Calcul atomiquement (un seul appel de kernel) le maximum d'un tableau en le
// préchargeant en mémoire partagée
__global__ void maximum_atomic(float* read, float* write, int d) {
    __shared__ float A[MAX_THREADS_PER_BLOCK];
    __shared__ float B[MAX_THREADS_PER_BLOCK];

    unsigned int threadId = threadIdx.x;

    // Chargement en mémoire partagée
    A[threadId] = read[threadId];
    __syncthreads();

    // Calcul du maximum
    unsigned int mask = 1;
    float* R = A;
    float* W = B;
    for (int i = 0; i < d; i++) {
        // Calcul
        W[threadId] = R[threadId ^ mask] + R[threadId];
        mask <<= 1;
        // Echange des buffers
        float* T = R;
        R = W;
        W = T;
        // Syncronisation
        __syncthreads();
    }

    // Ecriture en mémoire centrale
    write[threadId] = R[threadId];
}

float maximum_dim_seq(float* h_v, size_t d, float* elapsed_ms) {
    // Gestion des erreurs
    cudaError_t err = cudaSuccess;

    // Mesure temporelle
    cudaEvent_t start = 0, end = 0;

    // Mémoire GPU
    float *d_A, *d_B;
    size_t len = 1 << d;
    size_t bytes = sizeof(float) * len;

    // Création des événements
    err = cudaEventCreate(&start);
    EXPECT(err, "Could not create start event");

    err = cudaEventCreate(&end);
    EXPECT(err, "Could not create start end");

    // Allocation memoire GPU
    err = cudaMalloc((void**)&d_A, bytes);
    EXPECT(err, "Could not allocate d_A on device memory");

    err = cudaMalloc((void**)&d_B, bytes);
    EXPECT(err, "Could not allocate d_B on device memory");

    // Copie des données CPU vers GPU
    err = cudaMemcpy(d_A, h_v, bytes, cudaMemcpyHostToDevice);
    EXPECT(err, "Could not copy h_A to d_A");

    // Execution du kernel
    size_t threadsPerBlock = MIN(MAX_THREADS_PER_BLOCK, len);
    size_t blocksPerGrid = (len + threadsPerBlock - 1) / threadsPerBlock;
    dim3 dimBlocks(blocksPerGrid, 1, 1);
    dim3 dimThreads(threadsPerBlock, 1, 1);

    // Exécution du kernel
    err = cudaEventRecord(start);
    EXPECT(err, "Couldn't not record start event");

    printf("CUDA kernel launch with %ld blocks of %ld threads\n", blocksPerGrid,
           threadsPerBlock);

    float* read = d_A;
    float* write = d_B;
    bool swapped = false;
    if (dimBlocks.x == 1) {
        printf("Launched atomic mode\n");
        maximum_atomic<<<dimBlocks, dimThreads>>>(read, write, 1);
    } else {
        printf("Launched complex mode\n");
        for (int i = 0; i < d; i++) {
            maximum_dim_seq_kernel<<<dimBlocks, dimThreads>>>(read, write, i);
            // Swap the buffers
            float* tmp = read;
            read = write;
            write = tmp;
            swapped = !swapped;

            err = cudaDeviceSynchronize();
            EXPECT(err, "Error while synchronizing device");
        }
    }

    err = cudaGetLastError();
    EXPECT(err, "Error while executing kernel on GPU");

    err = cudaEventRecord(end);
    EXPECT(err, "Couln't not record end event");

    // Récupération tps d'exécution
    err = cudaEventSynchronize(end);
    EXPECT(err, "Error while syncronizing end event");

    err = cudaEventElapsedTime(elapsed_ms, start, end);
    EXPECT(err, "Error while measuring elapsed time");

    // Copie de vecteur d_A/B[0] vers h_max
    float h_max = 0;
    if (!swapped)
        err = cudaMemcpy(&h_max, d_A, sizeof(float), cudaMemcpyDeviceToHost);
    else
        err = cudaMemcpy(&h_max, d_B, sizeof(float), cudaMemcpyDeviceToHost);
    EXPECT(err, "Could not copy d_A/d_B[0] to h_max [swapped = %d]", swapped);

    return h_max;
}
