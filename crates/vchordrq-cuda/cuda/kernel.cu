#include <assert.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define QUOTE(x) vchordrq_assign_##x

extern "C" {

typedef float fp32;
typedef __half fp16;

typedef enum {
  vecf32_dot = 0,
  vecf32_l2s = 1,
  vecf16_dot = 2,
  vecf16_l2s = 3,
} op_t;

inline size_t size(op_t op) {
  switch (op) {
  case vecf32_dot:
    return 4;
  case vecf32_l2s:
    return 4;
  case vecf16_dot:
    return 2;
  case vecf16_l2s:
    return 2;
  default:
    return 0;
  }
}

typedef struct {
  op_t op;
  size_t d;
  size_t n;
  void *centroids;
} server_t;

typedef struct {
  cudaStream_t stream;
  op_t op;
  size_t d;
  size_t n;
  void *centroids;
  size_t m;
  void *vectors;
  void *buffer;
  uint32_t *results;
} client_t;

server_t *QUOTE(server_alloc)(op_t op, size_t d, size_t n, void *centroids) {
  server_t *server = (server_t *)malloc(sizeof(server_t));
  server->op = op;
  server->d = d;
  server->n = n;
  server->centroids = NULL;
  if (cudaMalloc(&server->centroids, n * d * size(op)) != cudaSuccess) {
    free(server);
    return NULL;
  }
  if (cudaMemcpy(server->centroids, centroids, n * d * size(op),
                 cudaMemcpyHostToDevice) != cudaSuccess) {
    cudaFree(server->centroids);
    free(server);
    return NULL;
  }
  return server;
}

void QUOTE(server_free)(server_t *server) {
  cudaFree(server->centroids);
  free(server);
}

client_t *QUOTE(client_alloc)(op_t op, size_t d, size_t n, void *centroids,
                              size_t m) {
  client_t *client = (client_t *)malloc(sizeof(client_t));
  client->stream = NULL;
  client->op = op;
  client->d = d;
  client->n = n;
  client->centroids = NULL;
  client->m = m;
  client->vectors = NULL;
  client->buffer = NULL;
  client->results = NULL;
  client->centroids = centroids;
  if (cudaStreamCreate(&client->stream) != cudaSuccess) {
    free(client);
    return NULL;
  }
  if (cudaMalloc(&client->vectors, m * d * size(op)) != cudaSuccess) {
    cudaStreamDestroy(client->stream);
    free(client);
    return NULL;
  }
  if (cudaMalloc(&client->buffer, m * n * size(op)) != cudaSuccess) {
    cudaStreamDestroy(client->stream);
    cudaFree(client->vectors);
    free(client);
    return NULL;
  }
  if (cudaMalloc(&client->results, m * sizeof(uint32_t)) != cudaSuccess) {
    cudaStreamDestroy(client->stream);
    cudaFree(client->buffer);
    cudaFree(client->vectors);
    free(client);
    return NULL;
  }
  return client;
}

void QUOTE(client_free)(client_t *client) {
  cudaStreamDestroy(client->stream);
  cudaFree(client->results);
  cudaFree(client->buffer);
  cudaFree(client->vectors);
  free(client);
}

__global__ void QUOTE(kernel_0)(fp32 *centroids, size_t d, size_t n,
                                fp32 *vectors, uint32_t *results, size_t k) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx >= k) {
    return;
  }

  fp32 best_distance = INFINITY;
  size_t best_index = 0xffffffff;

  for (size_t index = 0; index < n; index++) {
    fp32 sum = 0.0f;
    for (size_t j = 0; j < d; j++) {
      fp32 x = vectors[idx * d + j];
      fp32 y = centroids[index * d + j];
      sum += x * y;
    }
    fp32 distance = -sum;
    if (distance < best_distance) {
      best_distance = distance;
      best_index = index;
    }
  }

  results[idx] = best_index;
}

__global__ void QUOTE(kernel_1)(fp32 *centroids, size_t d, size_t n,
                                fp32 *vectors, uint32_t *results, size_t k) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx >= k) {
    return;
  }

  fp32 best_distance = INFINITY;
  size_t best_index = 0xffffffff;

  for (size_t index = 0; index < n; index++) {
    fp32 sum = 0.0f;
    for (size_t j = 0; j < d; j++) {
      fp32 x = vectors[idx * d + j];
      fp32 y = centroids[index * d + j];
      fp32 diff = x - y;
      sum += diff * diff;
    }
    fp32 distance = sum;
    if (distance < best_distance) {
      best_distance = distance;
      best_index = index;
    }
  }

  results[idx] = best_index;
}

__global__ void QUOTE(kernel_2)(fp16 *centroids, size_t d, size_t n,
                                fp16 *vectors, uint32_t *results, size_t k) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx >= k) {
    return;
  }

  fp16 best_distance = INFINITY;
  size_t best_index = 0xffffffff;

  for (size_t index = 0; index < n; index++) {
    fp16 sum = 0.0f;
    for (size_t j = 0; j < d; j++) {
      fp16 x = vectors[idx * d + j];
      fp16 y = centroids[index * d + j];
      sum += x * y;
    }
    fp16 distance = -sum;
    if (distance < best_distance) {
      best_distance = distance;
      best_index = index;
    }
  }

  results[idx] = best_index;
}

__global__ void QUOTE(kernel_3)(fp16 *centroids, size_t d, size_t n,
                                fp16 *vectors, uint32_t *results, size_t k) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx >= k) {
    return;
  }

  fp16 best_distance = INFINITY;
  size_t best_index = 0xffffffff;

  for (size_t index = 0; index < n; index++) {
    fp16 sum = 0.0f;
    for (size_t j = 0; j < d; j++) {
      fp16 x = vectors[idx * d + j];
      fp16 y = centroids[index * d + j];
      fp16 diff = x - y;
      sum += diff * diff;
    }
    fp16 distance = sum;
    if (distance < best_distance) {
      best_distance = distance;
      best_index = index;
    }
  }

  results[idx] = best_index;
}

int QUOTE(client_query)(client_t *client, size_t k, void *vectors,
                        uint32_t *results) {
  assert(k <= client->m);

  int threads = 256;
  int blocks = (k + threads - 1) / threads;

  if (cudaMemcpyAsync(client->vectors, vectors,
                      k * client->d * size(client->op), cudaMemcpyHostToDevice,
                      client->stream) != cudaSuccess) {
    return -1;
  }
  switch (client->op) {
  case vecf32_dot:
    QUOTE(kernel_0)<<<blocks, threads, 0, client->stream>>>(
        (fp32 *)client->centroids, client->d, client->n,
        (fp32 *)client->vectors, client->results, k);
    break;
  case vecf32_l2s:
    QUOTE(kernel_1)<<<blocks, threads, 0, client->stream>>>(
        (fp32 *)client->centroids, client->d, client->n,
        (fp32 *)client->vectors, client->results, k);
    break;
  case vecf16_dot:
    QUOTE(kernel_2)<<<blocks, threads, 0, client->stream>>>(
        (fp16 *)client->centroids, client->d, client->n,
        (fp16 *)client->vectors, client->results, k);
    break;
  case vecf16_l2s:
    QUOTE(kernel_3)<<<blocks, threads, 0, client->stream>>>(
        (fp16 *)client->centroids, client->d, client->n,
        (fp16 *)client->vectors, client->results, k);
    break;
  default:
    return -1;
  }
  if (cudaMemcpyAsync(results, client->results, k * sizeof(uint32_t),
                      cudaMemcpyDeviceToHost, client->stream) != cudaSuccess) {
    return -1;
  }

  if (cudaStreamSynchronize(client->stream) != cudaSuccess) {
    return -1;
  }

  return 0;
}
}
