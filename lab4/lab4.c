#include <malloc.h>
#include <math.h>
#include <mpi.h>
#include <stdio.h>
typedef struct {
  double *last_layer;
  double *current_layer;
} layers_t;
typedef struct {
  double *upper_edge;
  double *lower_edge;
} edges_t;
typedef struct {
  int rank;
  int size;
} process_info_t;
typedef struct {
  int *layer_height;
  int *offset;
} layers_info_t;
typedef struct {
  double start;
  double end;
} time_measure_t;
typedef struct {
  MPI_Request up;
  MPI_Request down;
} async_req;
// input coords
const double x_0 = -1.0;
const double y_0 = -1.0;
const double z_0 = -1.0;
// dims sizes
const double D_x = 2.0;
const double D_y = 2.0;
const double D_z = 2.0;
// Grid sizes

const int N_x = 400;
const int N_y = 400;
const int N_z = 400;
// step sizes
const double h_x = (D_x / (N_x - 1));
const double h_y = (D_y / (N_y - 1));
const double h_z = (D_z / (N_z - 1));
// sqr step sizes
const double sqr_H_x = h_x * h_x;
const double sqr_H_y = h_y * h_y;
const double sqr_H_z = h_z * h_z;
const double A = 1.0e5;
const double EPS = 1.0e-8;
const double inverse_constant = 2 / sqr_H_x + 2 / sqr_H_y + 2 / sqr_H_z + A;
process_info_t process_info;
layers_info_t layers_info;
async_req send_req;
async_req recv_req;
double max_diff = 0.0;
void get_process_info();
void get_layers_info();
void init_layers(layers_t *layers, int height, int offset);
void init_edges(edges_t *edges);
void swap_layers(layers_t *layers);
void free_resources(layers_t *layers, edges_t *edges);
double phi(double x, double y, double z);
double rho(double x, double y, double z);
void calc_phi_j_phi_k_func(layers_t *layers, double phi_i, int i, int j, int k);
void calc_inner(layers_t *layers);
void calc_edges(layers_t *layers, edges_t *edges);
double calc_max_delta(layers_t *layers);
void send_edges(layers_t *layers, edges_t *edges);
void wait_edges();
double get_x(int i);
double get_y(int j);
double get_z(int k);
int get_index(int i, int j, int k);

int main(int argc, char *argv[]) {
  layers_t layers;
  edges_t edges;
  time_measure_t time_measure;
  MPI_Init(&argc, &argv);
  get_process_info();
  get_layers_info();
  init_layers(&layers, layers_info.layer_height[process_info.rank],
              layers_info.offset[process_info.rank]);
  init_edges(&edges);
  time_measure.start = MPI_Wtime();
  do {
    max_diff = 0.0;
    swap_layers(&layers);
    send_edges(&layers, &edges);
    calc_inner(&layers);
    wait_edges();
    calc_edges(&layers, &edges);
    double tmp = 0.0;
    MPI_Allreduce(&max_diff, &tmp, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    max_diff = tmp;
  } while (max_diff >= EPS);
  double max_delta = calc_max_delta(&layers);
  time_measure.end = MPI_Wtime();
  if (process_info.rank == 0) {
    printf("%d,%lf,%le\n", process_info.size,
           time_measure.end - time_measure.start, max_delta);
  }
  free_resources(&layers, &edges);
  MPI_Finalize();
  return 0;
}
double phi(double x, double y, double z) { return x * x + y * y + z * z; }
double rho(double x, double y, double z) { return 6 - A * phi(x, y, z); }
double get_x(int i) { return x_0 + i * h_x; }
double get_y(int j) { return y_0 + j * h_y; }
double get_z(int k) { return z_0 + k * h_z; }
int get_index(int i, int j, int k) { return i * N_y * N_z + j * N_z + k; }
void get_process_info() {
  MPI_Comm_size(MPI_COMM_WORLD, &process_info.size);
  MPI_Comm_rank(MPI_COMM_WORLD, &process_info.rank);
}
void get_layers_info() {
  layers_info.layer_height = (int *)malloc(process_info.size * sizeof(int));
  layers_info.offset = (int *)malloc(process_info.size * sizeof(int));
  int offset = 0, bufSize = N_x / process_info.size;
  int bufRemainder = N_x % process_info.size;
  for (int i = 0; i < process_info.size; ++i) {
    layers_info.offset[i] = offset;
    layers_info.layer_height[i] = bufSize;
    if (i < bufRemainder)
      layers_info.layer_height[i] += 1;
    offset += layers_info.layer_height[i];
  }
}
void init_layers(layers_t *layers, int height, int offset) {
  layers->current_layer = (double *)malloc(height * N_y * N_z * sizeof(double));
  layers->last_layer = (double *)malloc(height * N_y * N_z * sizeof(double));
  for (int i = 0; i < height; ++i) {
    for (int j = 0; j < N_y; ++j) {
      for (int k = 0; k < N_z; ++k) {
        if ((offset + i == 0) || (j == 0) || (k == 0) ||
            (offset + i == N_x - 1) || (j == N_y - 1) || (k == N_z - 1)) {
          layers->current_layer[get_index(i, j, k)] =

              phi(get_x(offset + i), get_y(j), get_z(k));

          layers->last_layer[get_index(i, j, k)] = phi(get_x(offset

                                                             + i),
                                                       get_y(j), get_z(k));
        } else {
          layers->current_layer[get_index(i, j, k)] = 0;
          layers->last_layer[get_index(i, j, k)] = 0;
        }
      }
    }
  }
}
void init_edges(edges_t *edges) {
  edges->lower_edge = (double *)malloc(sizeof(double) * N_y * N_z);
  edges->upper_edge = (double *)malloc(sizeof(double) * N_y * N_z);
}
void send_edges(layers_t *layers, edges_t *edges) {
  if (process_info.rank != 0) {
    MPI_Isend(layers->last_layer, N_y * N_z, MPI_DOUBLE, process_info.rank - 1,
              0, MPI_COMM_WORLD, &send_req.up);
    MPI_Irecv(edges->upper_edge, N_y * N_z, MPI_DOUBLE, process_info.rank - 1,
              1, MPI_COMM_WORLD, &recv_req.up);
  }
  if (process_info.rank != process_info.size - 1) {
    double *prev_down_border =
        layers->last_layer +
        (layers_info.layer_height[process_info.rank] - 1) * N_y * N_z;
    MPI_Isend(prev_down_border, N_y * N_z, MPI_DOUBLE, process_info.rank + 1, 1,
              MPI_COMM_WORLD, &send_req.down);
    MPI_Irecv(edges->lower_edge, N_y * N_z, MPI_DOUBLE, process_info.rank + 1,
              0, MPI_COMM_WORLD, &recv_req.down);
  }
}
void swap_layers(layers_t *layers) {
  double *tmp = layers->last_layer;
  layers->last_layer = layers->current_layer;
  layers->current_layer = tmp;
}
void calc_inner(layers_t *layers) {
  for (int i = 1; i < layers_info.layer_height[process_info.rank] - 1; ++i) {
    for (int j = 1; j < N_y - 1; ++j) {
      for (int k = 1; k < N_z - 1; ++k) {
        double phi_i = (layers->last_layer[get_index(i - 1, j, k)] +

                        layers->last_layer[get_index(i + 1, j, k)]) /
                       sqr_H_x;
        calc_phi_j_phi_k_func(layers, phi_i, i, j, k);
      }
    }
  }
}
void calc_phi_j_phi_k_func(layers_t *layers, double phi_i, int i, int j,
                           int k) {
  double phi_j, phi_k;
  double tmp_max;
  phi_j = (layers->last_layer[get_index(i, j - 1, k)] +
           layers->last_layer[get_index(i, j + 1, k)]) /
          sqr_H_y;
  phi_k = (layers->last_layer[get_index(i, j, k - 1)] +
           layers->last_layer[get_index(i, j, k + 1)]) /
          sqr_H_z;
  layers->current_layer[get_index(i, j, k)] =
      (phi_i + phi_j + phi_k -
       rho(get_x(layers_info.offset[process_info.rank] + i), get_y(j),
           get_z(k))) /

      inverse_constant;
  tmp_max = fabs(layers->current_layer[get_index(i, j, k)] -
                 layers->last_layer[get_index(i, j, k)]);
  if (tmp_max > max_diff)
    max_diff = tmp_max;
}
void wait_edges() {
  if (process_info.rank != 0) {
    MPI_Wait(&send_req.up, MPI_STATUS_IGNORE);
    MPI_Wait(&recv_req.up, MPI_STATUS_IGNORE);
  }
  if (process_info.rank != process_info.size - 1) {
    MPI_Wait(&send_req.down, MPI_STATUS_IGNORE);
    MPI_Wait(&recv_req.down, MPI_STATUS_IGNORE);
  }
}
void calc_edges(layers_t *layers, edges_t *edges) {
  for (int j = 1; j < N_y - 1; ++j) {
    for (int k = 1; k < N_z - 1; ++k) {
      if (process_info.rank != 0) {
        int i = 0;
        double phi_i = (layers->last_layer[get_index(i + 1, j, k)] +
                        edges->upper_edge[get_index(0, j, k)]) /
                       sqr_H_x;
        calc_phi_j_phi_k_func(layers, phi_i, i, j, k);
      }
      if (process_info.rank != process_info.size - 1) {
        int i = layers_info.layer_height[process_info.rank] - 1;
        double phi_i = (layers->last_layer[get_index(i - 1, j, k)] +
                        edges->lower_edge[get_index(0, j, k)]) /
                       sqr_H_x;
        calc_phi_j_phi_k_func(layers, phi_i, i, j, k);
      }
    }
  }
}
void free_resources(layers_t *layers, edges_t *edges) {
  free(layers->last_layer);
  free(layers->current_layer);
  free(edges->lower_edge);
  free(edges->upper_edge);
  free(layers_info.layer_height);
  free(layers_info.offset);
}
double calc_max_delta(layers_t *layers) {
  double max_delta = 0.0;
  double tmp;
  double proc_max = 0.0;
  for (int i = 0; i < layers_info.layer_height[process_info.rank]; ++i) {
    for (int j = 0; j < N_y; ++j) {
      for (int k = 0; k < N_z; ++k) {
        tmp = fabs(layers->current_layer[get_index(i, j, k)] -
                   phi(get_x(i), get_y(j), get_z(k)));
        if (tmp > proc_max)
          proc_max = tmp;
      }
    }
  }
  MPI_Allreduce(&proc_max, &max_delta, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  return max_delta;
}
