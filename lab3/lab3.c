#include <errno.h>
#include <limits.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N_DIMS 2
#define x 0
#define y 1

void check_size_args(int n);
void init_cart(MPI_Comm *comm_grid, int *dims);
void init_coords(MPI_Comm *comm_grid, int *coords);
void init_dimension_communicators(MPI_Comm *comm_row, MPI_Comm *comm_col, MPI_Comm *comm_grid, int *coords);
void init_communicators(MPI_Comm *comm_row, MPI_Comm *comm_col, MPI_Comm *comm_grid, int *coords, int *dims);
void fill_test_diag_matrix(double *matrix, int n1, int n2, double diag, double value);
void init_matrices(double **A, double **B, double **C, int n1, int n2, int n3, char *argv[]);
void init_send_matrices(double **a, double **b, double **c, int n1, int n2, int n3, const int *dims);
void scatter_B(const double *B, double *b, MPI_Comm comm_row, const int *dims, int n2, int n3);
void send_matrices(const double *A, const double *B, double *a, double *b,
                   MPI_Comm comm_row, MPI_Comm comm_col,
                   const int dims[N_DIMS], const int coords[N_DIMS],
                   int n1, int n2, int n3);
void multiplication(const double *a, const double *b, double *c, int n1, int n2, int n3);
void gather(double *C, double *c, MPI_Comm comm_grid, const int dims[N_DIMS], int coords[N_DIMS], int comm_size, int n1, int n3);
void free_resources(double *A, double *B, double *C, double *a, double *b, double *c, MPI_Comm *row, MPI_Comm *col, MPI_Comm *grid, int rank);
void print_matrix(const double *matrix, int n1, int n2);

/*
 * This program provides only test demonstration of multiplication. In current implementation
 * it is assumed that multiplication performed on the two identical diagonal matrices.
 * To provide multiplication of arbitrary matrices it is necessary to implement reading from
 * file and adjust (void) (*init_matrices(double**, double**, double**, int, int, int)) function
 */
int main(int argc, char *argv[]) {
    if (argc != 6) {
        fprintf(stderr, "Required 3 arguments: n1 n2 n3 diag value, provided: %d\n", argc);
        return EXIT_FAILURE;
    }
    MPI_Init(&argc, &argv);
    double start_time, end_time, elapsed_time;
    int dims[N_DIMS] = {0};
    int coords[N_DIMS];
    int size;
    int rank;
    MPI_Comm comm_row;
    MPI_Comm comm_col;
    MPI_Comm comm_grid;

    int n1 = (int) strtol(argv[1], NULL, 10);
    int n2 = (int) strtol(argv[2], NULL, 10);
    int n3 = (int) strtol(argv[3], NULL, 10);

    check_size_args(n1);
    check_size_args(n2);
    check_size_args(n3);

    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Dims_create(size, N_DIMS, dims);
    init_communicators(&comm_row, &comm_col, &comm_grid, coords, dims);
    MPI_Comm_rank(comm_grid, &rank);

    double *A = NULL;
    double *B = NULL;
    double *C = NULL;
    if (rank == 0) {
        init_matrices(&A, &B, &C, n1, n2, n3, argv);
    }

    double *a = NULL;
    double *b = NULL;
    double *c = NULL;
    init_send_matrices(&a, &b, &c, n1, n2, n3, dims);
    if (rank == 0) {
        start_time = MPI_Wtime();
    }
    send_matrices(A, B, a, b, comm_row, comm_col, dims, coords, n1, n2, n3);
    multiplication(a, b, c, n1 / dims[y], n2, n3 / dims[x]);
    gather(C, c, comm_grid, dims, coords, size, n1, n3);
    if (rank == 0) {
        end_time = MPI_Wtime();
        elapsed_time = end_time - start_time;
        printf("%d,%lf\n", size, elapsed_time);
        //print_matrix(C, n1, n3);
    }

    free_resources(A, B, C, a, b, c, &comm_row, &comm_col, &comm_grid, rank);
    MPI_Finalize();

    return 0;
}
void check_size_args(int n) {
    if ((errno == ERANGE && (n == LONG_MAX || n == LONG_MIN)) || (errno != 0 && n == 0)) {
        perror("strtol");
        exit(EXIT_FAILURE);
    }
}
void init_cart(MPI_Comm *comm_grid, int *dims) {
    int periods[N_DIMS] = {0};
    MPI_Cart_create(MPI_COMM_WORLD, N_DIMS, dims, periods, 0, comm_grid);
}

void init_coords(MPI_Comm *comm_grid, int *coords) {
    int rank;
    MPI_Comm_rank(*comm_grid, &rank);
    MPI_Cart_coords(*comm_grid, rank, N_DIMS, coords);
}

void init_dimension_communicators(MPI_Comm *comm_row, MPI_Comm *comm_col, MPI_Comm *comm_grid, int *coords) {
    MPI_Comm_split(*comm_grid, coords[y], coords[x], comm_row);
    MPI_Comm_split(*comm_grid, coords[x], coords[y], comm_col);
}

void init_communicators(MPI_Comm *comm_row, MPI_Comm *comm_col, MPI_Comm *comm_grid, int *coords, int *dims) {
    init_cart(comm_grid, dims);
    init_coords(comm_grid, coords);
    init_dimension_communicators(comm_row, comm_col, comm_grid, coords);
}

void fill_test_diag_matrix(double *matrix, int n1, int n2, double diag, double value) {
    for (int i = 0; i < n1; ++i) {
        for (int j = 0; j < n2; ++j) {
            if (i == j) {
                matrix[i * n1 + j] = diag;
            } else {
                matrix[i * n1 + j] = value;
            }
        }
    }
}

void init_matrices(double **A, double **B, double **C, int n1, int n2, int n3, char *argv[]) {
    *A = (double *) malloc(n1 * n2 * sizeof(double));
    *B = (double *) malloc(n2 * n3 * sizeof(double));
    *C = (double *) malloc(n1 * n3 * sizeof(double));
    double diag = strtod(argv[4], NULL);
    double value = strtod(argv[5], NULL);
    fill_test_diag_matrix(*A, n1, n2, diag, value);
    fill_test_diag_matrix(*B, n2, n3, diag, value);
}

void init_send_matrices(double **a, double **b, double **c, int n1, int n2, int n3, const int *dims) {
    *a = (double *) malloc(n1 * n2 * sizeof(double) / dims[y]);
    *b = (double *) malloc(n2 * n3 * sizeof(double) / dims[x]);
    *c = (double *) malloc(n1 * n3 * sizeof(double) / (dims[x] * dims[y]));
}

void scatter_B(const double *B, double *b, MPI_Comm comm_row, const int *dims, int n2, int n3) {
    MPI_Datatype column_t;
    MPI_Datatype resized_column_t;
    MPI_Datatype receive_t;

    MPI_Type_vector(n2, n3 / dims[x], n3, MPI_DOUBLE, &column_t);
    MPI_Type_commit(&column_t);

    //https://rookiehpc.org/mpi/docs/mpi_type_create_resized/index.html это причина почему использую
    MPI_Type_create_resized(column_t, 0, n3 / dims[x] * sizeof(double), &resized_column_t);
    MPI_Type_commit(&resized_column_t);

    MPI_Type_contiguous(n2 * n3 / dims[x], MPI_DOUBLE, &receive_t);
    MPI_Type_commit(&receive_t);

    MPI_Scatter(B, 1, resized_column_t, b, 1, receive_t, 0, comm_row);

    MPI_Type_free(&column_t);
    MPI_Type_free(&resized_column_t);
    MPI_Type_free(&receive_t);
}

void send_matrices(const double *A, const double *B, double *a, double *b,
                   MPI_Comm comm_row, MPI_Comm comm_col,
                   const int dims[N_DIMS], const int coords[N_DIMS],
                   int n1, int n2, int n3) {
    if (coords[x] == 0) {
        MPI_Scatter(A, n1 * n2 / dims[y], MPI_DOUBLE,
                    a, n1 * n2 / dims[y], MPI_DOUBLE, 0, comm_col);
    }

    if (coords[y] == 0) {
        scatter_B(B, b, comm_row, dims, n2, n3);
    }

    MPI_Bcast(a, n1 * n2 / dims[y], MPI_DOUBLE, 0, comm_row);
    MPI_Bcast(b, n2 * n3 / dims[x], MPI_DOUBLE, 0, comm_col);
}

void multiplication(const double *A, const double *B, double *C, int n1, int n2, int n3) {
    for (int i = 0; i < n1; ++i) {
        double *c = C + i * n3;
        for (int j = 0; j < n3; ++j)
            c[j] = 0;
        for (int k = 0; k < n2; ++k) {
            const double *b = B + k * n3;
            double a = A[i * n2 + k];
            for (int j = 0; j < n3; ++j)
                c[j] += a * b[j];
        }
    }
}

void gather(double *C, double *c, MPI_Comm comm_grid, const int dims[N_DIMS], int coords[N_DIMS], int comm_size, int n1, int n3) {
    int *recvcounts = (int *) malloc(sizeof(int) * comm_size);
    int *displs = (int *) malloc(sizeof(int) * comm_size);

    MPI_Datatype recv_block_t, resized_recv_block_t, send_block_t;
    MPI_Type_contiguous(n1 * n3 / (dims[x] * dims[y]), MPI_DOUBLE, &send_block_t);
    MPI_Type_commit(&send_block_t);

    MPI_Type_vector(n1 / dims[y], n3 / dims[x], n3, MPI_DOUBLE, &recv_block_t);
    MPI_Type_commit(&recv_block_t);

    MPI_Type_create_resized(recv_block_t, 0, n3 / dims[x] * sizeof(double), &resized_recv_block_t);
    MPI_Type_commit(&resized_recv_block_t);

    for (int i = 0; i < comm_size; ++i) {
        recvcounts[i] = 1;
        MPI_Cart_coords(comm_grid, i, N_DIMS, coords);
        displs[i] = dims[x] * (n1 / dims[y]) * coords[y] + coords[x];
    }

    MPI_Gatherv(c, 1, send_block_t, C, recvcounts, displs, resized_recv_block_t, 0, comm_grid);

    MPI_Type_free(&recv_block_t);
    MPI_Type_free(&resized_recv_block_t);
    MPI_Type_free(&send_block_t);
}

void free_resources(double *A, double *B, double *C, double *a, double *b, double *c, MPI_Comm *row, MPI_Comm *col, MPI_Comm *grid, int rank) {
    if (rank == 0) {
        free(A);
        free(B);
        free(C);
    }
    free(a);
    free(b);
    free(c);

    MPI_Comm_free(grid);
    MPI_Comm_free(row);
    MPI_Comm_free(col);
}

void print_matrix(const double *matrix, int n1, int n2) {
    for (int i = 0; i < n1; ++i) {
        for (int j = 0; j < n2; ++j) {
            printf("%lf ", matrix[n1 * i + j]);
        }
        printf("\n");
    }
}
