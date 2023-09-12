#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#include<unistd.h>

int getDispls(int rank, int maxLines, int extraLines, int lines) {
        int displs;
        if (rank < extraLines) {
                displs = rank * maxLines;
        } else {
                displs = maxLines * extraLines + (rank - extraLines) * lines;
        }
        return displs;
}

void debug(int rank, const char* msg) {
        printf("DEBUG: Rank: %d, %s\n",rank,msg);
}
int main(int argc, char *argv[]) {
        int N = atoi(argv[1]);
        double eps = strtod(argv[2], NULL);
        double tao = strtod(argv[3], NULL);

        int size, rank;
        double *linesA;
        double *b;
        double *x;
        double *lastX;
        double *curX;
        double *mulAB;

        double localSqNorm, sqrNorm, localBNorm, bNorm;

        MPI_Init(&argc, &argv);
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

        int lines = N / size;
        int extraLines = N % size;
        int maxLines = lines;
        if (extraLines > 0) {
                maxLines++;
        }
        if (rank < extraLines) {
                lines++;
        }
        int displs = getDispls(rank, maxLines, extraLines, N / size);

        linesA = (double*) malloc((lines * N) * sizeof(double));
        b = (double*) malloc(lines * sizeof(double));
        lastX = (double*) malloc(lines * sizeof(double));
        curX = (double*) malloc((maxLines + 2) * sizeof(double));//last for size
        mulAB = (double*) malloc(lines * sizeof(double));


        localBNorm = 0;
        for (int i = 0; i < lines; ++i)
        {
                for(int j = 0; j < N; ++j) {
                        if ((displs + i) == j) {
                                linesA[i * N + j] = 2.0;
                        } else {
                                linesA[i * N + j] = 1.0;
                        }
                }
                b[i] = N + 1;
                curX[i] = 0;
                localBNorm += b[i] * b[i];
        }

        curX[maxLines] = lines;
        curX[maxLines + 1] = rank;

        MPI_Allreduce(&localBNorm, &bNorm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        double start_time;
        if (rank == 0) {
        start_time = MPI_Wtime();
        }

        do {
                for (int i = 0; i < lines; ++i)
                {
                        mulAB[i] = 0;
                }
                for (int rk = 0; rk < size; rk++) {
                int curDispls = getDispls(curX[maxLines + 1], maxLines, extraLines, N / size);
                        for (int i = 0; i < lines; ++i)
                        {
                                for (int j = 0; j < curX[maxLines]; j++) {
                                        mulAB[i] += linesA[i * N + j + curDispls] * curX[j];
                                }
                        }
                        MPI_Barrier(MPI_COMM_WORLDRLD);
                        MPI_Status status;
                        MPI_Sendrecv_replace(curX, maxLines + 2, MPI_DOUBLE, (rank + 1) % size, 0, (size + rank - 1) % size, 0, MPI_COMM_WORLD, &status);
                }
                //update curX with algo and count its norm
                localSqNorm = 0;

                for (int i = 0; i < lines; ++i)
                {
                        curX[i] = curX[i] - tao * (mulAB[i] - b[i]);
                        localSqNorm += (mulAB[i] - b[i]) * (mulAB[i] - b[i]);
                }

                MPI_Allreduce(&localSqNorm, &sqrNorm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        } while (sqrNorm / bNorm > eps * eps);
        if (rank == 0) {
                double end_time = MPI_Wtime();
                double elapsed_time = end_time - start_time;
                printf("%d,%lf\n", size, elapsed_time);
        }
        free(linesA);
        free(b);
        free(lastX);
        free(curX);
        free(x);
        free(mulAB);

        MPI_Finalize();

        return 0;
}
