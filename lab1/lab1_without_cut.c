#include &lt; mpi.h & gt;
#include &lt; stdio.h & gt;
#include &lt; stdlib.h & gt;
#include &lt; sys / time.h & gt;
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
  double localSqNorm, sqrNorm, bNorm;
  struct timeval st, end;
  MPI_Init(&amp; argc, &amp; argv);
  MPI_Comm_size(MPI_COMM_WORLD, &amp; size);
  MPI_Comm_rank(MPI_COMM_WORLD, &amp; rank);
  int *lines = (int *)malloc(size * sizeof(int));
  int *displs = (int *)malloc(size * sizeof(int));
  int extraLines = N % size;
  for (int i = 0; i & lt; size; ++i) {
    lines[i] = N / size;
    if (i & lt; extraLines) {
      lines[i]++;
    }
  }
  displs[0] = 0;
  for (int i = 1; i & lt; size; ++i) {
    displs[i] = displs[i - 1] + lines[i - 1];
  }

  linesA = (double *)malloc((lines[rank] * N) * sizeof(double));
  b = (double *)malloc(N * sizeof(double)); // b дублируется для
  каждого процесса x = (double *)malloc(N * sizeof(double));
  lastX = (double *)malloc(N * sizeof(double));
  curX = (double *)malloc(N * sizeof(double));
  mulAB = (double *)malloc(lines[rank] * sizeof(double));
  //инициализация b и расчет нормы b
  //инициализация curX
  bNorm = 0.0;
  for (int i = 0; i & lt; N; ++i) {
    b[i] = N + 1;
    bNorm += b[i] * b[i];
    curX[i] = 0;
  }

  for (int i = 0; i & lt; lines[rank]; ++i) {
    for (int j = 0; j & lt; N; ++j) {
      if ((displs[rank] + i) == j) {
        linesA[i * N + j] = 2.0;
      } else {
        linesA[i * N + j] = 1.0;
      }
    }
  }
  gettimeofday(&amp; st, NULL);
  do {
    for (int i = 0; i & lt; N; ++i) {
      lastX[i] = curX[i];
    }
    MPI_Allreduce(lastX, x, N, MPI_DOUBLE, MPI_SUM,

                  MPI_COMM_WORLD);

    localSqNorm = 0;
    for (int i = 0; i & lt; lines[rank]; ++i) {
      mulAB[i] = 0;
      for (int j = 0; j & lt; N; ++j) {
        mulAB[i] += linesA[i * N + j] * x[j];
      }
      curX[displs[rank] + i] =
          lastX[displs[rank] + i] - tao

                                        * (mulAB[i] - b[displs[rank] + i]);

      localSqNorm += (mulAB[i] - b[displs[rank] + i]) *

                     (mulAB[i] - b[displs[rank] + i]);
    }
    MPI_Allreduce(&amp; localSqNorm, &amp; sqrNorm, 1, MPI_DOUBLE, MPI_SUM,

                                           MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
  } while (sqrNorm / bNorm & gt; eps * eps);
  gettimeofday(&amp; end, NULL);
  printf(&quot; % d, % ld\n & quot;
         , rank, (end.tv_sec - st.tv_sec) * 1000000 + end.tv_usec - st.tv_usec);
  printf(&quot; #rank
         : % d | lines
         : % d / % lf, % lf, % lf, % lf\n & quot;
         , rank, lines[rank], lastX[displs[rank] + 0], lastX[displs[rank] + 1],
         lastX[displs[rank] + 2], lastX[displs[rank] + 3]);

  free(lines);
  free(displs);
  free(linesA);
  free(b);
  free(lastX);
  free(curX);
  free(x);
  free(mulAB);
  MPI_Finalize();
  return 0;
}
