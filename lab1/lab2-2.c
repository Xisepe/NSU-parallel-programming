#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

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

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	int *lines = (int*) malloc(size * sizeof(int));
	int *displs = (int*) malloc(size * sizeof(int));
	int extraLines = N % size;

	for (int i = 0; i < size; ++i)
	{
		lines[i] = N / size;
		if (i < extraLines) {
		lines[i]++;
		}
	}

	displs[0] = 0;
	for (int i = 1; i < size; ++i)
	{
		displs[i] = displs[i - 1] + lines[i - 1];
	}
	

	linesA = (double*) malloc((lines[rank] * N) * sizeof(double));
	b = (double*) malloc(N * sizeof(double));//b дублируется для каждого процесса
	x = (double*) malloc(N * sizeof(double));
	lastX = (double*) malloc(N * sizeof(double));
	curX = (double*) malloc(N * sizeof(double));
	mulAB = (double*) malloc(lines[rank] * sizeof(double));

	//инициализация b и расчет нормы b
	//инициализация curX
	bNorm = 0.0;
	for (int i = 0; i < N; ++i)
	{
		b[i] = N + 1;
		bNorm += b[i] * b[i];
		curX[i] = 0;
	}

	for (int i = 0; i < lines[rank]; ++i)
	{
		for(int j = 0; j < N; ++j) {
			if ((displs[rank] + i) == j) {
				linesA[i * N + j] = 2.0;
			} else {
				linesA[i * N + j] = 1.0;
			}
		}
	}	

	double start_time;
	if (rank == 0) {
		start_time = MPI_Wtime();
	}

	do {
		for (int i = 0; i < N; ++i)
		{
			lastX[i] = curX[i];			
		}
		MPI_Allreduce(lastX, x, N, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		localSqNorm = 0;
		for (int i = 0; i < lines[rank]; ++i)
		{	
			mulAB[i] = 0;
			for (int j = 0; j < N; ++j)
			{
				mulAB[i] += linesA[i * N + j] * x[j];
			}
			curX[displs[rank] + i] = lastX[displs[rank] + i] - tao * (mulAB[i] - b[displs[rank] + i]);
			localSqNorm += (mulAB[i] - b[displs[rank] + i]) * (mulAB[i] - b[displs[rank] + i]);
		}

		MPI_Allreduce(&localSqNorm, &sqrNorm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	} while (sqrNorm / bNorm > eps * eps);

	if (rank == 0) {
		double end_time = MPI_Wtime();
		double elapsed_time = end_time - start_time;
		printf("%d,%lf\n", size, elapsed_time);
	}

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
