#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <omp.h>

constexpr int N = 30000;
constexpr double t = 0.00001;
constexpr double eps = 0.0000000001;


double A[N][N];
double b[N];
double x[N];
double tmp[N];

void initArrays() {
    for (int i = 0; i < N; i++) {
        b[i] = N + 1;
        for (int j = 0; j < N; j++) {
            A[i][j] = i == j ? 2.0 : 1.0;
        }
    }
}

double getNormB() {
    double normb = 0.0;
#pragma omp parallel for reduction(+ \
                                   : normb)
    for (int i = 0; i < N; i++) {
        normb += b[i] * b[i];
    }
    normb = std::sqrt(normb);
    return normb;
}

void printControl() {
    std::cout << std::setprecision(15) << x[0] << " " << x[1] << " " << x[2] << " " << x[3] << "\n";
}

void oneParallelBlock(int numberOfThreads, double normb, int n) {
    omp_set_num_threads(numberOfThreads);
    double norm = 0.0;
    bool stop = false;
#pragma omp parallel
    while (!stop) {
#pragma omp for schedule(static, n) reduction(+ \
                                               : norm)
        for (int i = 0; i < N; i++) {
            tmp[i] = -b[i];
            for (int j = 0; j < N; j++) {
                tmp[i] += A[i][j] * x[j];
            }
            norm += tmp[i] * tmp[i];
            tmp[i] = t * tmp[i];
        }
#pragma omp for schedule(static, n)
        for (int i = 0; i < N; i++) {
            x[i] -= tmp[i];
        }
#pragma omp single
        {
            if (norm < eps * normb) {
                stop = true;
            }
            norm = 0.0;
        }
    }
}

void parallelForForEachParallelBlock(int numberOfThreads, double normb, int n) {
    omp_set_num_threads(numberOfThreads);

    double norm = 0.0;
    bool flag = true;
    while (flag) {
#pragma omp parallel for schedule(static, n) reduction(+ \
                                                        : norm)
        for (int i = 0; i < N; i++) {
            tmp[i] = -b[i];
            for (int j = 0; j < N; j++) {
                tmp[i] += A[i][j] * x[j];
            }
            norm += tmp[i] * tmp[i];
            tmp[i] = t * tmp[i];
        }
#pragma omp parallel for schedule(static, n)
        for (int i = 0; i < N; i++) {
            x[i] = x[i] - tmp[i];
        }
        if (norm < eps * normb) {
            flag = false;
        }
        norm = 0.0;
    }
}

int main(int argc, char *argv[]) {
    initArrays();
    double normb = getNormB();
    int numberOfThreads = std::stoi(argv[1]);
    int program = std::stoi(argv[2]);
	int chunks = N/std::stoi(argv[3]);
    auto t0 = std::chrono::steady_clock::now();
    if (program == 1) {
        oneParallelBlock(numberOfThreads, normb, chunks);
    } else {
        parallelForForEachParallelBlock(numberOfThreads, normb, chunks);
    }
    auto t1 = std::chrono::steady_clock::now();
    std::cout << numberOfThreads << "," << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() / 1000.0 << "\n";

    return 0;
}
