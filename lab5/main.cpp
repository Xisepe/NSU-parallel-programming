#include "ConcurrentIntQueue.hpp"
#include <condition_variable>
#include <iostream>
#include <mpi.h>
#include <mutex>
#include <random>
#include <thread>

enum AppState {
    QUEUE_SIZE = 10000,
    TASK_LOWER_BOUND = 10000,
    TASK_UPPER_BOUND = 1000000,
    REQUEST_TASK_TAG = 777,
    RESPONSE_TASK_TAG = 888,
    STOP_WORK = -1
};


typedef struct {
    int threadNumber;
    int rank;
    std::unique_ptr<ConcurrentIntQueue> queue;
    std::mutex mutex;
    std::condition_variable executeCond;
    std::condition_variable waitCond;
    bool running;
    bool taskReload;
    std::condition_variable reloadCond;
    int iterations;
    int executedTaskNumber;
    double taskResult;
} ApplicationContext;

void fillQueue(const ApplicationContext *ctx, size_t queueSize, int lowBound, int highBound) {
    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_int_distribution<std::mt19937::result_type> dist(lowBound, highBound);
    for (size_t i = 0; i < queueSize; ++i) {
        ctx->queue->push(dist(rng));
    }
}

void taskRequestListener(ApplicationContext *ctx) {
    MPI_Status status;
    int sendTo;
    while (ctx->running) {
        MPI_Recv(&sendTo, 1, MPI_INT, MPI_ANY_SOURCE, REQUEST_TASK_TAG, MPI_COMM_WORLD, &status);
        if (sendTo == STOP_WORK) continue;
        int task = ctx->queue->pop();
        MPI_Send(&task, 1, MPI_INT, sendTo, RESPONSE_TASK_TAG, MPI_COMM_WORLD);
    }
}

void iterateThenFinish(ApplicationContext *ctx) {
    MPI_Barrier(MPI_COMM_WORLD);
    if (ctx->iterations > 0) {
        ctx->taskReload = true;
        ctx->iterations--;
        fillQueue(ctx, QUEUE_SIZE, TASK_LOWER_BOUND, TASK_UPPER_BOUND);
        MPI_Barrier(MPI_COMM_WORLD);
        ctx->taskReload = false;
        ctx->reloadCond.notify_one();
    } else {
        ctx->running = false;
        int stopTask = STOP_WORK;
        MPI_Send(&stopTask, 1, MPI_INT, ctx->rank, REQUEST_TASK_TAG, MPI_COMM_WORLD);
        std::unique_lock<std::mutex> lock(ctx->mutex);
        ctx->executeCond.notify_one();
    }
}

void taskRequestProvider(ApplicationContext *ctx) {
    MPI_Status status;
    int task;

    while (ctx->running) {
        while (!ctx->queue->empty()) {
            std::unique_lock<std::mutex> lock(ctx->mutex);
            ctx->executeCond.notify_one();
            ctx->waitCond.wait(lock);
        }

        int finished = 0;

        for (int i = 0; i < ctx->threadNumber; ++i) {
            if (i == ctx->rank) continue;
            MPI_Send(&ctx->rank, 1, MPI_INT, i, REQUEST_TASK_TAG, MPI_COMM_WORLD);
            MPI_Recv(&task, 1, MPI_INT, i, RESPONSE_TASK_TAG, MPI_COMM_WORLD, &status);
            if (task != ALL_TASK_FINISHED) {
                ctx->queue->push(task);
            } else {
                finished++;
            }
        }

        if (finished == ctx->threadNumber - 1) {
            iterateThenFinish(ctx);
        }
    }
}

void doTask(int task, ApplicationContext *ctx) {
    for (int i = 0; i < task; ++i) {
        ctx->taskResult += sqrt(i);
    }
}

void taskExecutor(ApplicationContext *ctx) {
    while (ctx->running) {
        while (true) {
            int task = ctx->queue->pop();
            if (task == ALL_TASK_FINISHED) break;
            doTask(task, ctx);
            ctx->executedTaskNumber++;
        }
        while (ctx->running && ctx->queue->empty()) {
            std::unique_lock<std::mutex> lock(ctx->mutex);
            if (ctx->taskReload) {
                ctx->reloadCond.wait(lock);
                continue;
            }
            ctx->waitCond.notify_one();
            ctx->executeCond.wait(lock);
        }
    }
    std::cout << "rank:" << ctx->rank << ",executed:" << ctx->executedTaskNumber << "\n";
}

void initContext(ApplicationContext *ctx, int rank, int threadNumber, int iterations) {
    ctx->rank = rank;
    ctx->threadNumber = threadNumber;
    ctx->queue = std::make_unique<ConcurrentIntQueue>(&ctx->mutex, &ctx->executeCond);
    ctx->running = true;
    ctx->taskResult = 0;
    ctx->executedTaskNumber = 0;
    ctx->taskReload = false;
    ctx->iterations = iterations;
}

void runThreads(int rank, int size) {
    ApplicationContext context;
    initContext(&context, rank, size, 3);
    fillQueue(&context, QUEUE_SIZE, TASK_LOWER_BOUND, TASK_UPPER_BOUND);
    MPI_Barrier(MPI_COMM_WORLD);
    std::thread worker(taskExecutor, &context);
    std::thread listener(taskRequestListener, &context);
    std::thread provider(taskRequestProvider, &context);

    worker.join();
    listener.join();
    provider.join();
}

int main(int argc, char *argv[]) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double start = MPI_Wtime();
    runThreads(rank, size);
    double end = MPI_Wtime();

    if (rank == 0) {
        std::cout << size << "," << end - start << "\n";
    }


    MPI_Finalize();
    return 0;
}
