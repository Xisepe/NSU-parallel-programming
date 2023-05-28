//
// Created by Maxim on 24.05.2023.
//

#ifndef LAB4_CONCURRENTINTQUEUE_HPP
#define LAB4_CONCURRENTINTQUEUE_HPP


#include <queue>
#include <mutex>
#include <condition_variable>

enum States {
    ALL_TASK_FINISHED = -1
};

class ConcurrentIntQueue {
private:
    std::queue<int> queue;
    std::mutex *mtx;
    std::condition_variable *cond;
public:
    ConcurrentIntQueue(std::mutex *lock, std::condition_variable *cond) {
        this->mtx = lock;
        this->cond = cond;
    }

    void push(const int task) {
        std::lock_guard<std::mutex> lock(*mtx);
        queue.push(task);
        cond->notify_one();
    }

    int pop() {
        int task;
        std::lock_guard<std::mutex> lock(*mtx);
        if (!queue.empty()) {
            task = queue.front();
            queue.pop();
        } else {
            task = ALL_TASK_FINISHED;
        }
        return task;
    }

    bool empty() {
        std::lock_guard<std::mutex> lock(*mtx);
        return queue.empty();
    }

    size_t size() {
        std::lock_guard<std::mutex> lock(*mtx);
        return queue.size();
    }
};

#endif//LAB4_CONCURRENTINTQUEUE_HPP
