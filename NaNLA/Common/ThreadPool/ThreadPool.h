//
// Created by Steven Roddan on 8/21/2024.
//

#ifndef CUPYRE_THREADPOOL_H
#define CUPYRE_THREADPOOL_H

#include <cstdint>
#include <functional>
#include <future>
#include <iostream>
#include <queue>
#include <thread>
#include <vector>
#include "../Common.h"


namespace NaNLA {
    namespace Common {

        class ThreadPool {
        private:
            std::atomic<bool> isShutdown{false};
            std::vector<std::thread> threadPool;
            std::queue<std::function<void()>> functionQueue;

            std::mutex queueMutex;
            std::condition_variable mutexCondition;

            DECLSPEC void threadLoop();

        public:
            explicit DECLSPEC ThreadPool(uint64_t threads);

            template<class F, class... Args>
            inline auto queue(F&& f, Args&&... args) -> std::future<decltype(f(args...))> {
                using Return_T = decltype(f(args...));

                auto job = std::make_shared<std::packaged_task<Return_T()>>(
                        std::bind(std::forward<F>(f), std::forward<Args>(args)...)
                        );

                std::future<Return_T> result = job->get_future();
                {
                    std::unique_lock<std::mutex> queueLock(this->queueMutex);
                    this->functionQueue.emplace([job]() {
                        (*job)();
                    });
                }
                mutexCondition.notify_one();
                return result;
            }

            DECLSPEC ~ThreadPool();
        };

    } // Kernels
} // NaNLA


#endif //CUPYRE_THREADPOOL_H
