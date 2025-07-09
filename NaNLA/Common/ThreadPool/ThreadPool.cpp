//
// Created by Steven Roddan on 8/21/2024.
//

#include "ThreadPool.h"

namespace NaNLA {
    namespace Common {
        DECLSPEC void ThreadPool::threadLoop() {
            while(true) {
                std::function<void()> job;
                {
                    std::unique_lock<std::mutex> queueLock(this->queueMutex);
                    mutexCondition.wait(queueLock, [this] {
                        return !functionQueue.empty() || isShutdown;
                    });

                    if (isShutdown) {
                        return;
                    }

                    auto threadId = std::this_thread::get_id();
                    job = functionQueue.front();
                    functionQueue.pop();
                }
                job();
            }
        }

        DECLSPEC ThreadPool::ThreadPool(uint64_t threads) {
            for(uint64_t i = 0; i < threads; i++) {
                this->threadPool.emplace_back(&ThreadPool::threadLoop, this);
            }
        }

        DECLSPEC ThreadPool::~ThreadPool() {
            {
                std::unique_lock<std::mutex> lock(queueMutex);
                isShutdown = true;
            }
            this->mutexCondition.notify_all();

            for(auto& thread : threadPool) {
                thread.join();
            }
        }
    } // Kernels
} // NaNLA