//
// Created by Steven Roddan on 10/8/2024.
//

#ifndef CUPYRE_PERFORMANCETABLE_H
#define CUPYRE_PERFORMANCETABLE_H
#include "../Common.h"

#include <chrono>
#include <map>
#include <queue>
#include <string>
#include <sstream>

class DECLSPEC PerformanceTable {
private:
    class DECLSPEC TestData {
    public:
        std::string testName;
        std::map<std::string, std::chrono::duration<double>> testTimeMap;
        explicit TestData(std::string testName);
        void add(std::string subTestName, std::chrono::duration<double> timePoint);
    };

    std::map<std::string, TestData> tests;
    std::vector<std::string> headers;
public:

    void add(std::string testName, std::string subTestName, std::chrono::duration<double> timePoint);
    void print(std::ostream &ostream);
};

extern DECLSPEC PerformanceTable PTable;

#endif //CUPYRE_PERFORMANCETABLE_H
