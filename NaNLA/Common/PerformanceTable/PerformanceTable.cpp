//
// Created by Steven Roddan on 10/8/2024.
//

#include "PerformanceTable.h"

PerformanceTable::TestData::TestData(std::string testName) : testName(testName){ ; }

void PerformanceTable::TestData::add(std::string subTestName, std::chrono::duration<double> timePoint) {
    this->testTimeMap.insert(std::pair<std::string, std::chrono::duration<double>>(subTestName, timePoint));
}

void PerformanceTable::add(std::string testName, std::string subTestName,
                           std::chrono::duration<double> timePoint) {
    auto headerItt = std::find(headers.begin(), headers.end(), subTestName);
    if(headerItt == headers.end()) {
        headers.push_back(subTestName);
    }

    auto it = this->tests.find(testName);
    if(it != this->tests.end()) {
        it->second.add(subTestName, timePoint);
    } else {
        auto x = this->tests.insert(std::pair(testName, TestData(testName)));
        x.first->second.add(subTestName, timePoint);
    }
}

void PerformanceTable::print(std::ostream &ostream) {
    const int SPACING_SIZE = 3;
    const int STR_PRECISION = 4;
    const double MICROSECONDS_TO_SECONDS = 1000000.00;

    std::vector<int> columnWidths;
    columnWidths.resize(headers.size() + 1);

    ostream << "Size: " << headers.size() << std::endl;

    // get max spacing for test names
    int maxTestNameSpace = 0;
    for(const auto& kv : tests) {
        maxTestNameSpace = kv.first.size() > maxTestNameSpace ? kv.first.size() : maxTestNameSpace;
    }
    columnWidths.at(0) = maxTestNameSpace + SPACING_SIZE;

    int columnIncr = 1;
    for(const auto& header : headers) {
        int maxColumnSpace = header.size();
        for(const auto& testKV : tests) {
            auto timeMap = testKV.second;
            std::stringstream ss;

            auto it = testKV.second.testTimeMap.find(header);
            if(it != testKV.second.testTimeMap.end()) {
                ss << std::fixed << std::setprecision(STR_PRECISION)
                   << testKV.second.testTimeMap.at(header).count() / MICROSECONDS_TO_SECONDS;
            }
            maxColumnSpace = ss.str().size() > maxColumnSpace ? ss.str().size() : maxColumnSpace;
        }
        columnWidths.at(columnIncr++) = maxColumnSpace + SPACING_SIZE;
    }

    ostream << std::setw(columnWidths.at(0)) << "|";
    int printColumnIncr = 1;
    for(const auto& header : headers) {
        ostream << std::setw(columnWidths.at(printColumnIncr++)) << (header + " |");
    }
    ostream << std::endl;

    // prints border ------
    for(uint64_t i = 0; i < columnWidths.size(); i++) {
        ostream << std::right << std::setfill('-') << std::setw(columnWidths.at(i)) << "+";
    }
    ostream << std::endl;

    for(const auto& testKV : tests) {
        ostream << std::setfill(' ') << std::setw(columnWidths.at(0)) << (testKV.first + " |");
        for(uint64_t i = 1; i < columnWidths.size(); i++) {
            auto it = testKV.second.testTimeMap.find(headers.at(i-1));
            if(it != testKV.second.testTimeMap.end()) {
                ostream << std::setw(columnWidths.at(i) - SPACING_SIZE) << std::fixed << std::setprecision(STR_PRECISION) << it->second.count() << "s |";
            } else {
                ostream << std::setw(columnWidths.at(i)) << "n/a. |";
            }
        }
        ostream << std::endl;
    }

    ostream << std::endl;
}

DECLSPEC PerformanceTable PTable;
