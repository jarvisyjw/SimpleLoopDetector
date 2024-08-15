#include <iostream>
#include <chrono>
#include <thread>
#include <fstream>
#include <string>


int countLinesInFile(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Could not open the file: " << filename << std::endl;
        return -1;
    }

    int lineCount = 0;
    std::string line;
    while (std::getline(file, line)) {
        ++lineCount;
    }

    file.close();
    return lineCount;
}


void showProgressBar(int progress, int total) {
    int barWidth = 50;
    float progressRatio = static_cast<float>(progress) / total;

    std::cout << "[";
    int pos = barWidth * progressRatio;
    for (int i = 0; i < barWidth; ++i) {
        if (i < pos) {
            std::cout << "=";
        } else if (i == pos) {
            std::cout << ">";
        } else {
            std::cout << " ";
        }
    }
    std::cout << "] " << int(progressRatio * 100.0) << " %\r";
    std::cout.flush();
}

// Usage
// int main() {
//     int total = 100;

//     for (int i = 0; i <= total; ++i) {
//         showProgressBar(i, total);
//         std::this_thread::sleep_for(std::chrono::milliseconds(50));  // Simulate work being done
//     }

//     std::cout << std::endl;
//     return 0;
// }