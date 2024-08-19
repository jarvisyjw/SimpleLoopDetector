#include <iostream>
#include <chrono>
#include <thread>
#include <fstream>
#include <string>
#include <yaml-cpp/yaml.h>
#include <stdexcept>

// Global variables
std::string vocab_path = "ORBvoc.txt";
std::string image_path = "images/";
std::string file_name = "timestamp_kf.txt";
std::string output_path = "DBoW.txt";


// Parse the arguments from a ymal file
void parser(const std::string &config_file) {
    try {
        YAML::Node config = YAML::LoadFile(config_file);
        
        if (!config["vocab_path"] || !config["image_path"] || !config["file_name"]) {
            throw std::runtime_error("Missing required fields in YAML file");
        }

        vocab_path = config["vocab_path"].as<std::string>();
        image_path = config["image_path"].as<std::string>();
        file_name = config["file_name"].as<std::string>();
        output_path = config["output_path"].as<std::string>();

        std::cout << "Parsed arguments from YAML file: " << config_file << std::endl;
        std::cout << "vocab_path: " << vocab_path << std::endl;
        std::cout << "image_path: " << image_path << std::endl;
        std::cout << "file_name: " << file_name << std::endl;
        std::cout << "output_path: " << output_path << std::endl;

    } catch (const YAML::Exception &e) {
        std::cerr << "Error parsing YAML file: " << e.what() << std::endl;
        throw std::invalid_argument("invalid file path or content");
    }
}

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

// A function for output the similarity score to a txt file
void output_to_file(const std::string file_path, std::tuple<float, int, double, double> &output){
  std::ofstream file(file_path);
  if (!file.is_open()) {
      std::cerr << "Error opening file: " << file_path << std::endl;
      throw std::invalid_argument( "invalid file path" );
  }
  for (const auto &o : output){
    file << std::get<0>(o) << " " << std::get<1>(o) << " " << std::get<2>(o) << std::endl;
  }
  file.close();
  std::cout << "Output to file: " << file_path << std::endl;
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