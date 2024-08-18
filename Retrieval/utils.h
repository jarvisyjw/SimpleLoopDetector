#include <iostream>
#include <chrono>
#include <thread>
#include <fstream>
#include <string>

// parse the arguments from a ymal file, contains std::string vocab_path = "ORBvoc.txt"; std::string image_path = "images/"; std::string file_name = "timestamp_kf.txt";
// set the vocab_path, image_path, file_name as global variables
// do not use opencv
void parser(const std::string &config_file, std::string &vocab_path, std::string &image_path, std::string &file_name){
    std::ifstream file(config_file);
    if (!file.is_open()) {
        std::cerr << "Could not open the file: " << config_file << std::endl;
        throw std::invalid_argument( "invalid file path" );
    }

    std::string line;
    while (std::getline(file, line)) {
        if (line.find("vocab_path") != std::string::npos) {
            vocab_path = line.substr(line.find("=") + 1);
        } else if (line.find("image_path") != std::string::npos) {
            image_path = line.substr(line.find("=") + 1);
        } else if (line.find("file_name") != std::string::npos) {
            file_name = line.substr(line.find("=") + 1);
        }
    }

    file.close();
}

void parser(const std::string &config_file, std::string &vocab_path, std::string &image_path, std::string &file_name){
    cv::FileStorage fs(config_file, cv::FileStorage::READ);
    if (!fs.isOpened()){
        std::cerr << "Error opening file: " << config_file << std::endl;
        throw std::invalid_argument( "invalid file path" );
    }
    fs["vocab_path"] >> vocab_path;
    fs["image_path"] >> image_path;
    fs["file_name"] >> file_name;
    fs.release();
}


void parse_args(const std::string &config_file, std::string &vocab_path, std::string &image_path, std::string &file_name){
  cv::FileStorage fs(config_file, cv::FileStorage::READ);
  if (!fs.isOpened()){
    std::cerr << "Error opening file: " << config_file << std::endl;
    throw std::invalid_argument( "invalid file path" );
  }
  fs["vocab_path"] >> vocab_path;
  fs["image_path"] >> image_path;
  fs["file_name"] >> file_name;
  fs.release();
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
void output_to_file(const std::string file_path, const std::vector<std::tuple<int, int, double>> &output){
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