#include <iostream>
#include <vector>
#include <yaml-cpp/yaml.h>

// DBoW2
#include "DBoW2.h" // defines OrbVocabulary and OrbDatabase
#include "tqdm.h"
// #include "detector.h"

// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/opencv.hpp>

using namespace DBoW2;
using namespace std;

vector<tuple<string, string, int>> filename_pairs;
tuple<string, string, int> filename_pair_tmp;
string filename00, filename01;
int label;
OrbVocabulary voc;
std::string vocab_path;
std::string image_path;
std::string pairs_path;
std::string scores_path;
vector<vector<cv::Mat > > features(2);

double computeScore();
void extractFeatures(string filename00, string filename01);
void loadFeatures(vector<vector<cv::Mat > > &features);
void changeStructure(const cv::Mat &plain, vector<cv::Mat> &out);
void testVocCreation(const vector<vector<cv::Mat > > &features);
void testDatabase(const vector<vector<cv::Mat > > &features);
void load_vocab_from_file(const std::string &file_path);
void parser(const std::string &config_file);
void load_pairs_from_file(const std::string file_path, const std::string image_path);
void showProgressBar(int progress, int total);
int countLinesInFile(const std::string& filename);

// Count the number of lines in a file
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

// A function for showing the progress bar
void showProgressBar(int progress, int total) {
  int barWidth = 100;
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
// yaml parser
void parser(const std::string &config_file) {
  try {
      YAML::Node config = YAML::LoadFile(config_file);
      if (!config["vocab_path"] || !config["image_path"] || !config["pairs_path"]) {
          throw std::runtime_error("Missing required fields in YAML file");
      }

      vocab_path = config["vocab_path"].as<std::string>();
      image_path = config["image_path"].as<std::string>();
      pairs_path = config["pairs_path"].as<std::string>();
      scores_path = config["scores_path"].as<std::string>();
      std::cout << "Parsed arguments from YAML file: " << config_file << std::endl;
      std::cout << "vocab_path: " << vocab_path << std::endl;
      std::cout << "image_path: " << image_path << std::endl;
      std::cout << "pairs_path: " << pairs_path << std::endl;
      std::cout << "scores_path: " << scores_path << std::endl;

  } catch (const YAML::Exception &e) {
      std::cerr << "Error parsing YAML file: " << e.what() << std::endl;
      throw std::invalid_argument("invalid file path or content");
  }
}

void wait()
{
  cout << endl << "Press enter to continue" << endl;
  getchar();
}

void load_vocab_from_file(const std::string &file_path)
{
  voc.loadFromTextFile(file_path);
  cout << "Loaded vocabulary from " << file_path << endl;
}

int main(int argc, char **argv)
{

  std::string config_file = "config/config.yaml";
  if (argc > 1) config_file = argv[1];
  if (argc > 2) std::cout << "Usage: " << argv[0] << " [config_file]" << std::endl;

  parser(config_file); // Load parameters from a YAML file

  load_pairs_from_file(pairs_path, image_path); // Load pairs from a file
  cout << "Loaded " << filename_pairs.size() << " pairs from " << pairs_path << endl;
  // debug output
  // cout << "first pair: " << std::get<0>(filename_pairs[0]) << " - " << std::get<1>(filename_pairs[0]) << " label:" << std::get<2>(filename_pairs[0]) << endl;
  // cout << "last pair: " << std::get<0>(filename_pairs[filename_pairs.size()-1]) << " - " << std::get<1>(filename_pairs[filename_pairs.size()-1]) << " label:" << std::get<2>(filename_pairs[filename_pairs.size()-1])<< endl;
  load_vocab_from_file(vocab_path); // Load vocabulary from a file
  cout << "Loaded vocabulary with " << voc.size() << " words." << endl;

  tqdm bar;
  int i = 0;
  int total_pairs = filename_pairs.size();
  bar.progress(i, total_pairs); // Initialize the progress bar
  // iterate through the pairs and extract features
  for (const auto &pair : filename_pairs) {
    bar.progress(i++, total_pairs); // Update the progress bar
    double score;
    extractFeatures(std::get<0>(pair), std::get<1>(pair));
    score = computeScore();
    // for debugging
    // cout << "Processing pair: " << std::get<0>(pair) << " - " << std::get<1>(pair) << " - "<< std::get<2>(pair) << " Score: " << score<< endl;
    // write the score to a file
    std::ofstream scores_file(scores_path, std::ios::app);
    if (!scores_file.is_open()) {
      std::cerr << "Error opening scores file: " << scores_path << std::endl;
      throw std::invalid_argument("invalid file path");
    }
    scores_file << std::get<0>(pair) << " " 
                << std::get<1>(pair) << " " 
                << std::get<2>(pair) << " "
                << score << endl;
    scores_file.close();
    // cout << "Score for pair " << i << ": " << score << endl;
  }
  // cout << "Extracted features from " << filename_pairs.size() << " pairs." << endl;
  
  // // vector<vector<cv::Mat > > features;
  // loadFeatures(features); // load features of images, with NIMAGES * len(features) matrix

  // testVocCreation(features);

  // wait();

  // testDatabase(features);

  return 0;
}

void load_pairs_from_file(const std::string file_path, const std::string image_path){
  std::ifstream file(file_path);
  if (!file.is_open()) {
      std::cerr << "Error opening file: " << file_path << std::endl;
      throw std::invalid_argument( "invalid file path" );
  }
  std::string line;
  int i = 0;
  int total_lines = countLinesInFile(file_path);
  tqdm bar;
  while (std::getline(file, line)) {
      // showProgressBar(i, total_lines);
      bar.progress(i, total_lines);
      std::istringstream iss(line);
      
      // Read the timestamp and filename from the line
      if (!(iss >> filename00 >> filename01 >> label)) {
          std::cerr << "Error parsing line: " << line << std::endl;
          continue;  // Skip to the next line if there's an error
      }
      std::stringstream ss00, ss01;
      ss00 << image_path <<'/'<< filename00;
      ss01 << image_path <<'/'<< filename01;
      filename_pair_tmp = std::make_tuple(ss00.str(), ss01.str(), label);
      filename_pairs.push_back(filename_pair_tmp);
      i++;
      }
      file.close();
}

void extractFeatures(string filename00, string filename01)
{
  cv::Mat image0 = cv::imread(filename00, 0);
  cv::Mat image1 = cv::imread(filename01, 0);
  // resize the images to a fixed size
  cv::resize(image0, image0, cv::Size(224, 224));
  cv::resize(image1, image1, cv::Size(224, 224));

  vector<cv::KeyPoint> keypoints0;
  vector<cv::KeyPoint> keypoints1;
  cv::Mat descriptors0;
  cv::Mat descriptors1;
  if (image0.empty() || image1.empty()) {
    cerr << "Could not open or find the images!" << endl;
    throw std::invalid_argument( "invalid image" );
  }
  // Detect and compute ORB features for the first image
  cv::Ptr<cv::ORB> orb = cv::ORB::create();
  cv::Mat mask0;
  cv::Mat mask1;
  orb->detectAndCompute(image0, mask0, keypoints0, descriptors0);
  orb->detectAndCompute(image1, mask1, keypoints1, descriptors1);
   // create a vector of 2 vectors of cv::Mat
  changeStructure(descriptors0, features[0]); // change the size of features to NIMAGES x len(decriptors)
  changeStructure(descriptors1, features[1]); // change the size of features to NIMAGES x len(decriptors)
}

double computeScore()
{
  BowVector v1, v2;
  voc.transform(features[0], v1);
  voc.transform(features[1], v2);
  double score = voc.score(v1, v2);
  return score;
}

// ----------------------------------------------------------------------------

void changeStructure(const cv::Mat &plain, vector<cv::Mat> &out)
{
  out.resize(plain.rows);

  for(int i = 0; i < plain.rows; ++i)
  {
    out[i] = plain.row(i);
  }
}

