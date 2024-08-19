#include <iostream>
#include <chrono>
#include <thread>
#include <fstream>
#include <string>
#include <yaml-cpp/yaml.h>
#include <stdexcept>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>  // Make sure this is included for findFundamentalMat
#include <vector>
#include <iostream>
#include <string>
#include <thread>
#include <tuple>
#include "DBoW2.h" // defines OrbVocabulary and OrbDatabase

// Global variables
std::string vocab_path = "ORBvoc.txt";
std::string image_path = "images/";
std::string file_name = "timestamp_kf.txt";
std::string output_path = "DBoW.txt";
std::ios_base::openmode write_mode = std::ios::out;
bool first_write = true;    // Flag for the first write to the file
typedef std::tuple<double,double> Match;

using namespace DBoW2;

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
void output_to_file(const std::string file_path, std::tuple<int, int, float, double, double> &output){
  
  // Check if the first time open the file
  // If yes, open the file with write mode (clean the existing content)
  // If no, open the file with append mode
    if (first_write) {
        write_mode = std::ios::out;
        first_write = false;
    } else {
        write_mode = std::ios::app;
    }
  
    std::ofstream file(file_path, write_mode);
    
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << file_path << std::endl;
        throw std::invalid_argument( "invalid file path" );
    }

    file << std::get<0>(output) << " " << std::get<1>(output) << " " << std::get<2>(output) << " "
    << std::get<3>(output) << " " << std::get<4>(output) << std::endl;
    
    file.close();
    // std::cout << "Output to file: " << file_path << std::endl;
}

void changeStructure(const cv::Mat &plain, std::vector<cv::Mat> &out)
{
  out.resize(plain.rows);

  for(int i = 0; i < plain.rows; ++i)
    out[i] = plain.row(i);
}

class Retrieval {       // The class
  private:
    std::vector<std::vector<cv::Mat > > features;
    std::vector<cv::Mat > descs;
    std::vector<std::vector<cv::KeyPoint > > kpts;
    // std::vector<double> num_mathces;
    OrbDatabase db;
    const int rad; // search radius

  public:             // Access specifier

    Retrieval(const std::string vocab_path, const int rad) : rad(rad) {

      std::cout << "Loading the vocabulary " << vocab_path << std::endl;

      // load the vocabulary from disk
      OrbVocabulary voc;
      voc.loadFromTextFile(vocab_path);

      db = OrbDatabase(voc, false, 0); // false = do not use direct index
      // (so ignore the last param)
      // The direct index is useful if we want to retrieve the features that
      // belong to some vocabulary node.
      // db creates a copy of the vocabulary, we may get rid of "voc" now

    }

    void load_images_from_file(const std::string file_path, const std::string image_path){
        std::ifstream file(file_path);
        if (!file.is_open()) {
            std::cerr << "Error opening file: " << file_path << std::endl;
            throw std::invalid_argument( "invalid file path" );
        }
        std::string line;
        int i = 0;
        int total_lines = countLinesInFile(file_path);
        while (std::getline(file, line)) {
            showProgressBar(i, total_lines);
            std::istringstream iss(line);
            std::string timestamp, filename;
            // Read the timestamp and filename from the line
            if (!(iss >> timestamp >> filename)) {
                std::cerr << "Error parsing line: " << line << std::endl;
                continue;  // Skip to the next line if there's an error
            }
            std::stringstream ss;
            ss << image_path <<'/'<< filename;
            insert_image(cv::imread(ss.str(), 0));
            i++;
            }
            file.close();
      };


    void insert_image(const cv::Mat &image, bool show = false){

        if (image.empty()) {
        std::cerr << "Could not open or find the image!" << std::endl;
        throw std::invalid_argument( "invalid image" );
        }

        if (show){
            cv::imshow("test", image);
            cv::waitKey(0);
        }

        cv::Mat mask;
        std::vector<cv::KeyPoint> keypoints;
        std::vector<cv::Mat> feats;
        cv::Mat descriptors;
        cv::Ptr<cv::ORB> orb = cv::ORB::create();

        orb->detectAndCompute(image, mask, keypoints, descriptors);
        changeStructure(descriptors, feats);
        kpts.push_back(keypoints);
        features.push_back(feats);
        descs.push_back(descriptors);
        db.add(features.back());
    
    }

    Match match_pair(const int r, const int q) const {

      cv::BFMatcher matcher(cv::NORM_HAMMING, true);  // true = cross check
      cv::Mat ref_descs = descs.at(r);
      cv::Mat query_descs = descs.at(q);

      std::vector<std::vector<cv::DMatch>> knn_matches;
      matcher.knnMatch(ref_descs, query_descs, knn_matches, 1);  // k = 2 for ratio test and set cross check to false
      
      // Geometric Verification
      std::vector<cv::Point2f> points1, points2;
      for (const auto& m : knn_matches)
      {
        if (!m.empty())
        {
          points1.push_back(kpts[r][m[0].queryIdx].pt);
          points2.push_back(kpts[q][m[0].trainIdx].pt);
        }
      }
    
      cv::Mat iliers;
      cv::Mat F = cv::findFundamentalMat(points1, points2, cv::FM_RANSAC, 3, 0.99, iliers);
      int num_inliers = static_cast<int>(cv::countNonZero(iliers));
      int nmatches = static_cast<int>(knn_matches.size());
      
      return std::make_tuple(num_inliers, nmatches);
    }

      // if (!train_descriptors.empty() && !query_descriptors.empty()) {
      //     matcher.knnMatch(train_descriptors, query_descriptors, knn_matches, 2);  // k = 2 for ratio test
      // }  // Finds the 2 best matches for each descriptor

      // // Apply the ratio test
      // const float ratio_thresh = 0.75f;  // Recommended by Lowe in the SIFT paper
      // std::vector<cv::DMatch> good_matches;
      // for (size_t i = 0; i < knn_matches.size(); i++) {
      //     if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance) {
      //       good_matches.push_back(knn_matches[i][0]);
      //     }
      // }

      // std::vector<cv::Point2f> points1, points2;
      // for (size_t i = 0; i < good_matches.size(); i++) {
      //     points1.push_back(kpts[ti][good_matches[i].trainIdx].pt);
      //     points2.push_back(kpts[qi][good_matches[i].queryIdx].pt);
      // }

      // cv::Mat iliers;
      // cv::Mat F = cv::findFundamentalMat(points1, points2, cv::FM_RANSAC, 3, 0.99, iliers);

      // int num_inliers = static_cast<int>(cv::countNonZero(iliers));
      // int nmatches = static_cast<int>(good_matches.size());
      
      // NumMatch output(num_inliers, nmatches);

      // return output;
      // output.emplace_back(num_inliers, nmatches);


      // Compute relative pose
      // cv::Mat inliers;
      // cv::Mat E = cv::findEssentialMat(points1, points2, 1.0, cv::Point2d(0, 0), cv::RANSAC, 0.999, 1.0, inliers);



      // // Draw matches
      // cv::Mat img_matches;
      // cv::drawMatches(img1, keypoints1, img2, keypoints2, good_matches, img_matches, 
      //                 cv::Scalar::all(-1), cv::Scalar::all(-1), 
      //                 std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

      // // Display matches
      // cv::imshow("Good Matches", img_matches);
      // cv::waitKey(0);
      // MatchList output;

      // for (const auto &pair_list : knn_matches){
      //   if (!pair_list.empty()){
      //     const auto &pair = pair_list.back();

      //     auto trainpt = (kps[ti][pair.trainIdx].pt);
      //     auto querypt = (kps[qi][pair.queryIdx].pt);

      //     output.emplace_back(trainpt.x, trainpt.y, querypt.x, querypt.y, pair.distance);
      //   }
      // }

      // return output;
    // }

    int get_num_images() const {
      return static_cast<int>(features.size());
    }

    std::tuple<int, int, float, double, double> query(const int i) const {
      /*** 
       Returns
       A tuple of
         (int) Query idx,
         (int) index of the matching image, 
         (float) similarity score, 
         (double) number of matches, 
         (double) number of inliers
      ***/ 
      
      if ((i >= static_cast<int>(features.size())) || (i < 0))
        throw std::invalid_argument( "index invalid" );

      QueryResults ret;
      db.query(features[i], ret, features.size(), features.size());
    //   std::cout << "Querying the database: " << std::endl;
      // std::cout << "Query Results: " << ret << std::endl;
      std::tuple<int, int, float, double, double> output(0, -1, -1, 0.0, 0.0);
      
      for (const auto &r : ret){
        int j = r.Id;
        // only forward search and avoid self matching
        if ((i-j>0) && (r.Score > std::get<0>(output)))
        {
          // std::cout << "Mathing Image " << i << " with Image " << j << std::endl;
          Match num_matches = match_pair(i, j);
          int num_inliers = std::get<0>(num_matches);
          int nmatches = std::get<1>(num_matches);
          output = std::make_tuple(i, j, r.Score, nmatches, num_inliers);
          // NumMatch matches = match_pair(i, j);
          // output = std::make_tuple(r.Score, j, matches);
        }
      }
      return output;
    }
};

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