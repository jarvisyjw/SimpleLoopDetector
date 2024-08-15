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
#include "utils.h"

using namespace DBoW2;

// struct Image
// {
//   std::string timestamp;
//   std::string filename;
//   // cv::Mat image;
// };


void changeStructure(const cv::Mat &plain, std::vector<cv::Mat> &out)
{
  out.resize(plain.rows);

  for(int i = 0; i < plain.rows; ++i)
    out[i] = plain.row(i);
}

// void estimateAffine3D(const py::array_t<uint8_t> &pointsA, const py::array_t<uint8_t> &pointsB){

//         const py::buffer_info buf = pointsA.request();
//         const cv::Mat lpts(buf.shape[0], 3, CV_64F, (double *)buf.ptr);
//         std::cout << lpts << std::endl;
//         // const cv::Mat image(buf.shape[0], buf.shape[1], CV_64F, (unsigned char*)buf.ptr);

// }

// typedef std::vector<std::tuple<double, double, double, double, double>> MatchList;
// typedef std::vector<std::tuple<double,double>> MatchList;
typedef std::tuple<int, int> NumMatch;


class Retrieval {       // The class
  private:
    std::vector<std::vector<cv::Mat > > features;
    std::vector<cv::Mat > descs;
    std::vector<std::vector<cv::KeyPoint > > kpts;
    // std::vector<double> num_mathces;
    OrbDatabase db;
    const int rad; // search radius

  public:             // Access specifier

    Retrieval(const std::string vocab_path, const int rad) : rad(rad){

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

    void load_from_file(const std::string file_path, const std::string image_path){
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
          // std::cout << "Loading image: " << ss.str() << std::endl;
          insert_image(cv::imread(ss.str(), 0));
          i++;
          }
          file.close();
          std::cout << "Loaded " << i << " images from file: " << file_path << std::endl;
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

    std::tuple<double,double> match_pair(const int r, const int q) const {

      cv::BFMatcher matcher(cv::NORM_HAMMING, true);  // true = cross check
      cv::Mat ref_descs = descs.at(r);
      cv::Mat query_descs = descs.at(q);

      std::vector<std::vector<cv::DMatch>> knn_matches;
      matcher.knnMatch(ref_descs, query_descs, knn_matches, 1);  // k = 2 for ratio test and set cross check to false
      // std::cout << "Number of matches: " << knn_matches.size() << std::endl;
      // const float ratio_thresh = 0.75f;  // Recommended by Lowe in the SIFT paper
      // std::vector<cv::DMatch> good_matches;
      // for (size_t i = 0; i < knn_matches.size(); i++) {
      //     if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance) {
      //       good_matches.push_back(knn_matches[i][0]);
      //     }
      // }
      // return good_matches.size();
      // num_mathces.push_back(good_matches.size());
      std::cout << "Number of matches: " << knn_matches.size() << std::endl;
      std::cout << "Start Geometric Verification" << std::endl;

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


      // for (size_t i = 0; i < knn_matches.size(); i++) {
      //     points1.push_back(kpts[r][knn_matches[i].queryIdx].pt);
      //     points2.push_back(kpts[q][knn_matches[i].trainIdx].pt);
      // }

      std::cout << "Points1: " << points1.size() 
      << " Points2: " << points2.size() << std::endl;

      cv::Mat iliers;
      cv::Mat F = cv::findFundamentalMat(points1, points2, cv::FM_RANSAC, 3, 0.99, iliers);
      std::cout << "Fundamental Matrix: " << F << std::endl;
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

    std::tuple<float, int, double, double> query(const int i) const {

      if ((i >= static_cast<int>(features.size())) || (i < 0))
        throw std::invalid_argument( "index invalid" );

      QueryResults ret;
      db.query(features[i], ret, features.size());
      std::cout << "Querying the database: " << std::endl;
      // std::cout << "Query Results: " << ret << std::endl;
      std::tuple<float, int, double, double> output(-1, -1, 0.0, 0.0);
      for (const auto &r : ret){
        int j = r.Id;
        // only forward search and avoid self matching
        if ((i-j>0) && (r.Score > std::get<0>(output)))
        {
          std::cout << "Mathing Image " << i << " with Image " << j << std::endl;
          std::tuple<double,double> num_matches = match_pair(i, j);
          int num_inliers = std::get<0>(num_matches);
          int nmatches = std::get<1>(num_matches);
          output = std::make_tuple(r.Score, j, nmatches, num_inliers);
          // NumMatch matches = match_pair(i, j);
          // output = std::make_tuple(r.Score, j, matches);
        }
      }
      return output;
    }
};


const int NIMAGES = 4;

int main(int argc, char **argv)
{
  std::string vocab_path = "ORBvoc.txt";
  std::string image_path = "images/";
  std::string file_name = "timestamp_kf.txt";

  if (argc > 1)
    vocab_path = argv[1];
  if (argc > 2)
    image_path = argv[2];
  if (argc > 3)
    file_name = argv[3];

  // std::cout << "Loading the vocabulary " << vocab_path << std::endl;

  Retrieval LoopDetector(vocab_path, 1);

  LoopDetector.load_from_file(file_name, image_path);


  // for(int i = 0; i < NIMAGES; ++i)
  // {
  //   std::stringstream ss;
  //   ss << image_path <<"image" << i << ".png";
  //   LoopDetector.insert_image(cv::imread(ss.str(), 0));
  // }

    for(int i = 0; i < NIMAGES; ++i)
    { 
      std::cout << "Start Retrieval." << std::endl;
        auto output = LoopDetector.query(i);
        std::cout << "Searching for Image " << i << " Score: " << std::get<0>(output) 
        << " Index: " << std::get<1>(output) << " Num Matches: " << std::get<2>(output)
        << "Inliers after GV" << std::get<3>(output) << std::endl;
        // << std::get<1>(std::get<2>(output)) << std::endl;
        // << std::endl;
    }

    return 0;
  // load the vocabulary from disk
//   OrbVocabulary voc;
//   voc.loadFromTextFile(vocab_path);

//   OrbDatabase db(voc, false, 0); // false = do not use direct index
  // (so ignore the last param)
  // The direct index is useful if we want to retrieve the features that
  // belong to some vocabulary node.
  // db creates a copy of the vocabulary, we may get rid of "voc" now

//   cv::Mat image = cv::imread(image_path, 0);

//   cv::imshow("test", image);
//   cv::waitKey(0);

//   cv::Mat mask;
//   std::vector<cv::KeyPoint> keypoints;
//   cv::Mat descriptors;

//   cv::Ptr<cv::ORB> orb = cv::ORB::create();
//   orb->detectAndCompute(image, mask, keypoints, descriptors);

//   std::vector<cv::Mat > feats;
//   changeStructure(descriptors, feats);

//   db.add(feats);

//   // and query the database
//   std::cout << "Querying the database: " << std::endl;

//   QueryResults ret;
//   db.query(feats, ret, 4);

//   // ret[0] is always the same image in this case, because we added it to the
//   // database. ret[1] is the second best match.

//   std::cout << "Searching for Image 0. " << ret << std::endl;

//   return 0;
}
