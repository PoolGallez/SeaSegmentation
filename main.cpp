/**
 * Bag of Words Semantic Segmentation
 * @author Paolo Galletta
 * @version 1.0
 **/

// Library inclusion
#include <stdio.h>
#include <stdlib.h>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include <opencv2/core/utils/filesystem.hpp>
#include <opencv2/ml/ml.hpp>

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <dirent.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>

// Libraries for Graph Segmentation Algorithm
#include <cstdio>
#include <cstdlib>
#include <image.h>
#include <misc.h>
#include <pnmfile.h>
#include "segment-image.h"

#include "Loader.h"

#include "SemanticSegmentor.h"
//LBP descriptor
//#include "lbp.hpp"

#include "BagOfWords.h"

// Namespaces used
using namespace cv::xfeatures2d;
using namespace cv::ml;
//using namespace lbp;

// Constants
const std::string TRAIN_DIR = "../training";
const std::string WATER_DIR = "/water/";
const std::string NON_WATER_DIR = "/non_water/";
const int CLUSTER_COUNT = 200; // 80 words per class

const std::string TEST_FOLDER = "../test/";
const std::string PHOTOS = "photos/";
const std::string GROUND_TRUTH = "ground_truth/";

const int WATER_CLASS = 1;
const int NON_WATER_CLASS = -1;

/* void descriptorsExtractionLBP(vector<Mat> waterImages, vector<Mat> nonWaterImages, int *waterDescriptorsNumber,
                              int *nonWaterDescriptorsNumber,
                              Mat *totalDescriptors)
{
    cout << "*** Feature Extraction LBP" << endl;
    Ptr<SIFT> detector = SIFT::create();
    int i = 0;
    int waterDescNumb, nonWaterDescNumb;
    try
    {
        for (Mat image : waterImages)
        {
            if (image.empty())
            {
                cout << "Index " << i << "empty image, removing it " << endl;
                waterImages.erase(waterImages.begin() + i);
            }
            else
            {
                GaussianBlur(image, image, Size(7, 7), 5, 3, BORDER_CONSTANT); // tiny bit of smoothing is always a good idea

                image = lbp::ELBP(image);
                normalize(image, image, 0, 255, NORM_MINMAX, CV_8UC1);
                vector<KeyPoint> keypoints;
                Mat descriptors;
                detector->detect(image, keypoints);
                detector->compute(image, keypoints, descriptors);
                if (!descriptors.empty())
                {
                    totalDescriptors->push_back(descriptors);
                }
                i++;
            }
        }

        waterImages.clear();
        waterDescNumb = i;
        i = 0;
        for (Mat image : nonWaterImages)
        {
            GaussianBlur(image, image, Size(7, 7), 5, 3, BORDER_CONSTANT); // tiny bit of smoothing is always a good idea
            image = lbp::ELBP(image);
            normalize(image, image, 0, 255, NORM_MINMAX, CV_8UC1);
            vector<KeyPoint> keypoints;
            Mat descriptors;
            detector->detect(image, keypoints);
            detector->compute(image, keypoints, descriptors);
            if (!descriptors.empty())
            {
                totalDescriptors->push_back(descriptors);
            }
            i++;
        }
        cout << "*** Feature Extracted LBP" << endl;
        nonWaterImages.clear();
        nonWaterDescNumb = i;
        *nonWaterDescriptorsNumber = nonWaterDescNumb;
        *waterDescriptorsNumber = waterDescNumb;
    }
    catch (cv::Exception e)
    {
        cout << e.what() << endl;
        cout << "Index of problems: " << i << endl;
        exit(e.code);
    }
}
 */

/* void computeHistogramsLBP(Mat *histograms, Mat *labels, vector<Mat> waterImages, vector<Mat> nonWaterImages, BOWImgDescriptorExtractor bowDE)
{
    cout << "=================== Histogram Computed ===================" << endl;
    cout << "*** Histogram Computation for water images " << endl;
    Ptr<SIFT> detector = SIFT::create();
    Mat labelsMat;
    int count = 0;
    for (Mat waterImage : waterImages)
    {
        GaussianBlur(waterImage, waterImage, Size(7, 7), 5, 3, BORDER_CONSTANT); // tiny bit of smoothing is always a good idea
        waterImage = lbp::ELBP(waterImage);
        normalize(waterImage, waterImage, 0, 255, NORM_MINMAX, CV_8UC1);
        vector<KeyPoint> keypoints;
        Mat waterDescriptor;
        detector->detect(waterImage, keypoints);
        detector->compute(waterImage, keypoints, waterDescriptor);
        Mat histogram;
        if (!waterDescriptor.empty())
        {
            bowDE.compute(waterDescriptor, histogram);
            histograms->push_back(histogram);
            count++;
            labelsMat.push_back(Mat(1, 1, CV_32SC1, WATER_CLASS));
        }
        else
        {
            cout << "Empty water descriptor" << endl;
        }
        keypoints.clear();
        waterDescriptor.release();
    }
    waterImages.clear();
    cout << "*** Histogram Computation for non water images " << endl;
    for (Mat nonWaterImage : nonWaterImages)
    {
        GaussianBlur(nonWaterImage, nonWaterImage, Size(7, 7), 5, 3, BORDER_CONSTANT); // tiny bit of smoothing is always a good idea
        nonWaterImage = lbp::ELBP(nonWaterImage);
        normalize(nonWaterImage, nonWaterImage, 0, 255, NORM_MINMAX, CV_8UC1);
        vector<KeyPoint> keypoints;
        Mat nonWaterDescriptor;
        detector->detect(nonWaterImage, keypoints);
        detector->compute(nonWaterImage, keypoints, nonWaterDescriptor);
        Mat histogram;
        if (!nonWaterDescriptor.empty())
        {
            bowDE.compute(nonWaterDescriptor, histogram);
            histograms->push_back(histogram);
            count++;
            labelsMat.push_back(Mat(1, 1, CV_32SC1, NON_WATER_CLASS));
        }
        else
        {
            cout << "Empty non water descriptor" << endl;
        }
        keypoints.clear();
        nonWaterDescriptor.release();
    }
    nonWaterImages.clear();
    *labels = labelsMat;
    cout << "I have pushed " << count << "histograms " << endl;
    cout << "=================== Histogram Computed ===================" << endl;
}
 */

int main()
{

    std::cout << "Test of the images in the test folder: " << std::endl;

    FeatureExtractor extractor(cv::SIFT::create(), cv::makePtr<cv::BFMatcher>(cv::NORM_L2));
    BagOfWords bow(extractor);
    std::vector<cv::Mat> waterDescriptors, nonWaterDescriptors, waterImages, nonWaterImages;
    cv::Mat totalDescriptors;
    if (std::fopen("../dictionary.yml", "r"))
    {
        std::cout << "Dictionary found! Loading..." << std::endl;
        bow.loadDictionary("../dictionary.yml");
        std::cout << "Dictionary loaded" << std::endl;
    }
    else
    {
        std::cout << "No dictionary found, starting the training process..." << std::endl;
        /**
         * Steps for the creation of the visual vocabulary: 
         * 1. Extract SIFT descriptors from the training set
         * 2. Quantize the Visual Words with K-Means Clustering
         **/

        std::cout << "*************************** Dictionary Creation ***************************" << std::endl;
        //1.a Image loading
        Loader imageLoader(TRAIN_DIR + WATER_DIR, TRAIN_DIR + NON_WATER_DIR, "*.*");
        std::cout << "** Loading images ..." <<std::endl;
        imageLoader.loadImages(waterImages, nonWaterImages);

        //1.b Descriptors Extraction

        std::cout << "** Extracting features... " <<std::endl;
        extractor.descriptorExtraction(waterImages, nonWaterImages, waterDescriptors, nonWaterDescriptors, totalDescriptors, false);
        std::cout << "Extracted" << std::endl;

        // 2. Clustering to obtain the visual vocabulary
        std::cout << "** Starting the clustering process" << std::endl;
        cv::TermCriteria termCriteria(cv::TermCriteria::Type::COUNT | cv::TermCriteria::Type::EPS, 100, 0.01);
        bow.cluster(CLUSTER_COUNT, termCriteria, totalDescriptors);
        bow.saveDictionary("../dictionary.yml");
        std::cout << "** Codebook computed and stored in ${PROJECT_ROOT}/dictionary.yml" << std::endl;
    }
    if (std::fopen("../svm.yml", "r"))
    {
        std::cout << "SVM found! Loading...";
        bow.loadSVM("../svm.yml");
        std::cout << " Done!" << std::endl;
    }
    else
    {

        /**
         * Steps for the creation for the SVM training: 
         * 1. Load training images and extract the descriptors
         * 2. For each descriptor compute its bag of words representation (occurence histogram)
         * 3. Train the SVM on the collected histograms
         **/
        std::cout << "No trained SVM was detected!" << std::endl;
        std::cout << "*************************** SVM Training ***************************" << std::endl;

        if (waterImages.size() <= 0 || nonWaterImages.size() <= 0 || waterDescriptors.size() <= 0 || nonWaterDescriptors.size() <= 0)
        {
            //1.a Image Loading
            Loader imageLoader(TRAIN_DIR + WATER_DIR, TRAIN_DIR + NON_WATER_DIR, "*.*");
            std::cout << "** Loading images ..." <<std::endl;
            imageLoader.loadImages(waterImages, nonWaterImages);
            std::cout << " Done !" << std::endl;
            //1.b Feature extraction
            std::cout << "** Extracting features... " << std::endl;
            extractor.descriptorExtraction(waterImages, nonWaterImages, waterDescriptors, nonWaterDescriptors, totalDescriptors, false);
            std::cout << "Extracted" << std::endl;
        }
        // 2. Bag of words representation computation
        cv::Mat histograms,labels;
        std::cout << "** Histogram computation... " << std::endl;
        bow.computeHistograms(histograms,labels,waterDescriptors,nonWaterDescriptors);
        std::cout << " Done!" << std::endl;

        // 3. SVM training
        std::cout << "** SVM training ..." <<std::endl;
        cv::Ptr<cv::ml::TrainData> trainData = cv::ml::TrainData::create(histograms,cv::ml::ROW_SAMPLE,labels);
        float missClassified = bow.trainSVM(trainData);
        std::cout <<" Done! Missclassified on the training set : "<<missClassified << std::endl;
        bow.saveSVM("../svm.yml");
        std::cout << "The trained SVM has been stored in ${PROJECT_ROOT}/svm.yml" << std::endl;
    }


    int num;
    std::string name = "";
    std::cout << "The files will be searched in ${PROJECT_ROOT}/test/photos" << std::endl;
    std::cout << "The ground truth will be searched in ${PROJECT_ROOT}/test/ground_truth/" << std::endl;

    std::cout << "Enter the filename (with extension): ";
    std::cin >> name;
    std::cout << std::endl;
    cv::Mat inputImage = cv::imread(TEST_FOLDER + PHOTOS + name);
    std::string filename = name.substr(0, name.find("."));
    std::string ext = name.substr(name.find("."), name.length());

    std::cout << filename << " " << ext << std::endl;
    if (ext.find(".png") == std::string::npos)
    {
        ext = ".png";
        cv::imwrite(TEST_FOLDER + PHOTOS + filename + ext, inputImage);
        inputImage = cv::imread(TEST_FOLDER + PHOTOS + filename + ext);
    }

    cv::Mat groundTruth = cv::imread(TEST_FOLDER + GROUND_TRUTH + filename + ext, cv::IMREAD_GRAYSCALE);

    // Converts in PPM

    imwrite("../imbuff/convertTest.ppm", inputImage);
    image<rgb> *inputNative = loadPPM("../imbuff/convertTest.ppm");
    cv::cvtColor(inputImage, inputImage, cv::COLOR_BGR2GRAY);
    std::vector<cv::Mat> masks;
    image<rgb> *segmented = segment_image(inputNative, SemanticSegmentor::SIGMA_DEF, SemanticSegmentor::K_DEF, SemanticSegmentor::MINSIZE_DEF, &num, &masks);
    cv::Mat segmentedMat = convertNativeToMat(segmented);
    SemanticSegmentor segmentor(inputImage, segmentedMat, masks);
    cv::Mat out = segmentor.getSemanticSegmented(bow);

    double seaAcc = 0, nonSeaAcc = 0, totAcc;
    SemanticSegmentor::getPixelAccuracy(out, groundTruth, seaAcc, nonSeaAcc, totAcc);

    std::cout << "Sea Pixel Accuracy: " << seaAcc << " "
              << "Non Sea Pixel Accuracy: " << nonSeaAcc << std::endl;

    std::cout << "Total Pixel Accuracy: "<< totAcc << std::endl;

    for(cv::Mat mask : masks){
        cv::imshow("Mask", mask);
        cv::waitKey(0);
    }

    cv::imwrite("../test_results/semantic_segmentation/" + filename + ext, out);
    cv::imwrite("../test_results/segmentation/" + filename + ext, segmentedMat);
    return 0;
}
