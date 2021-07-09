/**
 * Bag of Words Segmentation
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

//LBP descriptor
#include "lbp.hpp"

// Namespaces used
using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;
using namespace cv::ml;
using namespace lbp;
// Constants
const string TRAIN_DIR = "../training";
const string WATER_DIR = "/water/";
const string NON_WATER_DIR = "/non_water/";
const int CLUSTER_COUNT = 160; // 80 words per class

const int WATER_CLASS = 1;
const int NON_WATER_CLASS = -1;

void loadImages(vector<Mat> *waterImages, vector<Mat> *nonWaterImages)
{
    cout << "=================== Dataset Loading ===================" << endl;
    vector<String> waterNames;
    vector<String> nonWaterNames;
    glob("../training/water/*.*", waterNames);
    glob("../training/non_water/*.*", nonWaterNames);
    for (String name : waterNames)
    {
        waterImages->push_back(imread(name, IMREAD_GRAYSCALE));
    }
    for (String name : nonWaterNames)
    {
        nonWaterImages->push_back(imread(name, IMREAD_GRAYSCALE));
    }
    cout << "=================== Dataset Loaded ===================" << endl;
}

void descriptorsExtraction(vector<Mat> waterImages, vector<Mat> nonWaterImages, vector<Mat> *waterDescriptors,
                           vector<Mat> *nonWaterDescriptors,
                           Mat *totalDescriptors)
{
    cout << "*** Feature Extraction" << endl;
    int minHessian = 100;
    Ptr<SIFT> detector = SIFT::create();
    int i = 0;
    for (Mat image : waterImages)
    {
        vector<KeyPoint> keypoints;
        Mat descriptors;
        detector->detect(image, keypoints);
        detector->compute(image, keypoints, descriptors);
        if (!descriptors.empty())
        {
            waterDescriptors->push_back(descriptors);
            totalDescriptors->push_back(descriptors);
        }
        else
        {
            cout << "Index " << i << " has an empty descriptor, removing the image" << endl;
            waterImages.erase(waterImages.begin() + i);
        }
        i++;
    }
    i = 0;
    for (Mat image : nonWaterImages)
    {
        vector<KeyPoint> keypoints;
        Mat descriptors;
        detector->detect(image, keypoints);
        detector->compute(image, keypoints, descriptors);
        if (!descriptors.empty())
        {
            nonWaterDescriptors->push_back(descriptors);
            totalDescriptors->push_back(descriptors);
        }
        else
        {
            cout << "Index " << i << " has an empty descriptor, removing the image" << endl;
            nonWaterImages.erase(nonWaterImages.begin() + i);
        }
        i++;
    }
    cout << "*** Feature Extracted" << endl;
}

void descriptorsExtractionLBP(vector<Mat> waterImages, vector<Mat> nonWaterImages, vector<Mat> *waterDescriptors,
                              vector<Mat> *nonWaterDescriptors,
                              Mat *totalDescriptors)
{
    cout << "*** Feature Extraction LBP" << endl;
    Ptr<SIFT> detector = SIFT::create();
    int i = 0;
    try
    {
        for (Mat image : waterImages)
        {
            if (image.empty())
            {
                cout << "Index "<< i << "empty image, removing it " << endl;
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
                    waterDescriptors->push_back(descriptors);
                    totalDescriptors->push_back(descriptors);
                }
                else
                {
                    cout << "Index " << i << " has an empty descriptor, removing the image" << endl;
                    waterImages.erase(waterImages.begin() + i);
                }
                i++;
            }
        }
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
                nonWaterDescriptors->push_back(descriptors);
                totalDescriptors->push_back(descriptors);
            }
            else
            {
                cout << "Index " << i << " has an empty descriptor, removing the image" << endl;
                nonWaterImages.erase(nonWaterImages.begin() + i);
            }
            i++;
        }
        cout << "*** Feature Extracted LBP" << endl;
    }
    catch (cv::Exception e)
    {
        cout << e.what() << endl;
        cout << "Index of problems: " << i << endl;
        exit(e.code);
    }
}

void computeHistograms(Mat *histograms, Mat *labels, vector<Mat> waterDescriptors, vector<Mat> nonWaterDescriptors, BOWImgDescriptorExtractor bowDE)
{
    cout << "=================== Histogram Computed ===================" << endl;
    cout << "*** Histogram Computation for water images " << endl;
    int minHessian = 100;
    Mat labelsMat;

    for (Mat waterDescriptor : waterDescriptors)
    {
        Mat histogram;
        bowDE.compute(waterDescriptor, histogram);
        histograms->push_back(histogram);
        labelsMat.push_back(Mat(1, 1, CV_32SC1, WATER_CLASS));
    }
    cout << "*** Histogram Computation for non water images " << endl;
    for (Mat nonWaterDescriptor : nonWaterDescriptors)
    {
        Mat histogram;
        bowDE.compute(nonWaterDescriptor, histogram);
        histograms->push_back(histogram);
        labelsMat.push_back(Mat(1, 1, CV_32SC1, NON_WATER_CLASS));
        //cout << histogram.type()<<endl;
    }
    *labels = labelsMat;
    cout << "=================== Histogram Computed ===================" << endl;
}

void getImageHistogram(Mat image, vector<KeyPoint> keypoints, Mat *bowDescriptor, BOWImgDescriptorExtractor bowDE)
{
    Mat histogram;
    Mat histogramConverted;
    bowDE.compute(image, keypoints, histogram);
    *bowDescriptor = histogram;
}

int main()
{
    int TEST_LBP = 1;
    cout << "Program which classifies an image with the Bag of Words framework" << endl;
    vector<Mat> waterImages, nonWaterImages, waterDescriptors, nonWaterDescriptors;
    Mat totalDescriptors, histograms, labels, bagOfWords;
    Ptr<SVM> svm;
    Ptr<DescriptorExtractor> detector = SIFT::create();
    Ptr<BFMatcher> matcher = makePtr<BFMatcher>(NORM_L2);
    BOWImgDescriptorExtractor bowDE(detector, matcher);
    if (!fopen("../dictionary.yml", "r"))
    {
        cout << "No dictionary was found, start the creation of the dictionary" << endl;
        // Dictionary Creation

        /**
     * Steps for the creation of the visual vocabulary: 
     * 1. Extract SIFT descriptors from the training set
     * 2. Quantize the Visual Words with K-Means Clustering
     **/

        // 1. Feature Extraction (SURF features are used to speed up computation)

        loadImages(&waterImages, &nonWaterImages);

        cout << "=================== Dictionary Creation ===================" << endl;

        if (!TEST_LBP)
        {
            descriptorsExtraction(waterImages, nonWaterImages, &waterDescriptors, &nonWaterDescriptors, &totalDescriptors);
        }
        else
        {
            descriptorsExtractionLBP(waterImages, nonWaterImages, &waterDescriptors, &nonWaterDescriptors, &totalDescriptors);
        }
        /*       cout << size(waterDescriptors) << endl;
        cout << size(nonWaterDescriptors) << endl; */
        cout << "*** Clustering Visual Words" << endl;

        TermCriteria termCriteria(TermCriteria::Type::COUNT | TermCriteria::Type::EPS, 100, 0.01);
        BOWKMeansTrainer trainer(CLUSTER_COUNT, termCriteria, 1, KMEANS_PP_CENTERS);
        bagOfWords = trainer.cluster(totalDescriptors);

        bowDE.setVocabulary(bagOfWords);

        FileStorage fs("../dictionary.yml", FileStorage::WRITE);

        fs << "codebook" << bagOfWords;
        fs.release();

        cout << "*** Dictionary has been created and stored in the project's root as dictionary.yml " << endl;

        cout << "=================== Dictionary Created ===================" << endl;
    }
    else
    {
        cout << "A dictionary has already been created, loading dictionary.yml ..." << endl;
        FileStorage fs("../dictionary.yml", FileStorage::READ);
        fs["codebook"] >> bagOfWords;
        fs.release();
        bowDE.setVocabulary(bagOfWords);
        cout << "Dictionary loaded! " << endl;
    }
    if (!fopen("../svm.yml", "r"))
    {

        cout << "No svm model already trained found, start training..." << endl;

        cout << "=================== SVM Training ===================" << endl;

        if (size(waterImages) <= 0 || size(nonWaterImages) <= 0 || size(waterDescriptors) <= 0 || size(nonWaterDescriptors) <= 0)
        {
            // To train the SVM it is necessary to extract the histograms of the images represented in terms of BagOfWords
            loadImages(&waterImages, &nonWaterImages);
            if (TEST_LBP)
            {
                descriptorsExtraction(waterImages, nonWaterImages, &waterDescriptors, &nonWaterDescriptors, &totalDescriptors);
            }
            else
            {
                descriptorsExtractionLBP(waterImages, nonWaterImages, &waterDescriptors, &nonWaterDescriptors, &totalDescriptors);
            }
        }

        // Translate the labels in a Mat format

        svm = SVM::create();
        svm->setType(SVM::C_SVC);
        svm->setKernel(SVM::LINEAR);
        svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER | TermCriteria::EPS, 100, 1e-6));

        computeHistograms(&histograms, &labels, waterDescriptors, nonWaterDescriptors, bowDE);

        Ptr<TrainData> trainData = TrainData::create(histograms, ROW_SAMPLE, labels);
        svm->trainAuto(trainData);
        float missclassified = svm->calcError(trainData, false, noArray());
        cout << "SVM trained, percentage of missclassified: " << missclassified << endl;
        svm->save("../svm.yml");
    }
    else
    {
        cout << "SVM model already trained found! Loading..." << endl;
        svm = SVM::load("../svm.yml");
    }

    float sigma = 1.5;
    float k = 300;
    float min_size = 400;
    int radius = 1;
    int neighbors = 8;

    image<rgb> *input = loadPPM("../blue-boat-freedom-horizon-ocean-2878.ppm");

    cout << "Start processing the image.. " << endl;
    int num_ccs;
    std::vector<cv::Mat> masks;
    image<rgb> *seg = segment_image(input, sigma, k, min_size, &num_ccs, &masks);
    cout << masks.size() << endl;
    // Input Image semantic segmentation part
    Mat testSegmentation = imread("../blue-boat-freedom-horizon-ocean-2878.ppm", IMREAD_GRAYSCALE);

    Mat testSmoothed;
    GaussianBlur(testSegmentation, testSmoothed, Size(7, 7), 5, 3, BORDER_CONSTANT); // tiny bit of smoothing is always a good idea
    Mat lbp_desc = lbp::ELBP(testSmoothed, radius, neighbors);
    normalize(lbp_desc, lbp_desc, 0, 255, NORM_MINMAX, CV_8UC1);
    imshow("LBP", lbp_desc);

    Range x_crop_mask(1, masks[0].cols - 1), y_crop_mask(1, masks[0].rows - 1);
    Mat mask(masks[0], y_crop_mask, x_crop_mask);

    cout << "Size LBP: " << lbp_desc.size() << endl;
    cout << "Size Mask: " << mask.size() << endl;

    bitwise_and(lbp_desc, mask, lbp_desc);
    imshow("LBP", lbp_desc);
    waitKey(0);
    Mat output = testSegmentation.clone();

    // Test LBP
    Mat lbpTestTraining = imread("../training/water/w1021.png", IMREAD_GRAYSCALE);
    Mat lbpImg = lbp::ELBP(lbpTestTraining);
    cout << lbpImg.size() << endl;
    normalize(lbpImg, lbpImg, 0, 255, NORM_MINMAX, CV_8UC1);
    cout << lbpTestTraining.size() << endl;
    cout << lbpImg.size() << endl;

    imshow("lbp inputimg", lbpTestTraining);
    imshow("lbp image", lbpImg);
    waitKey(0);
    /* 
        Test with SIFT on single cut images
      Mat partialTest = imread("../test/non_water/aida-ship-driving-cruise-ship-sea-144796_micro_ship.png", IMREAD_GRAYSCALE);
    vector<KeyPoint> tests;
    detector->detect(partialTest, tests);
    Mat histogram;
    getImageHistogram(partialTest, tests, &histogram, bowDE);
    int prediction = svm->predict(histogram);
    if (prediction == WATER_CLASS)
    {
        cout << "This is water" << endl;
    }
    else
    {
        cout << "This is not water" << endl;
    } */

    Mat partialTest = imread("../test/non_water/aida-ship-driving-cruise-ship-sea-144796_micro_ship.png", IMREAD_GRAYSCALE);
    partialTest = lbp::ELBP(partialTest);
    normalize(partialTest, partialTest, 0, 255, NORM_MINMAX, CV_8UC1);

    vector<KeyPoint> tests;
    detector->detect(partialTest, tests);

    Mat histogram;
    getImageHistogram(partialTest, tests, &histogram, bowDE);
    int prediction = svm->predict(histogram);
    if (prediction == WATER_CLASS)
    {
        cout << "This is water" << endl;
    }
    else
    {
        cout << "This is not water" << endl;
    }

    Mat img = convertNativeToMat(seg);
    imshow("Segmentation Result", img);
    imshow("Input Image", testSegmentation);
    waitKey(0);

    for (Mat mask : masks)
    {
        // Extract SIFT features for masked region
        int prediction = -1;
        vector<KeyPoint> keypoints;
        detector->detect(testSegmentation, keypoints, mask);
        // Represent features with an Histogram
        if (!keypoints.empty())
        {
            Mat histogram;
            getImageHistogram(testSegmentation, keypoints, &histogram, bowDE);
            // Perform classification
            prediction = svm->predict(histogram);
        }
        cout << prediction << endl;

        // Assign a "color" to each segment dependently on their class
        if (prediction == WATER_CLASS)
        {
            output.setTo(255, mask);
        }
        else
        {
            output.setTo(0, mask);
        }
    }

    imshow("Test Semantic Segmentation", output);
    waitKey(0);
    return 0;
}