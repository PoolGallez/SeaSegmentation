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

// Namespaces used
using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;
using namespace cv::ml;

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
    bowDE.compute(image, keypoints, histogram);
    *bowDescriptor = histogram;
}

int main()
{
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

        descriptorsExtraction(waterImages, nonWaterImages, &waterDescriptors, &nonWaterDescriptors, &totalDescriptors);
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

        // To train the SVM it is necessary to extract the histograms of the images represented in terms of BagOfWords
        loadImages(&waterImages, &nonWaterImages);

        descriptorsExtraction(waterImages, nonWaterImages, &waterDescriptors, &nonWaterDescriptors, &totalDescriptors);

        // Translate the labels in a Mat format

        svm = SVM::create();
        svm->setType(SVM::C_SVC);
        svm->setKernel(SVM::LINEAR);
        svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER | TermCriteria::EPS, 100, 1e-6));

        computeHistograms(&histograms, &labels, waterDescriptors, nonWaterDescriptors, bowDE);

        Ptr<TrainData> trainData = TrainData::create(histograms, ROW_SAMPLE, labels);
        svm->train(trainData);
        svm->save("../svm.yml");
    }
    else
    {
        cout << "SVM model already trained found! Loading..." << endl;
        svm = SVM::load("../svm.yml");
    }

    // Test phase
    Mat testImage = imread("../test/non_water/ship-tanker-cargo-sea-2573453.png", IMREAD_GRAYSCALE), histogram;
    vector<KeyPoint> keypoints;
    /* 
    Ptr<ORB> dect = ORB::create();
    dect->detect(testImage, keypoints);
    Mat keypointsDrawn;
    drawKeypoints(testImage, keypoints, keypointsDrawn);
    imshow("ORB", keypointsDrawn.clone());

    detector->detect(testImage, keypoints);
    drawKeypoints(testImage, keypoints, keypointsDrawn);
    imshow("SIFT", keypointsDrawn.clone());

    waitKey(0);
 */

    detector->detect(testImage, keypoints);
    Mat keypointsDrawn;
    drawKeypoints(testImage, keypoints, keypointsDrawn);
    imshow("Detected Keypoints", keypointsDrawn);
    waitKey(0);

    getImageHistogram(testImage, keypoints, &histogram, bowDE);
    Mat output;
    int predict = svm->predict(histogram);

    if (prediction == WATER_CLASS)
    {
        cout << "The class of this image is: water" << endl;
    }
    else
    {
        cout << "The class of this image is: non water" << endl;
    }

    return 0;
}