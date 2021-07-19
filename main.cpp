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

#include "SemanticSegmentor.h"
//LBP descriptor
//#include "lbp.hpp"

#include "BagOfWords.h"

// Namespaces used
using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;
using namespace cv::ml;
//using namespace lbp;
// Constants
const string TRAIN_DIR = "../training";
const string WATER_DIR = "/water/";
const string NON_WATER_DIR = "/non_water/";
const int CLUSTER_COUNT = 200; // 80 words per class

const string TEST_FOLDER = "../test/";
const string PHOTOS = "photos/";
const string GROUND_TRUTH = "ground_truth/";

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
    cout << "* Extraction from water images" << endl;
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
            cout << "Index " << i << " has an empty descriptor water, removing the image" << endl;
            waterImages.erase(waterImages.begin() + i);
        }
        i++;
    }
    i = 0;
    cout << "* Extraction from non water images" << endl;
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
            cout << "Index " << i << " has an empty non water descriptor, removing the image" << endl;
            nonWaterImages.erase(nonWaterImages.begin() + i);
        }
        i++;
    }
    cout << "*** Feature Extracted" << endl;
}

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
void computeHistograms(Mat *histograms, Mat *labels, vector<Mat> waterDescriptors, vector<Mat> nonWaterDescriptors, BOWImgDescriptorExtractor bowDE)
{
    cout << "=================== Histogram Computation ===================" << endl;
    cout << "*** Histogram Computation for water images " << endl;
    int minHessian = 100;
    Mat labelsMat;
    int i = 0;
    try
    {

        for (Mat waterDescriptor : waterDescriptors)
        {
            Mat histogram;
            bowDE.compute(waterDescriptor, histogram);
            histograms->push_back(histogram);
            labelsMat.push_back(Mat(1, 1, CV_32SC1, WATER_CLASS));
            i++;
        }
        waterDescriptors.clear();
        cout << "*** Histogram Computation for non water images " << endl;
        for (Mat nonWaterDescriptor : nonWaterDescriptors)
        {
            Mat histogram;
            bowDE.compute(nonWaterDescriptor, histogram);
            histograms->push_back(histogram);
            labelsMat.push_back(Mat(1, 1, CV_32SC1, NON_WATER_CLASS));
            //cout << histogram.type()<<endl;
        }
        nonWaterDescriptors.clear();
        *labels = labelsMat;
    }
    catch (cv::Exception e)
    {
        cout << "Ho avuto problemi " << i << endl;
        cout << waterDescriptors[i].type() << endl;
        cout << e.what() << endl;
        cout << waterDescriptors[i].size() << endl;
    }

    cout << "=================== Histogram Computed ===================" << endl;
}

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
void getImageHistogram(Mat image, vector<KeyPoint> keypoints, Mat *bowDescriptor, BOWImgDescriptorExtractor bowDE)
{
    Mat histogram;
    Mat histogramConverted;
    bowDE.compute(image, keypoints, histogram);
    *bowDescriptor = histogram;
}

int main()
{
    /* int TEST_LBP = 0;
    cout << "Program which classifies an image with the Bag of Words framework" << endl;
    vector<Mat> waterImages, nonWaterImages, waterDescriptors, nonWaterDescriptors;
    int waterDescriptorsNumber, nonWaterDescriptorsNumber;
    Mat totalDescriptors, histograms, labels, bagOfWords;
    Ptr<SVM> svm;
    Ptr<DescriptorExtractor> detector = SIFT::create();
    cout << detector->descriptorSize() << endl;
    Ptr<BFMatcher> matcher = makePtr<BFMatcher>(NORM_L2);
    BOWImgDescriptorExtractor bowDE(detector, matcher);
    if (!fopen("../dictionary.yml", "r"))
    {
        cout << "No dictionary was found, start the creation of the dictionary" << endl;
        // Dictionary Creation */

    /**
     * Steps for the creation of the visual vocabulary: 
     * 1. Extract SIFT descriptors from the training set
     * 2. Quantize the Visual Words with K-Means Clustering
     **/

    /*        // 1. Feature Extraction (SURF features are used to speed up computation)

        loadImages(&waterImages, &nonWaterImages);

        cout << "=================== Dictionary Creation ===================" << endl;

        if (!TEST_LBP)
        {
            descriptorsExtraction(waterImages, nonWaterImages, &waterDescriptors, &nonWaterDescriptors, &totalDescriptors);
        }
        else
        {
            //descriptorsExtractionLBP(waterImages, nonWaterImages, &waterDescriptorsNumber, &nonWaterDescriptorsNumber, &totalDescriptors);
        }
  */
    /*       cout << size(waterDescriptors) << endl;
        cout << size(nonWaterDescriptors) << endl; */
    /*        cout << "*** Clustering Visual Words" << endl;

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

        if (waterImages.size() <= 0 || nonWaterImages.size() <= 0 || waterDescriptors.size() <= 0 || nonWaterDescriptors.size() <= 0)
        {
            // To train the SVM it is necessary to extract the histograms of the images represented in terms of BagOfWords
            loadImages(&waterImages, &nonWaterImages);
            if (!TEST_LBP)
            {
                descriptorsExtraction(waterImages, nonWaterImages, &waterDescriptors, &nonWaterDescriptors, &totalDescriptors);
                computeHistograms(&histograms, &labels, waterDescriptors, nonWaterDescriptors, bowDE);
            }
            else
            {
                //computeHistogramsLBP(&histograms, &labels, waterImages, nonWaterImages, bowDE);
            }
        }

        // Translate the labels in a Mat format

        svm = SVM::create();
        svm->setType(SVM::C_SVC);
        svm->setKernel(SVM::LINEAR);
        svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER | TermCriteria::EPS, 1000, 1e-6));

        //computeHistograms(&histograms, &labels, waterDescriptors, nonWaterDescriptors, bowDE);

        Ptr<TrainData> trainData = TrainData::create(histograms, ROW_SAMPLE, labels);

        cout << histograms.size() << endl;
        cout << labels.size() << endl;
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
    float sigma = 2;
    float k = 600;
    float min_size = 600;
    int radius = 4;
    int neighbors = 16;

    Mat testSegmentation = imread("../04.png");
    imshow("Original Image",testSegmentation);
    imwrite("../imbuff/convertedPPM.ppm",testSegmentation);
    cvtColor(testSegmentation,testSegmentation,COLOR_BGR2GRAY);
    image<rgb> *input = loadPPM("../imbuff/convertedPPM.ppm");

    cout << "Start processing the image.. " << endl;
    int num_ccs;
    std::vector<cv::Mat> masks;
    image<rgb> *seg = segment_image(input, sigma, k, min_size, &num_ccs, &masks);
    Mat graphSegmented = convertNativeToMat(seg);
    SemanticSegmentor test(testSegmentation, graphSegmented, masks);

    FeatureExtractor extractor(SIFT::create(), makePtr<BFMatcher>(NORM_L2));
    BagOfWords bow(extractor);
    if (fopen("../dictionary.yml", "r"))
    {
        cout << "OBj Oriented: Dictionary found! Loading..." << endl;
        bow.loadDictionary("../dictionary.yml");
        cout << "Dictionary loaded" << endl;
    }
    if (fopen("../svm.yml", "r"))
    {
        cout << "OBj Oriented: SVM found! Loading..." << endl;
        bow.loadSVM("../svm.yml");
        cout << "SVM loaded" << endl;
    }

    Mat output = test.getSemanticSegmented(bow);
    imshow("GraphSegmented", test.getGraphSegmented());
    imshow("Semantic Segmentation", output);
    waitKey(0); */
    /* if (TEST_LBP)
    {
        // Input Image semantic segmentation part
        imshow("input image", testSegmentation);
        GaussianBlur(testSegmentation, testSegmentation, Size(9, 9), 7, 7); // tiny bit of smoothing is always a good idea
        testSegmentation = lbp::ELBP(testSegmentation);
        normalize(testSegmentation, testSegmentation, 0, 255, NORM_MINMAX, CV_8UC1);
        Mat output = testSegmentation.clone();
        Range x_cut(1, masks[0].cols - 1), y_cut(1, masks[0].rows - 1);

        Mat img = convertNativeToMat(seg);
        imshow("Segmentation Result", img);
        imshow("Input LBP", testSegmentation);
        waitKey(0);

        cout << testSegmentation.size() << endl;

        for (Mat mask : masks)
        {
            mask = mask(y_cut, x_cut);
            float prediction = -1;
            Mat drawnKeypoints;
            // Extract SIFT features for masked region
            vector<KeyPoint> keypoints;
            detector->detect(testSegmentation, keypoints, mask);
            // Represent features with an Histogram
            Mat histogram;
            getImageHistogram(testSegmentation, keypoints, &histogram, bowDE);

            if (!histogram.empty())
            {
                prediction = svm->predict(histogram);
            }
            // Perform classification

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
        resize(output, output, Size(), 0.5, 0.5);
        imshow("Test Semantic Segmentation", output);
        waitKey(0);
    }
    else
    {
        imshow("input image", testSegmentation);
        Mat img = convertNativeToMat(seg);
        imshow("Segmentation Result", img);
        imshow("Input", testSegmentation);
        waitKey(0);
        Mat output = testSegmentation.clone();
        cout << "# Segment: " << num_ccs << "found, classifying..." << endl;
        for (Mat mask : masks)
        {
            float prediction = -1;
            // Extract SIFT features for masked region
            vector<KeyPoint> keypoints;
            Mat tmp;
            detector->detect(testSegmentation, keypoints, mask);
            // Represent features with an Histogram
            Mat histogram;
            getImageHistogram(testSegmentation, keypoints, &histogram, bowDE);

            if (!histogram.empty())
            {
                prediction = svm->predict(histogram);
            }
            // Perform classification

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
        resize(output, output, Size(), 0.5, 0.5);
        imshow("Test Semantic Segmentation", output);
        waitKey(0);
    }

    cout << "Test with the objects" << endl;
    FeatureExtractor extractor(SIFT::create(), makePtr<BFMatcher>(NORM_L2));
    BagOfWords bow(extractor);
    if (fopen("../dictionary.yml", "r"))
    {
        cout << "OBj Oriented: Dictionary found! Loading..." << endl;
        bow.loadDictionary("../dictionary.yml");
        cout << "Dictionary loaded" << endl;
    }
    if(fopen("../svm.yml","r")){
        cout << "OBj Oriented: SVM found! Loading..." << endl;
        bow.loadSVM("../svm.yml");
        cout << "SVM loaded" << endl; 
    }

    Mat output = bow.getSemanticSegmented(testSegmentation,masks);
    imshow("OBJ oriented result",output);
    waitKey(0);
 */
    cout << "Test of the images in the test folder: " << endl;

    FeatureExtractor extractor(SIFT::create(), makePtr<BFMatcher>(NORM_L2));
    BagOfWords bow(extractor);
    if (fopen("../dictionary.yml", "r"))
    {
        cout << "OBj Oriented: Dictionary found! Loading..." << endl;
        bow.loadDictionary("../dictionary.yml");
        cout << "Dictionary loaded" << endl;
    }
    if (fopen("../svm.yml", "r"))
    {
        cout << "OBj Oriented: SVM found! Loading..." << endl;
        bow.loadSVM("../svm.yml");
        cout << "SVM loaded" << endl;
    }
    int num;
    string name = "";
    cout << "The files will be searched in ${PROJECT_ROOT}/test/photos" << endl;
    cout << "The ground truth will be searched in ${PROJECT_ROOT}/test/ground_truth/" << endl;

    cout << "Enter the filename (with extension): ";
    cin >> name;
    cout << endl;
    Mat inputImage = imread(TEST_FOLDER + PHOTOS + name);
    string filename = name.substr(0,name.find("."));
    string ext = name.substr(name.find("."), name.length());

    cout << filename << " " << ext << endl;
    if (ext.find(".png") == std::string::npos)
    {
        ext = ".png";
        imwrite(TEST_FOLDER + PHOTOS + filename + ext, inputImage );
        inputImage = imread(TEST_FOLDER + PHOTOS + filename + ext);
    }

    Mat groundTruth = imread(TEST_FOLDER + GROUND_TRUTH + filename + ext, IMREAD_GRAYSCALE);
    imwrite("../imbuff/convertTest.ppm", inputImage);
    image<rgb> *inputNative = loadPPM("../imbuff/convertTest.ppm");
    cvtColor(inputImage, inputImage, COLOR_BGR2GRAY);
    vector<Mat> masks;
    image<rgb> *segmented = segment_image(inputNative, SemanticSegmentor::SIGMA_DEF, SemanticSegmentor::K_DEF, SemanticSegmentor::MINSIZE_DEF, &num, &masks);
    Mat segmentedMat = convertNativeToMat(segmented);
    SemanticSegmentor segmentor(inputImage, segmentedMat, masks);
    Mat out = segmentor.getSemanticSegmented(bow);

    double seaAcc = 0, nonSeaAcc = 0;
    SemanticSegmentor::getPixelAccuracy(out, groundTruth, seaAcc, nonSeaAcc);

    cout << "Sea Pixel Accuracy: " << seaAcc << " "
         << "Non Sea Pixel Accuracy: " << nonSeaAcc << endl;

    imwrite("../test_results/semantic_segmentation/"+filename+ext, out);
    imwrite("../test_results/segmentation/"+filename+ext, segmentedMat);
    return 0;
}
