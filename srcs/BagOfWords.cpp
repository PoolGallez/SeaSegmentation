#include <BagOfWords.h>

/**
 * Default Constructor
 **/
BagOfWords::BagOfWords(): bowDE(extractor.getExtractor(),extractor.getMatcher()) { }

/**
 * Bag of Words Constructor
 * @param extractor the Feature Extractor class used to match input histogram with the bag of words
 **/
BagOfWords::BagOfWords(FeatureExtractor extractor) : bowDE(extractor.getExtractor(), extractor.getMatcher())
{
    this->extractor = extractor;
    svm = cv::ml::SVM::create();
}

/**
 * Bag of Words Constructor
 * @param extractor the Feature Extractor class used to match input histogram with the bag of words
 * @param svm the SVM classifier used to detect the sea
 **/
BagOfWords::BagOfWords(FeatureExtractor extractor, cv::Ptr<cv::ml::SVM> svm) : bowDE(extractor.getExtractor(), extractor.getMatcher())
{
    this->extractor = extractor;
    this->svm = svm;
}

/**
 * Extractor getter
 * @return the feature extractor involved
 **/
FeatureExtractor BagOfWords::getExtractor()
{
    return extractor;
}

/**
 * Bow descriptor extractor getter
 * @return the descriptor extractor class to get the histogram representation
 **/
cv::BOWImgDescriptorExtractor BagOfWords::getBowDescriptorExtractor()
{
    return bowDE;
}

/**
 * SVM getter
 * @return the SVM used to detect water
 **/
cv::Ptr<cv::ml::SVM> BagOfWords::getSVM()
{
    return svm;
}

/**
 * Compute the histograms for the training images
 * @param histograms the output mat containing the resulting histograms
 * @param labels the output mat containig the training labels
 * @param positives the descriptor's vector for the positive class (label +1)
 * @param negatives the descriptor's vector for the negative class (label -1)
 **/
void BagOfWords::computeHistograms(cv::Mat &histograms, cv::Mat &labels, std::vector<cv::Mat> positives, std::vector<cv::Mat> negatives)
{
    cv::Mat labelsMat;
    int i = 0;
    for (cv::Mat positive : positives)
    {
        cv::Mat histogram;
        bowDE.compute(positive, histogram);
        histograms.push_back(histogram);
        labelsMat.push_back(cv::Mat(1, 1, CV_32SC1, BagOfWords::POSITIVE_CLASS));
        i++;
    }
    positives.clear();
    for (cv::Mat negative : negatives)
    {
        cv::Mat histogram;
        bowDE.compute(negative, histogram);
        histograms.push_back(histogram);
        labelsMat.push_back(cv::Mat(1, 1, CV_32SC1, BagOfWords::NEGATIVE_CLASS));
    }
    negatives.clear();
    labels = labelsMat;

}


/**
 * Get the Bag of Words representation of an image
 * @param image the image to be analyzed
 * @param keypoints the extracted keypoints
 * @param bowDescriptor the output mat that will contain the bag of words representation of the image
 **/
void BagOfWords::getImageHistogram(cv::Mat image, std::vector<cv::KeyPoint> keypoints, cv::Mat &bowDescriptor)
{
    cv::Mat histogram;
    bowDE.compute(image, keypoints, histogram);
    bowDescriptor = histogram;
}

/**
 * Cluster the descriptors to obtain the codebook
 * @param clusterNumber number of clusters
 * @param termCriteria convergence termination criteria
 * @param totalDescriptors all the descriptors of the training set
 **/
void BagOfWords::cluster(int clusterNumber, cv::TermCriteria termCriteria, cv::Mat totalDescriptors)
{
    cv::BOWKMeansTrainer trainer(clusterNumber, termCriteria, 1, cv::KMEANS_PP_CENTERS);
    cv::Mat bagOfWords = trainer.cluster(totalDescriptors);
    bowDE.setVocabulary(bagOfWords);
}

/**
 * SVM Trainer
 * @param trainData training data used
 **/
float BagOfWords::trainSVM(cv::Ptr<cv::ml::TrainData> trainData)
{
    svm->trainAuto(trainData);
    float missclassified = svm->calcError(trainData, false, cv::noArray());
    return missclassified;
}

/**
 * Save the dictionary on the disk
 * @param path where to save the dictionary
 **/
bool BagOfWords::saveDictionary(std::string path)
{
    cv::FileStorage fs(path, cv::FileStorage::WRITE);
    fs << "codebook" << bowDE.getVocabulary();
    fs.release();
    return true;
}

/**
 * Saves the SVM
 * @param path where to save the SVM
 **/
bool BagOfWords::saveSVM(std::string path)
{
    svm->save(path);
    return true;
}


/**
 * Loads the dictionary
 * @param path the path of the dictionary
 **/
bool BagOfWords::loadDictionary(std::string path)
{
    cv::FileStorage fs("../dictionary.yml", cv::FileStorage::READ);
    cv::Mat bagOfWords;
    fs["codebook"] >> bagOfWords;
    fs.release();
    bowDE.setVocabulary(bagOfWords);
    return true;
}

/**
 * Loads the SVM
 * @param path the path of the SVM
 **/
bool BagOfWords::loadSVM(std::string path)
{
    svm = cv::ml::SVM::load(path);
    return true;
}

/**
 * Predict the selected segment
 * @param image the image to be analyzed
 * @param mask the mask from which extract the features
 * @return whether the selected segment is water or not
 **/
int BagOfWords::predict(cv::Mat image, cv::Mat mask)
{
    // By default the segment is negative
    int prediction = -1;
    cv::Ptr<cv::DescriptorExtractor> detector = extractor.getExtractor();
    std::vector<cv::KeyPoint> keypoints;

    // Extract the features from the part of the image selected by mask
    detector->detect(image, keypoints, mask);
    cv::Mat tmp;
    cv::drawKeypoints(image,keypoints, tmp);
    cv::imshow("Filtered Keypoints",tmp);
    cv::waitKey(0);

    // Represent features with an Histogram
    cv::Mat histogram;
    getImageHistogram(image, keypoints, histogram);

    if (!histogram.empty())
    {
        prediction = svm->predict(histogram);
    }
    // Perform classification
    return prediction;
}
