#include <BagOfWords.h>
BagOfWords::BagOfWords(): bowDE(extractor.getExtractor(),extractor.getMatcher()) { }
BagOfWords::BagOfWords(FeatureExtractor extractor) : bowDE(extractor.getExtractor(), extractor.getMatcher())
{
    this->extractor = extractor;
    svm = cv::ml::SVM::create();
}
BagOfWords::BagOfWords(FeatureExtractor extractor, cv::Ptr<cv::ml::SVM> svm) : bowDE(extractor.getExtractor(), extractor.getMatcher())
{
    this->extractor = extractor;
    this->svm = svm;
}

FeatureExtractor BagOfWords::getExtractor()
{
    return extractor;
}

cv::BOWImgDescriptorExtractor BagOfWords::getBowDescriptorExtractor()
{
    return bowDE;
}

cv::Ptr<cv::ml::SVM> BagOfWords::getSVM()
{
    return svm;
}

void BagOfWords::computeHistograms(cv::Mat &histograms, cv::Mat &labels, std::vector<cv::Mat> positives, std::vector<cv::Mat> negatives)
{
    std::cout << "=================== Histogram Computation ===================" << std::endl;
    std::cout << "*** Histogram Computation for water images " << std::endl;
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
    std::cout << "*** Histogram Computation for non water images " << std::endl;
    for (cv::Mat negative : negatives)
    {
        cv::Mat histogram;
        bowDE.compute(negative, histogram);
        histograms.push_back(histogram);
        labelsMat.push_back(cv::Mat(1, 1, CV_32SC1, BagOfWords::NEGATIVE_CLASS));
    }
    negatives.clear();
    labels = labelsMat;

    std::cout << "=================== Histogram Computed ===================" << std::endl;
}
void BagOfWords::getImageHistogram(cv::Mat image, std::vector<cv::KeyPoint> keypoints, cv::Mat &bowDescriptor)
{
    cv::Mat histogram;
    bowDE.compute(image, keypoints, histogram);
    bowDescriptor = histogram;
}

void BagOfWords::cluster(int clusterNumber, cv::TermCriteria termCriteria, cv::Mat totalDescriptors)
{
    cv::BOWKMeansTrainer trainer(clusterNumber, termCriteria, 1, cv::KMEANS_PP_CENTERS);
    cv::Mat bagOfWords = trainer.cluster(totalDescriptors);
    bowDE.setVocabulary(bagOfWords);
}

float BagOfWords::trainSVM(cv::Ptr<cv::ml::TrainData> trainData)
{
    svm->trainAuto(trainData);
    float missclassified = svm->calcError(trainData, false, cv::noArray());
    return missclassified;
}

bool BagOfWords::saveDictionary(std::string path)
{
    cv::FileStorage fs(path, cv::FileStorage::WRITE);
    fs << "codebook" << bowDE.getVocabulary();
    fs.release();
    return true;
}

bool BagOfWords::saveSVM(std::string path)
{
    svm->save(path);
    return true;
}

bool BagOfWords::loadDictionary(std::string path)
{
    cv::FileStorage fs("../dictionary.yml", cv::FileStorage::READ);
    cv::Mat bagOfWords;
    fs["codebook"] >> bagOfWords;
    fs.release();
    bowDE.setVocabulary(bagOfWords);
    return true;
}

bool BagOfWords::loadSVM(std::string path)
{
    svm = cv::ml::SVM::load(path);
    return true;
}

int BagOfWords::predict(cv::Mat image, cv::Mat mask)
{
    int prediction = -1;
    cv::Ptr<cv::DescriptorExtractor> detector = extractor.getExtractor();
    std::vector<cv::KeyPoint> keypoints;
    detector->detect(image, keypoints, mask);
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
