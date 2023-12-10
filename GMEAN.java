package weka1;

import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.Evaluation;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.Random;

import weka.classifiers.functions.*;


import java.util.Arrays;

public class GMEAN {
    public static void main(String[] args) throws Exception {
        // Load your dataset
        DataSource source = new DataSource("C:\\Users\\hella\\Desktop\\ALL\\results\\avro_result.arff"); // Replace with your dataset path
        Instances data = source.getDataSet();
        if (data.classIndex() == -1) {
            data.setClassIndex(data.numAttributes() - 1);
        }
        int trainSize = (int) Math.round(data.numInstances() * 0.8);
        int testSize = data.numInstances() - trainSize;
        Instances trainingData = new Instances(data, 0, trainSize);
        Instances testingData = new Instances(data, trainSize, testSize); // Create a testing set

        Logistic classifier = new Logistic(); // Replace with your classifier setup
        classifier.buildClassifier(trainingData);

        int nEpochs =60;
        int nIterations = 10; // Number of iterations
        int kFolds = 10;      // Number of folds

        // Initialize arrays to store cross-validation results
        double[][] cvScores = new double[nIterations][kFolds];
        double[] precisionScores = new 
        		double[nIterations];
        double[] f1Scores = new double[nIterations];
        double[] mccScores = new double[nIterations];
        
        
        
        double[] aucScores = new double[nIterations];
        double[] gMeanScores = new double[nIterations];
        double[] stdScores = new double[nIterations]; // Array to store standard deviation scores

        double[] epochAverageMeanScores = new double[nEpochs]; // Array to store average mean scores for each epoch
        double[] epochAverageStdScores = new double[nEpochs]; // Array to store average standard deviation scores for each epoch

        for (int epoch = 1; epoch <= nEpochs; epoch++) {
            System.out.println("Epoch " + epoch + ":");

            for (int i = 0; i < nIterations; i++) {
                // Create an Evaluation object
                Evaluation eval = new Evaluation(trainingData);

                // Perform cross-validation
                eval.crossValidateModel(classifier, trainingData, kFolds, new Random());
                
                gMeanScores[i] = Math.sqrt(eval.weightedTruePositiveRate() * eval.weightedTrueNegativeRate());

                // Store the scores
                for (int j = 0; j < kFolds; j++) {
                    cvScores[i][j] = eval.pctCorrect();
                }
            }

           
        Evaluation trainEval = new Evaluation(trainingData);
        trainEval.crossValidateModel(classifier, trainingData, kFolds, new Random());

        System.out.println("Training Set Evaluation Results:");
       
        System.out.println("Weighted G-Mean: " + Math.sqrt(trainEval.weightedTruePositiveRate() * trainEval.weightedTrueNegativeRate()));

        
     // Create an Evaluation object for testing
        Evaluation testEval = new Evaluation(testingData);

        // Evaluate the classifier on the testing set
        testEval.crossValidateModel(classifier, testingData, kFolds, new Random());
        System.out.println("Testing Set Evaluation Results:");
        
        System.out.println("Weighted G-Mean: " + Math.sqrt(testEval.weightedTruePositiveRate() * testEval.weightedTrueNegativeRate()));

    }
}
}
