package org.example;

import java.io.File;
import net.sf.javaml.clustering.Clusterer;
import net.sf.javaml.clustering.KMeans;
import net.sf.javaml.clustering.DensityBasedSpatialClustering;
import net.sf.javaml.clustering.Cobweb;
import net.sf.javaml.clustering.evaluation.SumOfSquaredErrors;
import net.sf.javaml.core.Dataset;
import net.sf.javaml.core.Instance;
import net.sf.javaml.distance.DistanceMeasure;
import net.sf.javaml.tools.data.FileHandler;
import net.sf.javaml.clustering.evaluation.ClusterEvaluation;
import net.sf.javaml.clustering.evaluation.AICScore;
import net.sf.javaml.clustering.evaluation.Gamma;
import net.sf.javaml.clustering.evaluation.CIndex;
import net.sf.javaml.distance.EuclideanDistance;

public class App {

    // Main method to execute clustering algorithms and display their results.
    public static void main(String[] args) throws Exception {
        Dataset data = loadDataset("C:src/resources/iris.data", 4, ",");

        performClustering("KMeans Clusters", new KMeans(), data);
        performClustering("Cobweb Clusters", new Cobweb(), data);
        performClustering("Density Based Space Clusters", new DensityBasedSpatialClustering(), data);
    }

    // Method to load the dataset from a file.
    private static Dataset loadDataset(String filePath, int attributes, String delimiter) throws Exception {
        return FileHandler.loadDataset(new File(filePath), attributes, delimiter);
    }

    // Method to perform clustering using a specific algorithm and display results.
    private static void performClustering(String header, Clusterer clusterer, Dataset data) {
        System.out.println("\n" + header + ":\n");
        Dataset[] clusters = clusterer.cluster(data);
        displayClusters(clusters);
        displayEvaluationScores(clusters);
        System.out.println("\n********************************************\n");
    }

    // Method to display clusters and their instances.
    private static void displayClusters(Dataset[] clusters) {
        for (int i = 0; i < clusters.length; i++) {
            System.out.println("Cluster " + i + ":");
            for (Instance instance : clusters[i]) {
                System.out.println(instance);
            }
            System.out.println();
        }
    }

    // Method to calculate and display evaluation scores for the clusters.
    private static void displayEvaluationScores(Dataset[] clusters) {
        DistanceMeasure calcEucDistance = new EuclideanDistance();
        ClusterEvaluation calcSumOfSqrErr = new SumOfSquaredErrors();
        ClusterEvaluation calcCindex = new CIndex(calcEucDistance);
        ClusterEvaluation calcGamma = new Gamma(calcEucDistance);
        ClusterEvaluation calcAIC = new AICScore();

        System.out.println("SumOfSquaredErrors score of Clusters: " + calcSumOfSqrErr.score(clusters));
        System.out.println("Cindex score of Clusters: " + calcCindex.score(clusters));
        System.out.println("Gamma score of Clusters: " + calcGamma.score(clusters));
        System.out.println("AICScore of Clusters: " + calcAIC.score(clusters));
    }
}
