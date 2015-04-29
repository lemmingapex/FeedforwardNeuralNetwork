package sw.neuralnetwork;

import java.io.PrintStream;

/*
 * @author scott
 */
public class FeedforwardNeuralNetwork {

    // determines how quickly weights are updated and altered as a low pass filter
    private final static double RHO = 0.1;

    private double[] inputNeurons;
    private double[] outputNeurons;

    private double[] hiddenNeurons;

    private double[][] weightsHiddenToInput;
    private double[][] weightsOutputToHidden;

    private final double[][] data;

    public FeedforwardNeuralNetwork(double[][] data) {

        if (data.length == 0) {
            throw new IllegalArgumentException("data is empty.");
        }

        if (data[0].length == 0) {
            throw new IllegalArgumentException("data is empty.");
        }

        this.data = data;

        inputNeurons = new double[this.data[0].length];
        outputNeurons = new double[this.data.length];

        // TODO : what size should this be?
        hiddenNeurons = new double[Math.max(Math.max(inputNeurons.length, outputNeurons.length), 2)];

        weightsHiddenToInput = new double[hiddenNeurons.length][inputNeurons.length];
        weightsOutputToHidden = new double[outputNeurons.length][hiddenNeurons.length];

        for (int i = 0; i < inputNeurons.length; i++) {
            inputNeurons[i] = 1.0;
        }

        for (int i = 0; i < hiddenNeurons.length; i++) {
            hiddenNeurons[i] = 1.0;
        }

        for (int j = 0; j < hiddenNeurons.length; j++) {
            for (int i = 0; i < inputNeurons.length; i++) {
                weightsHiddenToInput[j][i] = Math.random() - 0.5;
            }
        }

        for (int j = 0; j < outputNeurons.length; j++) {
            for (int i = 0; i < hiddenNeurons.length; i++) {
                weightsOutputToHidden[j][i] = Math.random() - 0.5;
            }
        }
    }

    private double currentMeanSquaredError(int index) {
        if (index < 0 || index >= outputNeurons.length) {
            throw new IllegalArgumentException("Index is not within bounds.  Must be greater than of equal to " + 0 + " and less than " + outputNeurons.length + ".");
        }

        double mse = 0.0;

        for (int i = 0; i < outputNeurons.length; i++) {
            double input = (index == i) ? 1.0 : 0.0;
            double e = input - outputNeurons[i];
            mse += e * e;
        }

        return mse / (double) outputNeurons.length;
    }

    private void setInputNeurons(double[] input) {
        if (inputNeurons.length != input.length) {
            throw new IllegalArgumentException("Input is not the right dimension.  Expected " + inputNeurons.length + ". Was " + input.length + ".");
        }

        for (int i = 0; i < inputNeurons.length; i++) {
            inputNeurons[i] = input[i];
        }
    }

    private void feedForward() {
        for (int i = 0; i < hiddenNeurons.length; i++) {
            hiddenNeurons[i] = 0.0;
            for (int j = 0; j < inputNeurons.length; j++) {
                hiddenNeurons[i] += (weightsHiddenToInput[i][j] * inputNeurons[j]);
            }

            hiddenNeurons[i] = ANNMath.sigmoid(hiddenNeurons[i]);
        }

        for (int i = 0; i < outputNeurons.length; i++) {
            outputNeurons[i] = 0.0;
            for (int j = 0; j < hiddenNeurons.length; j++) {
                outputNeurons[i] += (weightsOutputToHidden[i][j] * hiddenNeurons[j]);
            }

            outputNeurons[i] = ANNMath.sigmoid(outputNeurons[i]);
        }
    }

    private void printOutputNeurons(PrintStream s) {
        for (int i = 0; i < outputNeurons.length; i++) {
            s.println(i + " : " + outputNeurons[i]);
        }
    }

    private void backpropagate(int index) {
        if (index < 0 || index >= outputNeurons.length) {
            throw new IllegalArgumentException("Index is not within bounds.  Must be greater than of equal to " + 0 + " and less than " + outputNeurons.length + ".");
        }

        double[] errOutput = new double[outputNeurons.length];
        double[] errHidden = new double[hiddenNeurons.length];

        for (int i = 0; i < outputNeurons.length; i++) {
            double input = (index == i) ? 1.0 : 0.0;
            errOutput[i] = (input - outputNeurons[i]) * ANNMath.sigmoidDerivative(outputNeurons[i]);
        }

        for (int i = 0; i < hiddenNeurons.length; i++) {
            errHidden[i] = 0.0;

            // Include error contribution for all output neurons
            for (int j = 0; j < outputNeurons.length; j++) {
                errHidden[i] += errOutput[j] * weightsOutputToHidden[j][i];
            }

            errHidden[i] *= ANNMath.sigmoidDerivative(hiddenNeurons[i]);
        }

        for (int i = 0; i < outputNeurons.length; i++) {
            for (int j = 0; j < hiddenNeurons.length; j++) {
                weightsOutputToHidden[i][j] += RHO * errOutput[i] * hiddenNeurons[j];
            }
        }

        for (int i = 0; i < hiddenNeurons.length; i++) {
            for (int j = 0; j < inputNeurons.length; j++) {
                weightsHiddenToInput[i][j] += RHO * errHidden[i] * inputNeurons[j];
            }
        }
    }

    public double train(double meanSquaredErrorThreshold) {

        meanSquaredErrorThreshold = Math.max(meanSquaredErrorThreshold, Double.MIN_VALUE);

        double mse;
        int count = 0;
        do {
            // random input
            int dataIndex = Math.min((int) (Math.random() * data.length), data.length - 1);
            double[] input = data[dataIndex];

            setInputNeurons(input);
            feedForward();
            backpropagate(dataIndex);
            mse = currentMeanSquaredError(dataIndex);
            count++;
        } while (mse > meanSquaredErrorThreshold);

        return mse;
    }

    public double train() {
        return train(0.0001);
    }

    public int classify(double[] input) {
        setInputNeurons(input);
        feedForward();

        int bestIndex = 0;
        double max = outputNeurons[bestIndex];

        for (int i = 1; i < outputNeurons.length; i++) {
            if (outputNeurons[i] > max) {
                max = outputNeurons[i];
                bestIndex = i;
            }
        }

        return bestIndex;
    }
}
