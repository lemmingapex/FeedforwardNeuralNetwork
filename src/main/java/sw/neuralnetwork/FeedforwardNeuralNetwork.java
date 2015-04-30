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

    private double[][] inputs;
    private double[][] expectedOutputs;

    private void init(double[][] inputs, double[][] expectedOutputs) {
        if (inputs.length == 0) {
            throw new IllegalArgumentException("input is empty.");
        }

        if (inputs[0].length == 0) {
            throw new IllegalArgumentException("input is empty.");
        }

        if (expectedOutputs.length == 0) {
            throw new IllegalArgumentException("output is empty.");
        }

        if (expectedOutputs[0].length == 0) {
            throw new IllegalArgumentException("output is empty.");
        }

        if (inputs.length != expectedOutputs.length) {
            throw new IllegalArgumentException("input length does not match output length.");
        }

        this.inputs = inputs;
        this.expectedOutputs = expectedOutputs;

        inputNeurons = new double[inputs[0].length];
        outputNeurons = new double[expectedOutputs.length];

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
            double e = expectedOutputs[index][i] - outputNeurons[i];
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
            errOutput[i] = (expectedOutputs[index][i] - outputNeurons[i]) * ANNMath.sigmoidDerivative(outputNeurons[i]);
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

    public double train(double[][] inputs, double[][] expectedOutputs, double meanSquaredErrorThreshold) {
        init(inputs, expectedOutputs);
        meanSquaredErrorThreshold = Math.max(meanSquaredErrorThreshold, Double.MIN_VALUE);

        double mse;
        int count = 0;
        do {
            // random input
            int dataIndex = Math.min((int) (Math.random() * this.inputs.length), this.inputs.length - 1);
            double[] input = this.inputs[dataIndex];

            setInputNeurons(input);
            feedForward();
            backpropagate(dataIndex);
            mse = currentMeanSquaredError(dataIndex);
            count++;
        } while (mse > meanSquaredErrorThreshold);

        return mse;
    }

    public double train(double[][] inputs, double[][] expectedOutputs) {
        return train(inputs, expectedOutputs, 0.0001);
    }


    public double[] getOuput(double[] input) {
        setInputNeurons(input);
        feedForward();

        return outputNeurons;
    }

    public int classify(double[] input) {
        int bestIndex = 0;
        double max = getOuput(input)[bestIndex];

        for (int i = 1; i < outputNeurons.length; i++) {
            if (outputNeurons[i] > max) {
                max = outputNeurons[i];
                bestIndex = i;
            }
        }

        return bestIndex;
    }
}
