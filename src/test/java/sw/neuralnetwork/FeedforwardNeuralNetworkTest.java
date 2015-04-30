package sw.neuralnetwork;

import org.junit.Test;

import static org.junit.Assert.*;

/*
 * @author scott
 */
public class FeedforwardNeuralNetworkTest {

    @Test
    public void testSimpleClassification() {

        final double[][] inputs = new double[][]{
                {0, 0},
                {1, 1}
        };

        final double[][] expectedOutputs = new double[][]{
                {1, 0},
                {0, 1}
        };

        FeedforwardNeuralNetwork feedforwardNeuralNetwork = new FeedforwardNeuralNetwork();
        feedforwardNeuralNetwork.train(inputs, expectedOutputs);

        for (int i = 0; i < inputs.length; i++) {
            assertArrayEquals(expectedOutputs[i], feedforwardNeuralNetwork.getOuput(inputs[i]), 0.1);
            assertEquals(i, feedforwardNeuralNetwork.classify(inputs[i]));
        }
    }

    @Test
    public void testSimpleClassification1() {

        final double[][] inputs = new double[][]{
                {0, 1, 1, 1, 0, // 0
                        1, 0, 0, 0, 1,
                        1, 0, 0, 0, 1,
                        1, 0, 0, 0, 1,
                        1, 0, 0, 0, 1,
                        1, 0, 0, 0, 1,
                        0, 1, 1, 1, 0},
                {0, 0, 1, 0, 0, // 1
                        0, 1, 1, 0, 0,
                        0, 0, 1, 0, 0,
                        0, 0, 1, 0, 0,
                        0, 0, 1, 0, 0,
                        0, 0, 1, 0, 0,
                        0, 1, 1, 1, 0},
                {0, 1, 1, 1, 0, // 2
                        1, 0, 0, 0, 1,
                        0, 0, 0, 0, 1,
                        0, 0, 1, 1, 0,
                        0, 1, 0, 0, 0,
                        1, 0, 0, 0, 0,
                        1, 1, 1, 1, 1},
                {0, 1, 1, 1, 0, // 3
                        1, 0, 0, 0, 1,
                        0, 0, 0, 0, 1,
                        0, 0, 1, 1, 0,
                        0, 0, 0, 0, 1,
                        1, 0, 0, 0, 1,
                        0, 1, 1, 1, 0},
                {0, 0, 0, 1, 0, // 4
                        0, 0, 1, 1, 0,
                        0, 1, 0, 1, 0,
                        1, 1, 1, 1, 1,
                        0, 0, 0, 1, 0,
                        0, 0, 0, 1, 0,
                        0, 0, 0, 1, 0},
                {1, 1, 1, 1, 1, // 5
                        1, 0, 0, 0, 0,
                        1, 0, 0, 0, 0,
                        1, 1, 1, 1, 0,
                        0, 0, 0, 0, 1,
                        1, 0, 0, 0, 1,
                        0, 1, 1, 1, 0},
                {0, 1, 1, 1, 0, // 6
                        1, 0, 0, 0, 0,
                        1, 0, 0, 0, 0,
                        1, 1, 1, 1, 0,
                        1, 0, 0, 0, 1,
                        1, 0, 0, 0, 1,
                        0, 1, 1, 1, 0},
                {1, 1, 1, 1, 1, // 7
                        1, 0, 0, 0, 1,
                        0, 0, 0, 0, 1,
                        0, 0, 0, 1, 0,
                        0, 0, 1, 0, 0,
                        0, 1, 0, 0, 0,
                        0, 1, 0, 0, 0},
                {0, 1, 1, 1, 0, // 8
                        1, 0, 0, 0, 1,
                        1, 0, 0, 0, 1,
                        0, 1, 1, 1, 0,
                        1, 0, 0, 0, 1,
                        1, 0, 0, 0, 1,
                        0, 1, 1, 1, 0},
                {0, 1, 1, 1, 0, // 9
                        1, 0, 0, 0, 1,
                        1, 0, 0, 0, 1,
                        0, 1, 1, 1, 1,
                        0, 0, 0, 0, 1,
                        0, 0, 0, 1, 0,
                        0, 1, 1, 0, 0}
        };

        final double[][] expectedOutputs = new double[][]{
                {1, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                {0, 1, 0, 0, 0, 0, 0, 0, 0, 0},
                {0, 0, 1, 0, 0, 0, 0, 0, 0, 0},
                {0, 0, 0, 1, 0, 0, 0, 0, 0, 0},
                {0, 0, 0, 0, 1, 0, 0, 0, 0, 0},
                {0, 0, 0, 0, 0, 1, 0, 0, 0, 0},
                {0, 0, 0, 0, 0, 0, 1, 0, 0, 0},
                {0, 0, 0, 0, 0, 0, 0, 1, 0, 0},
                {0, 0, 0, 0, 0, 0, 0, 0, 1, 0},
                {0, 0, 0, 0, 0, 0, 0, 0, 0, 1}
        };

        FeedforwardNeuralNetwork feedforwardNeuralNetwork = new FeedforwardNeuralNetwork();
        feedforwardNeuralNetwork.train(inputs, expectedOutputs);

        for (int i = 0; i < inputs.length; i++) {
            assertArrayEquals(expectedOutputs[i], feedforwardNeuralNetwork.getOuput(inputs[i]), 0.2);
            assertEquals(i, feedforwardNeuralNetwork.classify(inputs[i]));
        }
    }

    // it is impossible for a single-layer perceptron network to learn an XOR function.
    /*
    @Test
    public void testXOR() {

        final double[][] inputs = new double[][]{
                {0, 0},
                {0, 1},
                {1, 0},
                {1, 1}
        };

        final double[][] expectedOutputs = new double[][]{
                {0, 0, 0, 0},
                {1, 1, 1, 1},
                {1, 1, 1, 1},
                {0, 0, 0, 0}
        };

        FeedforwardNeuralNetwork feedforwardNeuralNetwork = new FeedforwardNeuralNetwork();
        feedforwardNeuralNetwork.train(inputs, expectedOutputs);

        for (int i = 0; i < inputs.length; i++) {
            assertArrayEquals(expectedOutputs[i], feedforwardNeuralNetwork.getOuput(inputs[i]), 0.1);
            assertEquals(i, feedforwardNeuralNetwork.classify(inputs[i]));
        }
    }
    */

}
