package sw.neuralnetwork;

import org.junit.Test;

import static org.junit.Assert.*;

/*
 * @author scott
 */
public class FeedforwardNeuralNetworkTest {

    @Test
    public void testSimpleClassification() {

        final double[][] data = new double[][]{
                {0, 0},
                {1, 1}
        };

        FeedforwardNeuralNetwork feedforwardNeuralNetwork = new FeedforwardNeuralNetwork(data);
        feedforwardNeuralNetwork.train();

        for (int i = 0; i < data.length; i++) {
            assertEquals(i, feedforwardNeuralNetwork.classify(data[i]));
        }
    }

    @Test
    public void testSimpleClassification1() {

        final double[][] data = new double[][]{
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

        FeedforwardNeuralNetwork feedforwardNeuralNetwork = new FeedforwardNeuralNetwork(data);
        feedforwardNeuralNetwork.train();

        for (int i = 0; i < data.length; i++) {
            assertEquals(i, feedforwardNeuralNetwork.classify(data[i]));
        }
    }
}
