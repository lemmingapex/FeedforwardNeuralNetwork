package sw.neuralnetwork;

import org.junit.Test;
import static org.junit.Assert.*;

/*
 * @author scott
 */
public class FeedforwardNeuralNetworkTest {

  @Test
  public void testTraining() {

    final double[][] data = new double[][]{
            {0,0,0,0,0},
            {1,1,1,1,1}
    };

    FeedforwardNeuralNetwork feedforwardNeuralNetwork = new FeedforwardNeuralNetwork(data);
    feedforwardNeuralNetwork.train(0.1);

    assertEquals(0, feedforwardNeuralNetwork.classify(data[0]));
    assertEquals(1, feedforwardNeuralNetwork.classify(data[1]));
  }
}
