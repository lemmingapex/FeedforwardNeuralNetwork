package sw.neuralnetwork;

/*
 * @author scott, @date 4/27/15 7:35 PM
 */
public class FeedforwardNeuralNetwork {

  public double[] inputNeurons;
  public double[] outputNeurons;

  public double[] hiddenNeurons;

  public double[][] weightsHiddenToInput;
  public double[][] weightsOutputToHidden;

  public FeedforwardNeuralNetwork() {
    inputNeurons = new double[35];
    outputNeurons = new double[10];

    hiddenNeurons = new double[10];

    weightsHiddenToInput = new double[10][35];
    weightsOutputToHidden = new double[10][10];

    double RAND_WEIGHT = Math.random() - 0.5;

    for(int i = 0; i < inputNeurons.length; i++) {
      inputNeurons[i] = 1.0;
    }

    for(int i = 0; i < hiddenNeurons.length; i++) {
      hiddenNeurons[i] = 1.0;
    }

    for (int j = 0 ; j < hiddenNeurons.length; j++) {
      for (int i = 0 ; i < inputNeurons.length; i++) {
        weightsHiddenToInput[j][i] = RAND_WEIGHT;
      }
    }

    for (int j = 0 ; j < outputNeurons.length; j++) {
      for (int i = 0 ; i < hiddenNeurons.length; i++) {
        weightsOutputToHidden[j][i] = RAND_WEIGHT;
      }
    }
  }

  public boolean someLibraryMethod() {
      return true;
  }
}
