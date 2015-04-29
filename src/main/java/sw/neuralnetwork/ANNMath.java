package sw.neuralnetwork;

public class ANNMath {
    public static double sigmoid(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }

    public static double sigmoidDerivative(double x) {
        double y = sigmoid(x);
        return y * (1.0 - y);
    }
}
