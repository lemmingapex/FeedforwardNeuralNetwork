package sw.neuralnetwork;

import org.junit.Test;
import static org.junit.Assert.*;

/*
 * @author scott
 */
public class ANNMathTest {

    @Test
    public void testSigmoid() {
        assertEquals(0.5, ANNMath.sigmoid(0), Double.MIN_VALUE);
        assertEquals(0.7310585786300049, ANNMath.sigmoid(1), Double.MIN_VALUE);

        assertEquals(0.25, ANNMath.sigmoidDerivative(0), Double.MIN_VALUE);
        assertEquals(0.19661193324148185, ANNMath.sigmoidDerivative(1), Double.MIN_VALUE);
    }
}
