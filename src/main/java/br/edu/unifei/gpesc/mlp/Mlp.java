/*
 * Copyright (C) 2015 Universidade Federal de Itajuba
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
package br.edu.unifei.gpesc.mlp;

import br.edu.unifei.gpesc.mlp.layer.PatternLayer;
import br.edu.unifei.gpesc.mlp.layer.ConnectionLayer;
import br.edu.unifei.gpesc.mlp.layer.NeuronLayer;
import br.edu.unifei.gpesc.mlp.math.TanSig;
import br.edu.unifei.gpesc.mlp.math.Function;
import br.edu.unifei.gpesc.mlp.math.LogSig;
import br.edu.unifei.gpesc.mlp.layer.NeuronLayer.Neuron;
import br.edu.unifei.gpesc.mlp.log.MlpLogger;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;

/**
 *
 * @author Isaac Caldas Ferreira
 */
public class Mlp {

    /**
     * The maximum difference between the expected result and the result.
     */
    private static final double MAX_DIFFERENCE = 0.4;

    /**
     * The layer enumerator
     */
    public static enum Layer {

        /**
         * First hidden layer.
         */
        HIDDEN_1,

        /**
         * Second hidden layer.
         */
        HIDDEN_2,

        /**
         * Output layer
         */
        OUTPUT;
    }

    /**
     * The input layer.
     */
    protected final NeuronLayer mInputLayer;

    /**
     * The connections layers (hidden and output)
     */
    protected final ConnectionLayer[] mLayerArray;

    /**
     * The logger (is optional for training).
     */
    protected MlpLogger mLogger;

    /**
     * Creates a MLP.
     *
     * @param inLen The length of the input layer.
     * @param h1Len The length of the first hidden layer.
     * @param h2Len The length of the second hidden layer.
     * @param outLen The length of the output layer.
     */
    protected Mlp(int inLen, int h1Len, int h2Len, int outLen) {
        mInputLayer = new NeuronLayer(inLen);

        int h1 = Layer.HIDDEN_1.ordinal();
        int h2 = Layer.HIDDEN_2.ordinal();
        int out = Layer.OUTPUT.ordinal();

        mLayerArray = new ConnectionLayer[3];
        mLayerArray[h1] = new ConnectionLayer(h1Len, mInputLayer, new TanSig());
        mLayerArray[h2] = new ConnectionLayer(h2Len, mLayerArray[h1], new TanSig());
        mLayerArray[out] = new ConnectionLayer(outLen, mLayerArray[h2], new LogSig());
    }

    /**
     * Sets the transfer function to network layer.
     * @param layer The layer id.
     * @param function The transfer function.
     */
    public void setLayerFunction(Layer layer, Function function) {
        mLayerArray[layer.ordinal()].setFunction(function);
    }

    public void setLogger(MlpLogger logger) {
        mLogger = logger;
    }

    /**
     * Easy way to get the last layer.
     * @return The output layer.
     */
    protected ConnectionLayer getOutputLayer() {
        return mLayerArray[Layer.OUTPUT.ordinal()];
    }

    /**
     * Gets the length of the input layer.
     * @return The input layer length.
     */
    public int getInputLayerLength() {
        return mInputLayer.getLength();
    }

    /**
     * Computes the activation (output) for all connection layers.
     */
    protected void computeActivationOutput() {
        for (ConnectionLayer layer : mLayerArray) {
            layer.computeActivationOutput();
        }
    }

    /**
     * Logs a pattern using the {@link MlpLogger}.
     * @param index The index number of the pattern in the array (base 1).
     * @param expected The expected results.
     * @param result The results.
     */
    private boolean logPattern(int index, Neuron[] expected, Neuron[] result) {
        boolean correct = true;
        boolean onLimbo = false;

        double e, r;

        for (int i=0; i<expected.length; i++) {
            e = expected[i].activation;
            r = result[i].activation;
            correct &= compare(e, r);
            onLimbo |= onLimbo(r);
        }

        mLogger.logPattern(index, correct);

        if (!correct) {
            mLogger.logErrorType(onLimbo);
        }

        for (int i=0; i<expected.length; i++) {
            e = expected[i].activation;
            r = result[i].activation;
            mLogger.logResult(e, r);
        }

        return correct;
    }

    /**
     * Executes de MLP. <br>
     *
     * The {@link MlpLogger} must be set or {@link NullPointerException} will be throw.
     *
     * @param patterns The patterns to be processed.
     * @return The percent of correct patterns.
     */
    public float runTestSup(PatternLayer[] patterns) {
        // optimization
        MlpLogger logger = mLogger;

        int index = 0;      // pattern number
        double error;   // erro de um padrao de teste
        double totalError = 0.0;   // erro total dos padroes de teste

        int correct = 0;
        int incorrect = 0;

        boolean isCorrect;

        // run
        NeuronLayer inputLayer = mInputLayer;
        ConnectionLayer outputLayer = getOutputLayer();

        for (PatternLayer pattern : patterns) {
            inputLayer.setNeurons(pattern.inputLayer);
            computeActivationOutput();
            error = outputLayer.getDifferenceTotal(pattern.outputLayer);
            totalError += error;

            // log
            isCorrect = logPattern(++index, pattern.outputLayer.getNeurons(), outputLayer.getNeurons());
            logger.logError(error);

            if (isCorrect) {
                correct++;
            } else {
                incorrect++;
            }
        }

        // log
        logger.logTotalError(totalError);
        logger.logCorrectPatterns(correct);
        logger.logIncorrectPatterns(incorrect);

        return correct / (float) (correct + incorrect);
    }

    /**
     * Runs a not supervisioned test.
     * @param neurons The input activation neurons.
     * @return The output array.
     */
    public double[] runTestNonSup(double[] neurons) {
        mInputLayer.setNeurons(new NeuronLayer(neurons));
        computeActivationOutput();

        Neuron[] outNeurons = getOutputLayer().getNeurons();
        double[] output = new double[outNeurons.length];

        for (int i=0; i<output.length; i++) {
            output[i] = outNeurons[i].activation;
        }

        return output;
    }

    /**
     * Checks if the result is what is expected.
     * @param e The expected result.
     * @param r The result.
     * @return true if the absolute difference between e and r do not exceed
     * the {@link #MAX_DIFFERENCE}. False, otherwise.
     */
    public static boolean compare(double e, double r) {
        return (Math.abs(e - r) <= MAX_DIFFERENCE);
    }

    public static boolean onLimbo(double r) {
        return (0.4 < r) && (r < 0.6);
    }

    public void saveMlp(File file) throws IOException {
        // output stream
        FileOutputStream outStream = new FileOutputStream(file);
        FileChannel fileOut = outStream.getChannel();

        // calculate the size of the file (optimization)
        int bufferSize = Integer.BYTES; // reserve first layer
        for (ConnectionLayer layer : mLayerArray) {
            bufferSize += Integer.BYTES; // reserve for neurons size info
            bufferSize += layer.getLength() * Double.BYTES; // reserve for neurons
            bufferSize += layer.getConnectionsLength() * Double.BYTES; // reserve for matrix
        }

        // write to the file
        ByteBuffer outBuffer = ByteBuffer.allocate(bufferSize);

        // layers length
        outBuffer.putInt(mInputLayer.getLength());
        for (ConnectionLayer layer : mLayerArray) {
            outBuffer.putInt(layer.getLength());
        }

        // neuron array and connection matrix
        for (ConnectionLayer layer : mLayerArray) {
            layer.toByteBuffer(outBuffer);
        }

        outBuffer.flip();
        fileOut.write(outBuffer);

        fileOut.close();
        outStream.close();
    }

    /**
     * Loads the MLP data from a file, the creates the network. <br>
     * This method is meant to be used with a saved {@link MlpTrain}.
     * @param file The file to the MLP data.
     * @return The previously saved MLP.
     * @throws IOException If any IO error occurs.
     */
    public static Mlp loadMlp(File file) throws IOException {
        FileInputStream inStream = new FileInputStream(file);
        FileChannel fileIn = inStream.getChannel();

        ByteBuffer inBuffer = ByteBuffer.allocate((int) file.length());
        fileIn.read(inBuffer);
        inBuffer.flip();

        inStream.close();
        fileIn.close();

        int inLen = inBuffer.getInt();
        int h1Len = inBuffer.getInt();
        int h2Len = inBuffer.getInt();
        int outLen = inBuffer.getInt();

        Mlp mlp = new Mlp(inLen, h1Len, h2Len, outLen);

        for (ConnectionLayer layer : mlp.mLayerArray) {
            layer.loadFromByteBuffer(inBuffer);
        }

        return mlp;
    }
}
