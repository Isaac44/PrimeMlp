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
package br.edu.unifei.gpesc.run;

import br.edu.unifei.gpesc.mlp.TrainMlp;
import br.edu.unifei.gpesc.mlp.layer.*;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;

/**
 *
 * @author Isaac Caldas Ferreira
 */
public class MlpFactory {

    /**
     * The HAM output neuron layer.
     */
    private static final NeuronLayer HAM = new NeuronLayer(Spam.HAM);

    /**
     * The SPAM output neuron layer.
     */
    private static final NeuronLayer SPAM = new NeuronLayer(Spam.SPAM);

    /**
     * The generated input layer for future MLP builds.
     */
    private PatternLayer[] mInputLayers;

    /**
     * The generated validation layer for future MLP builds.
     */
    private PatternLayer[] mValidationLayers;

    /**
     * Builds a MLP factory.
     *
     * @param hamVectors The file with the not spam vectors.
     * @param spamVectors The file with the spam vectors.
     * @param percentToValidate The percentual of vectors (spam and not spam) to
     * be used to generate the validation patterns.
     *
     * @throws IOException
     */
    public MlpFactory(File hamVectors, File spamVectors, double percentToValidate) throws IOException {
        createMlpLayers(hamVectors, spamVectors, percentToValidate);
    }

    /**
     * Builds a {@link TrainMlp}.
     *
     * @param h1Len The first hidden layer length.
     * @param h2Len The second hidden layer length.
     *
     * @return A {@link TrainMlp}.
     */
    public TrainMlp buildWith(int h1Len, int h2Len) {
        TrainMlp trainMlp = new TrainMlp(mInputLayers[0].inputLayer.getLength(), h1Len, h2Len, 2);
        trainMlp.setInputArray(mInputLayers);
        trainMlp.setValidationArray(mValidationLayers);
        return trainMlp;
    }

    public static PatternLayer[] loadTrainMlp(File vectors, NeuronLayer output) throws IOException {
        // open file
        FileChannel fileIn = new FileInputStream(vectors).getChannel();

        // load data
        int bufferLen = (int) vectors.length();
        ByteBuffer inBuffer = ByteBuffer.allocate(bufferLen);

        int readed = fileIn.read(inBuffer);

        if (readed != bufferLen) {
            throw new IOException("Not all data was readed.");
        }

        fileIn.close();
        inBuffer.flip(); // reset pointer

        // create input layer
        PatternLayer[] layers = new PatternLayer[inBuffer.getInt()];
        double[] activations = new double[inBuffer.getInt()];

        for (int i=0; i<layers.length; i++) {

            // load neuron activation
            for (int k=0; k<activations.length; k++) {
                activations[k] = inBuffer.getDouble();
            }

            // add new pattern.
            layers[i] = new PatternLayer(new NeuronLayer(activations), output);
        }

        return layers;
    }

    private static PatternLayer[] replicate(PatternLayer[] array, int newSize) {
        // new array
        PatternLayer[] newArray = new PatternLayer[newSize];
        int offset = 0;

        // full copies
        int fullCopies = newSize / array.length;

        for (int k=0; k < fullCopies; k++) {
            System.arraycopy(array, 0, newArray, offset, array.length);
            offset += array.length;
        }

        // remain copies
        int remainCopies = newSize % array.length;
        System.arraycopy(array, 0, newArray, offset, remainCopies);

        // return
        return newArray;
    }

    private static PatternLayer[][] split(PatternLayer[] array, int length1) {
        // arrays
        PatternLayer[] array1 = new PatternLayer[length1];
        PatternLayer[] array2 = new PatternLayer[array.length - length1];

        // copy
        System.arraycopy(array, 0, array1, 0, array1.length);
        System.arraycopy(array, array1.length, array2, 0, array2.length);

        // return
        return new PatternLayer[][] {array1, array2};
    }

    public static PatternLayer[] merge(PatternLayer[] array1, PatternLayer[] array2) {
        // array
        PatternLayer[] arrayMerged = new PatternLayer[array1.length + array2.length];

        // merge
        System.arraycopy(array1, 0, arrayMerged, 0, array1.length);
        System.arraycopy(array2, 0, arrayMerged, array1.length, array2.length);

        // return
        return arrayMerged;
    }

    private void createMlpLayers(File hamVectors, File spamVectors, double percentToValidate) throws IOException {
        // open
        final int length;

        PatternLayer[] hamLayers = loadTrainMlp(hamVectors, HAM);
        PatternLayer[] spamLayers = loadTrainMlp(spamVectors, SPAM);

        // replicate
        if (spamLayers.length < hamLayers.length) {
            length = hamLayers.length;
            spamLayers = replicate(spamLayers, length);
        }
        else if (hamLayers.length < spamLayers.length) {
            length = spamLayers.length;
            hamLayers = replicate(hamLayers, spamLayers.length);
        }
        else {
            length = hamLayers.length; // equals
        }

        // split
        int inLength = (int) (length * (1 - percentToValidate));

        PatternLayer[][] hamSplited = split(hamLayers, inLength);
        PatternLayer[][] spamSplited = split(spamLayers, inLength);

        // merge
        mInputLayers = merge(spamSplited[0], hamSplited[0]);
        mValidationLayers = merge(spamSplited[1], hamSplited[1]);
    }

    public static void main(String[] args) throws IOException {
        String path = "/home/isaac/Unifei/Mestrado/SAS/Mail_Test/August/vectors/vector_chi2_500/";
        MlpFactory t = new MlpFactory(new File(path, "ham"), new File(path, "spam"), 0.3f);
        TrainMlp mlp = t.buildWith(10, 10);
        mlp.runTrainByEpoch();
        mlp.saveMlp(new File(path, "500_nn"));

    }

}
