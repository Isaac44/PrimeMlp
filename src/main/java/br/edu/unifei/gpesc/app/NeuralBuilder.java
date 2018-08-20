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
package br.edu.unifei.gpesc.app;

import br.edu.unifei.gpesc.mlp.layer.NeuronLayer;
import br.edu.unifei.gpesc.mlp.layer.PatternLayer;
import br.edu.unifei.gpesc.mlp.TrainMlp;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;

/**
 *
 * @author Isaac Caldas Ferreira
 */
public class NeuralBuilder {

    public static final NeuronLayer HAM = new NeuronLayer(1.0, 0.0);
    public static final NeuronLayer SPAM = new NeuronLayer(0.0, 1.0);

//    public static final NeuronLayer HAM = new NeuronLayer(1.0, 0.0);
//    public static final NeuronLayer SPAM = new NeuronLayer(0.0, 1.0);

    private PatternLayer[] mInputLayers;
    private PatternLayer[] mValidationLayers;

    public NeuralBuilder(File hamVectors, File spamVectors, double valPercent) throws IOException {
        createMlpLayers(hamVectors, spamVectors, valPercent);
    }

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

    public static PatternLayer[] replicate(PatternLayer[] array, int newSize) {
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

    private void createMlpLayers(File hamVectors, File spamVectors, double valPercent) throws IOException {
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
        int inLength = (int) (length * (1 - valPercent));

        PatternLayer[][] hamSplited = split(hamLayers, inLength);
        PatternLayer[][] spamSplited = split(spamLayers, inLength);

        // merge
        mInputLayers = merge(spamSplited[0], hamSplited[0]);
        mValidationLayers = merge(spamSplited[1], hamSplited[1]);
    }
}
