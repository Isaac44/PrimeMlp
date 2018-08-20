/*
 * Copyright (C) 2015 - GPESC - Universidade Federal de Itajuba
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
import br.edu.unifei.gpesc.mlp.layer.NeuronLayer.Neuron;
import java.util.Random;

/**
 *
 * @author Isaac Caldas Ferreira
 */
public class TrainMlp extends Mlp {

    /**
     * The max absolute value of the weight.
     */
    private double mAbsMaxInitialWeight = 0.00001;

    /**
     * The quantity of epochs to be processed.
     */
    private int mEpochs = 20;

    /**
     * The momentum rate.
     */
    private double mMomentum = 0.9;

    /**
     * The network seed. Preferred to be a prime number
     */
    private long mPrimeSeed = 7;//PrimeNumber.getRandomPrimeNumber();

    /**
     * The learn rate.
     */
    private double mLearnRate = 0.00001;

    /**
     * The output layer used by validation test.
     */
    private final NeuronLayer mValidationOutputLayer;

    /**
     * The train pattern array.
     */
    private PatternLayer[] mTrainInputArray;

    /**
     * The validation pattern array.
     */
    private PatternLayer[] mValidationArray;

    /**
     * Creates a Train MLP.
     *
     * @param inLen The length of the input layer.
     * @param h1Len The length of the first hidden layer.
     * @param h2Len The length of the second hidden layer.
     * @param outLen The length of the output layer.
     */
    public TrainMlp(int inLen, int h1Len, int h2Len, int outLen) {
        super(inLen, h1Len, h2Len, outLen);
        mValidationOutputLayer = new NeuronLayer(outLen);
    }

    /**
     * Sets the train pattern array.
     *
     * @param trainPattern The train pattern array.
     */
    public void setInputArray(PatternLayer[] trainPattern) {
        mTrainInputArray = trainPattern;
    }

    /**
     * Sets the validation pattern array.
     *
     * @param validationArray The validation pattern array.
     */
    public void setValidationArray(PatternLayer[] validationArray) {
        mValidationArray = validationArray;
    }

    /**
     * Sets the seed for the randomizer. Preferreable set a prime number.
     *
     * @param seed The seed.
     */
    public void setPrimeSeed(long seed) {
        mPrimeSeed = seed;
    }

    /**
     * Sets the number of epochs to be executed.
     *
     * @param epochs The number of epochs.
     */
    public void setEpochs(int epochs) {
        mEpochs = epochs;
    }

    /**
     * Sets the momentum.
     *
     * @param momentum The momentum.
     */
    public void setMomentum(double momentum) {
        mMomentum = momentum;
    }

    /**
     * Set the learn rate.
     *
     * @param learnRate The learn rate.
     */
    public void setLearnRate(double learnRate) {
        mLearnRate = learnRate;
    }

    /**
     * Sets the absolute maximum weight.
     *
     * @param maxWeight The maximum weight.
     */
    public void setAbsMaxInitialWeight(double maxWeight) {
        mAbsMaxInitialWeight = Math.abs(maxWeight);
    }

    /**
     * Initialize the bias and the weights for all connection layers.
     */
    private void initBiasAndWeights() {
        Random random = new Random(mPrimeSeed);
        for (ConnectionLayer layer : mLayerArray) {
            layer.initBiasAndWeights(random, mAbsMaxInitialWeight);
        }
    }

    /**
     * Runs the validation.
     *
     * @return The total error obtained.
     */
    private double runValidation() {
        NeuronLayer inputLayer = mInputLayer;
        ConnectionLayer outputLayer = mLayerArray[Layer.OUTPUT.ordinal()];

        Neuron[] outputNeurons_bkp = outputLayer.getNeurons();
        outputLayer.setNeurons(mValidationOutputLayer);
        outputLayer.reset();

        double error;
        double totalError = 0.0;

        for (PatternLayer validationLayer : mValidationArray) {
            inputLayer.setNeurons(validationLayer.inputLayer);
            computeActivationOutput();

            error = outputLayer.getDifferenceTotal(validationLayer.outputLayer);
            totalError += error;
        }

        // reset output layer
        outputLayer.setNeurons(outputNeurons_bkp);

        // Retorna o erro total sobre os padroes de validacao
        return totalError;
    }

    /**
     * Resets all connection layers.
     */
    private void resetLayers() {
        for (ConnectionLayer layer : mLayerArray) {
            layer.reset();
        }
    }

    /**
     * Computes the error of this training, via backpropagation.
     *
     * @param trainOutput The output layer of a train pattern.
     */
    private void computeError(NeuronLayer trainOutput) {
        ConnectionLayer[] layers = mLayerArray;

        // Computes the error for the output layer
        ConnectionLayer.computeOutputError(trainOutput, layers[Layer.OUTPUT.ordinal()]);

        // Computes the error for the hidden layers.
        // Out->H2, H2->H1, ...
        for (int i = layers.length - 1; i >= 1; i--) {
            layers[i].computeError();
        }
    }

    /**
     * Computes the bed and the wed increment for all connection layers.
     */
    private void computeBedAndWedIncrement() {
        for (ConnectionLayer layer : mLayerArray) {
            layer.computeBedAndWedIncrement();
        }
    }

    /**
     * Computes the bias delta and the weight delta, based on the learn rate and
     * the momentum, for all connection layers.
     *
     * @param learnRate The learn rate.
     * @param momentum The momentum.
     */
    private void computeBiasAndWeightsDeltas(double learnRate, double momentum) {
        for (ConnectionLayer layer : mLayerArray) {
            layer.computeBiasAndWeightsDeltas(learnRate, momentum);
        }
    }

    /**
     * Computes the bias and the weights for all connection layers.
     */
    private void computeBiasAndWeights() {
        for (ConnectionLayer layer : mLayerArray) {
            layer.computeBiasAndWeights();
        }
    }

    /**
     * Changes the value of the momentum based on the current epoch error and
     * the previous one.
     *
     * @param momentum The current momentum.
     * @param epochError The current epoch error.
     * @param prevEpochError The previous epoch error.
     * @return The new value of the momentum.
     */
    private double changeMomentum(double momentum, double epochError, double prevEpochError) {
        return (epochError >= prevEpochError) ? 0.0 : momentum;
    }

    /**
     * Changes the value of the momentum based on the current epoch error and
     * the previous one.
     *
     * @param momentum The current momentum.
     * @param epochError The current epoch error.
     * @param prevEpochError The previous epoch error.
     * @return The new value of the momentum.
     */
    private double changeLearnRate(double learnRate, double errtot, double lerrtot) {
        return (errtot >= lerrtot) ? (learnRate / 2.0) : (learnRate * 1.02);
    }

    /**
     * Log the current step.
     * @param step The current value of step.
     */
    private void logStep(int step) {
        if (mLogger != null) {
            mLogger.logStep(step);
        }
    }

    /**
     * Log the last computed epoch.
     * @param epoch The epoch.
     * @param error The validation error.
     * @param momentum The end momentum.
     * @param learnRate The end learn rate.
     */
    private void logEpoch(int epoch, double error, double momentum, double learnRate) {
        if (mLogger != null) {
            mLogger.logEpoch(epoch, error, momentum, learnRate);
        }
    }

    /**
     * Closes the log.
     */
    private void logClose() {
        if (mLogger != null) {
            mLogger.close();
        }
    }

    /**
     * Runs the training.
     */
    public void runTrainByEpoch() {
        // optimization
        int maxEpochs = mEpochs;
        double momentum = mMomentum;
        double learnRate = mLearnRate;

        NeuronLayer inputLayer = mInputLayer;
        ConnectionLayer outputLayer = mLayerArray[Layer.OUTPUT.ordinal()];

        // init
        initBiasAndWeights();

        double validationError = runValidation();

        double patError, epochError, prevValidationError;
        double prevEpochError = Double.MAX_VALUE;

        // start process
        int step = 0;
        do {
            // log
            logStep(++step);

            // epochs
            for (int epoch = 1; epoch <= maxEpochs; epoch++) {
                // reset
                epochError = 0.0;
                resetLayers();

                // compute for each input pattern
                for (PatternLayer trainLayer : mTrainInputArray) {

                    // insert pattern input in the network
                    inputLayer.setNeurons(trainLayer.inputLayer);

                    // compute activation (through  the layers)
                    computeActivationOutput();

                    // compute the error on the pattern output
                    computeError(trainLayer.outputLayer);

                    // compute the bed and the wed increments
                    computeBedAndWedIncrement();

                    // get the sum of the difference between the pattern
                    //output and the network output
                    patError = outputLayer.getDifferenceTotal(trainLayer.outputLayer);

                    // increment the epoch error
                    epochError += patError;
                }

                // compute the bias and the weight errors (deltas)
                computeBiasAndWeightsDeltas(learnRate, momentum);

                // compute the bias and weight
                computeBiasAndWeights();

                // update the momentum and learn rate
                momentum = changeMomentum(momentum, epochError, prevEpochError);
                learnRate = changeLearnRate(learnRate, epochError, prevEpochError);

                // save the current value
                prevEpochError = epochError;

                // log
                logEpoch(epoch, epochError, momentum, learnRate);
            }

            // save the current validation error
            prevValidationError = validationError;

            // run validation
            validationError = runValidation();

            // run until the current error is small then the previous
        } while (validationError < prevValidationError);
    }
}
