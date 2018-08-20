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
package br.edu.unifei.gpesc.mlp.layer;

import br.edu.unifei.gpesc.mlp.math.Function;
import java.nio.ByteBuffer;
import java.util.Random;

/**
 * This class connects the neurons of a Layer to this own neurons.
 *
 * @author Isaac Caldas Ferreira
 */
public class ConnectionLayer extends NeuronLayer {

    /**
     * The layer that comes previously of this one.
     */
    private final NeuronLayer mPreviousLayer;

    /**
     * The transfer function of the previous layer to this layer.
     */
    private Function mFunction;

    /**
     * The connections between this layer and the previous one. <br>
     * Each line corresponds this layer and each column corresponds the previous
     * layer.
     */
    private final Connection[][] mConnectionMatrix;

    /**
     * Creates a Connection Layer.
     *
     * @param length The size of the neuron array.
     * @param prevLayer The layer that comes before this one.
     * @param function The transfer function.
     * @see Function
     */
    public ConnectionLayer(int length, NeuronLayer prevLayer, Function function) {
        super(length);
        mFunction = function;
        mPreviousLayer = prevLayer;
        mConnectionMatrix = new Connection[length][prevLayer.getLength()];

        int i, j;
        Connection[][] connections = mConnectionMatrix;

        for (i = 0; i < connections.length; i++) {
            for (j = 0; j < connections[i].length; j++) {
                connections[i][j] = new Connection();
            }
        }
    }

    /**
     * Sets the transfer function.
     *
     * @param function The transfer function.
     */
    public void setFunction(Function function) {
        mFunction = function;
    }

    /**
     * Returns the length of the matrix of connections.
     * @return The length of the neuro-connections.
     */
    public int getConnectionsLength() {
        return mConnectionMatrix.length * mConnectionMatrix[0].length;
    }

    /**
     * First, this method will put the bias of the neurons and than the
     * neuron connections weights.
     * <p> The values are: all the {@link NeuronLayer.Neuron#bias} in the
     * neuron array and the all the {@link Connection#weight} in the connection matrix.
     *
     * @param buffer The output buffer.
     */
    public void toByteBuffer(ByteBuffer buffer) {
        for (Neuron neuron : mNeurons) {
            buffer.putDouble(neuron.bias);
        }

        for (Connection[] connectionArray : mConnectionMatrix) {
            for (Connection connection : connectionArray) {
                buffer.putDouble(connection.weight);
            }
        }
    }

    /**
     * Loads the bias and the weights from the buffer.
     * @param buffer The input buffer.
     */
    public void loadFromByteBuffer(ByteBuffer buffer) {
        for (Neuron neuron : mNeurons) {
            neuron.bias = buffer.getDouble();
        }

        for (Connection[] connectionArray : mConnectionMatrix) {
            for (Connection connection : connectionArray) {
                connection.weight = buffer.getDouble();
            }
        }
    }

    /**
     * Initialize the {@link NeuronLayer.Neuron#bias} and {@link Connection#weight}.
     *
     * @param rand The randomizer to be used to generate random values.
     * @param maxWeight The maximum absolute weight for the values.
     */
    public void initBiasAndWeights(Random rand, double maxWeight) {
        int i, j;

        Neuron[] neurons = mNeurons;
        Connection[][] connections = mConnectionMatrix;

        for (i = 0; i < neurons.length; i++) {
            neurons[i].bias = rand.nextDouble() * maxWeight;

            if (!rand.nextBoolean()) {
                neurons[i].bias *= -1;
            }

            for (j = 0; j < connections[i].length; j++) {
                connections[i][j].weight = rand.nextDouble() * maxWeight;

                if (!rand.nextBoolean()) {
                    connections[i][j].weight *= -1;
                }
            }
        }
    }

    /**
     * Computes the {@link NeuronLayer.Neuron#bias} and the
     * {@link Connection#weight}.
     */
    public void computeBiasAndWeights() {
        int i, j;

        Neuron[] neurons = mNeurons;
        Connection[][] connections = mConnectionMatrix;

        for (i = 0; i < neurons.length; i++) {
            for (j = 0; j < connections[i].length; j++) {
                connections[i][j].weight += connections[i][j].dweight;
            }
            neurons[i].bias += neurons[i].dbias;
        }
    }

    /**
     * Computes the {@link NeuronLayer.Neuron#bed} and the
     * {@link Connection#wed}.
     */
    public void computeBedAndWedIncrement() {
        int i, j;

        Neuron[] neurons = mNeurons;
        Neuron[] prevNeurons = mPreviousLayer.mNeurons;
        Connection[][] connections = mConnectionMatrix;

        for (i = 0; i < neurons.length; i++) {
            for (j = 0; j < prevNeurons.length; j++) {
                connections[i][j].wed += neurons[i].delta * prevNeurons[j].activation;
            }
            neurons[i].bed += neurons[i].delta;
        }
    }

    /**
     * Computes the {@link NeuronLayer.Neuron#dbias} and the
     * {@link Connection#dweight}, based on the learn rate and the momentum.
     *
     * @param learnRate The learn rate.
     * @param momentum The momentum.
     */
    public void computeBiasAndWeightsDeltas(double learnRate, double momentum) {
        int i, j;

        Neuron[] neurons = mNeurons;
        Connection[][] connections = mConnectionMatrix;

        for (i = 0; i < neurons.length; i++) {
            for (j = 0; j < connections[i].length; j++) {
                connections[i][j].dweight = (learnRate * connections[i][j].wed) + (momentum * connections[i][j].dweight);
            }
            neurons[i].dbias = (learnRate * neurons[i].bed) + (momentum * neurons[i].dbias);
        }
    }

    /**
     * Sets the values {@link NeuronLayer.Neuron#bed},
     * {@link NeuronLayer.Neuron#activation} and {@link Connection#wed} to zero.
     */
    public void reset() {
        int i, j;

        Neuron[] neurons = mNeurons;
        Connection[][] connections = mConnectionMatrix;

        for (i = 0; i < neurons.length; i++) {

            for (j = 0; j < connections[i].length; j++) {
                connections[i][j].wed = 0.0;
            }

            neurons[i].bed = 0.0;
            neurons[i].activation = 0.0;
        }
    }

    /**
     * Computes the {@link NeuronLayer.Neuron#activation}.
     */
    public void computeActivationOutput() {
        int i, j;

        Neuron[] neurons = mNeurons;
        Neuron[] prevNeurons = mPreviousLayer.mNeurons;
        Connection[][] connections = mConnectionMatrix;

        Function function = mFunction;

        double netinput;

        for (i = 0; i < neurons.length; i++) {
            netinput = neurons[i].bias;

            for (j = 0; j < prevNeurons.length; j++) {
                netinput += connections[i][j].weight * prevNeurons[j].activation;
            }

            neurons[i].activation = function.compute(netinput);
        }
    }

    /**
     * Computes the {@link NeuronLayer.Neuron#delta} error, via backpropagation.
     */
    public void computeError() {
        int j, i;
        double x;

        Neuron[] neurons = mNeurons;
        Connection[][] connections = mConnectionMatrix;

        Neuron[] prevNeurons = mPreviousLayer.mNeurons;
        Function function = ((ConnectionLayer) mPreviousLayer).mFunction;

        for (j = 0; j < prevNeurons.length; j++) {
            x = 0.0;

            for (i = 0; i < neurons.length; i++) {
                x += neurons[i].delta * connections[i][j].weight;
            }

            prevNeurons[j].delta = function.compute(x, prevNeurons[j].activation);
        }
    }

    /**
     * Computes the {@link NeuronLayer.Neuron#delta} error for the Output Layer.
     *
     * @param trainOutput
     * @param outputLayer
     */
    public static void computeOutputError(NeuronLayer trainOutput, ConnectionLayer outputLayer) {
        Neuron[] prevNeurons = outputLayer.mNeurons;
        Neuron[] neurons = trainOutput.mNeurons;

        Function function = outputLayer.mFunction;

        double x, y;

        for (int j = 0; j < prevNeurons.length; j++) {
            y = prevNeurons[j].activation;
            x = neurons[j].activation - y;

            prevNeurons[j].delta = function.compute(x, y);
        }
    }

    /**
     * This class represents the connections between all neurons of a
     * {@link ConnectionLayer} and a {@link NeuronLayer}.
     *
     * @author Otávio Augusto Salgado Carpinteiro
     */
    private static class Connection {

        /**
         * The weight.
         */
        double weight;

        /**
         * The weight's error derivative.
         */
        double wed;

        /**
         * The weight delta.
         */
        double dweight;
    }
}
