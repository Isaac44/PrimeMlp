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

/**
 * The neuron layer class for the Multi-Layer Perceptron Network.
 *
 * @author Isaac Caldas Ferreira
 */
public class NeuronLayer {

    /**
     * The neurons of this layer.
     */
    protected Neuron[] mNeurons;

    /**
     * Creates a empty Neuron Layer.
     */
    public NeuronLayer() {
    }

    /**
     * Creates a Neuron Layer initializing all neurons with an activation value.
     *
     * @param neurons The activation value array to initialize all
     * neurons in the neuron array.
     */
    public NeuronLayer(double... neurons) {
        mNeurons = new Neuron[neurons.length];

        Neuron[] array = mNeurons;
        for (int i = 0; i < array.length; i++) {
            array[i] = new Neuron(neurons[i]);
        }
    }

    /**
     * Creates a Neuron Layer setting the size of the neuron array.
     *
     * @param length The size of the neuron array.
     */
    public NeuronLayer(int length) {
        mNeurons = new Neuron[length];

        Neuron[] neurons = mNeurons;

        for (int i = 0; i < neurons.length; i++) {
            neurons[i] = new Neuron();
        }
    }

    /**
     * Sets the neuron array.
     *
     * @param neuronArray The new neuron array.
     */
    public void setNeurons(Neuron[] neuronArray) {
        mNeurons = neuronArray;
    }

    /**
     * Gets the neuron array.
     * @return The neuron array.
     */
    public Neuron[] getNeurons() {
        return mNeurons;
    }

    /**
     * Sets the neuron array.
     *
     * @param layer The layer with the new neuron array.
     */
    public void setNeurons(NeuronLayer layer) {
        mNeurons = layer.mNeurons;
    }

    /**
     * Gets the neuron array length.
     *
     * @return The neuron array length.
     */
    public int getLength() {
        return mNeurons.length;
    }

    /**
     * Computes the absolute difference between each {@link Neuron#activation}
     * and stores the result at {@link Neuron#activation} of this object's
     * neuron array.
     *
     * @param layer The other layer.
     */
    public void computeDifference(NeuronLayer layer) {
        Neuron[] neurons = mNeurons;
        Neuron[] otherNeurons = layer.mNeurons;

        for (int i = 0; i < neurons.length; i++) {
            neurons[i].activation = Math.abs(otherNeurons[i].activation - neurons[i].activation);
        }
    }

    /**
     * Gets the sum of the absolute difference between each
     * {@link Neuron#activation}.
     *
     * @param layer The other layer.
     * @return The sum of the difference of each Neuron of the layers.
     */
    public double getDifferenceTotal(NeuronLayer layer) {
        double error = 0.0;

        Neuron[] neurons = mNeurons;
        Neuron[] otherNeurons = layer.mNeurons;

        for (int i = 0; i < neurons.length; i++) {
            error += Math.abs(otherNeurons[i].activation - neurons[i].activation);
        }

        return error;
    }

    /**
     * This class represents the Neuron unity of the Multi-Layer Perceptron
     * Network.
     *
     * @author Otávio Augusto Salgado Carpinteiro
     */
    public static class Neuron {

        /**
         * The activation value.
         */
        public double activation;

        /**
         * The bias connection.
         */
        double bias;

        /**
         * The error derivative.
         */
        double delta;

        /**
         * The bias connection error derivative.
         */
        double bed;

        /**
         * The bias delta.
         */
        double dbias;

        /**
         * Creates a Neuron with activation equals to zero.
         */
        public Neuron() {
        }

        /**
         * Creates a Neuron, setting the activation value.
         *
         * @param activation The activation value.
         */
        public Neuron(double activation) {
            this.activation = activation;
        }
    }
}
