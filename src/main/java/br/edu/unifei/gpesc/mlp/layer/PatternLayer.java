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
 * This class represents a pattern.
 * @author Isaac Caldas Ferreira
 */
public class PatternLayer {

    /**
     * The input layer, with the input neurons.
     */
    public final NeuronLayer inputLayer;

    /**
     * The output layer, with the output neurons.
     */
    public final NeuronLayer outputLayer;

    /**
     * Creates a Pattern Layer setting the layers.
     * @param input The input layer.
     * @param output The output layer.
     */
    public PatternLayer(NeuronLayer input, NeuronLayer output) {
        inputLayer = input;
        outputLayer = output;
    }
}