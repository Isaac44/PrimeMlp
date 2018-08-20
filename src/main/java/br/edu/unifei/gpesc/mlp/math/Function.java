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
package br.edu.unifei.gpesc.mlp.math;

/**
 * Interface for the transfer function of the mlp neural network.
 * @author Isaac Caldas Ferreira
 */
public interface Function {

    /**
     * Computes the activation for the input value.
     * @param x The input value.
     * @return The activation result.
     */
    public double compute(double x);

    /**
     * Computes the delta for the input value.
     * @param x The X value.
     * @param y The Y value.
     * @return The activation result.
     */
    public double compute(double x, double y);
}
