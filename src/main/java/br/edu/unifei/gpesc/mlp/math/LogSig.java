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
 * This class computes the logsig transfer function: <br>
 * <b>logsig(x) = 1 / (1 + exp(-x))</b>
 * @author Isaac Caldas Ferreira
 */
public class LogSig implements Function {

    /**
     * Computes the logsig transfer function, which is given by the equation:
     * <b>logsig(x) = 1 / (1 + exp(-x))</b>
     *
     * @param x {@inheritDoc}
     * @return The logsig result.
     */
    @Override
    public double compute(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }

    @Override
    public double compute(double x, double y) {
        return x * y * (1.0 - y);
    }
}
