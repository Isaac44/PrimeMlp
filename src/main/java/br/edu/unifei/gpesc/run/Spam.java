/*
 * Copyright (C) 2015 - GEPESC - Universidade Federal de Itajuba
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

/**
 *
 * @author Isaac Caldas Ferreira
 */
public class Spam {

    /**
     * The maximun difference between the expected result and the result.
     */
    private static final double MAX_DIFFERENCE = 0.4;

    /**
     * The not-spam constant.
     */
    public static final double[] HAM = {0.0, 1.0};

    /**
     * The spam constant.
     */
    public static final double[] SPAM = {1.0, 0.0};

    /**
     * Checks if the values are compatible with the not spam (ham) pattern.
     *
     * @param v1 Activation 1
     * @param v2 Activation 2
     *
     * @return true if are compatible.
     */
    public static boolean isHam(double v1, double v2) {
        return compare(HAM[0], v1) && compare(HAM[1], v2);
    }

    /**
     * Checks if the values are compatible with the spam pattern.
     *
     * @param v1 Activation 1
     * @param v2 Activation 2
     *
     * @return true if are compatible.
     */
    public static boolean isSpam(double v1, double v2) {
        return compare(SPAM[0], v1) && compare(SPAM[1], v2);
    }


    /**
     * Compares two neuron activations.
     *
     * @param e The expected activation.
     * @param r The resulted activation.
     *
     * @return true, if the difference is bellow {@link #MAX_DIFFERENCE}.
     * And false, otherwise.
     */
    public static boolean compare(double e, double r) {
        return (Math.abs(e - r) <= MAX_DIFFERENCE);
    }

}
