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
package br.edu.unifei.gpesc.mlp.log;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.io.Writer;
import java.util.Properties;

/**
 *
 * @author Isaac Caldas Ferreira
 */
public class MlpLogger {

    /**
     * The step log pattern.
     */
    private final String STEP;

    /**
     * The epoch log pattern.
     */
    private final String EPOCH;

    /**
     * The error log pattern.
     */
    private final String ERROR;

    /**
     * The momentum log pattern.
     */
    private final String MOMENTUM;

    /**
     * The learn rate log pattern.
     */
    private final String LEARN_RATE;

    /**
     * The pattern log pattern.
     */
    private final String PATTERN;

    /**
     * The correct log pattern.
     */
    private final String CORRECT;

    /**
     * The incorrect log pattern.
     */
    private final String INCORRECT;

    /**
     * The total error log pattern.
     */
    private final String TOTAL_ERROR;

    /**
     * False-positive error log pattern.
     */
    private final String ERROR_FALSE;

    /**
     * Unknown error log pattern.
     */
    private final String ERROR_UNKNOWN;

    /**
     * The correct pattern log pattern.
     */
    private final String PATTERN_CORRECT;

    /**
     * The incorrect pattern log pattern.
     */
    private final String PATTERN_INCORRECT;

    /**
     * The result log pattern.
     */
    private final String PATTERN_RESULT;

    /**
     * File Writer.
     */
    private final Writer mWriter;

    /**
     * The internationalization properties.
     */
    private final Properties mInternationalization;

    /**
     * Creates an Async MLP Logger. All writer will occur on the
     * ExecutorService Thread.
     *
     * This uses the defaults internationalization strings. (current: PT-BR)
     *
     * @param file The file to store the log.
     *
     * @throws FileNotFoundException If the file does not exists.
     */
    public MlpLogger(File file) throws IOException {
        this(file, i18nDefaults_PT());
    }

    /**
     * Creates an Async MLP Logger. All writer will occur on the
     * ExecutorService Thread.
     *
     * This uses the defaults internationalization strings. (current: PT-BR)
     *
     * @param file The file to store the log.
     * @param i18n The internationalization strings for the output.
     *
     * @throws FileNotFoundException If the file does not exists.
     */
    public MlpLogger(File file, Properties i18n) throws IOException {
        mWriter = new BufferedWriter(new FileWriter(file));

        mInternationalization = i18n;

        STEP = i18nF("\n> ", "Step", " ");
        EPOCH = i18nF("\n\n\t> ", "Epoch", " "  );
        ERROR = i18nF("\n\t\t> ", "Error", " = ");
        MOMENTUM = i18nF("\n\t\t> ", "Momentum" , " = ");
        LEARN_RATE = i18nF("\n\t\t> ", "LearnRate", " = ");
        PATTERN = i18nF("\n\t> ", "Pattern", " ");
        CORRECT = i18nF(": ", "Correct", "");
        INCORRECT = i18nF(": ", "Incorrect", "");
        TOTAL_ERROR = i18nF("\n\n> ", "TotalError", " = ");
        ERROR_FALSE = i18nF("\n\t\t> ", "Error.False", " = ");
        ERROR_UNKNOWN = i18nF("\n\t\t> ", "Error.Unknown", " = ");
        PATTERN_CORRECT = i18nF("\n> ", "Pattern.Correct", " = ");
        PATTERN_INCORRECT = i18nF("\n> ", "Pattern.Incorrect", " = ");
        PATTERN_RESULT = i18nF("\n\t\t> ", "Pattern.Result", " = ");
    }

    private void append(String str) {
        try {
            mWriter.append(str);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void close() {
        try {
            mWriter.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    /**
     * Logs the current step.
     * @param step
     */
    public void logStep(int step) {
        append(STEP + step);
    }

    /**
     * Logs the current epoch, adding an "end line" character at end ("\n").
     * @param epoch
     * @param error
     * @param momentum
     * @param learnRate
     */
    public void logEpoch(int epoch, double error, double momentum, double learnRate) {
        logEpoch(epoch);
        logError(error);
        logMomentum(momentum);
        logLearnRate(learnRate);
        append("\n");
    }

    /**
     * Logs only the current epoch number.
     * @param epoch
     */
    public void logEpoch(int epoch) {
        append(EPOCH + epoch);
    }

    /**
     * Logs error.
     * @param error
     */
    public void logError(double error) {
        append(ERROR + error);
    }

    public void logErrorType(boolean onLimbo) {
        append(onLimbo? ERROR_UNKNOWN : ERROR_FALSE);
    }

    /**
     * Logs the current momentum.
     * @param momentum
     */
    public void logMomentum(double momentum) {
        append(MOMENTUM + momentum);
    }

    /**
     * Logs the current learn rate.
     * @param learnRate
     */
    public void logLearnRate(double learnRate) {
        append(LEARN_RATE + learnRate);
    }

    /**
     * Log a train header.
     * @param h1Len The first hidden layer length.
     * @param h2Len The second hidden layer length.
     * @param h1F The first hidden layer function name.
     * @param h2F The second hidden layer function name.
     * @param outF The output layer function name.
     * @param seed The seed.
     * @param epochs The number of epochs.
     * @param momentum The initial momentum.
     * @param learnRate The initial learn rate.
     */
    public void logTrainHead(int h1Len, int h2Len,
            String h1F, String h2F, String outF,
            long seed, int epochs, double momentum, double learnRate)
    {
        StringBuilder sb = new StringBuilder();

        sb.append("MLP");

        // first hidden layer
        sb.append("\n\t> ").append(i18nF("FirstHiddenLayer"))
            .append("\n\t\t> ").append(i18nF("Length")).append(" = ").append(h1Len)
            .append("\n\t\t> ").append(i18nF("Function")).append(" = ").append(h1F);

        // second hidden layer
        sb.append("\n\t> ").append(i18nF("SecondHiddenLayer"))
            .append("\n\t\t> ").append(i18nF("Length")).append(" = ").append(h2Len)
            .append("\n\t\t> ").append(i18nF("Function")).append(" = ").append(h2F);

        // output layer
        sb.append("\n\t> ").append(i18nF("OutputLayer"))
            .append("\n\t\t> ").append(i18nF("Function")).append(" = ").append(outF);

        // config
        sb.append("\n\t> ").append(i18nF("Seed")).append(" = ").append(seed);
        sb.append("\n\t> ").append(i18nF("Epochs")).append(" = ").append(epochs);
        sb.append("\n\t> ").append(i18nF("InitMomentum")).append(" = ").append(momentum);
        sb.append("\n\t> ").append(i18nF("InitLearnRate")).append(" = ").append(learnRate);

        // end
        sb.append("\n\n");

        // write log
        append(sb.toString());
    }

    /**
     * Log a separator.
     */
    public void logSeparator() {
        append("\n\n-------------------------------------------------------------------------------\n\n");
    }

    /**
     * Logs a result.
     * @param expected The expected result.
     * @param result The obtained result.
     */
    public void logResult(double expected, double result) {
        append(PATTERN_RESULT + expected + " [" + result + "]");
    }

    /**
     * Logs a pattern result.
     * @param index The pattern index.
     * @param correct The result.
     */
    public void logPattern(int index, boolean correct) {
        String result = correct ? CORRECT : INCORRECT;
        append(PATTERN + index + result);
    }

    /**
     * Logs a total error.
     * @param error
     */
    public void logTotalError(double error) {
        append(TOTAL_ERROR + error);
    }

    /**
     * Logs the quantity of correct patterns.
     * @param quantity
     */
    public void logCorrectPatterns(int quantity) {
        append(PATTERN_CORRECT + quantity);
    }

    /**
     * Logs the quantity of incorrect patterns.
     * @param quantity
     */
    public void logIncorrectPatterns(int quantity) {
        append(PATTERN_INCORRECT + quantity);
    }

    /**
     * Retrieves the internationalization string for the output log.
     * @param key
     * @return
     */
    private String i18nF(String key) {
        return mInternationalization.getProperty(key);
    }

    /**
     * Format and retrieves the internationalization string for the output log.
     * @param before
     * @param key
     * @param after
     * @return before + {@link #i18nF(String)} + after
     */
    private String i18nF(String before, String key, String after) {
        return before + i18nF(key) + after;
    }

    /**
     * See the source code for more details.
     * @return
     */
    private static Properties i18nDefaults_PT() {
        Properties p = new Properties();

        p.put("Step", "Passo");
        p.put("Epoch", "Época");
        p.put("Error", "erro");
        p.put("Error.Unknown", "Tipo de erro: sem classificação");
        p.put("Error.False", "Tipo de erro: falso positivo");
        p.put("Momentum", "momentum");
        p.put("LearnRate", "learn rate");
        p.put("FirstHiddenLayer", "Primeira camada escondida");
        p.put("SecondHiddenLayer", "Segunda camada escondida");
        p.put("OutputLayer", "Camada de saída");
        p.put("Length", "Neurônios");
        p.put("Function", "Função");
        p.put("Seed", "Semente");
        p.put("Epochs", "Épocas");
        p.put("InitMomentum", "Momentum inicial");
        p.put("InitLearnRate", "Taxa de aprendizado inicial (learn rate)");
        p.put("Pattern", "Padrão");
        p.put("Correct", "CORRETO");
        p.put("Incorrect", "INCORRETO");
        p.put("TotalError", "Erro Total");
        p.put("Pattern.Correct", "Padrões classificados corretamente");
        p.put("Pattern.Incorrect", "Padrões classificados incorretamente");
        p.put("Pattern.Result", "espera [obtido]");

        return p;
    }
}
