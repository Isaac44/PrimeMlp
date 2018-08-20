package br.edu.unifei.gpesc.app;

import br.edu.unifei.gpesc.mlp.Mlp;
import br.edu.unifei.gpesc.mlp.layer.PatternLayer;
import java.io.File;
import java.io.IOException;

/**
 *
 * @author Isaac Caldas Ferreira
 */
public class AppRunMlp {

    private static PatternLayer[] loadLayers(File folder) throws IOException {
        PatternLayer[] spams = NeuralBuilder.loadTrainMlp(new File(folder, "spam.dat"), NeuralBuilder.SPAM);
        PatternLayer[] hams = NeuralBuilder.loadTrainMlp(new File(folder, "ham.dat"), NeuralBuilder.HAM);
        return NeuralBuilder.merge(hams, spams);
    }

    public static void main(String[] args) throws IOException {
        // Carregar a rede neural com os pesos treinados.
        Mlp runMlp = Mlp.loadMlp(new File("vectors/train/weights.dat"));

        // Carregar os padroes de teste
        PatternLayer[] layers = loadLayers(new File("vectors/run/"));

        // Iniciar teste
        runMlp.runTestSup(layers);
    }

}
