package br.edu.unifei.gpesc.app;

import br.edu.unifei.gpesc.mlp.TrainMlp;
import java.io.File;
import java.io.IOException;

/**
 *
 * @author Isaac Caldas Ferreira
 */
public class AppTrain {

    public static void main(String[] args) throws IOException {
        // onde esta' os arquivos com os vetores de spam e ham
        String path = "vectors/train/";
        NeuralBuilder t = new NeuralBuilder(new File(path, "ham.dat"), new File(path, "spam.dat"), 1.0 / 3.0);

        // quantidade de neuronios das camadas escondidas
        // nota: a quantidade de neuronios da primeira camada
        //esta' dentro do arquivo com os vetores.
        int firstHiddenLayerLength = 10;
        int secondHiddenLayerLength = 12;

        // Criar mlp
        TrainMlp mlp = t.buildWith(firstHiddenLayerLength, secondHiddenLayerLength);

        // Treinar
        mlp.runTrainByEpoch();

        // Salvar os pesos finais.
        mlp.saveMlp(new File(path, "weights.dat"));
    }

}
