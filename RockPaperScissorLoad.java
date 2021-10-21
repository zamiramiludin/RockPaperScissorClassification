package ai.certifai.training.classification.RockPaperScissorClassification;

import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.common.io.ClassPathResource;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;

public class RockPaperScissorLoad {
    private static Logger log = LoggerFactory.getLogger(RockPaperScissorLoad.class);

    public static void main(String[] args) throws Exception {
        int height = 200;
        int width = 200;
        int noChannels = 3;

        File modelLoad = new File(System.getProperty("user.dir"), "generated-models/rps-classifier.zip");

        if(modelLoad.exists() == false)
        {
            System.out.println("Model not exist. Abort.");
            return;
        }
        File imageToTest = new ClassPathResource("Rock-Paper-Scissors/validation/paper3.png").getFile();

        MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork(modelLoad);

        NativeImageLoader loader = new NativeImageLoader(height, width, noChannels);
        INDArray image = loader.asMatrix(imageToTest);

        DataNormalization scaler = new ImagePreProcessingScaler(0,1);
        scaler.transform(image);

        int[] output2 = model.predict(image);
        INDArray output = model.output(image);
        log.info("Predict " + output2[0]);
        log.info("Label: " + Nd4j.argMax(output, 1));
        log.info("Probabilities: " + output.toString());
    }
}
