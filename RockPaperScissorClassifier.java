package ai.certifai.training.classification.RockPaperScissorClassification;

import org.deeplearning4j.core.storage.StatsStorage;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.InvocationType;
import org.deeplearning4j.optimize.listeners.EvaluativeListener;
import org.deeplearning4j.optimize.listeners.PerformanceListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.model.stats.StatsListener;
import org.deeplearning4j.ui.model.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;

public class RockPaperScissorClassifier {

    private static int nEpoch = 1;
    private static final int batchSize = 8;
    private static double lr = 1e-3;
    private static int seed = 123;
    private static final int height = 200;
    private static final int width = 200;
    private static final int channels = 3;
    private static final int numOutput = 3;

    public static void main(String[] args) throws IOException, IllegalAccessException {

        //NN Config
        MultiLayerConfiguration config = new NeuralNetConfiguration.Builder()
                .seed(123)
                .weightInit(WeightInit.RELU)
                .updater(new Adam(lr))
                .activation(Activation.RELU)
                .list()
                // Convolutional, ReLU, Pooling, FC layer, output layer
                .layer(new ConvolutionLayer.Builder()
                        .kernelSize(3,3)
                        .stride(1,1)
                        .nIn(channels)
                        .nOut(12) // filter count
                        .build())
                .layer(new SubsamplingLayer.Builder() //Pooling layer, doesnt need nIn
                        .kernelSize(2,2) //downsample the image into half
                        .stride(2,2)
                        .poolingType(SubsamplingLayer.PoolingType.MAX)
                        .build())
                .layer(new ConvolutionLayer.Builder()
                        .kernelSize(3,3)
                        .stride(1,1)
                        .nOut(24)
                        .build())
                .layer(new ConvolutionLayer.Builder()
                        .kernelSize(3,3)
                        .stride(1,1)
                        .nOut(36)
                        .build())
                .layer(new SubsamplingLayer.Builder()
                        .kernelSize(2,2)
                        .stride(2,2)
                        .poolingType(SubsamplingLayer.PoolingType.MAX)
                        .build())
                .layer(new DenseLayer.Builder()
                        .activation(Activation.RELU)
                        .nOut(20) //number of neurons
                        .build())
                .layer(new DenseLayer.Builder()
                        .activation(Activation.RELU)
                        .nOut(10)
                        .build())
                .layer(new OutputLayer.Builder()
                        .activation(Activation.SOFTMAX)
                        .lossFunction(LossFunctions.LossFunction.MCXENT)
                        .nOut(numOutput)
                        .build())
                .setInputType(InputType.convolutional(height, width, channels))
                .build();

        // Build & initialize model
        MultiLayerNetwork model = new MultiLayerNetwork(config);
        model.init();

        RockPaperScissorDataSet.setup(batchSize);
        DataSetIterator trainIter = RockPaperScissorDataSet.trainIteratorAugmented(true);
        DataSetIterator validIter = RockPaperScissorDataSet.validIterator();

//        Set up listener & train model
        StatsStorage storage = new InMemoryStatsStorage();
        UIServer server = UIServer.getInstance();
        server.attach(storage);
        model.setListeners(
                new StatsListener(storage),
                new ScoreIterationListener(5),
                new PerformanceListener(5),
                new EvaluativeListener(trainIter, 1, InvocationType.EPOCH_END),
                new EvaluativeListener(validIter, 1, InvocationType.EPOCH_END)
        );
        model.fit(trainIter, nEpoch);

//      Evaluate model performance
        Evaluation evalTrain = model.evaluate(trainIter);
        Evaluation evalTest = model.evaluate(validIter);

        System.out.println("Train Evaluation: " + evalTrain.stats());
        System.out.println("Test Evaluation: " + evalTest.stats());

        System.out.println("============= Save Trained Model =============");
        File modelFilename = new File(System.getProperty("user.dir"), "generated-models/rps-classifier16.zip");
        ModelSerializer.writeModel(model, modelFilename, true);
        System.out.println("============= Model Saved =============");
    }
}
