package ai.certifai.training.classification.RockPaperScissorClassification;

import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.filters.PathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.BaseImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.datavec.image.transform.*;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.common.io.ClassPathResource;
import org.nd4j.common.primitives.Pair;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.slf4j.Logger;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.io.File;

public class RockPaperScissorDataSet {
    private static final Logger log = org.slf4j.LoggerFactory.getLogger(RockPaperScissorDataSet.class);

    private static final String[] allowedFormats = BaseImageLoader.ALLOWED_FORMATS;
    private static Random rndseed = new Random(123);
    private static int batchSize;
    private static final int height = 200;
    private static final int width = 200;
    private static final int channels = 3;
    private static final int numOutput = 3;

    private static ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
    private static InputSplit trainData, validData;


    public static DataSetIterator trainIterator() throws IOException {
        return makeIterator(trainData);
    }

    public static DataSetIterator trainIteratorAugmented(boolean shuffle) throws IOException {
        //Image Augmentation
        ImageTransform hflip = new FlipImageTransform(1);
        ImageTransform rotate = new RotateImageTransform(45);

        //Usage to monitor the images being transformed
//        ImageTransform showImage = new ShowImageTransform("Transformed images", 50);

        List<Pair<ImageTransform, Double>> transform = Arrays.asList(
                new Pair<>(hflip, 0.5),
                new Pair<>(rotate, 0.3)
//                new Pair<>(showImage, 1.0)
// showImage use to Monitor image
        );

        ImageTransform pipeline = new PipelineImageTransform(transform, shuffle);

        return makeIterator(trainData, pipeline);
    }

    public static DataSetIterator validIterator() throws IOException {
        return makeIterator(validData);
    }

    private static DataSetIterator makeIterator (InputSplit split) throws IOException {
        ImageRecordReader rr = new ImageRecordReader(height, width, channels, labelMaker);
        rr.initialize(split);
        DataSetIterator iter = new RecordReaderDataSetIterator(rr, batchSize, 1, numOutput);

        DataNormalization scaler = new ImagePreProcessingScaler();
        scaler.fit(iter);
        iter.setPreProcessor(scaler);
        return iter;
    }

    private static DataSetIterator makeIterator (InputSplit split, ImageTransform pipeline) throws IOException {
        ImageRecordReader rr = new ImageRecordReader(height, width, channels, labelMaker);
        rr.initialize(split, pipeline);
        DataSetIterator iter = new RecordReaderDataSetIterator(rr, batchSize, 1, numOutput);

        DataNormalization scaler = new ImagePreProcessingScaler();
        scaler.fit(iter);
        iter.setPreProcessor(scaler);

//        ImageRecordReader recordReader = new ImageRecordReader(height,width,channels,labelMaker);
//        recordReader.initialize(split);
//        DataSetIterator iter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, numOutput);
//        iter.setPreProcessor(new VGG16ImagePreProcessor()); //copy from WeatherDataSetIterator
        return iter;
    }

    public static void setup(int batchSizeArg) throws IOException, IllegalAccessException {

        batchSize = batchSizeArg;

        File inputFileTrain = new ClassPathResource("Rock-Paper-Scissors/train").getFile();
        FileSplit fileSplitTrain = new FileSplit(inputFileTrain);
        FileSplit fileSplitValidation = new FileSplit(new ClassPathResource("Rock-Paper-Scissors/test").getFile());

        PathFilter pathFilter = new BalancedPathFilter(rndseed, allowedFormats, labelMaker);

        InputSplit[] sampleTrain = fileSplitTrain.sample(pathFilter, 1);
        InputSplit[] sampleValidation = fileSplitValidation.sample(pathFilter, 1);
        trainData = sampleTrain[0];
        validData = sampleValidation[0];


    }
}
