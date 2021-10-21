//package ai.certifai.training.classification.RockPaperScissorClassification;
//
//import org.bytedeco.opencv.opencv_core.*;
//
//import static org.bytedeco.opencv.global.opencv_core.flip;
//import static org.bytedeco.opencv.global.opencv_imgproc.*;
//
//import org.bytedeco.javacv.*;
//import org.datavec.image.loader.NativeImageLoader;
//import org.datavec.image.transform.ColorConversionTransform;
//import org.deeplearning4j.nn.graph.ComputationGraph;
//import org.deeplearning4j.nn.layers.objdetect.DetectedObject;
//import org.deeplearning4j.nn.layers.objdetect.YoloUtils;
//import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
//import org.deeplearning4j.util.ModelSerializer;
//import org.deeplearning4j.zoo.ZooModel;
//import org.deeplearning4j.zoo.model.TinyYOLO;
//import org.deeplearning4j.zoo.util.darknet.VOCLabels;
//import org.nd4j.linalg.api.ndarray.INDArray;
//import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
//import org.nd4j.linalg.factory.Nd4j;
//
//import java.awt.event.KeyEvent;
//import java.io.File;
//import java.util.List;
//
//public class RockPaperScissorWebcamClassifier {
//    private static String cameraPos = "front";
//
//    private  static int cameraNum = 0;
//    private static Thread thread;
//    private static final int gridWidth = 13;
//    private static final int gridHeight = 13;
//    private static double detectionThreshold = 0.5;
//    private static final int modelWidth = 200;
//    private static final int modelHeight = 200;
//
//    public static void main(String[] args) throws Exception {
//        if (!cameraPos.equals("front") && !cameraPos.equals("back")){
//            throw new Exception("Unknown argument for camera position. Choose between front and back");
//        }
//
//        FrameGrabber grabber = FrameGrabber.createDefault(cameraNum);
//        OpenCVFrameConverter.ToMat converter = new OpenCVFrameConverter.ToMat();
//
//        grabber.start();
//        String winName = "Object Detection";
//        CanvasFrame canvas = new CanvasFrame(winName);
//        int w = grabber.getImageHeight();
//        int h = grabber.getImageHeight();
//        canvas.setCanvasSize(w, h);
//
//        File locationToLoad = new File(System.getProperty("user.dir"), "generated-models/rps-classifier.zip");
//        //Load Trained Model
//        MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork(locationToLoad);
//        NativeImageLoader loader = new NativeImageLoader(modelHeight,modelWidth, 3);
//        ImagePreProcessingScaler scaler = new ImagePreProcessingScaler(0,1);
//
//        while (true)    {
//            Frame frame = grabber.grab();
//
//            //if thread is null, create new thread
//            if (thread == null) {
//                thread = new Thread(() ->
//                {
//                    while (frame != null)   {
//                        try {
//                            Mat rawImage = new Mat();
//                            //Flip camera if opening front camera
//                            if (cameraPos.equals("front"))  {
//                                Mat inputImage = converter.convert(frame);
//                                flip(inputImage, rawImage, 1);
//                            }   else {
//                                rawImage = converter.convert(frame);
//                            }
//                            Mat resizeImage = new Mat();
//                            resize(rawImage, resizeImage, new Size(modelWidth, modelHeight));
//                            INDArray inputImage = loader.asMatrix(resizeImage);
//                            scaler.transform(inputImage);
//                            INDArray outputs = model.output(inputImage);
//                            List<DetectedObject> objs = YoloUtils.getPredictedObjects(Nd4j.create(((TinyYOLO) model).getPriorBoxes()), outputs, detectionThreshold, 0.4);
//
//                            for (DetectedObject obj : objs) {
//                                double[] xy1 = obj.getTopLeftXY();
//                                double[] xy2 = obj.getBottomRightXY();
//                                String label = Nd4j.argMax(outputs, 1);
//                                int x1 = (int) Math.round(w * xy1[0] / gridWidth);
//                                int y1 = (int) Math.round(h * xy1[1] / gridHeight);
//                                int x2 = (int) Math.round(w * xy2[0] / gridWidth);
//                                int y2 = (int) Math.round(h * xy2[1] / gridHeight);
//                                rectangle(rawImage, new Point(x1, y1), new Point(x2, y2), Scalar.RED, 2, 0, 0);
//                                putText(rawImage, label, new Point(x1 + 2, y2 - 2), FONT_HERSHEY_DUPLEX, 1, Scalar.GREEN);
//                            }
//                            canvas.showImage(converter.convert(rawImage));
//                        } catch (Exception e) {
//                            throw new RuntimeException(e);
//                        }
//                    }
//                });
//                thread.start();
//            }
//            KeyEvent t = canvas.waitKey(33);
//            if ((t != null) && (t.getKeyCode() == KeyEvent.VK_Q)) {
//                break;
//            }
//        }
//        canvas.dispose();
//    }
//}
