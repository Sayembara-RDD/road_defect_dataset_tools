import numpy as np
import cv2
import argparse
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.get_logger().setLevel(3)

# Path to frozen detection graph. This is the actual model that is used for the object detection.
# Refer the model from here https://github.com/sekilab/RoadDamageDetector
PATH_TO_CKPT =  'trained_models/frozen_inference_graph_resnet.pb' 
# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = 'trained_models/crackLabelMap.pbtxt'
NUM_CLASSES = 8


class FrameExtractor():
    def __init__(self):
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.compat.v1.GraphDef()
            with tf.compat.v2.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

    def detect_object(self, frame):
        with self.detection_graph.as_default():
            with tf.compat.v1.Session(graph=self.detection_graph) as sess:
                # Definite input and output Tensors for detection_graph
                image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
                # Each box represents a part of the image where a particular object was detected.
                detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
                # Each score represent how level of confidence for each of the objects.
                # Score is shown on the result image, together with the class label.
                detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
                detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')
                # the array based representation of the image will be used later in order to prepare the
                # result image with boxes and labels on it.
                # image_np = load_image_into_numpy_array(image)
                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                # image_np_expanded = np.expand_dims(image_np, axis=0)
                image_np_expanded = np.expand_dims(frame, axis=0)
                # Actual detection.
                (boxes, scores, classes, num) = sess.run(
                    [detection_boxes, detection_scores, detection_classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})
                return np.squeeze(boxes), np.squeeze(scores)


    def is_object_detected(self, frame, threshold=0.3):
        boxes, scores = self.detect_object(frame)
        if (len(boxes) == 0 or len(scores) == 0):
            return False
        
        detected_obj_size = min(len(boxes), len(scores))
        for i in range(detected_obj_size):
            if scores[i] > threshold:
                return True
        return False


def extractFrames(input_folder_path, input_file_path, output_folder_path, steps, use_model):
    # Extract the name only
    file_name_without_ext = input_file_path.split('.')[0]
    output_video_folder_frames = os.path.join(output_folder_path, file_name_without_ext)
    if not os.path.exists(output_video_folder_frames):
        os.makedirs(output_video_folder_frames)

    targetted_file_name = os.path.join(output_video_folder_frames, file_name_without_ext)
    # Loop over the video and extract frame by frame
    cap = cv2.VideoCapture(os.path.join(input_folder_path, input_file_path))
    if (cap.isOpened()== False): 
        print("Error opening video stream or file")
        return

    frame_extractor = None    
    if use_model:
        frame_extractor = FrameExtractor()

    i = 0

    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            if use_model:
                if (frame_extractor.is_object_detected(frame)):
                    cv2.imwrite(f"{targetted_file_name}-{i}.jpg", frame)
                    i += 1
            else:
                if i%steps == 0:
                    cv2.imwrite(f"{targetted_file_name}-{i}.jpg", frame)
                i+=1
        else:
            break
    cap.release()
    print("Done")

def main(args):
    input_folder_path = args.input_folder_path
    output_folder_path = args.output_folder_path
    video_file_path = args.video_file_path 
    steps = int(args.steps)
    use_model = args.model

    # Create output folder if not exists
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    if not os.path.exists(input_folder_path):
        print("Input folder not exists!")
        return
    
    if (video_file_path != ""):
        extractFrames(input_folder_path, video_file_path, output_folder_path, steps, use_model)
    else:
        for f in os.listdir(input_folder_path):
            file_name = os.path.join(input_folder_path, f)
            if os.path.isfile(file_name):
                extractFrames(input_folder_path, f, output_folder_path, steps, use_model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract frames from collection of videos')
    parser.add_argument('-d', '--input_folder_path'),
    parser.add_argument('-o', '--output_folder_path')
    parser.add_argument('-f', '--video_file_path', default="")
    parser.add_argument('-s', "--steps", help="Value to define the frame to skip when doing frame extraction", default=1)
    parser.add_argument("--model", help="option to use model to detect frames with object of interest", action="store_true")
    args = parser.parse_args()

    main(args)