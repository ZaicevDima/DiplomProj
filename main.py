from imageai.Detection import VideoObjectDetection
from imageai.Detection import ObjectDetection
import os

execution_path = os.getcwd()

#detector = VideoObjectDetection()
detector = ObjectDetection()

#detector.setModelTypeAsRetinaNet()
detector.setModelTypeAsYOLOv3()
detector.setModelPath(os.path.join(execution_path, "Networks/yolo.h5"))
detector.loadModel()

detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path,
                                                                      "Experiments/OriginalFrames/videoplayback (2)_Moment.jpg"),
                                             output_image_path=os.path.join(execution_path,
                                                                            "out.jpg"),
                                             minimum_percentage_probability=60)
"""video_path = detector.detectObjectsFromVideo(
    input_file_path=os.path.join(execution_path, "CARLA_new.mp4"),
    output_file_path=os.path.join(execution_path, "camera_detected_video"),
    frames_per_second=20,
    log_progress=True)"""

"""detector = VideoObjectDetection()
detector.setModelTypeAsRetinaNet()
#detector.setModelTypeAsYOLOv3()
#detector.setModelPath(os.path.join(execution_path, "yolo.h5"))
detector.setModelPath(os.path.join(execution_path, "resnet50_coco_best_v2.1.0.h5"))
detector.loadModel()

video_path = detector.detectObjectsFromVideo(
    input_file_path=os.path.join(execution_path, "videoplayback (online-video-cutter.com).mp4"),
    output_file_path=os.path.join(execution_path, "RetinaNet"),
    frames_per_second=20,
    log_progress=True)
print(video_path)"""