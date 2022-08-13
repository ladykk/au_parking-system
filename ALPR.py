# ----------------------------------------------------------------------
# [AUTOMATIC LICENSE PLATE RECOGNITION MODULE]
# ----------------------------------------------------------------------
# > Import dependencies
import platform
import sys
import os
from threading import Thread
from multiprocessing import Process, Queue, Event
import logging
from pathlib import Path
from datetime import datetime
import cv2
import torch
import torch.backends.cudnn as cudnn
from utils.logger import getLogger
from multiprocessing import Queue

# > Import YOLOv5 dependencie
from bin.yolov5.models.common import DetectMultiBackend
from bin.yolov5.utils.dataloaders import LoadStreams
from bin.yolov5.utils.general import (check_img_size, check_imshow, colorstr, increment_path, non_max_suppression, scale_coords)
from bin.yolov5.utils.plots import Annotator, colors, save_one_box
from bin.yolov5.utils.torch_utils import  select_device, time_sync

# > Import utility module
from utils.ocr import image_to_license_id

# ----------------------------------------------------------------------
# Initialize Module Variable & Settings
# ----------------------------------------------------------------------
# > Initialize project path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0] # ROOT Directory
if str(ROOT) not in sys.path:
  sys.path.append(str(ROOT)) # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd())) # Relative path

# ----------------------------------------------------------------------
# Detection Process
# ----------------------------------------------------------------------
def detect_process(
    name='node', # Name of the node.
    source=0, # Path to the source. (Default: Webcam (0))
    share=Queue(), # Share memory between process and 
    logger_name="", # Logger of the node.
    debug=False, # If true, logger at debug level.
    show=False, # If true, show current image of object detection and ocr_result.
    event=Event(), # Stop event to detect stop signal from other process.
  ):
# > This function start object detection and try to apply OCR when a license plate is found.

  # > Get logger and setting the logging level.
    logger = getLogger(logger_name, logging.DEBUG if debug else logging.INFO)

  # > Initialze YOLOv5 settings.
    source = str(source)
    weights = ROOT / 'weights/tha-license-plate-detection.pt' # Detection model path.
    data = ROOT / 'bin/yolov5/data/coco128.yaml' # Dataset path.
    imgsz = (640, 640) # Inference size. (height, width)
    conf_thres = 0.25  # Confidence threshold.
    iou_thres = 0.45  # NMS IOU threshold.
    max_det = 1000 # Maximum detections per image.
    device = '' # Cuda device, i.e. 0 or 0,1,2,3 or cpu.
    classes = None  # Filter by class: --class 0, or --class 0 2 3.
    agnostic_nms = False  # Class-agnostic NMS.
    augment = False  # Augmented inference.
    project = ROOT / 'runs/detect' # Save folder.
    line_thickness = 2  # Bounding box thickness (pixels).
    half = False # Use FP16 half-precisiob inference.
    dnn = False # Use OpenCV DNN for ONNX inference.
    
  # Step 1: Loading model.
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz=imgsz, s=stride)

  # Step 2: Loading source.
    view_img = check_imshow() # check is image windows can be open in the environment.
    cudnn.benchmark = True # set True to speed up constant image size inference
    dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
    bs = len(dataset) # batch_size

  # Step 3: Run inference.
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz)) # warm up
    seen, windows, dt = 0, [], [0.0, 0.0, 0.0]
    for path, im, im0s, vid_cap, s in dataset:
      t1 = time_sync()
      im = torch.from_numpy(im).to(device)
      im = im.half() if model.fp16 else im.float() # uint8 to fp16/32
      im /= 255 # 0 - 255 to 0.0 - 1.0
      if len(im.shape) == 3:
        im = im[None] # expand for batch dim
      t2 = time_sync()
      dt[0] += t2 - t1

    # Inference
      pred = model(im, augment=augment, visualize=False)
      t3 = time_sync()
      dt[1] += t3 - t2

    # NMS
      pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
      dt[2] += time_sync() - t3

    # Process predictions
      for i, det in enumerate(pred): # per image
        seen += 1
      # Step 3.1: Setup save path and file_name.
        date = datetime.now().strftime("%Y_%m_%d")
        save_dir = increment_path(Path(project) / f'{date}/{name}/full', mkdir=True)
        crop_dir = increment_path(Path(project) / f'{date}/{name}/crops', mkdir=True)
        file_name = f'{name}-{datetime.now().strftime("%Y_%m_%d-%H_%M_%S")}'
        save_path = str(save_dir / f'{file_name}.jpg') # name-YYYY_mm_dd-HH_MM_SS.jpg
        crop_path = str(crop_dir / f'{file_name}-crop.jpg') # name-YYYY_mm_dd-HH_MM_SS-crop.jpg

      # Step 3.2: Setup predicted image and annotator.
        p, im0, frame = path[i], im0s[i].copy(), dataset.count
        s += '%gx%g ' % im.shape[2:] # print string
        annotator = Annotator(im0, line_width=line_thickness, example=str(names))
        imc = im0.copy()

      # Temp variable for detection
        license_id = None
        is_detect = None

        if len(det):
        # Rescale boxes from img_size to im0 size
          det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

        # Print results
          for c in det[:, -1].unique():
            n = (det[:, -1] == c).sum() # detection per class
            s += f"{n} {names[int(c)]}{'s' * (n > 1)}" # add to string
          
        # Write results
        # Step 3.2: Crop detected sections in images
          imcs = []
          for *xyxy, conf, cls in reversed(det):
            c = int(cls) # integer class
            label = None
            annotator.box_label(xyxy, label, color=colors(c, True))
            imcs.append(save_one_box(xyxy, imc, BGR=True, save=False))

        # Step 3.3: Find biggest crop section.
          imbc = None
          maxArea = 0
          for imc in imcs:
            area = imc.shape[0] * imc.shape[1]
            if area > maxArea:
              imbc = imc

        # Step 3.4: Apply OCR
          is_detect, license_id = image_to_license_id(imbc, debug, show)
          if is_detect:
            s += f' License ID found. ({license_id}) '
          else:
            s += f' License ID not found. '

        # Step 3.5: Update node values.
          share.put({"license_id": license_id, "is_detect": is_detect})
              
      # Stream results
        im0 = annotator.result()
        if view_img and show:
          if platform.system() == 'Linux' and p not in windows:
            windows.append(p)
            cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO) # allow windows resize (Linux)
            cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
          cv2.imshow(str(p), im0)
          cv2.waitKey(1) # 1 millisecond
        
      # Save result
      # Step 3.6: Save images if detected.
      # TODO: To be change for saving on motion sensor.
        if is_detect:
          cv2.imwrite(save_path, im0)
          cv2.imwrite(crop_path, imbc)
      
    # Print time (inference-only)
      logger.debug(f'{s} Done. ({t3 - t2:.3f}s)')

    # Check is stop event is set.
      if event and event.is_set():
        break;
    
  # Print results
    t = tuple(x / seen * 1E3 for x in dt) # speeds per image
    logger.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    logger.info(f"Results saved to {colorstr('bold', save_dir)}")

# ----------------------------------------------------------------------
# Automatic License Plate Recignition Class
# ----------------------------------------------------------------------
class ALPR:
  
# Constructor
  def __init__(
    self,
    name='node', # Node's name.
    source=0, # Source path. (Default: Webcam (0))
    debug=False, # Debug to console.
    show=False, # Show image windows.
  ):
  # Node configs
    self.__name = name
    self.__debug = debug
    self.__show = show

  # YOLOv5 configs
    self.__source = str(source)

  # Initilze logger
    logger_name = f'ALPR.{name.title()}'
    self.__logger = getLogger(logger_name, logging.DEBUG if debug else logging.INFO)
    
  # Node value variables
    self.__license_id = None
    self.__is_detect = False

  # Node thread & process variables
    self.__is_running = False
    self.__share = Queue()
    self.__stop_event = Event()
    self.__process = Process(
      target=detect_process,
      daemon=True,
      args=(self.__name, self.__source, self.__share, logger_name, self.__debug, self.__show, self.__stop_event)
    )
    self.__thread = Thread(
      target=self.__update,
      daemon=True
    )

# Get methods
  def license_id(self): return self.__license_id
  def is_detect(self): return self.__is_detect
  def is_running(self): return self.__is_running

# Thread & process methods
  def __update(self):
  # > This function keeps checking for an update from detection process.
    while self.__process.is_alive(): # Check if the detection process alive?
      while not self.__share.empty(): # Check if there is any update from the process.
        share = self.__share.get() # Get update infos from the share queue.
      # Update info.
        self.__license_id = share["license_id"]
        self.__is_detect = share["is_detect"]

  def start(self):
  # > This function attempts to start recognition process and update thread.
    try:
      self.__logger.info("Attempt to start recongition process.")
    # Process
      self.__logger.debug("Starting the process.")
      self.__process.start()
      if not self.__process.is_alive(): # Check if process is alive after start ?
        self.__logger.error("Cannot start recognition process.")
        raise
      self.__logger.debug("Process started.")
    # Thread
      self.__logger.debug("Strating the thread.")
      self.__thread.start()
      if not self.__thread.is_alive(): # Check if thread is avlice after start?
        self.__logger.error("Cannot start update thread.")
        raise
      self.__logger.debug("Thread started.")
      self.__is_running = True # Set modeule is running.
      self.__logger.info("Recognition process and update thread started.")
    except RuntimeError:
      self.__logger.warning("Recognition process or update thread is already running.")
    except:
      self.__logger.error("Cannot start recognition process and update thread.")
      self.__is_running = False
    
  def stop(self):
  # > This function attempt to stop update thread & recognition process.
    try:
      self.__logger.info("Attempt to stop recognition process.")
    # Process
      self.__logger.debug("Set stop event.")
      self.__stop_event.set()
      self.__logger.debug("Waiting for process to stop.")
      self.__process.join()
      self.__logger.debug("Process stopped.")
    # Thread
      self.__logger.debug("Waiting for thread to stop.")
      self.__thread.join()
      self.__logger.debug("Thread stopped.")
      self.__is_running = False
      self.__logger.info("Recognition thread and process stopped.")
    except:
      self.__logger.error("Cannot stop recognition process and update thread.")
    finally:
      self.__stop_event.clear() # clear stop_event.

def main():
# Create object.
  obj = ALPR(debug=True, show=True)
  obj.start()
  while obj.is_running():
    pass

if __name__ == "__main__":
  main()