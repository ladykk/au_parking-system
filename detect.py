import os
import platform
import sys
from pathlib import Path
from datetime import datetime
import cv2

import torch
import torch.backends.cudnn as cudnn

from utils.image import warp_image
from utils.ocr import image_to_license_id

# Initialize Variables
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0] # ROOT Directory
if str(ROOT) not in sys.path:
  sys.path.append(str(ROOT)) # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd())) # Relative path

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr, increment_path, non_max_suppression, scale_coords)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import  select_device, time_sync

# Settings
def run(
  weights=ROOT / 'weights/tha-license-plate-detection.pt', # License Dectection Model
  source=0, # Input Source (Default: Webcam (0))
  data=ROOT / 'data/coco128.yaml', # dataset
  imgsz=(640, 640), # inference size (height, width)
  conf_thres=0.25,  # confidence threshold
  iou_thres=0.45,  # NMS IOU threshold
  max_det=1000, # maximum detections per image
  device='', # cuda device, i.e. 0 or 0,1,2,3 or cpu
  classes=None,  # filter by class: --class 0, or --class 0 2 3
  agnostic_nms=False,  # class-agnostic NMS
  augment=False,  # augmented inference
  visualize=False,  # visualize features
  project=ROOT / 'runs/detect', # save folder
  name=datetime.now().strftime("%Y_%m_%d"), # save folder's name (Default: Current Date in "YYYY_MM_DD")
  line_thickness=3,  # bounding box thickness (pixels)
  half=False, # use FP16 half-precisiob inference
  dnn=False, # use OpenCV DNN for ONNX inference
):
  source = str(source)
  is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
  is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
  webcam = source.isnumeric() or (is_url and not is_file)
  if is_url and is_file:
    source = check_file(source)
  
  # Directories
  save_dir = increment_path(Path(project) / f'{name}/full', mkdir=True)
  crop_dir = increment_path(Path(project) / f'{name}/crops', mkdir=True)

  # Load model
  device = select_device(device)
  model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
  stride, names, pt = model.stride, model.names, model.pt
  imgsz = check_img_size(imgsz, s=stride)

  # Dataloader
  view_img = check_imshow()
  cudnn.benchmark = True # set True to speed up constant image size inference
  dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
  bs = len(dataset) # batch_size

  # Run inference
  model.warmup(imgsz=(1 if pt else bs, 3, *imgsz)) #warmup
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
    visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
    pred = model(im, augment=augment, visualize=visualize)
    t3 = time_sync()
    dt[1] += t3 - t2

    # NMS
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
    dt[2] += time_sync() - t3

    # Second-stage classifier (optional)
    # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

    # Process prefictions
    for i, det in enumerate(pred): # per image
      file_name = datetime.now().strftime("%Y_%m_%d %H_%M_%S")
      seen += 1
      if webcam: # batch_size >= 1
        p, im0, frame = path[i], im0s[i].copy(), dataset.count
        s += f'{i}: '
      else:
        p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
      
      p = Path(p) # to Path
      save_path = str(save_dir / f'{file_name}.jpg') # YYYY_mm_dd HH_MM_SS.jpg
      crop_path = str(crop_dir / f'{file_name}.jpg') # crops/YYYY_mm_dd HH_MM_SS.jpg
      s += '%gx%g ' % im.shape[2:] # print string
      annotator = Annotator(im0, line_width=line_thickness, example=str(names))

      license_id = None
      imc = None

      if len(det):
        # Apply warp to image
        imc = warp_image(im0, debug=True)
        license_id = image_to_license_id(imc)

        if license_id == None:
          s += f'OCR Failed'
          LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')
          continue;          

        # Rescale boxes from img_size to im0 size
        det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

        # Print result
        for c in det[:, -1].unique():
          n = (det[:, -1] == c).sum() # detections per class
          s += f"{n} {names[int(c)]}{'s' * (n > 1)}" # add to string
        
        # Write results
        for *xyxy, conf, cls in reversed(det):
          c = int(cls) # integer class
          label = None
          annotator.box_label(xyxy, label, color=colors(c, True))
          
      
      # Print License ID
      s += f' License ID: {license_id}'

      # Stream results
      im0 = annotator.result()
      if view_img:
        if platform.system() == 'Linux' and p not in windows:
          windows.append(p)
          cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO) # allow windows resize (Linux)
          cv2. resizeWindow(str(p), im0.shape[1], im0.shape[0])
        cv2.imshow(str(p), im0)
        cv2.waitKey(1) # 1 millisecond
      
      # Save results (image with License Plate Detection)
      if license_id != None:
        print(cv2.imwrite(save_path, im0))
        print(cv2.imwrite(crop_path, imc))
      
    # Print time (inference-only)
    LOGGER.info(f'{s} Done. ({t3 - t2:.3f}s)')
  
  # Print results
  t = tuple(x / seen * 1E3 for x in dt) # speeds per image
  LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
  LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}")

def main():
    check_requirements(exclude=('tensorboard', 'thop'))
    run()


if __name__ == "__main__":
    main()