import sys
import os
from threading import Thread
from multiprocessing import Process, Queue, Event
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
from multiprocessing import Queue
from utils.image import warp_image
from models.common import DetectMultiBackend
from utils.dataloaders import LoadStreams
from utils.general import (
    check_img_size, non_max_suppression, scale_coords)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync
from utils.ocr import image_to_license_number
from utils.logger import getLogger
from tkinter import Tk, Label
from PIL import Image, ImageTk
from firebase import TempDb
from firebase_admin.db import Event as dbEvent
from deepdiff import DeepDiff
from datetime import datetime
from utils.time import datetime_now, seconds_from_now

# > Initialize project path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # ROOT Directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # Relative path


def inference(
    name: str,  # name
    source: str,  # Path to the source. (Default: Webcam (0))
    queue: Queue,  # Share memory between process and
    stop_event: Event,
):
    # > Get logger and setting the logging level.
    logger = getLogger(f'ALPR.{name}')

    # > Initialze YOLOv5 settings.
    source = str(source)
    # Detection model path.
    weights = ROOT / 'models/tha-license-plate-detection.pt'
    data = ROOT / 'data/coco128.yaml'  # Dataset path.
    imgsz = (640, 640)  # Inference size. (height, width)
    conf_thres = 0.25  # Confidence threshold.
    iou_thres = 0.45  # NMS IOU threshold.
    max_det = 1000  # Maximum detections per image.
    device = ''  # Cuda device, i.e. 0 or 0,1,2,3 or cpu.
    classes = None  # Filter by class: --class 0, or --class 0 2 3.
    agnostic_nms = False  # Class-agnostic NMS.
    augment = False  # Augmented inference.
    line_thickness = 2  # Bounding box thickness (pixels).
    half = False  # Use FP16 half-precisiob inference.
    dnn = False  # Use OpenCV DNN for ONNX inference.

    # GUI settings
    gui = Tk()
    gui.title(f'ALPR: {name.capitalize()} Preview')
    gui.geometry("800x850+20+0" if name != 'exit' else "800x850+850+0")
    gui.minsize("800", "850")
    gui.maxsize("800", "850")
    gui.rowconfigure(0, minsize="450")
    gui.rowconfigure(2, minsize="144")
    gui.rowconfigure(4, minsize="144")

    # Widgets

    # video_feed
    video_feed = Label(gui, text="(source video)")
    video_feed.grid(row=0, column=0, columnspan=3)
    video_feed_label = Label(gui, text="Video Feed")
    video_feed_label.grid(row=1, column=0,  columnspan=3, pady=(5, 5))

    # input_feed
    input_feed = Label(gui, text="(wait for license plate detection)")
    input_feed.grid(row=2, column=0)
    input_feed_label = Label(gui, text="ALPR: Input")
    input_feed_label.grid(row=3, column=0, pady=(5, 5))

    # grayscale feed.
    gray_feed = Label(gui, text="(wait for license plate detection)")
    gray_feed.grid(row=2, column=1)
    gray_feed_label = Label(gui, text="ALPR: Grayscale")
    gray_feed_label.grid(row=3, column=1, pady=(5, 5))

    # threshold feed.
    thres_feed = Label(gui, text="(wait for license plate detection)")
    thres_feed.grid(row=2, column=2)
    thres_feed_label = Label(gui, text="ALPR: Threshold")
    thres_feed_label.grid(row=3, column=2, pady=(5, 5))

    # contour feed.
    contour_feed = Label(gui, text="(wait for license plate detection)")
    contour_feed.grid(row=4, column=0)
    contour_feed_label = Label(gui, text="ALPR: Contour")
    contour_feed_label.grid(row=5, column=0, pady=(5, 5))

    # corner feed.
    corner_feed = Label(gui, text="(wait for license plate detection)")
    corner_feed.grid(row=4, column=1)
    corner_feed_label = Label(gui, text="ALPR: Contour")
    corner_feed_label.grid(row=5, column=1, pady=((5, 5)))

    # warp feed.
    warp_feed = Label(gui, text="(wait for license plate detection)")
    warp_feed.grid(row=4, column=2)
    warp_feed_label = Label(gui, text="ALPR: Warp")
    warp_feed_label.grid(row=5, column=2, pady=((5, 5)))

    # Step 1: Loading model.
    device = select_device(device)
    model = DetectMultiBackend(
        weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz=imgsz, s=stride)

    # Step 2: Loading source.
    cudnn.benchmark = True  # set True to speed up constant image size inference
    dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
    bs = len(dataset)  # batch_size

    # Step 3: Run inference.
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warm up
    seen, dt = 0, [0.0, 0.0, 0.0]
    for path, im, im0s, vid_cap, s in dataset:
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        pred = model(im, augment=augment, visualize=False)
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        pred = non_max_suppression(
            pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t3

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1

            # Step 3.1: Setup predicted image and annotator.
            p, im0, frame = path[i], im0s[i].copy(), dataset.count
            s += '%gx%g ' % im.shape[2:]  # print string
            annotator = Annotator(
                im0, line_width=line_thickness, example=str(names))
            imc = im0.copy()

            iminput, imgray, imthres, imcontour, imcorner, imwarp = None, None, None, None, None, None
            video_feed_label.configure(
                text=f"No license plate detected.", background="red")
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(
                    im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detection per class
                    # add to string
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}"

                # Write results
                # Step 3.2: Crop detected sections in images
                imcs = []
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class
                    label = None
                    annotator.box_label(xyxy, label, color=colors(c, True))
                    imcs.append(save_one_box(
                        xyxy, imc, BGR=True, save=False))

                # Step 3.3: Find biggest crop section.
                imbc = None
                maxArea = 0
                for imc in imcs:
                    area = imc.shape[0] * imc.shape[1]
                    if area > maxArea:
                        imbc = imc

                # Step 3.4: Apply OCR
                iminput, imgray, imthres, imcontour, imcorner, imwarp = warp_image(
                    imbc)
                is_detect, license_number = image_to_license_number(
                    imthres if imthres is not None else iminput)

                # Step 3.5: Update node values.
                if is_detect:
                    queue.put(license_number)
                    video_feed_label.configure(
                        text=f"License plate detected. License number: {license_number}", background="green")
                    s += f' License ID found. ({license_number}) '
                else:
                    video_feed_label.configure(
                        text=f"License plate detected. No license number detected.", background="orange")
                    s += f' License ID not found. '

            # Stream results
            im0 = annotator.result()
            img_im0 = Image.fromarray(cv2.cvtColor(im0, cv2.COLOR_BGR2RGB)).resize(
                (800, 450), Image.ANTIALIAS)
            imgtk_im0 = ImageTk.PhotoImage(img_im0)
            video_feed.configure(image=imgtk_im0)
            if iminput is not None:
                img_iminput = Image.fromarray(iminput).resize(
                    (256, 144), Image.ANTIALIAS)
                imgtk_iminput = ImageTk.PhotoImage(img_iminput)
                input_feed.configure(image=imgtk_iminput)
            if imgray is not None:
                img_imgray = Image.fromarray(imgray).resize(
                    (256, 144), Image.ANTIALIAS)
                imgtk_imgray = ImageTk.PhotoImage(img_imgray)
                gray_feed.configure(image=imgtk_imgray)
            if imthres is not None:
                img_imthres = Image.fromarray(imthres).resize(
                    (256, 144), Image.ANTIALIAS)
                imgtk_imthres = ImageTk.PhotoImage(img_imthres)
                thres_feed.configure(image=imgtk_imthres)
            if imcontour is not None:
                img_imcontour = Image.fromarray(imcontour).resize(
                    (256, 144), Image.ANTIALIAS)
                imgtk_imcontour = ImageTk.PhotoImage(img_imcontour)
                contour_feed.configure(image=imgtk_imcontour)
            if imcorner is not None:
                img_imcorner = Image.fromarray(imcorner).resize(
                    (256, 144), Image.ANTIALIAS)
                imgtk_imcorner = ImageTk.PhotoImage(img_imcorner)
                corner_feed.configure(image=imgtk_imcorner)
            if imwarp is not None:
                img_imwarp = Image.fromarray(imwarp).resize(
                    (256, 144), Image.ANTIALIAS)
                imgtk_imwarp = ImageTk.PhotoImage(img_imwarp)
                warp_feed.configure(image=imgtk_imwarp)

            gui.update_idletasks()
            gui.update()

        # Print time (inference-only)
        # logger.info(f'{s} Done. ({t3 - t2:.3f}s)')

        # Check is stop event is set.
        if stop_event.is_set():
            break

    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    logger.info(
        f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)


class ALPR:
    def __init__(
        self,
        name='node',  # node name.
        source: str = '0',  # source path. (Default: Webcam (0))
    ):
        # > Local variables
        self.name = name
        logger_name = f'ALPR.{name.title()}'
        self._logger = getLogger(logger_name)

        # > YOLOv5 configuration
        self._source = str(source)

        # > ALPR variables
        self.license_numbers = {}

        # > Database
        self._connected_timestamp = datetime.now()
        self._status = {}
        self._command = ''
        self._db_ref = TempDb.reference(f"{self.name}/alpr")
        self._db_ref.child("status").listen(self._db_status_callback)
        self._db_ref.child("command").listen(self._db_command_callback)

        # > Process and thread
        self._queue = Queue()
        self._stop_event = Event()
        self._process = Process(
            target=inference,
            daemon=True,
            args=(self.name, self._source, self._queue, self._stop_event)
        )
        self._thread = Thread(
            target=self._update,
            daemon=True
        )

    # > Database functions
    def _format_status_db(self):
        return {
            "candidate_key": self.candidate_key(),
            "license_numbers": self.keys()
        }

    def _is_db_difference(self):
        return len(DeepDiff(self._format_status_db(), self._status)) != 0

    def _db_status_callback(self, event: dbEvent):
        self._status = event.data

    def _db_command_callback(self, event: dbEvent):
        self._command = event.data if event.data else ''

    # > Thread functions
    def start(self):
        if self._process.is_alive():
            return self._logger.warning("process is already running.")
        self._stop_event.clear()
        self._process.start()
        self._thread.start()

    def stop(self):
        self._stop_event.set()

    # > Thread logic functions
    def _update(self):
        # initialize value in the databse.
        self._db_ref.child("status").set(self._format_status_db())
        new_datetime, new_datetime_string = datetime_now()
        self._connected_timestamp = new_datetime
        self._db_ref.child("connected_timestamp").set(new_datetime_string)
        self._db_ref.child("command").set(self._command)

        while self._process.is_alive():  # while inference process is still running.
            while not self._queue.empty():  # if queue has some data.
                # update license_numbers.
                license_number = self._queue.get()
                old_license_number = self.license_numbers.get(license_number)
                self.license_numbers.update(
                    {license_number: old_license_number + 1 if old_license_number else 1})
                if self._is_db_difference():
                    self._db_ref.child("status").set(self._format_status_db())
            if self._command != '':
                self._command_exec()
            if seconds_from_now(self._connected_timestamp, 5):
                new_datetime, new_datetime_string = datetime_now()
                self._connected_timestamp = new_datetime
                self._db_ref.child("connected_timestamp").set(
                    new_datetime_string)

    def _command_exec(self):
        if self._command != '':
            input = self._command.split(':')
            self._logger.info(input)
            if hasattr(self, f'_c_{input[0]}'):
                if len(input) > 1:
                    getattr(self, f'_c_{input[0]}')(input[1])
                else:
                    getattr(self, f'_c_{input[0]}')()
            self._db_ref.child('command').set('')

    # > ALPR functions.
    def candidate_key(self):
        if len(list(self.license_numbers.keys())) == 0:
            return ""
        max_value = max(list(self.license_numbers.values()))
        for key, value in self.license_numbers.items():
            if value == max_value:
                return key
        return ""

    def keys(self):
        return list(self.license_numbers.keys())

    def clear(self):
        self.license_numbers.clear()

    def is_detect(self):
        return self.candidate_key() != ""

    def is_running(self):
        return self._process.is_alive()

    def _c_clear(self):
        self.clear()


def main():
    try:
        # Create object.
        alpr = ALPR()
        alpr.start()
        while alpr.is_running():
            pass
    except KeyboardInterrupt:
        alpr.stop()


if __name__ == "__main__":
    main()
