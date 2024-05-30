import math
import argparse
from pathlib import Path
import os
import cv2
import numpy as np
import tensorrt as trt
from cuda import cudart
from numpy import ndarray
import random
import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

os.environ['CUDA_MODULE_LOADING'] = 'LAZY'
warnings.filterwarnings(action='ignore', category=DeprecationWarning)

random.seed(42)
@dataclass
class Tensor:
    name: str
    dtype: np.dtype
    shape: Tuple
    cpu: ndarray
    gpu: int

# detection model classes
CLASSES_DET = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
           'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
           'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
           'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
           'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
           'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
           'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
           'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
           'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
           'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
           'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
           'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
           'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
           'scissors', 'teddy bear', 'hair drier', 'toothbrush')

# colors for per classes
COLORS = {
    cls: [random.randint(0, 255) for _ in range(3)]
    for i, cls in enumerate(CLASSES_DET)
}

class TRTEngine:
    def __init__(self, weight: Union[str, Path]) -> None:
        self.weight = Path(weight) if isinstance(weight, str) else weight
        status, self.stream = cudart.cudaStreamCreate()
        assert status.value == 0
        self.__init_engine()
        self.__init_bindings()
        self.__warm_up()

    def __init_engine(self) -> None:
        logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(logger, namespace='')
        with trt.Runtime(logger) as runtime:
            model = runtime.deserialize_cuda_engine(self.weight.read_bytes())

        context = model.create_execution_context()

        names = [model.get_binding_name(i) for i in range(model.num_bindings)]
        self.num_bindings = model.num_bindings
        self.bindings: List[int] = [0] * self.num_bindings
        num_inputs, num_outputs = 0, 0

        for i in range(model.num_bindings):
            if model.binding_is_input(i):
                num_inputs += 1
            else:
                num_outputs += 1

        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.model = model
        self.context = context
        self.input_names = names[:num_inputs]
        self.output_names = names[num_inputs:]

    def __init_bindings(self) -> None:
        dynamic = False
        inp_info = []
        out_info = []
        out_ptrs = []
        for i, name in enumerate(self.input_names):
            assert self.model.get_binding_name(i) == name
            dtype = trt.nptype(self.model.get_binding_dtype(i))
            shape = tuple(self.model.get_binding_shape(i))
            if -1 in shape:
                dynamic |= True
            if not dynamic:
                cpu = np.empty(shape, dtype)
                status, gpu = cudart.cudaMallocAsync(cpu.nbytes, self.stream)
                assert status.value == 0
                cudart.cudaMemcpyAsync(
                    gpu, cpu.ctypes.data, cpu.nbytes,
                    cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, self.stream)
            else:
                cpu, gpu = np.empty(0), 0
            inp_info.append(Tensor(name, dtype, shape, cpu, gpu))
        for i, name in enumerate(self.output_names):
            i += self.num_inputs
            assert self.model.get_binding_name(i) == name
            dtype = trt.nptype(self.model.get_binding_dtype(i))
            shape = tuple(self.model.get_binding_shape(i))
            if not dynamic:
                cpu = np.empty(shape, dtype=dtype)
                status, gpu = cudart.cudaMallocAsync(cpu.nbytes, self.stream)
                assert status.value == 0
                cudart.cudaMemcpyAsync(
                    gpu, cpu.ctypes.data, cpu.nbytes,
                    cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, self.stream)
                out_ptrs.append(gpu)
            else:
                cpu, gpu = np.empty(0), 0
            out_info.append(Tensor(name, dtype, shape, cpu, gpu))

        self.is_dynamic = dynamic
        self.inp_info = inp_info
        self.out_info = out_info
        self.out_ptrs = out_ptrs

    def __warm_up(self) -> None:
        if self.is_dynamic:
            print('You engine has dynamic axes, please warm up by yourself !')
            return
        for _ in range(10):
            inputs = []
            for i in self.inp_info:
                inputs.append(i.cpu)
            self.run(inputs)

    def set_profiler(self, profiler: Optional[trt.IProfiler]) -> None:
        self.context.profiler = profiler \
            if profiler is not None else trt.Profiler()

    def run(self, *inputs) -> Union[Tuple, ndarray]:

        assert len(inputs) == self.num_inputs
        contiguous_inputs: List[ndarray] = [
            np.ascontiguousarray(i) for i in inputs
        ]

        for i in range(self.num_inputs):

            if self.is_dynamic:
                self.context.set_binding_shape(
                    i, tuple(contiguous_inputs[i].shape))
                status, self.inp_info[i].gpu = cudart.cudaMallocAsync(
                    contiguous_inputs[i].nbytes, self.stream)
                assert status.value == 0
            cudart.cudaMemcpyAsync(
                self.inp_info[i].gpu, contiguous_inputs[i].ctypes.data,
                contiguous_inputs[i].nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, self.stream)
            self.bindings[i] = self.inp_info[i].gpu

        output_gpu_ptrs: List[int] = []
        outputs: List[ndarray] = []

        for i in range(self.num_outputs):
            j = i + self.num_inputs
            if self.is_dynamic:
                shape = tuple(self.context.get_binding_shape(j))
                dtype = self.out_info[i].dtype
                cpu = np.empty(shape, dtype=dtype)
                status, gpu = cudart.cudaMallocAsync(cpu.nbytes, self.stream)
                assert status.value == 0
                cudart.cudaMemcpyAsync(
                    gpu, cpu.ctypes.data, cpu.nbytes,
                    cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, self.stream)
            else:
                cpu = self.out_info[i].cpu
                gpu = self.out_info[i].gpu
            outputs.append(cpu)
            output_gpu_ptrs.append(gpu)
            self.bindings[j] = gpu

        self.context.execute_async_v2(self.bindings, self.stream)
        cudart.cudaStreamSynchronize(self.stream)

        for i, o in enumerate(output_gpu_ptrs):
            cudart.cudaMemcpyAsync(
                outputs[i].ctypes.data, o, outputs[i].nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, self.stream)

        return tuple(outputs) if len(outputs) > 1 else outputs[0]

class Yolov8SegTrt(TRTEngine):
    def __init__(self, engine_path, conf_threshold, iou_threshold):
        super().__init__(engine_path)
        self.num_masks = 32
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.input_height = 640
        self.input_width = 640
        self.mask_alpha = 0.5 
    
    def sigmoid(self, x: ndarray) -> ndarray:
        return 1. / (1. + np.exp(-x))

    def process_box_output(self, box_output):
        predictions = np.squeeze(box_output).T # (8400, 116)
        num_classes = box_output.shape[1] - self.num_masks - 4
        # Filter out object confidence scores below threshold
        scores = np.max(predictions[:, 4:4+num_classes], axis=1)
        predictions = predictions[scores > self.conf_threshold, :]
        scores = scores[scores > self.conf_threshold]
        if len(scores) == 0:
            return [], [], [], np.array([])
        box_predictions = predictions[..., :num_classes+4]
        mask_predictions = predictions[..., num_classes+4:]
        # Get the class with the highest confidence
        class_ids = np.argmax(box_predictions[:, 4:], axis=1)
        # Get bounding boxes for each object
        boxes = self.extract_boxes(box_predictions)
        # Apply non-maxima suppression to suppress weak, overlapping bounding boxes
        indices = self.nms(boxes, scores, self.iou_threshold)

        return boxes[indices], scores[indices], class_ids[indices], mask_predictions[indices]


    def process_mask_output(self, mask_predictions, mask_output):

        if mask_predictions.shape[0] == 0:
            return []

        mask_output = np.squeeze(mask_output)

        # Calculate the mask maps for each box
        num_mask, mask_height, mask_width = mask_output.shape  # CHW
        masks = self.sigmoid(mask_predictions @ mask_output.reshape((num_mask, -1)))
        masks = masks.reshape((-1, mask_height, mask_width))

        # Downscale the boxes to match the mask size
        scale_boxes = self.rescale_boxes(self.boxes,
                                   (self.img_height, self.img_width),
                                   (mask_height, mask_width))

        # For every box/mask pair, get the mask map
        mask_maps = np.zeros((len(scale_boxes), self.img_height, self.img_width))
        blur_size = (int(self.img_width / mask_width), int(self.img_height / mask_height))
        for i in range(len(scale_boxes)):

            scale_x1 = int(math.floor(scale_boxes[i][0]))
            scale_y1 = int(math.floor(scale_boxes[i][1]))
            scale_x2 = int(math.ceil(scale_boxes[i][2]))
            scale_y2 = int(math.ceil(scale_boxes[i][3]))

            x1 = int(math.floor(self.boxes[i][0]))
            y1 = int(math.floor(self.boxes[i][1]))
            x2 = int(math.ceil(self.boxes[i][2]))
            y2 = int(math.ceil(self.boxes[i][3]))

            scale_crop_mask = masks[i][scale_y1:scale_y2, scale_x1:scale_x2]
            crop_mask = cv2.resize(scale_crop_mask,
                              (x2 - x1, y2 - y1),
                              interpolation=cv2.INTER_CUBIC)

            crop_mask = cv2.blur(crop_mask, blur_size)

            crop_mask = (crop_mask > 0.5).astype(np.uint8)
            mask_maps[i, y1:y2, x1:x2] = crop_mask

        return mask_maps
    
    def extract_boxes(self, box_predictions):
        # Extract boxes from predictions
        boxes = box_predictions[:, :4]

        # Scale boxes to original image dimensions
        boxes = self.rescale_boxes(boxes,
                                   (self.input_height, self.input_width),
                                   (self.img_height, self.img_width))

        # Convert boxes to xyxy format
        boxes = self.xywh2xyxy(boxes)

        # Check the boxes are within the image
        boxes[:, 0] = np.clip(boxes[:, 0], 0, self.img_width)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, self.img_height)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, self.img_width)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, self.img_height)

        return boxes

    @staticmethod
    def rescale_boxes(boxes, input_shape, image_shape):
        # Rescale boxes to original image dimensions
        input_shape = np.array([input_shape[1], input_shape[0], input_shape[1], input_shape[0]])
        boxes = np.divide(boxes, input_shape, dtype=np.float32)
        boxes *= np.array([image_shape[1], image_shape[0], image_shape[1], image_shape[0]])

        return boxes
    
    def nms(self, boxes, scores, iou_threshold):
        # Sort by score
        sorted_indices = np.argsort(scores)[::-1]

        keep_boxes = []
        while sorted_indices.size > 0:
            # Pick the last box
            box_id = sorted_indices[0]
            keep_boxes.append(box_id)

            # Compute IoU of the picked box with the rest
            ious = self.compute_iou(boxes[box_id, :], boxes[sorted_indices[1:], :])

            # Remove boxes with IoU over the threshold
            keep_indices = np.where(ious < iou_threshold)[0]

            # print(keep_indices.shape, sorted_indices.shape)
            sorted_indices = sorted_indices[keep_indices + 1]

        return keep_boxes


    def compute_iou(self, box, boxes):
        # Compute xmin, ymin, xmax, ymax for both boxes
        xmin = np.maximum(box[0], boxes[:, 0])
        ymin = np.maximum(box[1], boxes[:, 1])
        xmax = np.minimum(box[2], boxes[:, 2])
        ymax = np.minimum(box[3], boxes[:, 3])
        # Compute intersection area
        intersection_area = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)
        # Compute union area
        box_area = (box[2] - box[0]) * (box[3] - box[1])
        boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        union_area = box_area + boxes_area - intersection_area

        # Compute IoU
        iou = intersection_area / union_area

        return iou

    def xywh2xyxy(self, x):
        # Convert bounding box (x, y, w, h) to bounding box (x1, y1, x2, y2)
        y = np.copy(x)
        y[..., 0] = x[..., 0] - x[..., 2] / 2
        y[..., 1] = x[..., 1] - x[..., 3] / 2
        y[..., 2] = x[..., 0] + x[..., 2] / 2
        y[..., 3] = x[..., 1] + x[..., 3] / 2
        return y    
    
    def prepare_input(self, image):
        self.img_height, self.img_width = image.shape[:2]
        input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Resize input image
        input_img = cv2.resize(input_img, (self.input_width, self.input_height))
        # Scale input pixel values to 0 to 1
        input_img = input_img / 255.0
        input_img = input_img.transpose(2, 0, 1)
        input_tensor = input_img[np.newaxis, :, :, :].astype(np.float32)
        return input_tensor

    def draw_masks(self, image, boxes, class_ids, mask_alpha=0.3, mask_maps=None):
        mask_img = image.copy()

        # Draw bounding boxes and labels of detections
        for i, (box, class_id) in enumerate(zip(boxes, class_ids)):
            color = COLORS[CLASSES_DET[class_id]]

            x1, y1, x2, y2 = box.astype(int)

            # Draw fill mask image
            if mask_maps is None:
                cv2.rectangle(mask_img, (x1, y1), (x2, y2), color, -1)
            else:
                crop_mask = mask_maps[i][y1:y2, x1:x2, np.newaxis]
                crop_mask_img = mask_img[y1:y2, x1:x2]
                crop_mask_img = crop_mask_img * (1 - crop_mask) + crop_mask * color
                mask_img[y1:y2, x1:x2] = crop_mask_img

        return cv2.addWeighted(mask_img, mask_alpha, image, 1 - mask_alpha, 0)


    def draw(self, image, boxes, scores, class_ids, mask_alpha=0.3, mask_maps=None):
        img_height, img_width = image.shape[:2]
        size = min([img_height, img_width]) * 0.0006
        text_thickness = int(min([img_height, img_width]) * 0.001)
        mask_img = self.draw_masks(image, boxes, class_ids, mask_alpha, mask_maps)
        # Draw bounding boxes and labels of detections
        for box, score, class_id in zip(boxes, scores, class_ids):
            color = COLORS[CLASSES_DET[class_id]]
            x1, y1, x2, y2 = box.astype(int)
            # Draw rectangle
            cv2.rectangle(mask_img, (x1, y1), (x2, y2), color, 2)
            label = CLASSES_DET[class_id]
            caption = f'{label} {int(score * 100)}%'
            (tw, th), _ = cv2.getTextSize(text=caption, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                        fontScale=size, thickness=text_thickness)
            th = int(th * 1.2)
            cv2.rectangle(mask_img, (x1, y1),
                        (x1 + tw, y1 - th), color, -1)
            cv2.putText(mask_img, caption, (x1, y1),
                        cv2.FONT_HERSHEY_SIMPLEX, size, (255, 255, 255), text_thickness, cv2.LINE_AA)
        return mask_img
        
        
    def forward(self, image):
        input_tensor = self.prepare_input(image)
        outputs = self.run(input_tensor)
        self.boxes, self.scores, self.class_ids, mask_pred = self.process_box_output(outputs[2])
        self.mask_maps = self.process_mask_output(mask_pred, outputs[0])
        
        combined_img = self.draw(image, self.boxes, self.scores,
                               self.class_ids, self.mask_alpha, mask_maps=self.mask_maps)
        
        return combined_img

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', type=str, help='Engine file')
    parser.add_argument('--imgs', type=str, help='Images file')
    parser.add_argument('--out-dir',
                        type=str,
                        default='./output',
                        help='Path to output file')
    parser.add_argument('--conf_threshold',
                        type=float,
                        default=0.25,
                        help='Confidence threshold')
    parser.add_argument('--iou_threshold',
                        type=float,
                        default=0.65,
                        help='Confidence threshold')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    segment = Yolov8SegTrt(args.engine, args.conf_threshold, args.iou_threshold)
    image_files = os.listdir(args.imgs)
    save_path = args.out_dir
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    for file in image_files:
        bgr = cv2.imread(args.imgs + '/' + str(file))
        # inference
        draw = segment.forward(bgr)
        print(draw.shape)
        cv2.imwrite(save_path + '/' + str(file), draw)
