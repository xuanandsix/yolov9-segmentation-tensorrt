# yolov9-segmentation-tensorrt
This is the tensorrt inference code for yolov9 instance segmentation. 

---

<img src="https://github.com/xuanandsix/yolov9-segmentation-tensorrt/raw/main/show/test.jpg" height="50%" width="50%">
<img src="https://github.com/xuanandsix/yolov9-segmentation-tensorrt/raw/main/show/bus.jpg" height="50%" width="50%">

---

Download [gelan-c-pan.pt](https://github.com/WongKinYiu/yolov9/releases/download/v0.1/gelan-c-pan.pt).

Prepare an onnx model:
```
git clone https://github.com/WongKinYiu/yolov9
pip install -r requirements.txt
python export.py --weights gelan-c-pan.pt --include onnx
```

Test tensorrt

1、Use trtexec tool convert onnx model to trt model. You can also try something else, please make sure to get the correct trt model.
```
/path/to/trtexec --onnx=gelan-c-pan.onnx --saveEngine=gelan-c-pan.engine --fp16
```

2、run python demo_trt.py, get image output. <br>

```
python demo_trt.py --engine gelan-c-pan.engine --imgs imgs --out-dir outputs
```

---
### Acknowledgement

This project is based on the following projects:

[YOLOv9](https://github.com/WongKinYiu/yolov9)

[YOLOv8-TensorRT](https://github.com/triple-Mu/YOLOv8-TensorRT)

[YOLOv9-ONNX-Segmentation](https://github.com/spacewalk01/yolov9-onnx-segmentation)
