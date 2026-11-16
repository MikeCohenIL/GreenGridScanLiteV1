# GreenGridScanLiteV1

Image classifier for street recycling containers in Israel. One picture in, one of four answers out: **blue** (paper), **orange** (packaging), **purple** (glass) or **other**.

**Status:** work in progress.

## Key figures

| | |
|---|---|
| Test accuracy | **96.7%** (206 of 213 held-out images) |
| Macro-averaged recall | 98.8% |
| Model size | 6.2 MB (ONNX) |
| Inference time | ~3 ms per image on CPU, no GPU (Intel Core Ultra 7 255H, onnxruntime) |
| Architecture | YOLO26-nano classifier (`yolo26n-cls`), 224 × 224 input |
| Runtime dependencies | `onnxruntime`, `numpy`, `Pillow` |
| Containers mapped | 17,522 across 5 municipalities, each with coordinates |
| Image crops collected | 52,535 |
| Hand-labelled images | 1,414 (987 train / 214 validation / 213 test) |

## Test results

213 hand-labelled images, never used in training or model selection.

| Class | Test images | Recall | Precision |
|---|---|---|---|
| blue (paper) | 25 | 100% (25 / 25) | 89.3% (25 / 28) |
| orange (packaging) | 38 | 100% (38 / 38) | 90.5% (38 / 42) |
| purple (glass) | 6 | 100% (6 / 6) | 100% (6 / 6) |
| other | 144 | 95.1% (137 / 144) | 100% (137 / 137) |

## Project overview

**Data.** Container locations and street-level photos come from Mapillary's open map data. For each municipality the pipeline reads the map tiles, keeps the detected waste containers inside the municipal boundary (OpenStreetMap), selects the three clearest photos of each container, and cuts the container out of each photo.

**Labelling.** Labels were assigned by a person in the loop: the model proposes, the reviewer corrects, the model is retrained on the corrections. 1,414 images were confirmed this way.

**Training.** Transfer learning from a pretrained YOLO26-nano classifier. Stratified 70 / 15 / 15 split. Early stopping on validation loss; the published weights are from epoch 7 of 27 (validation loss 0.189, validation accuracy 93.9%).

**Deployment.** Exported to ONNX. Packaged as an AWS Lambda container handler and as a Dockerised web page for uploading a picture and getting the answer.

## Scope

- Test images come from the same municipalities as the training images. Evaluation on an unseen municipality and on phone photos is in progress.
- One answer per image; an image showing several containers of different colors receives a single label.
- "Blue" covers paper recycling containers only.
- Trained on Israeli containers.

## Files

| File | Contents |
|---|---|
| `GreenGridScanLiteV1.onnx` | the model |
| `GreenGridScanLiteV1.json` | class names and input size |

## Usage

```bash
pip install onnxruntime numpy pillow
```

```python
import json
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps

meta = json.load(open("GreenGridScanLiteV1.json"))
session = ort.InferenceSession("GreenGridScanLiteV1.onnx", providers=["CPUExecutionProvider"])


def prepare(path, size):
    """Shorter side -> size, centre crop, values 0..1, channels first."""
    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    w, h = img.size
    nw, nh = (size, int(size * h / w)) if w <= h else (int(size * w / h), size)
    img = img.resize((nw, nh), Image.BILINEAR)
    left, top = round((nw - size) / 2), round((nh - size) / 2)
    x = np.asarray(img.crop((left, top, left + size, top + size)), dtype=np.float32) / 255.0
    return x.transpose(2, 0, 1)[None]


probs = session.run(None, {session.get_inputs()[0].name: prepare("bin.jpg", meta["imgsz"])})[0][0]
print(meta["names"][int(probs.argmax())], float(probs.max()))
```

## Data and license

- Imagery: [Mapillary](https://www.mapillary.com), CC-BY-SA 4.0. Boundaries: OpenStreetMap contributors.
- Built with [Ultralytics](https://github.com/ultralytics/ultralytics) YOLO26 (AGPL-3.0); this model is released under AGPL-3.0. See `LICENSE`.
