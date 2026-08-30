import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ultralytics import YOLO
import cv2
from resnet.classifier import ResNet18InferenceEngine

model = YOLO(r'runs\detect\sih27class\yolo11s_sih_27class\weights\best.pt')
test_imgs = list(Path('SIH_Dataset_27class/test/images').glob('*.png'))[:5]
print(f'Found {len(test_imgs)} test images')

for p in test_imgs:
    img = cv2.imread(str(p))
    res = model.predict(source=img, conf=0.25, verbose=False)[0]
    classes = [model.names[int(b.cls[0])] for b in res.boxes]
    confs = [float(b.conf[0]) for b in res.boxes]
    print(f'{p.name} -> Detections: {classes}, Confs: {[f"{c:.1%}" for c in confs]}')

resnet_eng = ResNet18InferenceEngine(weights_path='weights/resnet18_debris_best.pt', device='auto')
res_crop = resnet_eng.predict_roi(cv2.imread(str(test_imgs[0])), target_class_name='bottle')
print(f'ResNet prediction: {res_crop["pred_class"]} ({res_crop["pred_conf"]:.1%})')
