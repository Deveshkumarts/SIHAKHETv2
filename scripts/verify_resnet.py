import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from resnet.classifier import ResNet18InferenceEngine
import cv2

engine = ResNet18InferenceEngine(weights_path='weights/resnet18_debris_best.pt', device='auto')

test_classes = ['tire', 'small-tire', 'can', 'wrench', 'valve', 'Shipwrecks', 'pipeline or cable', 'plastic-bottle', 'bottle']
for cname in test_classes:
    matches = list(Path('SIH_Dataset_27class/test/images').glob(f'{cname}_*.*'))
    if matches:
        img = cv2.imread(str(matches[0]))
        res = engine.predict_roi(img, target_class_name=cname)
        top_str = ', '.join([f'{c}: {p:.1%}' for c, p in res['top3']])
        print(f"Target: {cname:<20s} -> Predicted: {res['pred_class']:<20s} ({res['pred_conf']:.1%}) | Top3: {top_str}")
