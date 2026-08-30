
### YOLOv11 (Object Detection)
- **Precision**: 0.8800 (88.00%)
- **Recall**: 0.8860 (88.60%)
- **mAP@50**: 0.9247 (92.47%)
- **mAP@50-95**: 0.8429 (84.29%)

### SegFormer (Edge & Boundary Segmentation)
- **mIoU**: 0.6350 (63.50%)
- **Dice Score**: 0.7687 (76.87%)

### ResNet18 (Feature Verification & Classification)
- **Accuracy**: 0.9987 (99.87%)
- **F1-Score (Weighted)**: 0.9987 (99.87%)
- **F1-Score (Macro)**: 0.9983 (99.83%)

### System (End-to-End Pipeline)
- **End-to-End Success Rate**: 50.51%
- **Preprocessing Latency**: 2.44 ms
- **YOLOv11 Inference Time**: 18.30 ms
- **SegFormer Inference Time**: 5.90 ms
- **ResNet18 + Grad-CAM Time**: 23.61 ms
- **Total Pipeline Inference Time**: 50.64 ms
- **System Throughput (FPS)**: 19.7 FPS (NVIDIA GeForce RTX 4050 Laptop GPU)
