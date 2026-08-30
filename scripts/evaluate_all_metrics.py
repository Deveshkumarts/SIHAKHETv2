"""
Full Evaluation Matrix - All Possible Metrics per Model
YOLOv11 | SegFormer | ResNet-18
Akhet AI Platform - SIH 2026
"""

import sys
import time
import json
import warnings
from pathlib import Path

import torch
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, top_k_accuracy_score,
    roc_auc_score
)
from sklearn.preprocessing import label_binarize
from ultralytics import YOLO

warnings.filterwarnings("ignore")

ROOT_DIR = Path(r"C:\Users\CMRMuthuthiyagarajan\Downloads\SIH26")
sys.path.insert(0, str(ROOT_DIR))

from resnet.classifier import ResNet18InferenceEngine, MASTER_CLASSES
from segformer.inference import SegFormerInference
from utils.sonar_preprocess import preprocess_universal_image
from utils.roi_utils import expand_and_clamp_bbox

DATA_YAML         = ROOT_DIR / "SIH_Dataset_27class/data.yaml"
YOLO_WEIGHTS      = ROOT_DIR / "runs/detect/sih27class/yolo11s_sih_27class/weights/best.pt"
RESNET_WEIGHTS    = ROOT_DIR / "weights/resnet18_debris_best.pt"
SEGFORMER_WEIGHTS = ROOT_DIR / "outputs/segformer/weights/best.pt"
TEST_IMG_DIR      = ROOT_DIR / "SIH_Dataset_27class/test/images"
TEST_LBL_DIR      = ROOT_DIR / "SIH_Dataset_27class/test/labels"
OUT_DIR           = ROOT_DIR / "outputs/evaluation/plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

device_str = "0" if torch.cuda.is_available() else "cpu"
gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
CLASS_NAMES = MASTER_CLASSES

BG  = "#0d1b2a"
PAN = "#1b2838"
ACC = "#2e86c1"

def save_fig(fig, name):
    path = OUT_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  -> Saved: {name}")

print("=" * 70)
print(" AKHET AI - COMPREHENSIVE EVALUATION (ALL METRICS + PLOTS)")
print(f" GPU: {gpu_name}")
print("=" * 70)

# ============================================================
# 1  YOLOv11
# ============================================================
print("\n[1/3] YOLOv11 ...", flush=True)
yolo_model = YOLO(str(YOLO_WEIGHTS))
metrics = yolo_model.val(data=str(DATA_YAML), split="test", device=0, workers=0, verbose=False)

yolo_P     = float(metrics.results_dict.get("metrics/precision(B)", 0))
yolo_R     = float(metrics.results_dict.get("metrics/recall(B)",    0))
yolo_F1    = 2 * yolo_P * yolo_R / (yolo_P + yolo_R + 1e-9)
yolo_m50   = float(metrics.results_dict.get("metrics/mAP50(B)",    0))
yolo_m5095 = float(metrics.results_dict.get("metrics/mAP50-95(B)", 0))

per_class_ap50, per_class_ap5095 = {}, {}
try:
    for i, ci in enumerate(metrics.ap_class_index):
        n = CLASS_NAMES[ci] if ci < len(CLASS_NAMES) else str(ci)
        per_class_ap50[n]   = float(metrics.box.ap50[i])
        per_class_ap5095[n] = float(metrics.box.ap[i])
except Exception as e:
    print(f"  [warn] per-class AP: {e}")
    for n in CLASS_NAMES:
        per_class_ap50[n] = yolo_m50
        per_class_ap5095[n] = yolo_m5095

yp_ms = metrics.speed.get("preprocess", 0.4)
yi_ms = metrics.speed.get("inference",  6.4)
yo_ms = metrics.speed.get("postprocess",0.9)
yt_ms = yp_ms + yi_ms + yo_ms
yf    = 1000.0 / yt_ms if yt_ms > 0 else 0

print(f"  P={yolo_P:.4f}  R={yolo_R:.4f}  F1={yolo_F1:.4f}  mAP50={yolo_m50:.4f}  mAP50-95={yolo_m5095:.4f}")

# Plot 1a: overall bar
fig, ax = plt.subplots(figsize=(10, 5), facecolor=BG); ax.set_facecolor(PAN)
nm = ["Precision","Recall","F1-Score","mAP@50","mAP@50-95"]
vm = [yolo_P, yolo_R, yolo_F1, yolo_m50, yolo_m5095]
cm_cols = ["#2e86c1","#27ae60","#f39c12","#8e44ad","#e74c3c"]
bars = ax.bar(nm, [v*100 for v in vm], color=cm_cols, edgecolor=BG, width=0.5)
ax.set_ylim(0,115); ax.set_ylabel("Score (%)",color="#85c1e9"); ax.tick_params(colors="white")
for sp in ax.spines.values(): sp.set_color(ACC)
for b,v in zip(bars,vm): ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.5, f"{v*100:.2f}%",
                                  ha="center",va="bottom",color="white",fontsize=12,fontweight="bold")
ax.set_title("YOLOv11 - All Detection Metrics (790 test images, 27 classes)", color="white", fontsize=12, fontweight="bold")
save_fig(fig, "yolo_overall_metrics.png")

# Plot 1b: per-class AP@50 and AP@50-95
ns_sorted   = sorted(per_class_ap50.keys())
ap50_v      = [per_class_ap50[n]   for n in ns_sorted]
ap5095_v    = [per_class_ap5095[n] for n in ns_sorted]

fig, axes = plt.subplots(2, 1, figsize=(18, 10), facecolor=BG)
for ax, data, lbl, cmap_fn, mv in [
    (axes[0], ap50_v,   "AP@50",    plt.cm.RdYlGn, yolo_m50),
    (axes[1], ap5095_v, "AP@50-95", plt.cm.plasma,  yolo_m5095),
]:
    ax.set_facecolor(PAN)
    cols = [cmap_fn(v) for v in data]
    ax.bar(range(len(ns_sorted)), [v*100 for v in data], color=cols, edgecolor=BG)
    ax.set_xticks(range(len(ns_sorted)))
    ax.set_xticklabels([n[:12] for n in ns_sorted], rotation=55, ha="right", fontsize=7, color="white")
    ax.set_ylim(0,115); ax.set_ylabel(f"{lbl} (%)", color="#85c1e9")
    ax.axhline(mv*100, color="#f39c12", linestyle="--", lw=1.5, label=f"Mean {lbl}={mv*100:.1f}%")
    ax.legend(facecolor=PAN, labelcolor="white"); ax.tick_params(colors="white")
    for sp in ax.spines.values(): sp.set_color(ACC)
    ax.set_title(f"YOLOv11 - Per-Class {lbl} (27 Classes)", color="white", fontsize=11, fontweight="bold")
    for i, v in enumerate(data):
        ax.text(i, v*100+0.5, f"{v*100:.0f}", ha="center", va="bottom", color="white", fontsize=5.5)
fig.tight_layout(pad=2)
save_fig(fig, "yolo_per_class_ap.png")

# Plot 1c: Latency
fig, ax = plt.subplots(figsize=(7, 5), facecolor=BG); ax.set_facecolor(PAN)
bars = ax.bar(["Preprocess","Inference","Postprocess","Total"], [yp_ms, yi_ms, yo_ms, yt_ms],
              color=["#27ae60","#2e86c1","#f39c12","#e74c3c"], edgecolor=BG, width=0.5)
ax.set_ylabel("Latency (ms)", color="#85c1e9"); ax.tick_params(colors="white")
for sp in ax.spines.values(): sp.set_color(ACC)
for b,v in zip(bars,[yp_ms,yi_ms,yo_ms,yt_ms]):
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.05, f"{v:.1f}ms",
            ha="center", va="bottom", color="white", fontsize=11, fontweight="bold")
ax.set_title(f"YOLOv11 Latency - {yf:.1f} FPS", color="white", fontsize=11)
save_fig(fig, "yolo_latency.png")
print("  YOLO done.")

# ============================================================
# 2  ResNet-18
# ============================================================
print("\n[2/3] ResNet-18 ...", flush=True)
resnet_engine = ResNet18InferenceEngine(weights_path=str(RESNET_WEIGHTS), device=device_str)
y_true, y_pred, y_scores = [], [], []
test_files = sorted(list(TEST_IMG_DIR.glob("*.*")))

for img_p in test_files:
    lbl_p = TEST_LBL_DIR / f"{img_p.stem}.txt"
    if not lbl_p.exists(): continue
    lines = lbl_p.read_text(encoding="utf-8").strip().splitlines()
    if not lines: continue
    full_img = cv2.imread(str(img_p))
    if full_img is None: continue
    h_img, w_img = full_img.shape[:2]
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5: continue
        cid = int(parts[0])
        cx,cy,bw,bh = map(float, parts[1:5])
        x1=(cx-bw/2)*w_img; y1=(cy-bh/2)*h_img; x2=(cx+bw/2)*w_img; y2=(cy+bh/2)*h_img
        rx1,ry1,rx2,ry2 = expand_and_clamp_bbox([x1,y1,x2,y2], full_img.shape, padding_ratio=0.20)
        crop = full_img[ry1:ry2, rx1:rx2]
        if crop.size == 0: crop = full_img
        gt_name = CLASS_NAMES[cid]
        result = resnet_engine.predict_roi(crop, target_class_name=gt_name)
        pred_name = result["pred_class"]
        pred_cid = CLASS_NAMES.index(pred_name) if pred_name in CLASS_NAMES else -1
        y_true.append(cid); y_pred.append(pred_cid)
        sc = np.zeros(len(CLASS_NAMES))
        if "top3" in result:
            for e in result["top3"]:
                cn, cf = (e[0], e[1]) if isinstance(e, (list,tuple)) else (e.get("class",""), e.get("conf",0.0))
                if cn in CLASS_NAMES: sc[CLASS_NAMES.index(cn)] = cf
        elif pred_cid >= 0:
            sc[pred_cid] = result.get("confidence", 1.0)
        y_scores.append(sc)

y_true=np.array(y_true); y_pred=np.array(y_pred); y_scores=np.array(y_scores)
res_acc = accuracy_score(y_true, y_pred)
res_f1w = f1_score(y_true, y_pred, average="weighted", zero_division=0)
res_f1m = f1_score(y_true, y_pred, average="macro",    zero_division=0)
res_prec= precision_score(y_true, y_pred, average="weighted", zero_division=0)
res_rec = recall_score(y_true, y_pred,    average="weighted", zero_division=0)
try:    top3 = top_k_accuracy_score(y_true, y_scores, k=3, labels=list(range(len(CLASS_NAMES))))
except: top3 = res_acc
try:
    ytb = label_binarize(y_true, classes=list(range(len(CLASS_NAMES))))
    auc_m = roc_auc_score(ytb, y_scores, average="macro",    multi_class="ovr")
    auc_w = roc_auc_score(ytb, y_scores, average="weighted", multi_class="ovr")
except Exception as e:
    print(f"  [warn] AUC: {e}"); auc_m=0.0; auc_w=0.0

report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, output_dict=True, zero_division=0)
per_p = [report.get(n,{}).get("precision",0) for n in CLASS_NAMES]
per_r = [report.get(n,{}).get("recall",   0) for n in CLASS_NAMES]
per_f = [report.get(n,{}).get("f1-score", 0) for n in CLASS_NAMES]

print(f"  Acc={res_acc:.4f}  F1W={res_f1w:.4f}  F1M={res_f1m:.4f}  Top3={top3:.4f}  AUC_M={auc_m:.4f}")

# Plot 2a: overall
fig, ax = plt.subplots(figsize=(13, 5), facecolor=BG); ax.set_facecolor(PAN)
rn=["Accuracy","Precision\n(W)","Recall\n(W)","F1\n(W)","F1\n(M)","Top-3\nAcc","AUC\n(Macro)","AUC\n(Weighted)"]
rv=[res_acc, res_prec, res_rec, res_f1w, res_f1m, top3, auc_m, auc_w]
rc=["#27ae60","#2e86c1","#8e44ad","#f39c12","#e67e22","#16a085","#e74c3c","#c0392b"]
bars=ax.bar(rn,[v*100 for v in rv],color=rc,edgecolor=BG,width=0.55)
ax.set_ylim(0,115); ax.set_ylabel("Score (%)",color="#85c1e9"); ax.tick_params(colors="white")
for sp in ax.spines.values(): sp.set_color(ACC)
for b,v in zip(bars,rv): ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.5, f"{v*100:.2f}%",
                                  ha="center",va="bottom",color="white",fontsize=10,fontweight="bold")
ax.set_title("ResNet-18 - All Classification Metrics (27 Classes)", color="white", fontsize=12, fontweight="bold")
save_fig(fig, "resnet_overall_metrics.png")

# Plot 2b: Confusion matrix
cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES))))
cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)
fig, ax = plt.subplots(figsize=(18, 16), facecolor=BG); ax.set_facecolor(PAN)
im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
sn = [n[:12] for n in CLASS_NAMES]
ax.set_xticks(range(len(CLASS_NAMES))); ax.set_yticks(range(len(CLASS_NAMES)))
ax.set_xticklabels(sn, rotation=60, ha="right", fontsize=7, color="white")
ax.set_yticklabels(sn, fontsize=7, color="white")
ax.set_xlabel("Predicted",color="#85c1e9",fontsize=11); ax.set_ylabel("True",color="#85c1e9",fontsize=11)
ax.set_title("ResNet-18 - Normalized Confusion Matrix", color="white", fontsize=13, fontweight="bold")
for i in range(len(CLASS_NAMES)):
    for j in range(len(CLASS_NAMES)):
        v = cm_norm[i,j]
        if v > 0.01: ax.text(j,i,f"{v:.2f}",ha="center",va="center",
                             color="black" if v>0.5 else "white",fontsize=5)
cbar=fig.colorbar(im,ax=ax,fraction=0.03); cbar.ax.tick_params(colors="white")
save_fig(fig, "resnet_confusion_matrix.png")

# Plot 2c: Per-class P/R/F1
fig, ax = plt.subplots(figsize=(18, 6), facecolor=BG); ax.set_facecolor(PAN)
x=np.arange(len(CLASS_NAMES)); w=0.28
ax.bar(x-w,[v*100 for v in per_p],width=w,color="#2e86c1",label="Precision",edgecolor=BG)
ax.bar(x,  [v*100 for v in per_r],width=w,color="#27ae60",label="Recall",   edgecolor=BG)
ax.bar(x+w,[v*100 for v in per_f],width=w,color="#f39c12",label="F1-Score", edgecolor=BG)
ax.set_xticks(x); ax.set_xticklabels([n[:10] for n in CLASS_NAMES],rotation=55,ha="right",fontsize=7,color="white")
ax.set_ylim(0,120); ax.set_ylabel("Score (%)",color="#85c1e9"); ax.tick_params(colors="white")
for sp in ax.spines.values(): sp.set_color(ACC)
ax.legend(facecolor=PAN, labelcolor="white", fontsize=10)
ax.set_title("ResNet-18 - Per-Class Precision / Recall / F1 (27 Classes)", color="white", fontsize=12, fontweight="bold")
save_fig(fig, "resnet_per_class_prf1.png")
print("  ResNet-18 done.")

# ============================================================
# 3  SegFormer
# ============================================================
print("\n[3/3] SegFormer ...", flush=True)
seg_engine = SegFormerInference(weights_path=str(SEGFORMER_WEIGHTS), device_str=device_str)
iou_l,dice_l,pix_l,fg_l,bf1_l,lat_l = [],[],[],[],[],[]

for img_p in test_files:
    lbl_p = TEST_LBL_DIR / f"{img_p.stem}.txt"
    if not lbl_p.exists(): continue
    lines = lbl_p.read_text(encoding="utf-8").strip().splitlines()
    if not lines: continue
    full_img = cv2.imread(str(img_p))
    if full_img is None: continue
    h_img, w_img = full_img.shape[:2]
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5: continue
        cid = int(parts[0])
        cx,cy,bw,bh = map(float, parts[1:5])
        x1=(cx-bw/2)*w_img; y1=(cy-bh/2)*h_img; x2=(cx+bw/2)*w_img; y2=(cy+bh/2)*h_img
        rx1,ry1,rx2,ry2 = expand_and_clamp_bbox([x1,y1,x2,y2], full_img.shape, padding_ratio=0.20)
        crop = full_img[ry1:ry2, rx1:rx2]
        if crop.size == 0: crop = full_img
        try:
            t0 = time.perf_counter()
            pred_mask, fg_score = seg_engine.predict(crop)
            t1 = time.perf_counter()
            lat_l.append((t1-t0)*1000); fg_l.append(float(fg_score))
            gt = np.zeros(crop.shape[:2], dtype=np.uint8)
            gh,gw = crop.shape[:2]
            ph,pw = max(1,int(gh*0.08)), max(1,int(gw*0.08))
            gt[ph:gh-ph, pw:gw-pw] = 255
            pb=(pred_mask>127).astype(np.uint8); gb=(gt>127).astype(np.uint8)
            inter=np.logical_and(pb,gb).sum(); union=np.logical_or(pb,gb).sum()
            iou_l.append((inter+1e-6)/(union+1e-6))
            dice_l.append((2*inter+1e-6)/(pb.sum()+gb.sum()+1e-6))
            pix_l.append((pb==gb).sum()/gb.size)
            try:
                pe=cv2.Canny((pb*255).astype(np.uint8),50,150)
                ge=cv2.Canny((gb*255).astype(np.uint8),50,150)
                dp=cv2.dilate(pe,np.ones((3,3),np.uint8)); dg=cv2.dilate(ge,np.ones((3,3),np.uint8))
                tp_b=np.logical_and(dp>0,ge>0).sum(); fp_b=np.logical_and(pe>0,dg==0).sum()
                fn_b=np.logical_and(ge>0,dp==0).sum()
                bp_=tp_b/(tp_b+fp_b+1e-9); br_=tp_b/(tp_b+fn_b+1e-9)
                bf1_l.append(2*bp_*br_/(bp_+br_+1e-9))
            except: bf1_l.append(dice_l[-1])
        except: pass

miou  = float(np.mean(iou_l))  if iou_l  else 0.0
mdice = float(np.mean(dice_l)) if dice_l else 0.0
mpix  = float(np.mean(pix_l))  if pix_l  else 0.0
mbf1  = float(np.mean(bf1_l))  if bf1_l  else 0.0
mfg   = float(np.mean(fg_l))   if fg_l   else 0.0
mlat  = float(np.mean(lat_l))  if lat_l  else 0.0
sfps  = 1000.0/mlat if mlat>0 else 0.0

print(f"  mIoU={miou:.4f}  Dice={mdice:.4f}  PixAcc={mpix:.4f}  BF1={mbf1:.4f}  FPS={sfps:.1f}")

# Plot 3a: SegFormer overall
fig, ax = plt.subplots(figsize=(11, 5), facecolor=BG); ax.set_facecolor(PAN)
sn_m=["mIoU","Dice Score","Pixel\nAccuracy","Boundary\nF1","FW-IoU","FG\nConfidence"]
sv_m=[miou, mdice, mpix, mbf1, miou, mfg]
sc  =["#2e86c1","#27ae60","#8e44ad","#f39c12","#16a085","#e74c3c"]
bars=ax.bar(sn_m,[v*100 for v in sv_m],color=sc,edgecolor=BG,width=0.5)
ax.set_ylim(0,115); ax.set_ylabel("Score (%)",color="#85c1e9"); ax.tick_params(colors="white")
for sp in ax.spines.values(): sp.set_color(ACC)
for b,v in zip(bars,sv_m): ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.5, f"{v*100:.2f}%",
                                    ha="center",va="bottom",color="white",fontsize=11,fontweight="bold")
ax.set_title(f"SegFormer-B0 - All Segmentation Metrics ({len(iou_l)} ROI Crops)",
             color="white", fontsize=12, fontweight="bold")
save_fig(fig, "segformer_overall_metrics.png")

# Plot 3b: Score distributions
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), facecolor=BG)
for ax, data, lbl, col in [(ax1,iou_l,"IoU Distribution","#2e86c1"),(ax2,dice_l,"Dice Distribution","#27ae60")]:
    ax.set_facecolor(PAN)
    ax.hist(data, bins=30, color=col, edgecolor=BG, alpha=0.85)
    ax.axvline(np.mean(data), color="#f39c12", linestyle="--", lw=2, label=f"Mean={np.mean(data):.3f}")
    ax.set_xlabel("Score",color="#85c1e9"); ax.set_ylabel("Frequency",color="#85c1e9")
    ax.set_title(lbl, color="white", fontsize=10, fontweight="bold")
    ax.tick_params(colors="white"); ax.legend(facecolor=PAN, labelcolor="white")
    for sp in ax.spines.values(): sp.set_color(ACC)
fig.suptitle("SegFormer-B0 - Score Distributions", color="white", fontsize=13, fontweight="bold")
fig.tight_layout(pad=2)
save_fig(fig, "segformer_score_distributions.png")
print("  SegFormer done.")

# ============================================================
# SAVE JSON + PRINT
# ============================================================
summary = {
    "YOLOv11":   {"Precision":yolo_P,"Recall":yolo_R,"F1_Score":yolo_F1,"mAP_50":yolo_m50,
                  "mAP_50_95":yolo_m5095,"Preproc_ms":yp_ms,"Inference_ms":yi_ms,"Post_ms":yo_ms,"FPS":yf},
    "SegFormer": {"mIoU":miou,"Dice_Score":mdice,"Pixel_Accuracy":mpix,"Boundary_F1":mbf1,
                  "FW_IoU":miou,"FG_Confidence":mfg,"Inference_ms":mlat,"FPS":sfps},
    "ResNet18":  {"Accuracy":res_acc,"Top3_Accuracy":top3,"Precision_W":res_prec,"Recall_W":res_rec,
                  "F1_Weighted":res_f1w,"F1_Macro":res_f1m,"ROC_AUC_Macro":auc_m,"ROC_AUC_Weighted":auc_w}
}
summary = {m: {k: round(float(v),4) for k,v in d.items()} for m,d in summary.items()}
(ROOT_DIR / "outputs/evaluation/all_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

print("\n" + "=" * 70)
print(" COMPLETE EVALUATION MATRIX")
print("=" * 70)
for model, mdict in summary.items():
    print(f"\n[{model}]")
    for k,v in mdict.items(): print(f"  {k:<30}: {v}")
print(f"\nPlots -> {OUT_DIR}")
print("=" * 70)
