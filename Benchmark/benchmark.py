"""
benchmark.py — run all parameter combinations and write a comparison report.

Combinations tested (2 × 2 × 2 = 8):
  DET_THRESH  : 0.30  0.20
  NMS_IOU     : 0.40  0.30
  TILE_SIZE   : 640   480
"""

import os
import glob
import time
import cv2
import numpy as np
from datetime import datetime
from itertools import product
from insightface.app import FaceAnalysis

REPORT_FILE = "benchmark_report.txt"
BASE_OUTPUT = "benchmark_runs"

DET_THRESHOLDS  = [0.30, 0.20]
NMS_IOU_VALUES  = [0.40, 0.30]
TILE_SIZES      = [640,  480]
OVERLAP         = 0.25


# ── helpers ──────────────────────────────────────────────────────────────────

def find_jpg_files(directory: str) -> list[str]:
    files = glob.glob(os.path.join(directory, "*.jpg")) + \
            glob.glob(os.path.join(directory, "*.JPG"))
    seen, unique = set(), []
    for f in sorted(files):
        key = os.path.normcase(f)
        if key not in seen:
            seen.add(key)
            unique.append(f)
    return unique


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float) -> list[int]:
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep  = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou   = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[np.where(iou <= iou_thresh)[0] + 1]
    return keep


def detect_faces_tiled(app, image: np.ndarray,
                        tile_size: int, nms_iou: float, det_thresh: float):
    h, w  = image.shape[:2]
    step  = int(tile_size * (1 - OVERLAP))
    all_boxes, all_scores = [], []

    for y in range(0, h, step):
        for x in range(0, w, step):
            x1 = min(x, w - tile_size)
            y1 = min(y, h - tile_size)
            x2, y2 = x1 + tile_size, y1 + tile_size
            tile  = image[y1:y2, x1:x2]
            faces = app.get(tile)
            for face in faces:
                if face.det_score < det_thresh:
                    continue
                bx1, by1, bx2, by2 = face.bbox
                all_boxes.append([bx1 + x1, by1 + y1, bx2 + x1, by2 + y1])
                all_scores.append(float(face.det_score))

    if not all_boxes:
        return [], []
    boxes  = np.array(all_boxes,  dtype=np.float32)
    scores = np.array(all_scores, dtype=np.float32)
    keep   = nms(boxes, scores, nms_iou)
    return boxes[keep].tolist(), scores[keep].tolist()


def run_combination(app, jpg_files: list[str], script_dir: str,
                    det_thresh: float, nms_iou: float, tile_size: int,
                    run_id: int, total_runs: int) -> dict:
    tag = f"det{int(det_thresh*100)}_nms{int(nms_iou*100)}_tile{tile_size}"
    out_dir = os.path.join(script_dir, BASE_OUTPUT, tag)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n[{run_id}/{total_runs}] det={det_thresh}  nms_iou={nms_iou}  tile={tile_size}px")

    # Re-prepare model with correct tile/det settings
    app.det_model.det_thresh = det_thresh
    app.prepare(ctx_id=-1, det_size=(tile_size, tile_size), det_thresh=det_thresh)

    file_counts = {}
    t0 = time.time()

    for filepath in jpg_files:
        filename = os.path.basename(filepath)
        image = cv2.imread(filepath)
        if image is None:
            continue

        boxes, scores = detect_faces_tiled(app, image, tile_size, nms_iou, det_thresh)
        count = len(boxes)
        file_counts[filename] = count
        print(f"  {filename}: {count} faces")

        annotated = image.copy()
        for (x1, y1, x2, y2), score in zip(boxes, scores):
            cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)
            cv2.putText(annotated, f"{score:.2f}", (int(x1), max(int(y1) - 8, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        cv2.imwrite(os.path.join(out_dir, filename), annotated)

    elapsed = time.time() - t0
    total   = sum(file_counts.values())
    print(f"  Total faces: {total}  |  Time: {elapsed:.1f}s")

    return {
        "tag":        tag,
        "det_thresh": det_thresh,
        "nms_iou":    nms_iou,
        "tile_size":  tile_size,
        "total":      total,
        "per_file":   file_counts,
        "elapsed_s":  elapsed,
        "out_dir":    out_dir,
    }


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    jpg_files  = find_jpg_files(script_dir)
    if not jpg_files:
        print("No .jpg files found.")
        return

    combinations = list(product(DET_THRESHOLDS, NMS_IOU_VALUES, TILE_SIZES))
    total_runs   = len(combinations)

    print(f"Loading face model (buffalo_l)...")
    app = FaceAnalysis(name="buffalo_l", allowed_modules=["detection"])
    app.prepare(ctx_id=-1, det_size=(640, 640))

    print(f"\nRunning {total_runs} parameter combinations...\n")
    all_results = []
    for i, (det, nms_iou, tile) in enumerate(combinations, 1):
        result = run_combination(app, jpg_files, script_dir,
                                 det, nms_iou, tile, i, total_runs)
        all_results.append(result)

    # ── build report ─────────────────────────────────────────────────────────
    all_results.sort(key=lambda r: r["total"], reverse=True)
    filenames = list(all_results[0]["per_file"].keys())
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    W_TAG  = 28
    W_DET  = 10
    W_NMS  = 10
    W_TILE = 8
    W_TIME = 10
    W_TOT  = 12
    SEP    = "-" * (W_TAG + W_DET + W_NMS + W_TILE + W_TIME + W_TOT + 10)

    header = (
        f"{'PARAMETERS':<{W_TAG}}"
        f"{'DET':>{W_DET}}"
        f"{'NMS IoU':>{W_NMS}}"
        f"{'TILE':>{W_TILE}}"
        f"{'TIME(s)':>{W_TIME}}"
        f"{'FACES':>{W_TOT}}"
    )

    lines = [
        "=" * len(SEP),
        "Face Detection Benchmark Report",
        f"Generated  : {ts}",
        f"Model      : insightface RetinaFace (buffalo_l)",
        f"Overlap    : {int(OVERLAP*100)}%  (fixed across all runs)",
        f"Images     : {', '.join(filenames)}",
        "",
        "Ranked by total face count (highest first)",
        "=" * len(SEP),
        header,
        SEP,
    ]

    baseline_total = None
    for rank, r in enumerate(all_results, 1):
        if baseline_total is None:
            baseline_total = r["total"]
            delta_str = "(best)"
        else:
            delta = r["total"] - baseline_total
            delta_str = f"({delta:+d})"

        lines.append(
            f"#{rank:<{W_TAG-1}}"
            f"{r['det_thresh']:>{W_DET}.2f}"
            f"{r['nms_iou']:>{W_NMS}.2f}"
            f"{r['tile_size']:>{W_TILE}}"
            f"{r['elapsed_s']:>{W_TIME}.1f}"
            f"{str(r['total']) + ' ' + delta_str:>{W_TOT}}"
        )

    lines += [
        SEP,
        "",
        "Per-file breakdown",
        "=" * len(SEP),
    ]

    for filename in filenames:
        lines.append(f"\n  {filename}")
        lines.append(f"  {'Parameters':<28} {'Faces':>8}  {'Time(s)':>8}")
        lines.append(f"  {'-'*46}")
        for r in all_results:
            count = r["per_file"].get(filename, "N/A")
            lines.append(f"  {r['tag']:<28} {count:>8}  {r['elapsed_s']:>8.1f}")

    lines += [
        "",
        "=" * len(SEP),
        "Annotated images saved per run:",
    ]
    for r in all_results:
        lines.append(f"  {r['tag']:<30}  {r['out_dir']}/")

    lines += ["", "=" * len(SEP)]

    # Print to terminal
    print("\n")
    for line in lines:
        print(line)

    # Write to file
    report_path = os.path.join(script_dir, REPORT_FILE)
    with open(report_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nReport written to: {report_path}")
    print(f"Open with:         vim {REPORT_FILE}")


if __name__ == "__main__":
    main()
