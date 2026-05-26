import os
import glob
import cv2
import numpy as np
from datetime import datetime
from insightface.app import FaceAnalysis

OUTPUT_DIR      = "processed_images"
REPORT_FILE     = "people_count_report.txt"
TILE_SIZE       = 480    # px — best from benchmark sweep (det20_nms40_tile480)
OVERLAP         = 0.25   # 25% tile overlap to avoid missing faces at edges
DET_THRESH      = 0.20   # best from benchmark: catches more distant/partial faces
NMS_IOU_THRESH  = 0.40   # best from benchmark: NMS IoU had minimal effect at 0.30


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
    """Non-maximum suppression to remove duplicate face detections across tiles."""
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
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


def detect_faces_tiled(app, image: np.ndarray) -> tuple[list, list]:
    """Tile the image, run RetinaFace on each tile, merge with NMS."""
    h, w   = image.shape[:2]
    step   = int(TILE_SIZE * (1 - OVERLAP))
    all_boxes, all_scores = [], []

    y_starts = list(range(0, h, step))
    x_starts = list(range(0, w, step))
    total_tiles = len(y_starts) * len(x_starts)
    processed  = 0

    for y in y_starts:
        for x in x_starts:
            # Clamp to image bounds while keeping tile full-size
            x1 = min(x, w - TILE_SIZE)
            y1 = min(y, h - TILE_SIZE)
            x2, y2 = x1 + TILE_SIZE, y1 + TILE_SIZE

            tile  = image[y1:y2, x1:x2]
            faces = app.get(tile)

            for face in faces:
                if face.det_score < DET_THRESH:
                    continue
                bx1, by1, bx2, by2 = face.bbox
                all_boxes.append([bx1 + x1, by1 + y1, bx2 + x1, by2 + y1])
                all_scores.append(float(face.det_score))

            processed += 1
            if processed % 50 == 0 or processed == total_tiles:
                print(f"  Tiles: {processed}/{total_tiles}  |  raw detections so far: {len(all_boxes)}")

    if not all_boxes:
        return [], []

    boxes  = np.array(all_boxes,  dtype=np.float32)
    scores = np.array(all_scores, dtype=np.float32)
    keep   = nms(boxes, scores, NMS_IOU_THRESH)
    return boxes[keep].tolist(), scores[keep].tolist()


def write_report(lines: list[str], path: str) -> None:
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)

    jpg_files = find_jpg_files(script_dir)
    if not jpg_files:
        print("No .jpg files found in the current directory.")
        return

    print("Loading face detection model (RetinaFace / buffalo_l)...")
    app = FaceAnalysis(name="buffalo_l", allowed_modules=["detection"])
    app.prepare(ctx_id=-1, det_size=(TILE_SIZE, TILE_SIZE), det_thresh=DET_THRESH)

    results_summary: list[tuple[str, int]] = []

    for filepath in jpg_files:
        filename = os.path.basename(filepath)
        print(f"\nProcessing: {filename}")
        image = cv2.imread(filepath)
        if image is None:
            print(f"  [WARN] Could not read {filename}, skipping.")
            continue

        h, w = image.shape[:2]
        step = int(TILE_SIZE * (1 - OVERLAP))
        n_tiles = len(range(0, h, step)) * len(range(0, w, step))
        print(f"  Image  : {w} x {h} px  |  Tiles: ~{n_tiles} at {TILE_SIZE}px / {int(OVERLAP*100)}% overlap")

        boxes, scores = detect_faces_tiled(app, image)
        count = len(boxes)
        print(f"  Faces detected: {count}")

        annotated = image.copy()
        for (x1, y1, x2, y2), score in zip(boxes, scores):
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(annotated, f"{score:.2f}", (x1, max(y1 - 8, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

        cv2.imwrite(os.path.join(output_dir, filename), annotated)
        results_summary.append((filename, count))

    # Build report
    col_w = max(len(name) for name, _ in results_summary) + 2
    bar   = "-" * (col_w + 20)
    total = sum(c for _, c in results_summary)
    ts    = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    report_lines = [
        "People (Face) Detection Report",
        f"Generated  : {ts}",
        f"Model      : insightface RetinaFace (buffalo_l)",
        f"Det thresh : {DET_THRESH}",
        f"NMS IoU    : {NMS_IOU_THRESH}",
        f"Tile size  : {TILE_SIZE}px  |  Overlap: {int(OVERLAP*100)}%",
        "",
        "=" * (col_w + 20),
        f"{'FILE NAME':<{col_w}} {'FACE COUNT':>10}",
        bar,
        *[f"{name:<{col_w}} {count:>10}" for name, count in results_summary],
        bar,
        f"{'TOTAL':<{col_w}} {total:>10}",
        "=" * (col_w + 20),
        "",
        f"Annotated images : {output_dir}/",
    ]

    print()
    for line in report_lines:
        print(line)

    report_path = os.path.join(script_dir, REPORT_FILE)
    write_report(report_lines, report_path)
    print(f"\nReport saved to : {report_path}")


if __name__ == "__main__":
    main()
