import os
import glob
import cv2
from datetime import datetime
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

OUTPUT_DIR  = "processed_images"
REPORT_FILE = "people_count_report.txt"
MODEL_NAME  = "yolov8x.pt"  # extra-large model — slowest, most accurate
CONFIDENCE  = 0.10           # low threshold to catch distant/partial people
SLICE_SIZE  = 640            # tile size in pixels
OVERLAP     = 0.2            # 20% tile overlap to avoid edge misses
PERSON_CLASS_ID = 0


def find_jpg_files(directory: str) -> list[str]:
    pattern_lower = os.path.join(directory, "*.jpg")
    pattern_upper = os.path.join(directory, "*.JPG")
    files = glob.glob(pattern_lower) + glob.glob(pattern_upper)
    seen, unique = set(), []
    for f in sorted(files):
        key = os.path.normcase(f)
        if key not in seen:
            seen.add(key)
            unique.append(f)
    return unique


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)

    jpg_files = find_jpg_files(script_dir)
    if not jpg_files:
        print("No .jpg files found in the current directory.")
        return

    print(f"Loading model: {MODEL_NAME}  (confidence={CONFIDENCE})")
    detection_model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=MODEL_NAME,
        confidence_threshold=CONFIDENCE,
        device="cpu",
    )

    results_summary: list[tuple[str, int]] = []

    for filepath in jpg_files:
        filename = os.path.basename(filepath)
        print(f"\nProcessing: {filename}")

        result = get_sliced_prediction(
            filepath,
            detection_model,
            slice_height=SLICE_SIZE,
            slice_width=SLICE_SIZE,
            overlap_height_ratio=OVERLAP,
            overlap_width_ratio=OVERLAP,
            verbose=0,
        )

        person_preds = [
            p for p in result.object_prediction_list
            if p.category.id == PERSON_CLASS_ID
        ]
        count = len(person_preds)
        print(f"  Detected: {count} people")

        image = cv2.imread(filepath)
        for pred in person_preds:
            b = pred.bbox
            x1, y1, x2, y2 = int(b.minx), int(b.miny), int(b.maxx), int(b.maxy)
            conf = pred.score.value
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{conf:.2f}"
            cv2.putText(image, label, (x1, max(y1 - 6, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

        out_path = os.path.join(output_dir, filename)
        cv2.imwrite(out_path, image)
        results_summary.append((filename, count))

    # Build report lines
    col_w = max(len(name) for name, _ in results_summary) + 2
    bar   = "-" * (col_w + 20)
    total = sum(c for _, c in results_summary)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    report_lines = [
        f"People Detection Report",
        f"Generated : {timestamp}",
        f"Model     : {MODEL_NAME}",
        f"Confidence: {CONFIDENCE}",
        f"",
        "=" * (col_w + 20),
        f"{'FILE NAME':<{col_w}} {'PEOPLE COUNT':>12}",
        bar,
        *[f"{name:<{col_w}} {count:>12}" for name, count in results_summary],
        bar,
        f"{'TOTAL':<{col_w}} {total:>12}",
        "=" * (col_w + 20),
        f"",
        f"Annotated images saved to: {output_dir}/",
    ]

    # Print to terminal
    print()
    for line in report_lines:
        print(line)

    # Write plain-text report file
    report_path = os.path.join(script_dir, REPORT_FILE)
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines) + "\n")
    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    main()
