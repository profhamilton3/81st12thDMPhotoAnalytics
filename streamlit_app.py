import io
from datetime import datetime

import cv2
import numpy as np
import streamlit as st
from insightface.app import FaceAnalysis

OVERLAP = 0.25
TILE_CHOICES = [320, 480, 640]

BACKSTORY = """
### The Challenge

During a large fraternity gathering, a photographer captured a sweeping group photo of hundreds of
brothers assembled on the steps of a historic building. The image was stunning — but no one could
agree on how many men were actually in the frame. Manual counting on a high-resolution photo with
overlapping figures, partial faces, and varying distances from the camera quickly became an
exercise in frustration.

### The Journey to the Algorithm

The first attempt used **YOLOv8**, a powerful general-purpose object detector. While capable of
detecting people and faces in many settings, it struggled here: distant figures in the upper rows
of the group shot were too small for the model to reliably catch.

The breakthrough came with **RetinaFace** (via insightface's `buffalo` model family), a detector
purpose-built for faces at a wide range of scales. But even RetinaFace had limits on a
single pass over a 20+ megapixel image — faces in the corners and edges of the frame were
still being missed.

The solution was **tiled detection**: slice the full image into overlapping square tiles, run
RetinaFace independently on each tile, translate the local bounding boxes back to global image
coordinates, and then merge all detections with **Non-Maximum Suppression (NMS)** to eliminate
duplicates that appeared in adjacent tiles. This dramatically improved recall across the entire
frame.

A systematic **benchmark sweep** across detection thresholds (0.20–0.30), NMS IoU values
(0.30–0.40), and tile sizes (480px–640px) identified the optimal default parameters now baked
into this app. The sliders on the left let you explore those same trade-offs on your own photos.
"""


@st.cache_resource(show_spinner="Loading face detection model — this happens once per session...")
def load_face_model(model_name: str, tile_size: int) -> FaceAnalysis:
    app = FaceAnalysis(name=model_name, allowed_modules=["detection"])
    # Use a low internal threshold so the model surfaces borderline faces;
    # the user-selected det_thresh is applied as a post-filter in detect_faces_tiled.
    app.prepare(ctx_id=-1, det_size=(tile_size, tile_size), det_thresh=0.10)
    return app


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float) -> list[int]:
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
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[np.where(iou <= iou_thresh)[0] + 1]
    return keep


def detect_faces_tiled(
    app: FaceAnalysis,
    image: np.ndarray,
    tile_size: int,
    det_thresh: float,
    nms_iou_thresh: float,
    progress_bar=None,
) -> tuple[list, list]:
    h, w = image.shape[:2]
    step = int(tile_size * (1 - OVERLAP))
    all_boxes, all_scores = [], []

    y_starts = list(range(0, h, step))
    x_starts = list(range(0, w, step))
    total_tiles = len(y_starts) * len(x_starts)
    processed = 0

    for y in y_starts:
        for x in x_starts:
            x1 = min(x, max(0, w - tile_size))
            y1 = min(y, max(0, h - tile_size))
            x2, y2 = x1 + tile_size, y1 + tile_size

            tile = image[y1:y2, x1:x2]
            if tile.shape[0] < 10 or tile.shape[1] < 10:
                processed += 1
                continue

            faces = app.get(tile)
            for face in faces:
                if face.det_score < det_thresh:
                    continue
                bx1, by1, bx2, by2 = face.bbox
                all_boxes.append([bx1 + x1, by1 + y1, bx2 + x1, by2 + y1])
                all_scores.append(float(face.det_score))

            processed += 1
            if progress_bar is not None:
                progress_bar.progress(processed / total_tiles)

    if not all_boxes:
        return [], []

    boxes = np.array(all_boxes, dtype=np.float32)
    scores = np.array(all_scores, dtype=np.float32)
    keep = nms(boxes, scores, nms_iou_thresh)
    return boxes[keep].tolist(), scores[keep].tolist()


def annotate_image(image: np.ndarray, boxes: list, scores: list) -> np.ndarray:
    annotated = image.copy()
    for (x1, y1, x2, y2), score in zip(boxes, scores):
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(
            annotated, f"{score:.2f}", (x1, max(y1 - 8, 12)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2,
        )
    return annotated


def encode_image_for_download(image: np.ndarray) -> bytes | None:
    success, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return buf.tobytes() if success else None


def main():
    st.set_page_config(
        page_title="Photo People Counter — Face Detection Analytics",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("Photo People Counter")
    st.caption("Face Detection Analytics — powered by RetinaFace (insightface)")

    # --- Sidebar controls ---
    with st.sidebar:
        st.header("Detection Settings")

        model_name = st.selectbox(
            "Model",
            options=["buffalo_sc", "buffalo_l"],
            index=0,
            help=(
                "**buffalo_sc** — lightweight, fast, recommended for cloud.\n\n"
                "**buffalo_l** — higher accuracy, ~500 MB, may be slow on free tier."
            ),
        )

        tile_size = st.select_slider(
            "Tile size (px)",
            options=TILE_CHOICES,
            value=480,
            help="Size of each detection tile. Larger tiles are slower but may catch more context.",
        )

        det_thresh = st.slider(
            "Detection threshold",
            min_value=0.10,
            max_value=0.70,
            value=0.20,
            step=0.05,
            help="Minimum confidence score to keep a detection. Lower = more faces found, more false positives.",
        )

        nms_iou = st.slider(
            "NMS IoU threshold",
            min_value=0.20,
            max_value=0.70,
            value=0.40,
            step=0.05,
            help="Overlap threshold for Non-Maximum Suppression. Lower = more aggressive duplicate removal.",
        )

        st.divider()
        st.caption(
            "Changing **Model** or **Tile size** reloads the model (once per combination). "
            "Threshold sliders take effect instantly."
        )

    # --- Back Story ---
    with st.expander("The Back Story", expanded=False):
        st.markdown(BACKSTORY)

    # --- Upload ---
    st.subheader("Upload a Photo")
    uploaded = st.file_uploader(
        "Choose a JPG or PNG image",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )

    if not uploaded:
        st.info("Upload a photo above to begin face detection.")
        return

    # Decode image
    file_bytes = np.frombuffer(uploaded.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image is None:
        st.error("Could not decode the uploaded image. Please try a different file.")
        return

    h, w = image.shape[:2]
    step = int(tile_size * (1 - OVERLAP))
    n_tiles = len(range(0, h, step)) * len(range(0, w, step))

    st.caption(f"**{uploaded.name}** — {w} × {h} px | ~{n_tiles} tiles at {tile_size} px / {int(OVERLAP * 100)}% overlap")

    # Load model (cached per model_name + tile_size)
    app = load_face_model(model_name, tile_size)

    # Run detection
    with st.spinner("Running tiled face detection..."):
        progress_bar = st.progress(0.0, text="Processing tiles...")
        boxes, scores = detect_faces_tiled(
            app, image, tile_size, det_thresh, nms_iou, progress_bar=progress_bar
        )
        progress_bar.empty()

    count = len(boxes)
    annotated = annotate_image(image, boxes, scores)

    # --- Results ---
    st.subheader(f"Results: {count} {'face' if count == 1 else 'faces'} detected")

    col1, col2 = st.columns(2)
    with col1:
        st.caption("Original")
        orig_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        st.image(orig_rgb, use_container_width=True)
    with col2:
        st.caption(f"Annotated — {count} detections")
        ann_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        st.image(ann_rgb, use_container_width=True)

    # Download button
    img_bytes = encode_image_for_download(annotated)
    if img_bytes:
        base, ext = uploaded.name.rsplit(".", 1) if "." in uploaded.name else (uploaded.name, "jpg")
        st.download_button(
            label="Download annotated image",
            data=img_bytes,
            file_name=f"{base}_detected.jpg",
            mime="image/jpeg",
        )

    # Detection report
    st.subheader("Detection Report")
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Faces detected", count)
    with col_b:
        avg_conf = f"{np.mean(scores):.3f}" if scores else "—"
        st.metric("Avg confidence", avg_conf)

    st.table(
        {
            "Parameter": [
                "File",
                "Dimensions",
                "Model",
                "Tile size",
                "Tile overlap",
                "Detection threshold",
                "NMS IoU threshold",
                "Faces detected",
                "Generated",
            ],
            "Value": [
                uploaded.name,
                f"{w} × {h} px",
                model_name,
                f"{tile_size} px",
                f"{int(OVERLAP * 100)}%",
                det_thresh,
                nms_iou,
                count,
                ts,
            ],
        }
    )


if __name__ == "__main__":
    main()
