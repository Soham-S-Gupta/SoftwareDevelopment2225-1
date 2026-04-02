import argparse
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2

from src.asl_translator.config import load_project_config
from src.asl_translator.hand_tracking import HandTracker


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--letter", required=True, help="Letter folder name, such as A, B, M, or space")
    parser.add_argument("--count", type=int, default=None)
    args = parser.parse_args()

    config = load_project_config(args.config)
    target_count = args.count or config["runtime"]["custom_capture_images_per_letter"]
    output_dir = Path(config["paths"]["custom_letter_train_dir"]) / args.letter
    output_dir.mkdir(parents=True, exist_ok=True)

    tracker = HandTracker()
    cap = cv2.VideoCapture(config["runtime"]["camera_index"])
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    saved = 0
    last_save_time = 0.0
    save_interval_seconds = 0.2

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        landmarks, bbox = tracker.detect(frame)
        display = frame.copy()

        if bbox is not None:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 220, 120), 2)

        cv2.putText(display, f"Capture letter: {args.letter.upper()}", (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(display, f"Saved: {saved}/{target_count}", (18, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 220, 90), 2)
        cv2.putText(display, "Hold the hand steady. Press S to save manually, A for auto-capture, Q to quit.", (18, 102), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (180, 180, 180), 2)

        cv2.imshow("Capture Custom Letters", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        manual_save = key == ord("s")
        auto_save = key == ord("a")
        should_save = manual_save or auto_save

        if auto_save and bbox is not None and time.time() - last_save_time < save_interval_seconds:
            should_save = False

        if should_save and bbox is not None:
            x1, y1, x2, y2 = bbox
            crop = frame[y1:y2, x1:x2]
            if crop.size > 0:
                filename = output_dir / f"{args.letter.lower()}_{saved:04d}.jpg"
                cv2.imwrite(str(filename), crop)
                saved += 1
                last_save_time = time.time()

        if saved >= target_count:
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Saved {saved} images to {output_dir}")


if __name__ == "__main__":
    main()
