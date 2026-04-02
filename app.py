import cv2

from src.asl_translator.config import load_project_config
from src.asl_translator.pipeline import ASLTranslatorPipeline
from src.asl_translator.ui import TranslatorUI, WINDOW_NAME


def main() -> None:
    config = load_project_config("configs/project_config.yaml")
    pipeline = ASLTranslatorPipeline(config)
    ui = TranslatorUI(config)

    camera_index = config["runtime"]["camera_index"]
    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Check your camera index in configs/project_config.yaml")

    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, ui.handle_mouse)

    last_result = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        if ui.state.started and not ui.state.paused:
            last_result = pipeline.process_frame(frame)

        display = ui.draw(frame, last_result)

        cv2.imshow(WINDOW_NAME, display)
        key = cv2.waitKey(1) & 0xFF
        ui.handle_key(key)

        if ui.state.clear_requested:
            pipeline.clear_text()
            ui.state.clear_requested = False
        if ui.state.clear_history_requested:
            pipeline.clear_history()
            ui.state.clear_history_requested = False
        if ui.state.should_quit:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
