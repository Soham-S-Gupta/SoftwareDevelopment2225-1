from dataclasses import dataclass, field
import ctypes
from pathlib import Path
import string

import cv2
import numpy as np


PANEL_BG = (24, 27, 32)
CARD_BG = (35, 40, 48)
TEXT = (245, 245, 245)
MUTED = (170, 178, 190)
ACCENT = (89, 196, 255)
SUCCESS = (98, 220, 160)
WARNING = (255, 208, 92)
DANGER = (100, 90, 235)
DARK = (12, 14, 18)
WINDOW_NAME = "ASL Translator"
FALLBACK_DISPLAY_WIDTH = 1280
FALLBACK_DISPLAY_HEIGHT = 720


@dataclass
class UIButton:
    name: str
    rect: tuple[int, int, int, int]
    label: str
    fill: tuple[int, int, int]
    text_color: tuple[int, int, int] = TEXT
    font_scale: float = 0.62


@dataclass
class UIState:
    started: bool = False
    paused: bool = False
    should_quit: bool = False
    clear_requested: bool = False
    clear_history_requested: bool = False
    history_view: bool = False
    guide_mode: str = "list"
    selected_letter: str = ""
    practice_text: str = ""
    practice_index: int = 0
    practice_focused: bool = False
    practice_box_rect: tuple[int, int, int, int] = (0, 0, 0, 0)
    buttons: list[UIButton] = field(default_factory=list)


class TranslatorUI:
    def __init__(self, config: dict) -> None:
        self.config = config
        self.state = UIState()
        self.display_scale = 1.0
        self.max_display_width, self.max_display_height = self._detect_display_limits()
        self.practice_text_max_width = 420
        self.guide_order = list(string.ascii_uppercase) + ["space", "del"]
        guide_image_dir = config["paths"].get("guide_image_dir", config["paths"]["letter_train_dir"])
        self.guide_images = self._load_guide_images(Path(guide_image_dir))

    def handle_mouse(self, event: int, x: int, y: int, flags: int, param) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if self.display_scale != 1.0:
            x = int(x / self.display_scale)
            y = int(y / self.display_scale)
        bx1, by1, bx2, by2 = self.state.practice_box_rect
        self.state.practice_focused = bx1 <= x <= bx2 and by1 <= y <= by2
        for button in self.state.buttons:
            x1, y1, x2, y2 = button.rect
            if x1 <= x <= x2 and y1 <= y <= y2:
                self._activate_button(button.name)
                break

    def handle_key(self, key: int) -> None:
        if key in (-1, 255):
            return
        if not self.state.practice_focused:
            return
        if key in (8, 127):
            if self.state.practice_text:
                self.state.practice_text = self.state.practice_text[:-1]
                self.state.practice_index = min(self.state.practice_index, max(len(self.state.practice_text) - 1, 0))
            return
        if key in (13, 83):
            self._advance_practice(1)
            return
        if key == 81:
            self._advance_practice(-1)
            return

        try:
            char = chr(key)
        except ValueError:
            return

        if char.isalpha():
            self._append_practice_character(char.upper())
        elif char == " ":
            self._append_practice_character(" ")

    def draw(self, frame: np.ndarray, result: dict | None) -> np.ndarray:
        self.state.buttons = []
        camera_height, camera_width = frame.shape[:2]
        side_width = self._clamp(int(camera_width * 0.34), 320, 430)
        bottom_height = self._clamp(int(camera_height * 0.68), 460, 560)
        canvas = np.zeros((camera_height + bottom_height, camera_width + side_width, 3), dtype=np.uint8)
        canvas[:, :, :] = PANEL_BG
        canvas[:camera_height, :camera_width] = frame

        if result and result["bbox"] is not None and not self.state.history_view:
            x1, y1, x2, y2 = result["bbox"]
            cv2.rectangle(canvas, (x1, y1), (x2, y2), SUCCESS, 3)

        if self.state.history_view:
            self._draw_history_view(canvas, result)
        else:
            self._draw_header(canvas, camera_width, result)
            self._draw_bottom(canvas, camera_width, camera_height, bottom_height, result)
            self._draw_sidebar(canvas, camera_width, side_width, camera_height + bottom_height)

        if not self.state.started:
            self._draw_start_overlay(canvas)

        return self._fit_display_to_screen(canvas)

    def _activate_button(self, name: str) -> None:
        if name == "start":
            self.state.started = True
            self.state.paused = False
            return
        if name == "pause_resume":
            self.state.paused = not self.state.paused
            return
        if name == "quit":
            self.state.should_quit = True
            return
        if name == "open_history":
            self.state.history_view = True
            return
        if name == "back_main":
            self.state.history_view = False
            return
        if name == "clear_history":
            self.state.clear_history_requested = True
            return
        if name == "guide_back":
            self.state.guide_mode = "list"
            self.state.selected_letter = ""
            return
        if name == "practice_prev":
            self._advance_practice(-1)
            return
        if name == "practice_next":
            self._advance_practice(1)
            return
        if name == "clear_text":
            self.state.practice_text = ""
            self.state.practice_index = 0
            return
        if name.startswith("guide_"):
            self.state.guide_mode = "detail"
            self.state.selected_letter = name.replace("guide_", "", 1)

    def _advance_practice(self, delta: int) -> None:
        characters = self._practice_characters()
        if not characters:
            self.state.practice_index = 0
            return
        self.state.practice_index = max(0, min(self.state.practice_index + delta, len(characters) - 1))

    def _load_guide_images(self, train_dir: Path) -> dict[str, np.ndarray | None]:
        guide_images: dict[str, np.ndarray | None] = {}
        for label in self.guide_order:
            folder = train_dir / label
            image = None
            if folder.exists():
                files = sorted(
                    [path for path in folder.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"}]
                )
                if files:
                    image = cv2.imread(str(files[0]))
            guide_images[label] = image
        return guide_images

    def _draw_header(self, canvas: np.ndarray, camera_width: int, result: dict | None) -> None:
        cv2.rectangle(canvas, (0, 0), (camera_width, 96), (0, 0, 0), -1)
        cv2.putText(canvas, "ASL Alphabet Translator", (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.78, TEXT, 2)
        status = "Paused" if self.state.paused else "Scanning" if self.state.started else "Ready to start"
        cv2.putText(canvas, f"Status: {status}", (18, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.72, SUCCESS, 2)

        live_text = "Live letter: --"
        if result and self.state.started and not self.state.paused:
            live_text = f"Live letter: {result['live_status']}"
        elif self.state.paused:
            live_text = "Live letter: paused"
        cv2.putText(canvas, live_text, (350, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WARNING, 2)

        pause_label = "Resume" if self.state.paused else "Pause"
        button_w = self._clamp(int(camera_width * 0.1), 105, 130)
        button_gap = 20
        quit_x1 = camera_width - button_w - 20
        pause_x1 = quit_x1 - button_w - button_gap
        bar_width = max(140, min(420, pause_x1 - 378))
        self._draw_confidence_bar(canvas, result, 350, 48, bar_width, 18)
        self._draw_button(canvas, UIButton("pause_resume", (pause_x1, 18, pause_x1 + button_w, 58), pause_label, ACCENT))
        self._draw_button(canvas, UIButton("quit", (quit_x1, 18, quit_x1 + button_w, 58), "Quit", DANGER))

    def _draw_bottom(self, canvas: np.ndarray, camera_width: int, camera_height: int, bottom_height: int, result: dict | None) -> None:
        top = camera_height + 16
        width = camera_width - 32

        current_h = self._clamp(int(bottom_height * 0.29), 116, 138)
        practice_h = self._clamp(int(bottom_height * 0.31), 126, 148)
        history_h = bottom_height - 64 - current_h - practice_h

        current_word = result["current_word"] if result else ""
        current_sentence = result["current_sentence"] if result else ""
        history = result["history"] if result else []

        cv2.rectangle(canvas, (16, top), (16 + width, top + current_h), CARD_BG, -1)
        cv2.putText(canvas, "Current Word / Sentence", (34, top + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, ACCENT, 2)
        self._put_wrapped_text(canvas, f"Current word: {current_word or '(still spelling...)'}", 34, top + 70, width - 40, 28, 0.82, TEXT, thickness=2)
        self._put_wrapped_text(canvas, f"Sentence: {current_sentence or '(nothing finished yet)'}", 34, top + 104, width - 40, 24, 0.68, MUTED, thickness=2)

        practice_top = top + current_h + 16
        cv2.rectangle(canvas, (16, practice_top), (16 + width, practice_top + practice_h), CARD_BG, -1)
        cv2.putText(canvas, "Type A Letter, Word, Or Sentence To Practice", (34, practice_top + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.84, ACCENT, 2)
        typed_display = self.state.practice_text or "(type here)"
        right_edge = 16 + width
        clear_y1 = practice_top + practice_h - 46
        clear_y2 = practice_top + practice_h - 14
        clear_x1 = right_edge - 170
        clear_x2 = right_edge - 40
        input_rect = (34, practice_top + 48, clear_x1 - 20, practice_top + 98)
        self.state.practice_box_rect = input_rect
        cv2.rectangle(canvas, (input_rect[0], input_rect[1]), (input_rect[2], input_rect[3]), DARK, -1)
        border_color = ACCENT if self.state.practice_focused else (70, 76, 86)
        cv2.rectangle(canvas, (input_rect[0], input_rect[1]), (input_rect[2], input_rect[3]), border_color, 2)
        self.practice_text_max_width = max(40, clear_x1 - 78)
        self._put_wrapped_text(canvas, typed_display, 46, practice_top + 82, self.practice_text_max_width, 24, 0.78, TEXT, thickness=2)
        self._draw_button(canvas, UIButton("clear_text", (clear_x1, clear_y1, clear_x2, clear_y2), "Clear", DANGER))

        history_top = practice_top + practice_h + 16
        cv2.rectangle(canvas, (16, history_top), (16 + width, history_top + history_h), CARD_BG, -1)
        cv2.putText(canvas, "Completed Words / Sentences", (34, history_top + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, ACCENT, 2)
        self._draw_button(canvas, UIButton("open_history", (16 + width - 180, history_top + 18, 16 + width - 34, history_top + 58), "Open History", ACCENT))
        if not history:
            self._put_wrapped_text(canvas, "No finished words or sentences yet.", 34, history_top + 72, width - 40, 24, 0.64, MUTED, thickness=2)
        else:
            preview = history[-3:]
            for index, item in enumerate(preview):
                y = history_top + 68 + index * 28
                number = len(history) - len(preview) + index + 1
                cv2.putText(canvas, f"{number}. {item}", (34, y), cv2.FONT_HERSHEY_SIMPLEX, 0.72, TEXT, 2)

    def _draw_sidebar(self, canvas: np.ndarray, camera_width: int, side_width: int, total_height: int) -> None:
        left = camera_width + 16
        right = camera_width + side_width - 16
        cv2.rectangle(canvas, (camera_width, 0), (camera_width + side_width, total_height), (18, 21, 25), -1)
        cv2.putText(canvas, "ASL Letter Guide", (left, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, ACCENT, 2)

        if self.state.guide_mode == "detail" and self.state.selected_letter:
            self._draw_button(canvas, UIButton("guide_back", (left, 48, left + 90, 84), "Back", ACCENT))
            label = self.state.selected_letter
            cv2.putText(canvas, f"Showing: {self._display_label(label)}", (left + 110, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.72, WARNING, 2)
            self._draw_guide_image(canvas, label, left, 108, right - left, total_height - 140)
        else:
            self._put_wrapped_text(canvas, "Click on any letter to see how it's signed.", left, 62, right - left - 4, 16, 0.48, MUTED, thickness=1)
            content_width = right - left
            cols = 4 if content_width >= 300 else 3
            gap = max(6, int(content_width * 0.025))
            button_w = (content_width - gap * (cols - 1)) // cols
            button_h = self._clamp(int(button_w * 0.56), 34, 42)
            start_y = 112
            for index, label in enumerate(self.guide_order):
                row = index // cols
                col = index % cols
                x1 = left + col * (button_w + gap)
                y1 = start_y + row * (button_h + gap)
                button = UIButton(
                    name=f"guide_{label}",
                    rect=(x1, y1, x1 + button_w, y1 + button_h),
                    label=self._display_label(label),
                    fill=CARD_BG,
                )
                self._draw_button(canvas, button)

            chars = self._practice_characters()
            current_char = chars[self.state.practice_index] if chars else ""
            if current_char:
                rows = (len(self.guide_order) + cols - 1) // cols
                grid_bottom = start_y + rows * (button_h + gap) - gap
                title_y = min(max(grid_bottom + 36, int(total_height * 0.45)), total_height - 150)
                cv2.putText(canvas, "Practice guide", (left, title_y), cv2.FONT_HERSHEY_SIMPLEX, 0.76, ACCENT, 2)
                cv2.putText(canvas, f"Showing: {self._display_label(current_char)}", (left, title_y + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.66, WARNING, 2)
                image_top = title_y + 44
                image_height = max(80, total_height - image_top - 78)
                self._draw_guide_image(canvas, current_char, left, image_top, right - left, image_height)
                self._draw_button(canvas, UIButton("practice_prev", (left + 34, total_height - 70, left + 114, total_height - 26), "<", ACCENT))
                self._draw_button(canvas, UIButton("practice_next", (right - 114, total_height - 70, right - 34, total_height - 26), ">", ACCENT))

    def _draw_history_view(self, canvas: np.ndarray, result: dict | None) -> None:
        canvas[:, :, :] = PANEL_BG
        cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 96), (0, 0, 0), -1)
        cv2.putText(canvas, "Completed Words / Sentences", (24, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.0, TEXT, 2)
        cv2.putText(canvas, "This page shows your full history from oldest to newest.", (24, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.62, MUTED, 2)
        self._draw_button(canvas, UIButton("back_main", (canvas.shape[1] - 280, 22, canvas.shape[1] - 160, 62), "Back", ACCENT))
        self._draw_button(canvas, UIButton("clear_history", (canvas.shape[1] - 170, 22, canvas.shape[1] - 20, 62), "Clear History", DANGER, font_scale=0.5))

        history = result["history"] if result else []
        cv2.rectangle(canvas, (24, 120), (canvas.shape[1] - 24, canvas.shape[0] - 24), CARD_BG, -1)
        if not history:
            cv2.putText(canvas, "No history yet.", (48, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.9, MUTED, 2)
            return

        for index, item in enumerate(history):
            y = 170 + index * 34
            if y > canvas.shape[0] - 48:
                break
            cv2.putText(canvas, f"{index + 1}. {item}", (48, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, TEXT, 2)

    def _draw_guide_image(self, canvas: np.ndarray, label: str, x: int, y: int, width: int, height: int) -> None:
        cv2.rectangle(canvas, (x, y), (x + width, y + height), CARD_BG, -1)
        image = self.guide_images.get(label)
        if image is None:
            cv2.putText(canvas, "No sample image here yet", (x + 18, y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, MUTED, 2)
            return

        fitted = self._fit_image(image, width - 24, height - 54)
        img_h, img_w = fitted.shape[:2]
        x_offset = x + (width - img_w) // 2
        y_offset = y + 16
        canvas[y_offset : y_offset + img_h, x_offset : x_offset + img_w] = fitted

    @staticmethod
    def _fit_image(image: np.ndarray, max_width: int, max_height: int) -> np.ndarray:
        height, width = image.shape[:2]
        scale = min(max_width / width, max_height / height)
        new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
        return cv2.resize(image, new_size)

    def _fit_display_to_screen(self, canvas: np.ndarray) -> np.ndarray:
        height, width = canvas.shape[:2]
        self.display_scale = min(self.max_display_width / width, self.max_display_height / height, 1.0)
        if self.display_scale == 1.0:
            return canvas
        new_size = (max(1, int(width * self.display_scale)), max(1, int(height * self.display_scale)))
        return cv2.resize(canvas, new_size, interpolation=cv2.INTER_AREA)

    def _draw_confidence_bar(
        self,
        canvas: np.ndarray,
        result: dict | None,
        x: int,
        y: int,
        width: int,
        height: int,
    ) -> None:
        confidence = 0.0
        if result and self.state.started and not self.state.paused:
            prediction = result.get("letter_prediction", {})
            if prediction.get("label"):
                confidence = max(0.0, min(float(prediction.get("confidence", 0.0)), 1.0))

        cv2.rectangle(canvas, (x - 2, y - 2), (x + width + 2, y + height + 2), CARD_BG, -1)
        for offset in range(width):
            amount = offset / max(width - 1, 1)
            red = int(230 * (1.0 - amount))
            green = int(220 * amount)
            cv2.line(canvas, (x + offset, y), (x + offset, y + height), (45, green, red), 1)

        slider_x = x + int(confidence * width)
        cv2.rectangle(canvas, (slider_x - 3, y - 5), (slider_x + 3, y + height + 5), TEXT, -1)
        cv2.rectangle(canvas, (x, y), (x + width, y + height), MUTED, 1)

    def _draw_start_overlay(self, canvas: np.ndarray) -> None:
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, 0), (canvas.shape[1], canvas.shape[0]), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, canvas, 0.45, 0, canvas)
        box_w = self._clamp(int(canvas.shape[1] * 0.42), 560, 700)
        box_h = self._clamp(int(canvas.shape[0] * 0.27), 260, 330)
        x1 = (canvas.shape[1] - box_w) // 2
        y1 = (canvas.shape[0] - box_h) // 2
        cv2.rectangle(canvas, (x1, y1), (x1 + box_w, y1 + box_h), CARD_BG, -1)
        title_scale = self._clamp_float(box_w / 600, 0.95, 1.18)
        subtitle_scale = self._clamp_float(box_w / 700, 0.78, 1.02)
        cv2.putText(canvas, "ASL Alphabet Translator", (x1 + 48, y1 + int(box_h * 0.25)), cv2.FONT_HERSHEY_SIMPLEX, title_scale, TEXT, 2)
        cv2.putText(canvas, "Press Start to begin scanning.", (x1 + 48, y1 + int(box_h * 0.43)), cv2.FONT_HERSHEY_SIMPLEX, subtitle_scale, MUTED, 2)
        button_w, button_h = 200, 56
        bx1 = x1 + (box_w - button_w) // 2
        by1 = y1 + box_h - 94
        self._draw_button(canvas, UIButton("start", (bx1, by1, bx1 + button_w, by1 + button_h), "Start", ACCENT))

    @staticmethod
    def _clamp(value: int, minimum: int, maximum: int) -> int:
        return max(minimum, min(value, maximum))

    @staticmethod
    def _clamp_float(value: float, minimum: float, maximum: float) -> float:
        return max(minimum, min(value, maximum))

    @staticmethod
    def _detect_display_limits() -> tuple[int, int]:
        try:
            user32 = ctypes.windll.user32
            screen_width = int(user32.GetSystemMetrics(0))
            screen_height = int(user32.GetSystemMetrics(1))
            return max(900, screen_width - 80), max(600, screen_height - 120)
        except (AttributeError, OSError, ValueError):
            return FALLBACK_DISPLAY_WIDTH, FALLBACK_DISPLAY_HEIGHT

    def _practice_characters(self) -> list[str]:
        characters: list[str] = []
        for char in self.state.practice_text:
            if char == " ":
                characters.append("space")
            elif char.isalpha():
                characters.append(char.upper())
        return characters

    def _append_practice_character(self, char: str) -> None:
        candidate = self.state.practice_text + char
        width = cv2.getTextSize(candidate or "(type here)", cv2.FONT_HERSHEY_SIMPLEX, 0.78, 2)[0][0]
        if width <= self.practice_text_max_width:
            self.state.practice_text = candidate

    @staticmethod
    def _display_label(label: str) -> str:
        return label.upper() if len(label) == 1 else label.upper()

    def _put_wrapped_text(
        self,
        canvas: np.ndarray,
        text: str,
        x: int,
        y: int,
        max_width: int,
        line_height: int,
        font_scale: float,
        color: tuple[int, int, int],
        thickness: int = 1,
    ) -> None:
        words = text.split()
        current = ""
        line_y = y
        for word in words:
            candidate = word if not current else f"{current} {word}"
            width = cv2.getTextSize(candidate, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0][0]
            if width <= max_width:
                current = candidate
                continue
            cv2.putText(canvas, current, (x, line_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
            current = word
            line_y += line_height
        if current:
            cv2.putText(canvas, current, (x, line_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

    def _draw_button(self, canvas: np.ndarray, button: UIButton) -> None:
        x1, y1, x2, y2 = button.rect
        cv2.rectangle(canvas, (x1, y1), (x2, y2), button.fill, -1)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (70, 76, 86), 1)
        text_size = cv2.getTextSize(button.label, cv2.FONT_HERSHEY_SIMPLEX, button.font_scale, 2)[0]
        tx = x1 + (x2 - x1 - text_size[0]) // 2
        ty = y1 + (y2 - y1 + text_size[1]) // 2
        cv2.putText(canvas, button.label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, button.font_scale, button.text_color, 2)
        self.state.buttons.append(button)
