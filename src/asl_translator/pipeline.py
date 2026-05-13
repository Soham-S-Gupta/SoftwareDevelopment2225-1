from collections import Counter, deque

from src.asl_translator.hand_tracking import HandTracker
from src.asl_translator.letter_model import LetterClassifier
from src.asl_translator.speech import SpeechEngine


class ASLTranslatorPipeline:
    def __init__(self, config: dict) -> None:
        self.config = config
        self.runtime = config["runtime"]
        self.paths = config["paths"]
        self.tracker = HandTracker()
        self.letter_history: deque[str] = deque(maxlen=self.runtime["smoothing_window"])
        self.current_word = ""
        self.completed_items: deque[str] = deque(maxlen=self.runtime["max_history_items"])
        self.current_sentence: list[str] = []
        self.last_committed_letter = ""
        self.repeat_letter_cooldown = 0
        self.frames_without_hand = 0
        self.last_spoken_word = ""
        self.speech = SpeechEngine(enabled=self.runtime["text_to_speech_enabled"])

        self.letter_model = None
        chosen_checkpoint = self.paths.get("custom_letter_checkpoint", self.paths["letter_checkpoint"])
        if not LetterClassifier.is_ready(chosen_checkpoint, self.paths["letter_labels"]):
            chosen_checkpoint = self.paths["letter_checkpoint"]

        if LetterClassifier.is_ready(chosen_checkpoint, self.paths["letter_labels"]):
            self.letter_model = LetterClassifier(
                checkpoint_path=chosen_checkpoint,
                labels_path=self.paths["letter_labels"],
                image_size=config["training"]["image_size"],
            )

    def clear_text(self) -> None:
        self.letter_history.clear()
        self.current_word = ""
        self.current_sentence.clear()
        self.last_committed_letter = ""
        self.repeat_letter_cooldown = 0
        self.frames_without_hand = 0
        self.last_spoken_word = ""

    def clear_history(self) -> None:
        self.clear_text()
        self.completed_items.clear()

    def process_frame(self, frame) -> dict:
        landmarks, bbox = self.tracker.detect(frame)
        hand_found = bool(landmarks)
        self.repeat_letter_cooldown = max(0, self.repeat_letter_cooldown - 1)

        if hand_found:
            self.frames_without_hand = 0
        else:
            self.frames_without_hand += 1
            self.letter_history.clear()
            if self.frames_without_hand >= self.runtime["frames_without_hand_to_finalize_word"]:
                self._finalize_word()
            if self.frames_without_hand >= self.runtime["frames_without_hand_to_finalize_sentence"]:
                self._finalize_sentence()

        letter_prediction = {"label": "", "confidence": 0.0, "top_k": []}
        if hand_found and self.letter_model is not None:
            letter_prediction = self.letter_model.predict(frame, bbox)
            self._consume_letter_prediction(letter_prediction)

        current_sentence = " ".join(self.current_sentence + ([self.current_word] if self.current_word else []))
        return {
            "bbox": bbox,
            "landmarks_found": hand_found,
            "current_word": self.current_word,
            "current_sentence": current_sentence,
            "history": list(self.completed_items),
            "latest_word": self._latest_text(),
            "letter_prediction": letter_prediction,
            "live_status": self._build_live_status(letter_prediction, hand_found),
            "models_ready": {"letters": self.letter_model is not None},
            "tts_enabled": self.runtime["text_to_speech_enabled"],
        }

    def _consume_letter_prediction(self, prediction: dict) -> None:
        if self._is_uncertain_prediction(prediction):
            return

        label = prediction["label"]
        self.letter_history.append(label)
        stable_letter = self._majority_vote(self.letter_history)
        stable_count = self.letter_history.count(stable_letter)

        if stable_count < self.runtime["stable_vote_min_count"]:
            return
        if self.repeat_letter_cooldown > 0 and stable_letter == self.last_committed_letter:
            return

        if stable_letter == "space":
            self._finalize_word()
        elif stable_letter == "del":
            self.current_word = self.current_word[:-1]
        elif stable_letter != "nothing":
            self.current_word += stable_letter.upper()

        self.last_committed_letter = stable_letter
        self.repeat_letter_cooldown = self.runtime["repeat_letter_cooldown_frames"]
        self.letter_history.clear()

    def _finalize_word(self) -> None:
        word = self.current_word.strip().upper()
        if not word:
            return

        self.current_sentence.append(word)
        self.current_word = ""

    def _finalize_sentence(self) -> None:
        sentence = " ".join(self.current_sentence).strip().upper()
        if sentence:
            self.completed_items.append(sentence)
            if self.runtime["text_to_speech_enabled"] and sentence != self.last_spoken_word:
                self.speech.speak(sentence)
                self.last_spoken_word = sentence
        self.current_word = ""
        self.current_sentence.clear()
        self.frames_without_hand = 0

    def _latest_text(self) -> str:
        if self.current_word:
            return self.current_word.upper()
        if self.completed_items:
            return self.completed_items[-1]
        return ""

    def _build_live_status(self, prediction: dict, hand_found: bool) -> str:
        if not hand_found:
            return "No hand detected"
        if not prediction["label"]:
            return "Scanning..."
        if self._is_uncertain_prediction(prediction):
            return "Uncertain"
        return prediction["label"].upper()

    def _is_uncertain_prediction(self, prediction: dict) -> bool:
        if prediction["confidence"] < self.runtime["letter_confidence_threshold"]:
            return True
        top_k = prediction.get("top_k", [])
        if len(top_k) >= 2:
            margin = top_k[0]["confidence"] - top_k[1]["confidence"]
            return margin < self.runtime["top2_margin_threshold"]
        return False

    @staticmethod
    def _majority_vote(values: deque[str]) -> str:
        if not values:
            return ""
        counts = Counter(values)
        return counts.most_common(1)[0][0]
