from collections import deque
from threading import Lock, Thread

import pyttsx3


class SpeechEngine:
    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self._queue: deque[str] = deque()
        self._lock = Lock()
        self._worker: Thread | None = None

    def speak(self, text: str) -> None:
        if not self.enabled or not text.strip():
            return
        with self._lock:
            self._queue.append(text)
            if self._worker is None or not self._worker.is_alive():
                self._worker = Thread(target=self._run, daemon=True)
                self._worker.start()

    def _run(self) -> None:
        engine = pyttsx3.init()
        engine.setProperty("rate", 170)
        while True:
            with self._lock:
                if not self._queue:
                    break
                text = self._queue.popleft()
            engine.say(text)
            engine.runAndWait()
        engine.stop()
