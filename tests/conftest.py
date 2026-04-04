import os
import sys
import types


# Keep tests deterministic and lightweight in environments without whisper.
if "whisper" not in sys.modules:
    whisper_stub = types.ModuleType("whisper")

    def _load_model(_name):
        class _FakeModel:
            def transcribe(self, _audio_path):
                return {"segments": []}

        return _FakeModel()

    whisper_stub.load_model = _load_model
    sys.modules["whisper"] = whisper_stub


# Ensure project root imports resolve regardless of how pytest is invoked.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
