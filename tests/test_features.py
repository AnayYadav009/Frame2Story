import pytest
from pathlib import Path
from unittest.mock import MagicMock
from utils.translator import translate_text
from utils.audio_generator import generate_tts_narration
from utils.speech_to_text import transcribe_audio

def test_translator_basic():
    # Test empty text
    assert translate_text("", "Spanish") == ""
    assert translate_text("  ", "Spanish") == ""
    
    # Test English returns original text without API call
    assert translate_text("Hello", "English") == "Hello"
    
    # Test invalid target language returns original text
    assert translate_text("Hello", "InvalidLanguage") == "Hello"

def test_translator_with_mocked_api(monkeypatch):
    class FakeTranslator:
        def __init__(self, source, target):
            self.source = source
            self.target = target
            
        def translate(self, text):
            return f"translated_{self.target}_{text}"
            
    monkeypatch.setattr("utils.translator.GoogleTranslator", FakeTranslator)
    
    result = translate_text("Hello", "Spanish")
    assert result == "translated_es_Hello"

def test_translator_failsafe_fallback(monkeypatch):
    def fake_translate(*args, **kwargs):
        raise RuntimeError("API rate limit exceeded or offline")
        
    monkeypatch.setattr("utils.translator.GoogleTranslator", lambda **k: MagicMock(translate=fake_translate))
    
    # Should fallback gracefully to original text
    result = translate_text("Hello world", "Spanish")
    assert result == "Hello world"

def test_audio_generator_raises_on_empty():
    with pytest.raises(ValueError):
        generate_tts_narration("", "Spanish")

def test_audio_generator_with_mocked_gtts(tmp_path, monkeypatch):
    written_file = tmp_path / "audio.mp3"
    
    saved_text = []
    saved_lang = []
    
    class FakeGTTS:
        def __init__(self, text, lang, slow=False):
            saved_text.append(text)
            saved_lang.append(lang)
            
        def save(self, path):
            Path(path).write_bytes(b"mock_mp3_data")
            
    monkeypatch.setattr("utils.audio_generator.gTTS", FakeGTTS)
    
    result_path = generate_tts_narration("My recap", "Spanish", output_path=str(written_file))
    
    assert result_path == str(written_file)
    assert written_file.exists()
    assert written_file.read_bytes() == b"mock_mp3_data"
    assert saved_text == ["My recap"]
    assert saved_lang == ["es"]

def test_speech_to_text_with_mocked_faster_whisper(tmp_path, monkeypatch):
    class MockSegment:
        def __init__(self, start, end, text):
            self.start = start
            self.end = end
            self.text = text

    class MockModel:
        def __init__(self, *args, **kwargs):
            pass
            
        def transcribe(self, audio_path, **kwargs):
            return [
                MockSegment(1.5, 3.0, "Hello"),
                MockSegment(3.5, 6.0, "World")
            ], object()
            
    monkeypatch.setattr("utils.speech_to_text.WhisperModel", MockModel)
    
    audio_file = tmp_path / "test.wav"
    audio_file.write_bytes(b"wav")
    
    output_srt = tmp_path / "subtitles.srt"
    
    res = transcribe_audio(str(audio_file), output_srt=str(output_srt))
    
    assert res == str(output_srt)
    assert output_srt.exists()
    
    content = output_srt.read_text(encoding="utf-8")
    assert "00:00:01,500 --> 00:00:03,000" in content
    assert "Hello" in content
    assert "00:00:03,500 --> 00:00:06,000" in content
    assert "World" in content
