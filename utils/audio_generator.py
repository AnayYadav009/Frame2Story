from pathlib import Path
from gtts import gTTS
from utils.translator import SUPPORTED_LANGUAGES

def generate_tts_narration(
    text: str,
    target_lang_name: str,
    output_path: str = "outputs/final/recap_narration.mp3"
) -> str:
    """Generate a Text-to-Speech MP3 audio file from the text.
    
    Args:
        text: Narrative text to synthesize.
        target_lang_name: Language of the text, matching languages in translator.py.
        output_path: Destination path for the audio narration file.
        
    Returns:
        The string path of the generated audio file.
    """
    if not text or not text.strip():
        raise ValueError("Cannot generate audio narration for empty text")
        
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Default to "en" if language is not supported or not found
    lang_code = SUPPORTED_LANGUAGES.get(target_lang_name, "en")
    
    # gTTS speaks Chinese using 'zh-CN' which matches our translator list
    tts = gTTS(text=text, lang=lang_code, slow=False)
    tts.save(str(out_path))
    
    return str(out_path)
