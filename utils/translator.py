from deep_translator import GoogleTranslator

# Map user-friendly language names to ISO codes supported by deep-translator
SUPPORTED_LANGUAGES = {
    "English": "en",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Italian": "it",
    "Portuguese": "pt",
    "Hindi": "hi",
    "Bengali": "bn",
    "Chinese (Simplified)": "zh-CN",
    "Japanese": "ja",
    "Korean": "ko"
}

def translate_text(text: str, target_lang_name: str) -> str:
    """Translate text to the specified target language name.
    
    Falls back to original text if translation fails or language is not supported.
    """
    if not text or not text.strip():
        return ""
    
    lang = (target_lang_name or "English").strip()
    target_code = SUPPORTED_LANGUAGES.get(lang)
    
    if not target_code or target_code == "en":
        return text.strip()
        
    try:
        # GoogleTranslator is free and does not require credentials/API keys.
        translated = GoogleTranslator(source="auto", target=target_code).translate(text)
        return translated.strip() if translated else text.strip()
    except Exception as e:
        # Fail-safe fallback to original text (e.g., offline or network error)
        print(f"Translation failed: {e}")
        return text.strip()
