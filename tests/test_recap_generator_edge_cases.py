from unittest.mock import MagicMock
from modules.summarization.recap_generator import generate_final_recap, _deduplicate_sentences

def test_generate_final_recap_short_text(monkeypatch):
    # Mock get_model_components
    mock_tokenizer = MagicMock()
    mock_tokenizer.model_max_length = 1024
    
    # For the `tokenizer(text, ...)` call to get truncated
    # Return a dict-like object for the inputs
    class MockInputs(dict):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.numel = lambda: 5
            self.shape = [1, 5]
        
        def to(self, device):
            return self

    mock_tokenizer.return_value = MockInputs(input_ids=MockInputs())
    mock_tokenizer.decode.return_value = "Mocked short recap result."
    
    mock_model = MagicMock()
    mock_model.generate.return_value = MagicMock(numel=lambda: 5)
    mock_model.generate.return_value.__getitem__.return_value = [1, 2, 3] # output_ids[0]

    def mock_get_model_components():
        return mock_tokenizer, None, mock_model, "cpu"
        
    monkeypatch.setattr("modules.summarization.recap_generator.get_model_components", mock_get_model_components)
    
    # Should not raise exception
    res = generate_final_recap("Short", max_length=100, min_length=40)
    assert res == "Mocked short recap result."
    assert len(res) > 0

def test_deduplicate_sentences_exact_duplicates():
    text = "Hello world. Hello world. How are you?"
    res = _deduplicate_sentences(text)
    assert res == "Hello world. How are you?"

def test_deduplicate_sentences_tautologies():
    text = "It is raining if it is raining. The sun is shining."
    res = _deduplicate_sentences(text)
    assert res == "The sun is shining."
