import torch

from modules.summarization import abstractive_summarizer as ab
from modules.summarization import model_cache as mc


class _FakeModel:
    def __init__(self):
        self.to_device = None
        self.eval_called = False

    def to(self, device):
        self.to_device = device
        return self

    def eval(self):
        self.eval_called = True
        return self

    def generate(self, *args, **kwargs):
        return torch.tensor([[7, 8, 9]])


def test_model_cache_reuses_same_components(monkeypatch):
    mc._MODEL_CACHE.clear()
    calls = {"tok": 0, "cfg": 0, "model": 0}

    class _FakeTokenizer:
        pass

    def _tok_loader(_name):
        calls["tok"] += 1
        return _FakeTokenizer()

    def _cfg_loader(_name):
        calls["cfg"] += 1
        return object()

    fake_model = _FakeModel()

    def _model_loader(_name):
        calls["model"] += 1
        return fake_model

    monkeypatch.setattr(mc.AutoTokenizer, "from_pretrained", _tok_loader)
    monkeypatch.setattr(mc.AutoConfig, "from_pretrained", _cfg_loader)
    monkeypatch.setattr(mc.AutoModelForSeq2SeqLM, "from_pretrained", _model_loader)
    monkeypatch.setattr(mc.torch.cuda, "is_available", lambda: False)

    a = mc.get_model_components("demo-model")
    b = mc.get_model_components("demo-model")

    assert a is b
    assert calls == {"tok": 1, "cfg": 1, "model": 1}
    assert fake_model.to_device == "cpu"
    assert fake_model.eval_called is True


def test_abstractive_summarizer_handles_empty_text():
    assert ab.summarize_text("   ") == ""


def test_abstractive_summarizer_with_mocked_components(monkeypatch):
    class _FakeTokenizer:
        model_max_length = 128

        def __call__(self, text, **kwargs):
            if kwargs.get("return_tensors") == "pt":
                return {
                    "input_ids": torch.tensor([[1, 2, 3, 4]]),
                    "attention_mask": torch.tensor([[1, 1, 1, 1]]),
                }
            return {"input_ids": [1, 2, 3, 4]}

        def decode(self, _ids, skip_special_tokens=True):
            return "decoded summary"

    fake_model = _FakeModel()

    monkeypatch.setattr(
        ab,
        "get_model_components",
        lambda: (_FakeTokenizer(), object(), fake_model, "cpu"),
    )

    output = ab.summarize_text("Some source text.")
    assert isinstance(output, str)
    assert output == "decoded summary"
