import modules.visual.object_detector
from modules.visual.object_detector import get_model

def test_object_detector_cache(monkeypatch):
    class DummyYOLO:
        def __init__(self, name):
            self.name = name

    monkeypatch.setattr(modules.visual.object_detector, "YOLO", DummyYOLO)
    
    # Reset cache for testing
    modules.visual.object_detector._MODELS = {}

    model1 = get_model("model1.pt")
    model2 = get_model("model2.pt")

    assert model1 is not model2
    assert model1.name == "model1.pt"
    assert model2.name == "model2.pt"
    
    # Assert cache has two entries
    assert len(modules.visual.object_detector._MODELS) == 2
    assert "model1.pt" in modules.visual.object_detector._MODELS
    assert "model2.pt" in modules.visual.object_detector._MODELS
    
    # Assert same model name returns same object
    model1_again = get_model("model1.pt")
    assert model1 is model1_again
