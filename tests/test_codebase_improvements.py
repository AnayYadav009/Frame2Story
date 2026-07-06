import pytest
import numpy as np
from modules.visual.key_frame_extractor import is_flat_frame, get_non_flat_frame
from modules.visual.motion_analyzer import compute_optical_flow_magnitude
from modules.visual.visual_analyzer import object_score, compute_importance_from_features


def test_is_flat_frame():
    # A completely black frame should be detected as flat
    black_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    assert is_flat_frame(black_frame) is True

    # A flat gray frame should be flat (low variance)
    gray_frame = np.ones((100, 100, 3), dtype=np.uint8) * 128
    assert is_flat_frame(gray_frame) is True

    # A high-contrast frame should NOT be flat
    high_contrast = np.zeros((100, 100, 3), dtype=np.uint8)
    high_contrast[:50, :, :] = 255
    assert is_flat_frame(high_contrast, std_threshold=10.0, mean_threshold=10.0) is False


def test_get_non_flat_frame():
    # Mock a reader that returns a flat frame at index 5, and a good frame at index 6
    flat_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    good_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    good_frame[:50, :, :] = 255

    class MockReader:
        def get_frame(self, idx):
            if idx == 5:
                return flat_frame
            if idx == 6:
                return good_frame
            return None

    reader = MockReader()
    # If we ask for index 5, it should find index 6 (since offset +1 is checked first)
    resolved_idx, frame = get_non_flat_frame(
        video_path="mock.mp4",
        target_idx=5,
        start_bound=0,
        end_bound=10,
        reader=reader
    )
    assert resolved_idx == 6
    assert np.array_equal(frame, good_frame)


def test_compute_optical_flow_magnitude():
    # Two identical frames should have 0 motion magnitude
    frame1 = np.zeros((100, 100, 3), dtype=np.uint8)
    frame1[30:70, 30:70, :] = 255
    frame2 = frame1.copy()

    mag_zero = compute_optical_flow_magnitude(frame1, frame2)
    assert mag_zero == pytest.approx(0.0, abs=1e-2)

    # Frame 2 shifted slightly should have non-zero motion magnitude
    frame3 = np.zeros((100, 100, 3), dtype=np.uint8)
    frame3[35:75, 35:75, :] = 255
    mag_motion = compute_optical_flow_magnitude(frame1, frame3)
    assert mag_motion > 0.0


def test_object_score_relevance_boost():
    # Without boost
    score_normal = object_score(["book"])
    assert score_normal == pytest.approx(0.3)

    # With boost (book is relevant)
    score_boosted = object_score(["book"], relevant_objects=["book"])
    assert score_boosted == pytest.approx(0.55)  # 0.3 + 0.25

    # Capped at 1.0
    score_capped = object_score(["gun", "weapon"], relevant_objects=["gun", "weapon"])
    assert score_capped == 1.0


def test_importance_fusion_relevance_boost():
    # Normal importance calculation
    imp_normal = compute_importance_from_features(
        motion_score=0.5,
        motion_level="MEDIUM",
        objects=["book"],
        duration=5.0,
        max_duration=10.0,
        relevant_objects=[]
    )

    # Boosted importance calculation
    imp_boosted = compute_importance_from_features(
        motion_score=0.5,
        motion_level="MEDIUM",
        objects=["book"],
        duration=5.0,
        max_duration=10.0,
        relevant_objects=["book"]
    )

    # Boosted importance should be higher because the object component score was boosted
    assert imp_boosted > imp_normal
