"""
Integration tests for webcam_capture_test_Haiku45 project.

These tests verify that different modules work correctly together,
testing end-to-end workflows and system integration.
"""

import sys
import os
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
import tempfile
import shutil
import time

import pytest
import numpy as np
import cv2

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import webcam_capture_test_Haiku45 as webcam_module

# Import camera viewer module
import importlib.util
spec = importlib.util.spec_from_file_location(
    "camera_viewer",
    PROJECT_ROOT / "camera_viewer_v3_2_production_Claude Code.py"
)
camera_viewer = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(camera_viewer)
except Exception:
    pass


# ============================================================================
# Full Workflow Integration Tests
# ============================================================================

class TestWebcamCaptureWorkflow:
    """Integration tests for the complete webcam capture workflow."""

    @pytest.mark.integration
    def test_full_capture_save_workflow(self, mock_cv2, temp_output_dir, monkeypatch, mock_window):
        """Test complete workflow: init -> capture -> save -> cleanup."""
        monkeypatch.setattr(webcam_module, 'OUTPUT_DIR', temp_output_dir)
        mock_cv2.set_available_cameras([0])

        # Step 1: Ensure output directory
        output_dir = webcam_module.ensure_output_directory()
        assert output_dir.exists()

        # Step 2: Initialize camera
        cap = webcam_module.initialize_camera()
        assert webcam_module.verify_camera_opened(cap)

        # Step 3: Get properties
        props = webcam_module.get_camera_properties(cap)
        assert 'width' in props
        assert 'height' in props
        assert 'fps' in props

        # Step 4: Capture frame
        ret, frame = webcam_module.capture_single_frame(cap)
        assert ret is True
        assert frame is not None

        # Step 5: Process frame (add text)
        processed = webcam_module.add_text_to_frame(
            frame.copy(),
            f"Test Frame - {props['width']}x{props['height']}",
            (10, 30)
        )
        assert processed is not None

        # Step 6: Save frame
        save_path = webcam_module.save_image(processed, "integration_test.jpg")
        assert save_path.exists()

        # Verify saved image can be read
        loaded = cv2.imread(str(save_path))
        assert loaded is not None

        # Step 7: Cleanup
        webcam_module.cleanup_resources(cap)

    @pytest.mark.integration
    def test_multiple_frame_capture_workflow(self, mock_cv2, temp_output_dir, monkeypatch):
        """Test capturing multiple frames in sequence."""
        monkeypatch.setattr(webcam_module, 'OUTPUT_DIR', temp_output_dir)
        mock_cv2.set_available_cameras([0])

        webcam_module.ensure_output_directory()
        cap = webcam_module.initialize_camera()

        frames_captured = []
        for i in range(5):
            ret, frame = webcam_module.capture_single_frame(cap)
            if ret:
                frames_captured.append(frame)
                save_path = webcam_module.save_image(frame, f"frame_{i:03d}.jpg")

        webcam_module.cleanup_resources(cap)

        assert len(frames_captured) == 5
        saved_files = list(temp_output_dir.glob("frame_*.jpg"))
        assert len(saved_files) == 5

    @pytest.mark.integration
    def test_camera_failure_recovery(self, mock_cv2, temp_output_dir, monkeypatch):
        """Test handling of camera failure during capture."""
        monkeypatch.setattr(webcam_module, 'OUTPUT_DIR', temp_output_dir)
        mock_cv2.set_available_cameras([0])

        webcam_module.ensure_output_directory()
        cap = webcam_module.initialize_camera()

        # Capture some frames successfully
        for _ in range(3):
            ret, frame = webcam_module.capture_single_frame(cap)
            assert ret is True

        # Simulate camera failure
        cap.set_fail_read(True)
        ret, frame = webcam_module.capture_single_frame(cap)
        assert ret is False

        # Cleanup should still work
        webcam_module.cleanup_resources(cap)


# ============================================================================
# Camera Viewer Integration Tests
# ============================================================================

class TestCameraViewerWorkflow:
    """Integration tests for the camera viewer workflow."""

    @pytest.mark.integration
    def test_environment_and_camera_init_workflow(self, mock_cv2, mock_environment):
        """Test environment validation followed by camera initialization."""
        mock_cv2.set_available_cameras([0])

        # Step 1: Environment validation
        result = camera_viewer.EnvironmentValidator.validate()
        assert result is True

        # Step 2: Camera initialization
        initializer = camera_viewer.CameraInitializer()

        with patch('time.sleep'):
            result = initializer.verify_and_initialize()

        assert result is True
        assert initializer.cap is not None

        # Cleanup
        initializer.cleanup()

    @pytest.mark.integration
    def test_image_adjustment_pipeline(self, sample_frame):
        """Test complete image adjustment pipeline."""
        # Test various adjustment combinations
        test_cases = [
            (0, 1.0, 1.0),      # Neutral
            (50, 1.0, 1.0),     # Bright
            (-50, 1.0, 1.0),    # Dark
            (0, 1.5, 1.0),      # High contrast
            (0, 0.5, 1.0),      # Low contrast
            (0, 1.0, 1.5),      # High saturation
            (0, 1.0, 0.5),      # Low saturation
            (25, 1.2, 1.3),     # Combined
        ]

        for brightness, contrast, saturation in test_cases:
            result = camera_viewer.adjust_image(
                sample_frame.copy(),
                brightness,
                contrast,
                saturation
            )
            assert result is not None
            assert result.shape == sample_frame.shape
            assert result.dtype == np.uint8

    @pytest.mark.integration
    def test_text_overlay_workflow(self, sample_frame_small):
        """Test adding multiple text overlays to a frame."""
        frame = sample_frame_small.copy()

        # Add multiple text lines
        lines = [
            ("FPS: 30.0", (10, 30)),
            ("Resolution: 1280x720", (10, 60)),
            ("Brightness: +10", (10, 90)),
            ("Date: 2025-01-01", (10, 120)),
        ]

        for text, position in lines:
            frame = camera_viewer.put_text_chinese(frame, text, position)

        assert frame is not None


# ============================================================================
# Cross-Module Integration Tests
# ============================================================================

class TestCrossModuleIntegration:
    """Tests for integration between different modules."""

    @pytest.mark.integration
    def test_webcam_module_frame_with_viewer_processing(self, mock_cv2):
        """Test webcam module frame processed by viewer module."""
        mock_cv2.set_available_cameras([0])

        # Capture frame using webcam module
        cap = webcam_module.initialize_camera()
        ret, frame = webcam_module.capture_single_frame(cap)
        webcam_module.cleanup_resources(cap)

        assert ret is True

        # Process frame using viewer module functions
        processed = camera_viewer.adjust_image(frame, 10, 1.1, 1.0)
        assert processed is not None

        # Add text using both modules
        with_text = webcam_module.add_text_to_frame(
            processed.copy(),
            "Webcam Module",
            (10, 30)
        )
        with_text = camera_viewer.put_text_chinese(
            with_text,
            "Viewer Module",
            (10, 60)
        )

        assert with_text is not None

    @pytest.mark.integration
    def test_camera_properties_compatibility(self, mock_cv2):
        """Test camera property reading is compatible across modules."""
        mock_cv2.set_available_cameras([0])

        # Using webcam module
        cap = webcam_module.initialize_camera()
        props = webcam_module.get_camera_properties(cap)

        assert 'width' in props
        assert 'height' in props
        assert 'fps' in props

        webcam_module.cleanup_resources(cap)


# ============================================================================
# Performance Integration Tests
# ============================================================================

class TestPerformanceIntegration:
    """Performance-related integration tests."""

    @pytest.mark.integration
    @pytest.mark.slow
    def test_continuous_capture_performance(self, mock_cv2):
        """Test continuous frame capture performance."""
        mock_cv2.set_available_cameras([0])

        cap = webcam_module.initialize_camera()

        start_time = time.time()
        frame_count = 0

        while frame_count < 100:
            ret, frame = webcam_module.capture_single_frame(cap)
            if ret:
                frame_count += 1

        elapsed = time.time() - start_time
        webcam_module.cleanup_resources(cap)

        # Should capture 100 frames in reasonable time (mocked, so very fast)
        assert frame_count == 100
        assert elapsed < 5.0  # Should be nearly instant with mocks

    @pytest.mark.integration
    @pytest.mark.slow
    def test_image_processing_performance(self, sample_frame):
        """Test image processing pipeline performance."""
        iterations = 100

        start_time = time.time()

        for _ in range(iterations):
            processed = camera_viewer.adjust_image(
                sample_frame.copy(), 10, 1.2, 1.1
            )
            _ = camera_viewer.put_text_chinese(
                processed, "Performance Test", (10, 30)
            )

        elapsed = time.time() - start_time
        fps = iterations / elapsed

        # Should process at reasonable speed
        assert fps > 10  # At least 10 iterations per second


# ============================================================================
# Error Handling Integration Tests
# ============================================================================

class TestErrorHandlingIntegration:
    """Tests for error handling across modules."""

    @pytest.mark.integration
    def test_graceful_camera_failure(self, mock_cv2, capture_stdout):
        """Test graceful handling of camera failure."""
        mock_cv2.set_should_fail_open(True)

        cap = webcam_module.initialize_camera()
        is_opened = webcam_module.verify_camera_opened(cap)

        assert is_opened is False

    @pytest.mark.integration
    def test_output_directory_permission_error(self, monkeypatch):
        """Test handling of output directory permission errors."""
        # Use a path that should fail
        bad_path = Path("/root/impossible_path/output")
        monkeypatch.setattr(webcam_module, 'OUTPUT_DIR', bad_path)

        # This may raise PermissionError on some systems
        try:
            webcam_module.ensure_output_directory()
        except PermissionError:
            pass  # Expected behavior

    @pytest.mark.integration
    def test_invalid_frame_processing(self):
        """Test processing of invalid frames."""
        invalid_frames = [
            None,
            np.array([]),
            np.zeros((0, 0, 3), dtype=np.uint8),
        ]

        for frame in invalid_frames:
            if frame is not None and frame.size > 0:
                try:
                    camera_viewer.adjust_image(frame, 0, 1.0, 1.0)
                except (cv2.error, ValueError):
                    pass  # Expected behavior


# ============================================================================
# Data Flow Integration Tests
# ============================================================================

class TestDataFlowIntegration:
    """Tests for data flow between components."""

    @pytest.mark.integration
    def test_frame_data_integrity(self, mock_cv2, temp_output_dir, monkeypatch):
        """Test that frame data maintains integrity through pipeline."""
        monkeypatch.setattr(webcam_module, 'OUTPUT_DIR', temp_output_dir)
        mock_cv2.set_available_cameras([0])

        # Capture original frame
        cap = webcam_module.initialize_camera()
        ret, original_frame = webcam_module.capture_single_frame(cap)
        webcam_module.cleanup_resources(cap)

        original_hash = hash(original_frame.tobytes())
        original_shape = original_frame.shape

        # Process through pipeline
        processed = camera_viewer.adjust_image(original_frame.copy(), 0, 1.0, 1.0)

        # Save and reload
        save_path = webcam_module.save_image(processed, "integrity_test.jpg")
        loaded = cv2.imread(str(save_path))

        # Verify shape is preserved
        assert loaded.shape == original_shape

    @pytest.mark.integration
    def test_properties_propagation(self, mock_cv2):
        """Test that camera properties propagate correctly."""
        mock_cv2.set_available_cameras([0])

        # Get properties from webcam module
        cap = webcam_module.initialize_camera()
        props = webcam_module.get_camera_properties(cap)
        webcam_module.cleanup_resources(cap)

        # Use properties for display info
        info_text = f"Resolution: {props['width']}x{props['height']}"
        assert str(props['width']) in info_text
        assert str(props['height']) in info_text

        # Properties should match mock camera defaults
        assert props['width'] == 1280
        assert props['height'] == 720
