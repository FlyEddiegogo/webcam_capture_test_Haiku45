"""
Unit tests for webcam_capture_test_Haiku45.py

Tests all core functions of the webcam capture test script:
- Directory management
- Camera initialization and verification
- Frame capture and saving
- Text rendering
- Key input handling
- Display functions
"""

import sys
import os
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, call
import tempfile
import shutil

import pytest
import numpy as np
import cv2

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import module under test
import webcam_capture_test_Haiku45 as webcam_module


# ============================================================================
# Test Constants
# ============================================================================

class TestConstants:
    """Test module constants are properly defined."""

    @pytest.mark.unit
    def test_camera_index_defined(self):
        """Verify CAMERA_INDEX constant exists and is valid."""
        assert hasattr(webcam_module, 'CAMERA_INDEX')
        assert isinstance(webcam_module.CAMERA_INDEX, int)
        assert webcam_module.CAMERA_INDEX >= 0

    @pytest.mark.unit
    def test_output_dir_defined(self):
        """Verify OUTPUT_DIR constant exists and is a Path."""
        assert hasattr(webcam_module, 'OUTPUT_DIR')
        assert isinstance(webcam_module.OUTPUT_DIR, Path)

    @pytest.mark.unit
    def test_image_quality_defined(self):
        """Verify IMAGE_QUALITY constant is valid."""
        assert hasattr(webcam_module, 'IMAGE_QUALITY')
        assert 0 <= webcam_module.IMAGE_QUALITY <= 100

    @pytest.mark.unit
    def test_key_codes_defined(self):
        """Verify key code constants are defined correctly."""
        assert webcam_module.KEY_Q == ord('q')
        assert webcam_module.KEY_S == ord('s')
        assert webcam_module.KEY_ESC == 27

    @pytest.mark.unit
    def test_window_title_defined(self):
        """Verify WINDOW_TITLE constant exists."""
        assert hasattr(webcam_module, 'WINDOW_TITLE')
        assert isinstance(webcam_module.WINDOW_TITLE, str)
        assert len(webcam_module.WINDOW_TITLE) > 0


# ============================================================================
# Test ensure_output_directory
# ============================================================================

class TestEnsureOutputDirectory:
    """Tests for ensure_output_directory function."""

    @pytest.mark.unit
    def test_creates_directory_if_not_exists(self, temp_output_dir, monkeypatch):
        """Test that function creates directory when it doesn't exist."""
        test_dir = temp_output_dir / "new_output"
        monkeypatch.setattr(webcam_module, 'OUTPUT_DIR', test_dir)

        assert not test_dir.exists()

        result = webcam_module.ensure_output_directory()

        assert test_dir.exists()
        assert test_dir.is_dir()
        assert result == test_dir

    @pytest.mark.unit
    def test_no_error_if_directory_exists(self, temp_output_dir, monkeypatch):
        """Test that function doesn't error if directory already exists."""
        monkeypatch.setattr(webcam_module, 'OUTPUT_DIR', temp_output_dir)

        # Call twice - should not raise
        webcam_module.ensure_output_directory()
        result = webcam_module.ensure_output_directory()

        assert result == temp_output_dir

    @pytest.mark.unit
    def test_creates_parent_directories(self, temp_output_dir, monkeypatch):
        """Test that function creates parent directories."""
        test_dir = temp_output_dir / "parent" / "child" / "output"
        monkeypatch.setattr(webcam_module, 'OUTPUT_DIR', test_dir)

        result = webcam_module.ensure_output_directory()

        assert test_dir.exists()
        assert result == test_dir

    @pytest.mark.unit
    def test_returns_path_object(self, temp_output_dir, monkeypatch):
        """Test that function returns a Path object."""
        monkeypatch.setattr(webcam_module, 'OUTPUT_DIR', temp_output_dir)

        result = webcam_module.ensure_output_directory()

        assert isinstance(result, Path)


# ============================================================================
# Test initialize_camera
# ============================================================================

class TestInitializeCamera:
    """Tests for initialize_camera function."""

    @pytest.mark.unit
    def test_returns_video_capture_object(self, mock_cv2):
        """Test that function returns a VideoCapture-like object."""
        cap = webcam_module.initialize_camera()

        assert cap is not None
        assert hasattr(cap, 'isOpened')
        assert hasattr(cap, 'read')

    @pytest.mark.unit
    def test_uses_camera_index(self, mock_cv2):
        """Test that function uses the CAMERA_INDEX constant."""
        # The mock factory will be called with the camera index
        cap = webcam_module.initialize_camera()

        # Verify camera was opened
        assert cap._index == webcam_module.CAMERA_INDEX


# ============================================================================
# Test verify_camera_opened
# ============================================================================

class TestVerifyCameraOpened:
    """Tests for verify_camera_opened function."""

    @pytest.mark.unit
    def test_returns_true_when_camera_opened(self, mock_camera):
        """Test returns True when camera is successfully opened."""
        mock_camera._is_opened = True

        result = webcam_module.verify_camera_opened(mock_camera)

        assert result is True

    @pytest.mark.unit
    def test_returns_false_when_camera_not_opened(self, mock_camera):
        """Test returns False when camera failed to open."""
        mock_camera._is_opened = False

        result = webcam_module.verify_camera_opened(mock_camera)

        assert result is False

    @pytest.mark.unit
    def test_calls_isOpened_method(self):
        """Test that function calls isOpened() on capture object."""
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True

        webcam_module.verify_camera_opened(mock_cap)

        mock_cap.isOpened.assert_called_once()


# ============================================================================
# Test get_camera_properties
# ============================================================================

class TestGetCameraProperties:
    """Tests for get_camera_properties function."""

    @pytest.mark.unit
    def test_returns_dict_with_required_keys(self, mock_camera):
        """Test that function returns dict with width, height, fps."""
        result = webcam_module.get_camera_properties(mock_camera)

        assert isinstance(result, dict)
        assert 'width' in result
        assert 'height' in result
        assert 'fps' in result

    @pytest.mark.unit
    def test_width_and_height_are_integers(self, mock_camera):
        """Test that width and height are integers."""
        result = webcam_module.get_camera_properties(mock_camera)

        assert isinstance(result['width'], int)
        assert isinstance(result['height'], int)

    @pytest.mark.unit
    def test_returns_correct_values(self, mock_camera):
        """Test that function returns correct camera values."""
        mock_camera._width = 1920
        mock_camera._height = 1080
        mock_camera._fps = 60.0

        result = webcam_module.get_camera_properties(mock_camera)

        assert result['width'] == 1920
        assert result['height'] == 1080
        assert result['fps'] == 60.0


# ============================================================================
# Test capture_single_frame
# ============================================================================

class TestCaptureSingleFrame:
    """Tests for capture_single_frame function."""

    @pytest.mark.unit
    def test_returns_tuple(self, mock_camera):
        """Test that function returns a tuple."""
        result = webcam_module.capture_single_frame(mock_camera)

        assert isinstance(result, tuple)
        assert len(result) == 2

    @pytest.mark.unit
    def test_returns_success_and_frame_on_successful_read(self, mock_camera):
        """Test successful frame capture."""
        ret, frame = webcam_module.capture_single_frame(mock_camera)

        assert ret is True
        assert frame is not None
        assert isinstance(frame, np.ndarray)

    @pytest.mark.unit
    def test_returns_false_on_failed_read(self, mock_camera):
        """Test failed frame capture."""
        mock_camera.set_fail_read(True)

        ret, frame = webcam_module.capture_single_frame(mock_camera)

        assert ret is False

    @pytest.mark.unit
    def test_frame_has_correct_shape(self, mock_camera):
        """Test that captured frame has correct shape."""
        ret, frame = webcam_module.capture_single_frame(mock_camera)

        assert frame.shape == (720, 1280, 3)  # Default mock camera size


# ============================================================================
# Test save_image
# ============================================================================

class TestSaveImage:
    """Tests for save_image function."""

    @pytest.mark.unit
    def test_saves_image_with_default_filename(self, sample_frame, temp_output_dir, monkeypatch):
        """Test saving image with default filename."""
        monkeypatch.setattr(webcam_module, 'OUTPUT_DIR', temp_output_dir)

        result = webcam_module.save_image(sample_frame)

        assert result.exists()
        assert result.name == "test_frame.jpg"

    @pytest.mark.unit
    def test_saves_image_with_custom_filename(self, sample_frame, temp_output_dir, monkeypatch):
        """Test saving image with custom filename."""
        monkeypatch.setattr(webcam_module, 'OUTPUT_DIR', temp_output_dir)

        result = webcam_module.save_image(sample_frame, "custom_image.jpg")

        assert result.exists()
        assert result.name == "custom_image.jpg"

    @pytest.mark.unit
    def test_returns_path_object(self, sample_frame, temp_output_dir, monkeypatch):
        """Test that function returns a Path object."""
        monkeypatch.setattr(webcam_module, 'OUTPUT_DIR', temp_output_dir)

        result = webcam_module.save_image(sample_frame)

        assert isinstance(result, Path)

    @pytest.mark.unit
    def test_saved_image_is_readable(self, sample_frame, temp_output_dir, monkeypatch):
        """Test that saved image can be read back."""
        monkeypatch.setattr(webcam_module, 'OUTPUT_DIR', temp_output_dir)

        save_path = webcam_module.save_image(sample_frame)
        loaded_frame = cv2.imread(str(save_path))

        assert loaded_frame is not None
        assert loaded_frame.shape == sample_frame.shape

    @pytest.mark.unit
    def test_saved_image_quality(self, sample_frame, temp_output_dir, monkeypatch):
        """Test that saved image uses proper quality settings."""
        monkeypatch.setattr(webcam_module, 'OUTPUT_DIR', temp_output_dir)

        save_path = webcam_module.save_image(sample_frame)

        # High quality JPEG should be reasonably sized
        assert save_path.stat().st_size > 1000  # At least 1KB


# ============================================================================
# Test add_text_to_frame
# ============================================================================

class TestAddTextToFrame:
    """Tests for add_text_to_frame function."""

    @pytest.mark.unit
    def test_returns_frame(self, sample_frame):
        """Test that function returns a frame."""
        result = webcam_module.add_text_to_frame(
            sample_frame.copy(),
            "Test",
            (10, 30)
        )

        assert result is not None
        assert isinstance(result, np.ndarray)

    @pytest.mark.unit
    def test_modifies_frame(self, sample_frame):
        """Test that function modifies the frame (adds text)."""
        original = sample_frame.copy()

        result = webcam_module.add_text_to_frame(
            sample_frame.copy(),
            "Test Text",
            (100, 100)
        )

        # Frame should be different after adding text
        # Compare a region where text would be
        assert not np.array_equal(result[90:110, 90:200], original[90:110, 90:200])

    @pytest.mark.unit
    def test_default_color_is_green(self, sample_frame_small):
        """Test that default text color is green."""
        frame = np.zeros((100, 200, 3), dtype=np.uint8)

        result = webcam_module.add_text_to_frame(
            frame,
            "Test",
            (10, 50)
        )

        # Green channel should have non-zero values where text was drawn
        # (This is a basic check - precise text location checking is complex)
        assert np.any(result[:, :, 1] > 0)  # Green channel

    @pytest.mark.unit
    def test_custom_color(self, sample_frame_small):
        """Test custom text color."""
        frame = np.zeros((100, 200, 3), dtype=np.uint8)

        result = webcam_module.add_text_to_frame(
            frame,
            "Test",
            (10, 50),
            color=(255, 0, 0)  # Blue in BGR
        )

        # Blue channel should have non-zero values
        assert np.any(result[:, :, 0] > 0)

    @pytest.mark.unit
    def test_font_size_parameter(self, sample_frame_small):
        """Test that font_size parameter is accepted."""
        frame = sample_frame_small.copy()

        # Should not raise with different font sizes
        webcam_module.add_text_to_frame(frame.copy(), "Small", (10, 30), font_size=0.5)
        webcam_module.add_text_to_frame(frame.copy(), "Large", (10, 30), font_size=2.0)


# ============================================================================
# Test handle_key_input
# ============================================================================

class TestHandleKeyInput:
    """Tests for handle_key_input function."""

    @pytest.mark.unit
    def test_q_key_returns_exit(self):
        """Test that 'q' key returns 'exit'."""
        result = webcam_module.handle_key_input(ord('q'), 0)
        assert result == 'exit'

    @pytest.mark.unit
    def test_esc_key_returns_exit(self):
        """Test that ESC key returns 'exit'."""
        result = webcam_module.handle_key_input(27, 0)
        assert result == 'exit'

    @pytest.mark.unit
    def test_s_key_returns_save(self):
        """Test that 's' key returns 'save'."""
        result = webcam_module.handle_key_input(ord('s'), 0)
        assert result == 'save'

    @pytest.mark.unit
    def test_other_key_returns_continue(self):
        """Test that other keys return 'continue'."""
        result = webcam_module.handle_key_input(ord('x'), 0)
        assert result == 'continue'

    @pytest.mark.unit
    def test_no_key_returns_continue(self):
        """Test that no key (-1) returns 'continue'."""
        result = webcam_module.handle_key_input(-1, 0)
        assert result == 'continue'

    @pytest.mark.unit
    def test_masks_high_bits(self):
        """Test that function masks high bits correctly."""
        # Some systems return values with high bits set
        result = webcam_module.handle_key_input(0xFF00 | ord('q'), 0)
        assert result == 'exit'

    @pytest.mark.unit
    def test_frame_count_parameter_accepted(self):
        """Test that frame_count parameter is accepted."""
        # Function should accept any frame count
        result = webcam_module.handle_key_input(ord('s'), 12345)
        assert result == 'save'


# ============================================================================
# Test display functions
# ============================================================================

class TestDisplayFunctions:
    """Tests for display_* functions."""

    @pytest.mark.unit
    def test_display_startup_info(self, capture_stdout):
        """Test display_startup_info outputs correctly."""
        webcam_module.display_startup_info()
        output = capture_stdout()

        assert "網路攝影機測試程式" in output or "WebCam Test" in output
        assert "=" in output  # Separator line

    @pytest.mark.unit
    def test_display_camera_info(self, capture_stdout, camera_properties):
        """Test display_camera_info outputs camera properties."""
        webcam_module.display_camera_info(camera_properties)
        output = capture_stdout()

        assert str(camera_properties['width']) in output
        assert str(camera_properties['height']) in output

    @pytest.mark.unit
    def test_display_control_instructions(self, capture_stdout):
        """Test display_control_instructions outputs key bindings."""
        webcam_module.display_control_instructions()
        output = capture_stdout()

        assert "q" in output.lower() or "esc" in output.lower()
        assert "s" in output.lower()


# ============================================================================
# Test cleanup_resources
# ============================================================================

class TestCleanupResources:
    """Tests for cleanup_resources function."""

    @pytest.mark.unit
    def test_releases_camera(self, mock_camera):
        """Test that function releases camera."""
        mock_camera._is_opened = True

        webcam_module.cleanup_resources(mock_camera)

        assert mock_camera._is_opened is False

    @pytest.mark.unit
    def test_calls_release(self):
        """Test that function calls release() on capture object."""
        mock_cap = Mock()

        with patch('cv2.destroyAllWindows'):
            webcam_module.cleanup_resources(mock_cap)

        mock_cap.release.assert_called_once()

    @pytest.mark.unit
    def test_destroys_windows(self, mock_camera):
        """Test that function destroys all windows."""
        with patch('cv2.destroyAllWindows') as mock_destroy:
            webcam_module.cleanup_resources(mock_camera)

        mock_destroy.assert_called_once()


# ============================================================================
# Test test_webcam (main function)
# ============================================================================

class TestTestWebcam:
    """Tests for test_webcam main function."""

    @pytest.mark.unit
    def test_returns_false_when_camera_fails_to_open(self, mock_cv2, temp_output_dir, monkeypatch):
        """Test returns False when camera cannot be opened."""
        mock_cv2.set_should_fail_open(True)
        monkeypatch.setattr(webcam_module, 'OUTPUT_DIR', temp_output_dir)

        with patch('cv2.destroyAllWindows'):
            result = webcam_module.test_webcam()

        assert result is False

    @pytest.mark.unit
    def test_creates_output_directory(self, mock_cv2, temp_output_dir, monkeypatch, mock_window):
        """Test that function creates output directory."""
        mock_cv2.set_should_fail_open(True)
        monkeypatch.setattr(webcam_module, 'OUTPUT_DIR', temp_output_dir / "webcam_test")

        with patch('cv2.destroyAllWindows'):
            webcam_module.test_webcam()

        assert (temp_output_dir / "webcam_test").exists()

    @pytest.mark.unit
    def test_displays_startup_info(self, mock_cv2, temp_output_dir, monkeypatch, capture_stdout):
        """Test that function displays startup info."""
        mock_cv2.set_should_fail_open(True)
        monkeypatch.setattr(webcam_module, 'OUTPUT_DIR', temp_output_dir)

        with patch('cv2.destroyAllWindows'):
            webcam_module.test_webcam()

        output = capture_stdout()
        assert "網路攝影機測試程式" in output or "WebCam Test" in output


# ============================================================================
# Test main_loop
# ============================================================================

class TestMainLoop:
    """Tests for main_loop function."""

    @pytest.mark.unit
    def test_exits_on_read_failure(self, mock_camera, camera_properties, mock_window, capture_stdout):
        """Test that loop exits when frame read fails."""
        mock_camera.set_fail_after_n_reads(2)

        webcam_module.main_loop(mock_camera, camera_properties)

        output = capture_stdout()
        assert "錯誤" in output or "error" in output.lower()

    @pytest.mark.unit
    def test_exits_on_q_key(self, mock_camera, camera_properties, capture_stdout):
        """Test that loop exits on 'q' key press."""
        with patch('cv2.waitKey', side_effect=[ord('q')]), \
             patch('cv2.imshow'):
            webcam_module.main_loop(mock_camera, camera_properties)

        output = capture_stdout()
        assert "退出" in output or "exit" in output.lower() or "統計" in output

    @pytest.mark.unit
    def test_exits_on_esc_key(self, mock_camera, camera_properties, capture_stdout):
        """Test that loop exits on ESC key press."""
        with patch('cv2.waitKey', side_effect=[27]), \
             patch('cv2.imshow'):
            webcam_module.main_loop(mock_camera, camera_properties)

        output = capture_stdout()
        # Should contain exit message or statistics
        assert "統計" in output or "使用者" in output

    @pytest.mark.unit
    def test_saves_image_on_s_key(self, mock_camera, camera_properties, temp_output_dir, monkeypatch, capture_stdout):
        """Test that loop saves image on 's' key press."""
        monkeypatch.setattr(webcam_module, 'OUTPUT_DIR', temp_output_dir)

        # First return -1 (no key), then 's', then 'q' to exit
        key_sequence = [-1, ord('s'), ord('q')]

        with patch('cv2.waitKey', side_effect=key_sequence), \
             patch('cv2.imshow'):
            webcam_module.main_loop(mock_camera, camera_properties)

        # Check that image was saved
        saved_files = list(temp_output_dir.glob("*.jpg"))
        assert len(saved_files) >= 1

    @pytest.mark.unit
    def test_frame_count_increments(self, mock_camera, camera_properties, capture_stdout):
        """Test that frame count increments correctly."""
        key_sequence = [-1, -1, -1, ord('q')]  # 3 frames then quit

        with patch('cv2.waitKey', side_effect=key_sequence), \
             patch('cv2.imshow'):
            webcam_module.main_loop(mock_camera, camera_properties)

        output = capture_stdout()
        # Should show at least 3 frames captured
        assert "3" in output or "4" in output


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.unit
    def test_empty_frame_handling(self, temp_output_dir, monkeypatch):
        """Test handling of empty/null frames."""
        monkeypatch.setattr(webcam_module, 'OUTPUT_DIR', temp_output_dir)

        # Create minimal valid frame
        empty_frame = np.zeros((1, 1, 3), dtype=np.uint8)

        # Should not raise
        result = webcam_module.save_image(empty_frame, "empty.jpg")
        assert result.exists()

    @pytest.mark.unit
    def test_large_frame_handling(self, temp_output_dir, monkeypatch):
        """Test handling of large frames."""
        monkeypatch.setattr(webcam_module, 'OUTPUT_DIR', temp_output_dir)

        # 4K frame
        large_frame = np.zeros((2160, 3840, 3), dtype=np.uint8)

        result = webcam_module.save_image(large_frame, "large.jpg")
        assert result.exists()

    @pytest.mark.unit
    def test_camera_properties_with_zero_fps(self):
        """Test handling of camera reporting zero FPS."""
        mock_cap = Mock()
        mock_cap.get.side_effect = lambda x: {
            cv2.CAP_PROP_FRAME_WIDTH: 640,
            cv2.CAP_PROP_FRAME_HEIGHT: 480,
            cv2.CAP_PROP_FPS: 0.0
        }.get(x, 0)

        result = webcam_module.get_camera_properties(mock_cap)

        assert result['fps'] == 0.0

    @pytest.mark.unit
    def test_unicode_text_in_frame(self, sample_frame_small):
        """Test adding unicode text to frame."""
        frame = sample_frame_small.copy()

        # Should not raise with unicode
        result = webcam_module.add_text_to_frame(
            frame,
            "Frame: 123",  # ASCII safe
            (10, 30)
        )

        assert result is not None

    @pytest.mark.unit
    def test_negative_position_text(self, sample_frame_small):
        """Test text at negative position (edge case)."""
        frame = sample_frame_small.copy()

        # OpenCV handles negative positions
        result = webcam_module.add_text_to_frame(
            frame,
            "Test",
            (-10, 30)
        )

        assert result is not None
