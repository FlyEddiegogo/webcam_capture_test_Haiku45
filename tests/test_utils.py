"""
Test utilities for webcam_capture_test_Haiku45 project.

This module provides helper functions and utilities for testing:
- Frame generation
- Image comparison
- Validation helpers
- Mock data generators
"""

import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock

import pytest
import numpy as np
import cv2

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================================
# Frame Generation Utilities
# ============================================================================

class FrameGenerator:
    """Utility class for generating test frames."""

    @staticmethod
    def solid_color(width=640, height=480, color=(128, 128, 128)):
        """
        Generate a solid color frame.

        Args:
            width: Frame width in pixels
            height: Frame height in pixels
            color: BGR color tuple

        Returns:
            numpy.ndarray: Solid color frame
        """
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:] = color
        return frame

    @staticmethod
    def gradient_horizontal(width=640, height=480):
        """
        Generate a horizontal gradient frame.

        Args:
            width: Frame width in pixels
            height: Frame height in pixels

        Returns:
            numpy.ndarray: Horizontal gradient frame
        """
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        for j in range(width):
            value = int(255 * j / width)
            frame[:, j] = [value, value, value]
        return frame

    @staticmethod
    def gradient_vertical(width=640, height=480):
        """
        Generate a vertical gradient frame.

        Args:
            width: Frame width in pixels
            height: Frame height in pixels

        Returns:
            numpy.ndarray: Vertical gradient frame
        """
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        for i in range(height):
            value = int(255 * i / height)
            frame[i, :] = [value, value, value]
        return frame

    @staticmethod
    def checkerboard(width=640, height=480, square_size=32):
        """
        Generate a checkerboard pattern frame.

        Args:
            width: Frame width in pixels
            height: Frame height in pixels
            square_size: Size of each square in pixels

        Returns:
            numpy.ndarray: Checkerboard frame
        """
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        for i in range(0, height, square_size):
            for j in range(0, width, square_size):
                if ((i // square_size) + (j // square_size)) % 2 == 0:
                    frame[i:i+square_size, j:j+square_size] = [255, 255, 255]
        return frame

    @staticmethod
    def noise(width=640, height=480):
        """
        Generate a random noise frame.

        Args:
            width: Frame width in pixels
            height: Frame height in pixels

        Returns:
            numpy.ndarray: Random noise frame
        """
        return np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)

    @staticmethod
    def with_circles(width=640, height=480, num_circles=5):
        """
        Generate a frame with random circles.

        Args:
            width: Frame width in pixels
            height: Frame height in pixels
            num_circles: Number of circles to draw

        Returns:
            numpy.ndarray: Frame with circles
        """
        frame = np.ones((height, width, 3), dtype=np.uint8) * 255

        for _ in range(num_circles):
            center = (
                np.random.randint(50, width - 50),
                np.random.randint(50, height - 50)
            )
            radius = np.random.randint(20, 100)
            color = tuple(np.random.randint(0, 256, 3).tolist())
            thickness = np.random.choice([-1, 2, 3])
            cv2.circle(frame, center, radius, color, thickness)

        return frame

    @staticmethod
    def webcam_simulation(width=1280, height=720):
        """
        Generate a frame simulating webcam output.

        Args:
            width: Frame width in pixels
            height: Frame height in pixels

        Returns:
            numpy.ndarray: Simulated webcam frame
        """
        # Create base frame with slight noise
        frame = np.random.randint(100, 156, (height, width, 3), dtype=np.uint8)

        # Add some structure
        cv2.rectangle(frame, (100, 100), (width-100, height-100), (180, 180, 180), -1)

        # Add simulated "face" region
        center = (width // 2, height // 2)
        cv2.ellipse(frame, center, (150, 200), 0, 0, 360, (200, 180, 160), -1)

        return frame


# ============================================================================
# Image Comparison Utilities
# ============================================================================

class ImageCompare:
    """Utility class for comparing images."""

    @staticmethod
    def are_equal(img1, img2):
        """
        Check if two images are exactly equal.

        Args:
            img1: First image
            img2: Second image

        Returns:
            bool: True if images are identical
        """
        if img1.shape != img2.shape:
            return False
        return np.array_equal(img1, img2)

    @staticmethod
    def are_similar(img1, img2, threshold=0.95):
        """
        Check if two images are similar using structural similarity.

        Args:
            img1: First image
            img2: Second image
            threshold: Similarity threshold (0-1)

        Returns:
            bool: True if images are similar above threshold
        """
        if img1.shape != img2.shape:
            return False

        # Simple similarity using normalized correlation
        img1_norm = img1.astype(float) / 255.0
        img2_norm = img2.astype(float) / 255.0

        diff = np.abs(img1_norm - img2_norm)
        similarity = 1.0 - np.mean(diff)

        return similarity >= threshold

    @staticmethod
    def mean_difference(img1, img2):
        """
        Calculate mean absolute difference between images.

        Args:
            img1: First image
            img2: Second image

        Returns:
            float: Mean absolute difference
        """
        if img1.shape != img2.shape:
            raise ValueError("Images must have same shape")

        return np.mean(np.abs(img1.astype(float) - img2.astype(float)))

    @staticmethod
    def max_difference(img1, img2):
        """
        Calculate maximum pixel difference between images.

        Args:
            img1: First image
            img2: Second image

        Returns:
            int: Maximum pixel difference
        """
        if img1.shape != img2.shape:
            raise ValueError("Images must have same shape")

        return int(np.max(np.abs(img1.astype(int) - img2.astype(int))))

    @staticmethod
    def histogram_similarity(img1, img2, bins=256):
        """
        Calculate histogram-based similarity.

        Args:
            img1: First image
            img2: Second image
            bins: Number of histogram bins

        Returns:
            float: Histogram correlation (0-1)
        """
        # Convert to grayscale if needed
        if len(img1.shape) == 3:
            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        else:
            gray1 = img1

        if len(img2.shape) == 3:
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        else:
            gray2 = img2

        # Calculate histograms
        hist1 = cv2.calcHist([gray1], [0], None, [bins], [0, 256])
        hist2 = cv2.calcHist([gray2], [0], None, [bins], [0, 256])

        # Normalize
        cv2.normalize(hist1, hist1)
        cv2.normalize(hist2, hist2)

        # Compare using correlation
        return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)


# ============================================================================
# Validation Utilities
# ============================================================================

class Validator:
    """Utility class for validation."""

    @staticmethod
    def is_valid_frame(frame):
        """
        Validate that a frame is properly formatted.

        Args:
            frame: Frame to validate

        Returns:
            tuple: (is_valid, error_message)
        """
        if frame is None:
            return False, "Frame is None"

        if not isinstance(frame, np.ndarray):
            return False, f"Frame is not numpy array: {type(frame)}"

        if frame.dtype != np.uint8:
            return False, f"Frame dtype is {frame.dtype}, expected uint8"

        if len(frame.shape) != 3:
            return False, f"Frame has {len(frame.shape)} dimensions, expected 3"

        if frame.shape[2] != 3:
            return False, f"Frame has {frame.shape[2]} channels, expected 3"

        if frame.shape[0] == 0 or frame.shape[1] == 0:
            return False, f"Frame has zero dimension: {frame.shape}"

        return True, "Valid"

    @staticmethod
    def is_valid_camera_properties(props):
        """
        Validate camera properties dictionary.

        Args:
            props: Properties dictionary

        Returns:
            tuple: (is_valid, error_message)
        """
        if not isinstance(props, dict):
            return False, "Properties is not a dictionary"

        required_keys = ['width', 'height', 'fps']
        for key in required_keys:
            if key not in props:
                return False, f"Missing required key: {key}"

        if not isinstance(props['width'], int) or props['width'] <= 0:
            return False, f"Invalid width: {props['width']}"

        if not isinstance(props['height'], int) or props['height'] <= 0:
            return False, f"Invalid height: {props['height']}"

        if not isinstance(props['fps'], (int, float)) or props['fps'] < 0:
            return False, f"Invalid fps: {props['fps']}"

        return True, "Valid"

    @staticmethod
    def is_valid_image_file(path):
        """
        Validate that a file is a valid image.

        Args:
            path: Path to image file

        Returns:
            tuple: (is_valid, error_message)
        """
        path = Path(path)

        if not path.exists():
            return False, f"File does not exist: {path}"

        if path.stat().st_size == 0:
            return False, f"File is empty: {path}"

        # Try to read the image
        img = cv2.imread(str(path))
        if img is None:
            return False, f"Could not read image: {path}"

        return True, "Valid"


# ============================================================================
# Mock Data Generators
# ============================================================================

class MockGenerator:
    """Utility class for generating mock objects and data."""

    @staticmethod
    def camera_capture(is_opened=True, width=1280, height=720, fps=30.0):
        """
        Generate a mock VideoCapture object.

        Args:
            is_opened: Whether camera is opened
            width: Frame width
            height: Frame height
            fps: Frames per second

        Returns:
            Mock: Mock VideoCapture object
        """
        mock_cap = Mock()
        mock_cap.isOpened.return_value = is_opened
        mock_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FRAME_WIDTH: float(width),
            cv2.CAP_PROP_FRAME_HEIGHT: float(height),
            cv2.CAP_PROP_FPS: fps
        }.get(prop, 0.0)

        frame = FrameGenerator.webcam_simulation(width, height)
        mock_cap.read.return_value = (True, frame)

        return mock_cap

    @staticmethod
    def camera_properties(width=1280, height=720, fps=30.0):
        """
        Generate camera properties dictionary.

        Args:
            width: Frame width
            height: Frame height
            fps: Frames per second

        Returns:
            dict: Camera properties
        """
        return {
            'width': width,
            'height': height,
            'fps': fps
        }

    @staticmethod
    def key_sequence(keys):
        """
        Generate a key sequence for waitKey mock.

        Args:
            keys: List of key codes or characters

        Returns:
            list: List of key codes
        """
        result = []
        for key in keys:
            if isinstance(key, str):
                result.append(ord(key))
            elif key is None:
                result.append(-1)
            else:
                result.append(key)
        return result


# ============================================================================
# Test Helper Functions
# ============================================================================

def create_temp_image(width=640, height=480, format='jpg'):
    """
    Create a temporary image file.

    Args:
        width: Image width
        height: Image height
        format: Image format (jpg, png, etc.)

    Returns:
        tuple: (path, frame) where path is temporary file path
    """
    import tempfile

    frame = FrameGenerator.webcam_simulation(width, height)
    temp_fd, temp_path = tempfile.mkstemp(suffix=f'.{format}')

    cv2.imwrite(temp_path, frame)

    import os
    os.close(temp_fd)

    return temp_path, frame


def measure_execution_time(func, *args, **kwargs):
    """
    Measure execution time of a function.

    Args:
        func: Function to measure
        *args: Positional arguments
        **kwargs: Keyword arguments

    Returns:
        tuple: (result, elapsed_time)
    """
    import time

    start = time.time()
    result = func(*args, **kwargs)
    elapsed = time.time() - start

    return result, elapsed


def assert_frame_approximately_equal(frame1, frame2, tolerance=5):
    """
    Assert that two frames are approximately equal.

    Args:
        frame1: First frame
        frame2: Second frame
        tolerance: Maximum allowed mean difference
    """
    assert frame1.shape == frame2.shape, f"Shape mismatch: {frame1.shape} vs {frame2.shape}"

    diff = ImageCompare.mean_difference(frame1, frame2)
    assert diff <= tolerance, f"Mean difference {diff} exceeds tolerance {tolerance}"


# ============================================================================
# Pytest Fixtures from Utilities
# ============================================================================

@pytest.fixture
def frame_generator():
    """Provide FrameGenerator class."""
    return FrameGenerator


@pytest.fixture
def image_compare():
    """Provide ImageCompare class."""
    return ImageCompare


@pytest.fixture
def validator():
    """Provide Validator class."""
    return Validator


@pytest.fixture
def mock_generator():
    """Provide MockGenerator class."""
    return MockGenerator


# ============================================================================
# Test the Utilities
# ============================================================================

class TestFrameGenerator:
    """Tests for FrameGenerator utility."""

    @pytest.mark.unit
    def test_solid_color(self):
        """Test solid color frame generation."""
        frame = FrameGenerator.solid_color(100, 100, (255, 0, 0))

        assert frame.shape == (100, 100, 3)
        assert np.all(frame[:, :, 0] == 255)  # Blue channel
        assert np.all(frame[:, :, 1] == 0)    # Green channel
        assert np.all(frame[:, :, 2] == 0)    # Red channel

    @pytest.mark.unit
    def test_gradient_horizontal(self):
        """Test horizontal gradient generation."""
        frame = FrameGenerator.gradient_horizontal(100, 50)

        assert frame.shape == (50, 100, 3)
        # Left edge should be darker than right edge
        assert np.mean(frame[:, 0]) < np.mean(frame[:, -1])

    @pytest.mark.unit
    def test_gradient_vertical(self):
        """Test vertical gradient generation."""
        frame = FrameGenerator.gradient_vertical(50, 100)

        assert frame.shape == (100, 50, 3)
        # Top edge should be darker than bottom edge
        assert np.mean(frame[0, :]) < np.mean(frame[-1, :])

    @pytest.mark.unit
    def test_checkerboard(self):
        """Test checkerboard pattern generation."""
        frame = FrameGenerator.checkerboard(64, 64, 32)

        assert frame.shape == (64, 64, 3)
        # Check that alternating squares have different colors
        assert not np.array_equal(frame[0:32, 0:32], frame[0:32, 32:64])

    @pytest.mark.unit
    def test_noise(self):
        """Test noise frame generation."""
        frame1 = FrameGenerator.noise(100, 100)
        frame2 = FrameGenerator.noise(100, 100)

        # Two random frames should be different
        assert not np.array_equal(frame1, frame2)


class TestImageCompare:
    """Tests for ImageCompare utility."""

    @pytest.mark.unit
    def test_are_equal_identical(self):
        """Test equality check for identical images."""
        frame = FrameGenerator.solid_color(100, 100)

        assert ImageCompare.are_equal(frame, frame.copy()) is True

    @pytest.mark.unit
    def test_are_equal_different(self):
        """Test equality check for different images."""
        frame1 = FrameGenerator.solid_color(100, 100, (0, 0, 0))
        frame2 = FrameGenerator.solid_color(100, 100, (255, 255, 255))

        assert ImageCompare.are_equal(frame1, frame2) is False

    @pytest.mark.unit
    def test_are_similar_similar(self):
        """Test similarity check for similar images."""
        frame = FrameGenerator.solid_color(100, 100, (128, 128, 128))
        frame_noise = frame.copy()
        # Add small noise
        noise = np.random.randint(-5, 6, frame.shape, dtype=np.int16)
        frame_noise = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # Use == True to handle numpy bool types
        assert ImageCompare.are_similar(frame, frame_noise, threshold=0.9) == True


class TestValidator:
    """Tests for Validator utility."""

    @pytest.mark.unit
    def test_valid_frame(self):
        """Test validation of valid frame."""
        frame = FrameGenerator.solid_color(100, 100)
        is_valid, msg = Validator.is_valid_frame(frame)

        assert is_valid is True

    @pytest.mark.unit
    def test_invalid_frame_none(self):
        """Test validation of None frame."""
        is_valid, msg = Validator.is_valid_frame(None)

        assert is_valid is False
        assert "None" in msg

    @pytest.mark.unit
    def test_valid_camera_properties(self):
        """Test validation of valid camera properties."""
        props = {'width': 1280, 'height': 720, 'fps': 30.0}
        is_valid, msg = Validator.is_valid_camera_properties(props)

        assert is_valid is True

    @pytest.mark.unit
    def test_invalid_camera_properties_missing_key(self):
        """Test validation of properties with missing key."""
        props = {'width': 1280, 'height': 720}  # Missing fps
        is_valid, msg = Validator.is_valid_camera_properties(props)

        assert is_valid is False
        assert "fps" in msg
