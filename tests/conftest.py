"""
Pytest configuration and fixtures for webcam_capture_test_Haiku45 tests.

This module provides:
- Mock objects for camera hardware
- Test fixtures for frames and images
- Common test utilities and helpers
"""

import sys
import os
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
import tempfile
import shutil

import pytest
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================================
# Mock Classes for Camera Hardware
# ============================================================================

class MockVideoCapture:
    """
    Mock class for cv2.VideoCapture.

    Simulates camera behavior without requiring actual hardware.
    """

    def __init__(self, index=0, api_preference=None):
        """Initialize mock camera with configurable properties."""
        self._index = index
        self._api_preference = api_preference
        self._is_opened = True
        self._frame_count = 0
        self._width = 1280
        self._height = 720
        self._fps = 30.0
        self._fail_read = False
        self._fail_after_n_reads = None

    def isOpened(self):
        """Check if camera is opened."""
        return self._is_opened

    def read(self):
        """
        Read a frame from the mock camera.

        Returns:
            tuple: (success, frame) where frame is a numpy array
        """
        if self._fail_read:
            return False, None

        if self._fail_after_n_reads is not None:
            if self._frame_count >= self._fail_after_n_reads:
                return False, None

        self._frame_count += 1
        # Create a test frame with some variation
        frame = np.zeros((self._height, self._width, 3), dtype=np.uint8)
        # Add some pattern for testing
        frame[100:200, 100:200] = [255, 0, 0]  # Blue square
        frame[200:300, 200:300] = [0, 255, 0]  # Green square
        frame[300:400, 300:400] = [0, 0, 255]  # Red square
        return True, frame

    def get(self, prop_id):
        """Get camera property."""
        import cv2
        prop_map = {
            cv2.CAP_PROP_FRAME_WIDTH: self._width,
            cv2.CAP_PROP_FRAME_HEIGHT: self._height,
            cv2.CAP_PROP_FPS: self._fps,
        }
        return prop_map.get(prop_id, 0.0)

    def set(self, prop_id, value):
        """Set camera property."""
        import cv2
        if prop_id == cv2.CAP_PROP_FRAME_WIDTH:
            self._width = int(value)
            return True
        elif prop_id == cv2.CAP_PROP_FRAME_HEIGHT:
            self._height = int(value)
            return True
        elif prop_id == cv2.CAP_PROP_FPS:
            self._fps = value
            return True
        return False

    def release(self):
        """Release the camera resource."""
        self._is_opened = False

    def set_fail_read(self, fail=True):
        """Configure read to fail."""
        self._fail_read = fail

    def set_fail_after_n_reads(self, n):
        """Configure read to fail after n successful reads."""
        self._fail_after_n_reads = n


class MockVideoCaptureFactory:
    """
    Factory for creating MockVideoCapture instances.

    Allows configuration of different camera behaviors for testing.
    """

    def __init__(self):
        self.instances = []
        self.should_fail_open = False
        self.available_cameras = [0]

    def __call__(self, index=0, api_preference=None):
        """Create a new MockVideoCapture instance."""
        mock_cap = MockVideoCapture(index, api_preference)

        if self.should_fail_open or index not in self.available_cameras:
            mock_cap._is_opened = False

        self.instances.append(mock_cap)
        return mock_cap

    def set_available_cameras(self, indices):
        """Set which camera indices are available."""
        self.available_cameras = indices

    def set_should_fail_open(self, fail=True):
        """Set whether camera open should fail."""
        self.should_fail_open = fail


# ============================================================================
# Pytest Fixtures
# ============================================================================

@pytest.fixture
def mock_video_capture_factory():
    """
    Provide a MockVideoCaptureFactory for creating mock cameras.

    Usage:
        def test_something(mock_video_capture_factory):
            factory = mock_video_capture_factory
            factory.set_available_cameras([0, 1])
            # ... test code
    """
    return MockVideoCaptureFactory()


@pytest.fixture
def mock_camera(mock_video_capture_factory):
    """
    Provide a single mock camera instance.

    Usage:
        def test_camera(mock_camera):
            ret, frame = mock_camera.read()
            assert ret is True
    """
    return mock_video_capture_factory(0)


@pytest.fixture
def mock_cv2(mock_video_capture_factory):
    """
    Patch cv2.VideoCapture with mock factory.

    Usage:
        def test_with_mock_cv2(mock_cv2):
            # cv2.VideoCapture is now mocked
            import cv2
            cap = cv2.VideoCapture(0)  # Returns MockVideoCapture
    """
    with patch('cv2.VideoCapture', mock_video_capture_factory):
        yield mock_video_capture_factory


@pytest.fixture
def sample_frame():
    """
    Provide a sample BGR frame for testing.

    Returns:
        numpy.ndarray: A 720x1280x3 BGR frame
    """
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    # Add gradient for realistic testing
    for i in range(720):
        frame[i, :, 0] = int(255 * i / 720)  # Blue gradient
    for j in range(1280):
        frame[:, j, 1] = int(255 * j / 1280)  # Green gradient
    frame[:, :, 2] = 128  # Fixed red channel
    return frame


@pytest.fixture
def sample_frame_small():
    """
    Provide a small sample frame for faster tests.

    Returns:
        numpy.ndarray: A 480x640x3 BGR frame
    """
    frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    return frame


@pytest.fixture
def grayscale_frame():
    """
    Provide a grayscale frame for testing.

    Returns:
        numpy.ndarray: A 720x1280 grayscale frame
    """
    return np.random.randint(0, 256, (720, 1280), dtype=np.uint8)


@pytest.fixture
def temp_output_dir():
    """
    Provide a temporary directory for output files.

    The directory is automatically cleaned up after the test.

    Yields:
        Path: Path to temporary directory
    """
    temp_dir = tempfile.mkdtemp(prefix='webcam_test_')
    yield Path(temp_dir)
    # Cleanup
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)


@pytest.fixture
def mock_output_dir(temp_output_dir, monkeypatch):
    """
    Mock the OUTPUT_DIR constant to use a temporary directory.

    Usage:
        def test_save(mock_output_dir):
            # OUTPUT_DIR now points to temp directory
            save_image(frame, "test.jpg")
    """
    # This will be used to patch the module's OUTPUT_DIR
    return temp_output_dir


@pytest.fixture
def camera_properties():
    """
    Provide standard camera properties for testing.

    Returns:
        dict: Camera properties (width, height, fps)
    """
    return {
        'width': 1280,
        'height': 720,
        'fps': 30.0
    }


@pytest.fixture
def camera_properties_low_res():
    """
    Provide low resolution camera properties.

    Returns:
        dict: Camera properties (width, height, fps)
    """
    return {
        'width': 640,
        'height': 480,
        'fps': 30.0
    }


@pytest.fixture
def mock_window():
    """
    Mock OpenCV window functions.

    Patches imshow, namedWindow, destroyAllWindows, waitKey, etc.
    """
    with patch('cv2.imshow') as mock_imshow, \
         patch('cv2.namedWindow') as mock_named_window, \
         patch('cv2.destroyAllWindows') as mock_destroy, \
         patch('cv2.waitKey', return_value=-1) as mock_wait_key, \
         patch('cv2.resizeWindow') as mock_resize:

        yield {
            'imshow': mock_imshow,
            'namedWindow': mock_named_window,
            'destroyAllWindows': mock_destroy,
            'waitKey': mock_wait_key,
            'resizeWindow': mock_resize
        }


@pytest.fixture
def mock_trackbar():
    """
    Mock OpenCV trackbar functions.
    """
    with patch('cv2.createTrackbar') as mock_create, \
         patch('cv2.getTrackbarPos', return_value=100) as mock_get, \
         patch('cv2.setTrackbarPos') as mock_set:

        yield {
            'createTrackbar': mock_create,
            'getTrackbarPos': mock_get,
            'setTrackbarPos': mock_set
        }


@pytest.fixture
def mock_environment():
    """
    Mock environment for testing EnvironmentValidator.
    """
    with patch.dict(os.environ, {'VIRTUAL_ENV': '/path/to/venv'}):
        yield


@pytest.fixture
def capture_stdout(capsys):
    """
    Fixture to capture stdout for testing print statements.

    Returns a function that returns the captured output.
    """
    def get_output():
        captured = capsys.readouterr()
        return captured.out
    return get_output


# ============================================================================
# Pytest Hooks
# ============================================================================

def pytest_configure(config):
    """
    Configure pytest with custom markers.
    """
    config.addinivalue_line(
        "markers", "unit: mark test as unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow"
    )
    config.addinivalue_line(
        "markers", "camera: mark test as requiring camera hardware"
    )
    config.addinivalue_line(
        "markers", "gui: mark test as requiring GUI/display"
    )


def pytest_collection_modifyitems(config, items):
    """
    Modify test collection based on markers.

    Skip camera and gui tests by default unless explicitly requested.
    """
    skip_camera = pytest.mark.skip(reason="Requires camera hardware")
    skip_gui = pytest.mark.skip(reason="Requires GUI/display")

    for item in items:
        if "camera" in item.keywords:
            item.add_marker(skip_camera)
        if "gui" in item.keywords:
            item.add_marker(skip_gui)


# ============================================================================
# Helper Functions
# ============================================================================

def create_test_image(width=640, height=480, color=(128, 128, 128)):
    """
    Create a test image with specified dimensions and color.

    Args:
        width: Image width in pixels
        height: Image height in pixels
        color: BGR color tuple

    Returns:
        numpy.ndarray: Test image
    """
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:] = color
    return img


def create_circle_test_image(width=640, height=480):
    """
    Create a test image with circles for detection testing.

    Args:
        width: Image width in pixels
        height: Image height in pixels

    Returns:
        numpy.ndarray: Test image with circles
    """
    import cv2
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:] = (255, 255, 255)  # White background

    # Draw test circles
    cv2.circle(img, (width // 2, height // 2), 100, (0, 0, 0), 2)
    cv2.circle(img, (width // 4, height // 4), 50, (0, 0, 255), 2)
    cv2.circle(img, (3 * width // 4, 3 * height // 4), 75, (255, 0, 0), 2)

    return img


def assert_frame_valid(frame, expected_shape=None):
    """
    Assert that a frame is valid.

    Args:
        frame: Frame to validate
        expected_shape: Optional expected shape tuple
    """
    assert frame is not None, "Frame is None"
    assert isinstance(frame, np.ndarray), "Frame is not numpy array"
    assert frame.dtype == np.uint8, f"Frame dtype is {frame.dtype}, expected uint8"
    assert len(frame.shape) == 3, f"Frame has {len(frame.shape)} dimensions, expected 3"
    assert frame.shape[2] == 3, f"Frame has {frame.shape[2]} channels, expected 3 (BGR)"

    if expected_shape:
        assert frame.shape == expected_shape, f"Frame shape {frame.shape} != expected {expected_shape}"


def assert_image_saved(path, min_size=100):
    """
    Assert that an image was saved correctly.

    Args:
        path: Path to the saved image
        min_size: Minimum expected file size in bytes
    """
    path = Path(path)
    assert path.exists(), f"Image file does not exist: {path}"
    assert path.stat().st_size >= min_size, f"Image file too small: {path.stat().st_size} bytes"
