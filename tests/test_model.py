import pytest
import numpy as np

from ade.models.image.image import (
    fft_filter,
    fft_filter_id,
    make_beacon,
    make_background,
    create_synth_image_moving,
    align_images
)


def test_fft_filter():
    # Setup data with shape (time_steps, x_size, y_size)
    np.random.seed(42)
    data = np.random.normal(15.0, 2.0, (20, 32, 40))
    
    # Run vectorized fft_filter
    output = fft_filter(data)
    
    assert output.shape == (32, 40)
    assert output.dtype == int


def test_fft_filter_id():
    np.random.seed(42)
    data = np.random.normal(15.0, 2.0, (15, 32, 40))
    
    # Run vectorized fft_filter_id
    output_img, freq_img = fft_filter_id(data, stage=1)
    
    assert output_img.shape == (32, 40)
    assert output_img.dtype == int
    assert freq_img.shape == (32, 40)
    assert freq_img.dtype == int


def test_make_beacon():
    beacon = make_beacon()
    # It generates a 2D mesh grid of size bound_x*2 x bound_y*2 (step 1)
    # bounds are -15:15 -> 30 x 30
    assert beacon.shape == (30, 30)
    assert not np.isnan(beacon).any()


def test_create_synth_image_moving():
    # Generator of dicts
    gen = create_synth_image_moving()
    first_msg = next(gen)
    
    assert "data" in first_msg
    assert "timestamp" in first_msg
    assert first_msg["topic"] == "images"
    assert first_msg["name"] == "synth_ir"
    
    # Check data shape (resized to x_size=32, y_size=40)
    # wait: img = cv2.resize(synth_img, (y_size, x_size)) -> shape: (x_size, y_size) -> (32, 40)
    assert first_msg["data"].shape == (32, 40)


def test_align_images():
    # align_images expects a 3D array of shape (N, H, W)
    np.random.seed(42)
    images = np.random.normal(15.0, 2.0, (3, 100, 100)).astype(np.uint8)
    
    aligned = align_images(images)
    
    assert aligned.shape == (3, 100, 100)
    assert aligned.dtype == np.uint8
