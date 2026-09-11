"""
tests/test_state_buffer.py
==========================
Unit tests for systems/buffer.py (StateBuffer).
"""

import threading
import numpy as np
import pytest

from systems.buffer import StateBuffer, DEFAULT_WINDOW_SIZE, PDW_NUM_FIELDS


def test_buffer_initialization():
    """Buffer initializes with zeros of shape (window_size, 5)."""
    buf = StateBuffer(window_size=10)
    obs = buf.get_observation()
    assert obs.shape == (10, 5)
    assert obs.dtype == np.float32
    assert np.all(obs == 0.0)
    assert buf.count == 0
    assert not buf.is_full
    assert len(buf) == 0


def test_buffer_invalid_size():
    """Buffer raises ValueError on invalid window sizes."""
    with pytest.raises(ValueError):
        StateBuffer(window_size=0)
    with pytest.raises(ValueError):
        StateBuffer(window_size=-5)


def test_buffer_push_and_slide():
    """Pushing pulses updates buffer and slides past elements correctly."""
    buf = StateBuffer(window_size=3)

    p1 = np.array([10.0, 1000.0, 1.0, 45.0, -30.0, 0.0, 1.0, 1.0], dtype=np.float32)
    p2 = np.array([20.0, 2000.0, 2.0, 50.0, -25.0, 0.0, 1.0, 1.0], dtype=np.float32)
    p3 = np.array([30.0, 3000.0, 3.0, 55.0, -20.0, 0.0, 1.0, 1.0], dtype=np.float32)
    p4 = np.array([40.0, 4000.0, 4.0, 60.0, -15.0, 0.0, 1.0, 1.0], dtype=np.float32)

    buf.push(p1)
    obs = buf.get_observation()
    assert buf.count == 1
    assert len(buf) == 1
    assert not buf.is_full
    np.testing.assert_array_equal(obs[-1], p1[:5])

    buf.push(p2)
    buf.push(p3)
    assert buf.is_full
    assert buf.count == 3
    assert len(buf) == 3

    obs = buf.get_observation()
    np.testing.assert_array_equal(obs[0], p1[:5])
    np.testing.assert_array_equal(obs[1], p2[:5])
    np.testing.assert_array_equal(obs[2], p3[:5])

    # 4th push drops p1
    buf.push(p4)
    assert buf.count == 4
    obs = buf.get_observation()
    np.testing.assert_array_equal(obs[0], p2[:5])
    np.testing.assert_array_equal(obs[1], p3[:5])
    np.testing.assert_array_equal(obs[2], p4[:5])


def test_buffer_defensive_copy():
    """Mutating returned observation does not alter internal buffer state."""
    buf = StateBuffer(window_size=5)
    pulse = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    buf.push(pulse)

    obs = buf.get_observation()
    obs[0, 0] = 9999.0

    obs2 = buf.get_observation()
    assert obs2[0, 0] != 9999.0


def test_buffer_reset():
    """Reset clears observations back to zeros."""
    buf = StateBuffer(window_size=5)
    pulse = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    buf.push(pulse)
    assert buf.count == 1

    buf.reset()
    assert buf.count == 0
    assert not buf.is_full
    assert np.all(buf.get_observation() == 0.0)


def test_buffer_thread_safety():
    """Concurrent pushes from multiple threads execute without race crashes."""
    buf = StateBuffer(window_size=10)
    num_threads = 4
    pushes_per_thread = 250

    def worker(tid: int):
        for i in range(pushes_per_thread):
            p = np.array([float(i), float(tid * 1000), 1.0, 0.0, -40.0], dtype=np.float32)
            buf.push(p)
            _ = buf.get_observation()

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(num_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert buf.count == num_threads * pushes_per_thread
    assert buf.is_full
    assert buf.get_observation().shape == (10, 5)
