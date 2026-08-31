import numpy as np

from neurolive.data.webcam_events import FrameEventSimulator, FrameEventSimulatorConfig


def test_first_frame_produces_no_events():
    sim = FrameEventSimulator(FrameEventSimulatorConfig(downscale_to=(32, 32)))
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    events = sim.step(frame, t_us=0.0)
    assert events.shape == (0, 4)


def test_static_frames_produce_no_events():
    sim = FrameEventSimulator(FrameEventSimulatorConfig(downscale_to=(32, 32)))
    frame = np.full((64, 64, 3), 128, dtype=np.uint8)
    sim.step(frame, t_us=0.0)
    events = sim.step(frame, t_us=33000.0)  # identical frame, ~33ms later
    assert events.shape[0] == 0


def test_bright_square_produces_events_at_correct_location():
    sim = FrameEventSimulator(FrameEventSimulatorConfig(threshold=10, downscale_to=None))
    frame1 = np.full((32, 32, 3), 50, dtype=np.uint8)
    sim.step(frame1, t_us=0.0)

    frame2 = frame1.copy()
    frame2[10:15, 10:15] = 250  # bright square appears
    events = sim.step(frame2, t_us=33000.0)

    assert events.shape[0] > 0
    assert (events[:, 3] == 1).all()  # all positive-polarity (brightness increase)
    assert events[:, 0].min() >= 10 and events[:, 0].max() < 15
    assert events[:, 1].min() >= 10 and events[:, 1].max() < 15
    assert events[:, 2].min() >= 0.0 and events[:, 2].max() <= 33000.0


def test_events_are_time_sorted():
    sim = FrameEventSimulator(FrameEventSimulatorConfig(downscale_to=(32, 32)))
    rng = np.random.default_rng(0)
    frame1 = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
    frame2 = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
    sim.step(frame1, t_us=0.0)
    events = sim.step(frame2, t_us=33000.0)
    assert (np.diff(events[:, 2]) >= 0).all()


def test_max_events_cap_respected():
    cfg = FrameEventSimulatorConfig(threshold=1, max_events_per_frame_pair=100, downscale_to=(64, 64))
    sim = FrameEventSimulator(cfg)
    rng = np.random.default_rng(1)
    frame1 = np.zeros((64, 64, 3), dtype=np.uint8)
    frame2 = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
    sim.step(frame1, t_us=0.0)
    events = sim.step(frame2, t_us=33000.0)
    assert events.shape[0] <= 100
