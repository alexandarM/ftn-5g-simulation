"""
test_receivers.py
===================

Unit tests for `ntp_campus.receivers.generate_random_receiver_positions`:
rejection-sampling of random (x, y) positions inside a rectangular area
that must NOT fall inside any given 2D building bounding box.

Run with:
    pytest tests/test_receivers.py -v
"""

import random

import pytest

from ntp_campus.receivers import generate_random_receiver_positions


@pytest.fixture(autouse=True)
def _seed_random():
    """Make position generation reproducible across test runs."""
    random.seed(41)


def test_generates_requested_count_when_area_is_empty():
    positions = generate_random_receiver_positions(
        building_bboxes_2d=[],
        num_receivers=50,
        x_min=0, x_max=100, y_min=0, y_max=100,
        rx_height=1.5,
    )
    assert len(positions) == 50


def test_positions_stay_within_area_bounds():
    positions = generate_random_receiver_positions(
        building_bboxes_2d=[],
        num_receivers=200,
        x_min=-50, x_max=50, y_min=-20, y_max=20,
        rx_height=1.5,
    )
    for x, y, z in positions:
        assert -50 <= x <= 50
        assert -20 <= y <= 20


def test_all_positions_use_the_requested_height():
    positions = generate_random_receiver_positions(
        building_bboxes_2d=[],
        num_receivers=30,
        x_min=0, x_max=10, y_min=0, y_max=10,
        rx_height=2.75,
    )
    assert all(z == pytest.approx(2.75) for _, _, z in positions)


def test_no_position_falls_inside_a_building_bbox():
    # A single building covering most of the sampling area, leaving only
    # a narrow strip (x in [90, 100]) free.
    building_bboxes_2d = [(0, 90, 0, 100)]

    positions = generate_random_receiver_positions(
        building_bboxes_2d=building_bboxes_2d,
        num_receivers=100,
        x_min=0, x_max=100, y_min=0, y_max=100,
        rx_height=1.5,
        max_attempts=200_000,
    )

    assert len(positions) == 100
    for x, y, _ in positions:
        inside_building = 0 <= x <= 90 and 0 <= y <= 100
        assert not inside_building


def test_stops_at_max_attempts_when_area_is_fully_blocked():
    # Building bbox covers the entire sampling area -> no valid position exists.
    building_bboxes_2d = [(0, 100, 0, 100)]

    positions = generate_random_receiver_positions(
        building_bboxes_2d=building_bboxes_2d,
        num_receivers=10,
        x_min=0, x_max=100, y_min=0, y_max=100,
        rx_height=1.5,
        max_attempts=500,
    )

    # Can't raise/hang -- must return early with fewer positions than requested.
    assert len(positions) == 0
