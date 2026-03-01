import numpy as np

from robot_hand_control.geometry import angle_three_points, hand_translation


def test_angle_three_points_right_angle():
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 0.0, 0.0])
    c = np.array([0.0, 1.0, 0.0])
    ang = angle_three_points(a, b, c)
    assert np.isclose(ang, np.pi / 2, atol=1e-6)


def test_hand_translation_shape():
    l = np.zeros((21, 3), dtype=np.float32)
    l[0] = [0.1, 0.2, 0.3]
    l[5] = [0.5, 0.5, 0.5]
    l[9] = [0.7, 0.7, 0.7]
    l[13] = [0.9, 0.9, 0.9]
    l[17] = [1.1, 1.1, 1.1]
    t = hand_translation(l)
    assert t.shape == (3,)
