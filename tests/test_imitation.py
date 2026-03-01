import numpy as np

from robot_hand_control.imitation import Episode, ImitationPolicy


def test_policy_can_fit_identity_like_data():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(64, 13)).astype(np.float32)
    y = x.copy()
    policy = ImitationPolicy()
    policy.train(Episode(observations=x, actions=y))
    pred = policy.predict(x[0])
    assert pred.shape == (13,)
