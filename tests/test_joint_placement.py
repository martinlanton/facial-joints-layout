"""Tests for JointPlacementSolver: joints moved toward the skin they drive, within a bound."""

import numpy as np
import pytest

from facial_joints_layout import JointPlacementSolver

SIDE = 9  # a flat 9 x 9 grid of vertices, 1 unit apart, in the plane z = 0


def _grid():
    xs, ys = np.meshgrid(
        np.arange(SIDE, dtype=float) - (SIDE - 1) / 2, np.arange(SIDE, dtype=float)
    )
    verts = np.stack([xs.ravel(), ys.ravel(), np.zeros(SIDE * SIDE)], axis=1)
    index = np.arange(SIDE * SIDE).reshape(SIDE, SIDE)
    quads = np.stack(
        [index[:-1, :-1], index[:-1, 1:], index[1:, 1:], index[1:, :-1]], axis=-1
    ).reshape(-1, 4)
    return verts, quads


def _deltas(verts, moving=None):
    """One shape that lifts the skin along z by 1 (only where ``moving`` is true, if given)."""
    deltas = np.zeros((1, len(verts), 3))
    deltas[0, :, 2] = 1.0 if moving is None else moving.astype(float)
    return deltas


def _weights_to_nearest(verts, joints):
    """Every vertex fully on its nearest joint, shape (N, P)."""
    nearest = np.linalg.norm(verts[:, None] - joints[None], axis=2).argmin(axis=1)
    weights = np.zeros((len(verts), len(joints)))
    weights[np.arange(len(verts)), nearest] = 1.0
    return weights


def _solve(joints, names, weights=None, deltas=None, **settings):
    verts, quads = _grid()
    return JointPlacementSolver(**settings).solve(
        rest_verts=verts,
        rest_faces=quads,
        deltas=_deltas(verts) if deltas is None else deltas,
        weights=_weights_to_nearest(verts, joints) if weights is None else weights,
        joint_positions=joints,
        joint_names=names,
    )


def test_a_joint_moves_toward_the_centre_of_the_skin_it_drives():
    verts, _ = _grid()
    joints = np.array(
        [[-4.0, 0.0, -0.1]]
    )  # one joint at the grid's edge, driving all of it
    weights = np.ones((len(verts), 1))

    placement = _solve(
        joints, ["C_chin_01"], weights=weights, bound_fraction=0.5, max_move=None
    )

    np.testing.assert_allclose(
        placement.positions[0, 1], 4.0, atol=1e-9
    )  # pulled to the middle row


def test_a_joint_keeps_its_depth_under_the_skin():
    joints = np.array([[-2.0, 2.0, -0.3], [2.0, 6.0, -0.3]])

    placement = _solve(joints, ["L_cheek_01", "R_brow_01"])

    np.testing.assert_allclose(placement.positions[:, 2], joints[:, 2])


def test_skin_that_does_not_move_does_not_pull():
    verts, _ = _grid()
    joints = np.array([[0.0, 4.0, 0.0]])
    weights = np.ones((len(verts), 1))
    moving = verts[:, 1] >= 4  # only the upper half of the grid moves

    placement = _solve(
        joints,
        ["C_nose_01"],
        weights=weights,
        deltas=_deltas(verts, moving),
        max_move=None,
    )

    assert placement.positions[0, 1] > 4.5


def test_a_move_stops_at_the_bound_fraction_of_the_gap_to_the_nearest_joint():
    verts, _ = _grid()
    joints = np.array([[0.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    weights = np.zeros((len(verts), 2))
    weights[:, 0] = 1.0  # the first joint drives everything, so it wants to go far

    placement = _solve(
        joints, ["C_chin_01", "C_lip_01"], weights=weights, bound_fraction=0.25
    )

    assert np.linalg.norm(placement.moves[0]) == pytest.approx(0.25 * 2.0)
    assert placement.at_bound[0]


def test_mirrored_partners_move_as_mirror_images():
    joints = np.array(
        [[-2.0, 3.0, 0.0], [2.0, 3.0, 0.0], [-1.0, 6.0, 0.0], [1.0, 6.0, 0.0]]
    )
    names = ["L_cheek_01", "R_cheek_01", "L_brow_01", "R_brow_01"]
    verts, _ = _grid()
    weights = _weights_to_nearest(verts, joints)
    weights[verts[:, 1] < 2, 1] = (
        0.0  # make the right side's skin different from the left's
    )
    weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)

    placement = _solve(joints, names, weights=weights)

    mirror = np.array([-1.0, 1.0, 1.0])
    np.testing.assert_allclose(placement.moves[1], placement.moves[0] * mirror)
    np.testing.assert_allclose(placement.moves[3], placement.moves[2] * mirror)


def test_centre_joints_stay_on_the_mirror_plane():
    joints = np.array([[0.0, 2.0, 0.0], [-3.0, 2.0, 0.0]])

    placement = _solve(joints, ["C_lip_01", "L_cheek_01"])

    assert placement.moves[0, 0] == 0.0


def test_fixed_joints_do_not_move():
    joints = np.array([[0.0, 1.0, 0.0], [-3.0, 5.0, 0.0]])
    verts, quads = _grid()

    placement = JointPlacementSolver().solve(
        rest_verts=verts,
        rest_faces=quads,
        deltas=_deltas(verts),
        weights=_weights_to_nearest(verts, joints),
        joint_positions=joints,
        joint_names=["C_head_01", "L_cheek_01"],
        fixed_joints=[0],
    )

    np.testing.assert_array_equal(placement.moves[0], 0.0)


def test_an_absolute_bound_caps_every_move():
    joints = np.array([[0.0, 0.0, 0.0], [0.0, 8.0, 0.0]])
    verts, _ = _grid()
    weights = np.zeros((len(verts), 2))
    weights[:, 0] = 1.0

    placement = _solve(
        joints, ["C_chin_01", "C_brow_01"], weights=weights, max_move=0.3
    )

    assert np.linalg.norm(placement.moves[0]) == pytest.approx(0.3)


def test_triangles_work_as_well_as_quads():
    verts, quads = _grid()
    triangles = np.concatenate([quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]])
    joints = np.array([[-2.0, 2.0, -0.2], [2.0, 6.0, -0.2]])
    names = ["L_cheek_01", "R_brow_01"]
    common = {
        "deltas": _deltas(verts),
        "weights": _weights_to_nearest(verts, joints),
        "joint_positions": joints,
        "joint_names": names,
    }

    from_quads = JointPlacementSolver().solve(
        rest_verts=verts, rest_faces=quads, **common
    )
    from_triangles = JointPlacementSolver().solve(
        rest_verts=verts, rest_faces=triangles, **common
    )

    np.testing.assert_allclose(
        from_triangles.positions, from_quads.positions, atol=1e-9
    )


@pytest.mark.parametrize("fraction", [0.0, -0.1, 0.6])
def test_the_bound_fraction_is_above_zero_and_at_most_half(fraction):
    with pytest.raises(ValueError, match="bound_fraction"):
        JointPlacementSolver(bound_fraction=fraction)


def test_weights_must_have_one_column_per_joint():
    joints = np.array([[0.0, 0.0, 0.0], [0.0, 4.0, 0.0]])
    verts, _ = _grid()

    with pytest.raises(ValueError, match="weights"):
        _solve(joints, ["C_chin_01", "C_nose_01"], weights=np.ones((len(verts), 3)))
