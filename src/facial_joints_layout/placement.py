r"""Moving a rig's joints toward the skin they drive, symmetrically and within a bound.

A joint works best, both for the compression and as a tweaker an animator
grabs, when it sits over the skin it moves. Starting from a solve with the
rigger's joints, each joint's target is the centre of the skin it drives,
weighted by how much it drives each vertex and how far that vertex moves:

.. math::

    c_j = \frac{\sum_i w_{ij} m_i v_i}{\sum_i w_{ij} m_i},
    \qquad m_i = \max_k \lVert \Delta_{k,i} \rVert,

with :math:`w_{ij}` the weight of joint :math:`j` on vertex :math:`i`,
:math:`v_i` the rest position and :math:`\Delta_{k,i}` the vertex's offset in
shape :math:`k`. Skin that never moves does not pull. Only the part of
:math:`c_j - p_j` along the surface is kept: the component along the
weighted normal :math:`n_j` of that skin is removed, so the joint keeps its
depth under the skin.

Left and right partners, named ``L_<name>`` and ``R_<name>``, move as mirror
images across the plane :math:`x = 0` (each takes the average of its own move
and its partner's mirrored move), and centre joints (``C_``) do not move in
:math:`x`. Every move is then shortened to its bound: a fraction of the
distance from the joint to its nearest other joint, the smaller of the two for
a mirrored pair, so joints in a dense area such as the lips move less than
those in a sparse one such as the neck. With a fraction of at most one half
no two joints can pass each other, and below one half no two can meet.

Example:
    >>> solver = JointPlacementSolver(bound_fraction=0.5)
    >>> placement = solver.solve(
    ...     rest_verts=verts,
    ...     rest_faces=faces,
    ...     deltas=deltas,
    ...     weights=weights,
    ...     joint_positions=joints,
    ...     joint_names=names,
    ...     fixed_joints=[root],
    ... )
    >>> new_positions = placement.positions
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

_MIRROR = np.array([-1.0, 1.0, 1.0])  # reflection across the plane x = 0
_LARGEST_BOUND_FRACTION = 0.5


@dataclass(frozen=True)
class JointPlacement:
    """Where the joints go, and how far each one moved.

    Attributes:
        positions: New joint positions, shape (P, 3).
        moves: Each joint's move, shape (P, 3).
        wanted: How far each joint wanted to move before symmetry and the
            bound, shape (P,).
        bounds: Each joint's largest allowed move, shape (P,); infinite where
            nothing bounds it.
        at_bound: Whether each joint's move was shortened by its bound, shape (P,).
    """

    positions: np.ndarray
    moves: np.ndarray
    wanted: np.ndarray
    bounds: np.ndarray
    at_bound: np.ndarray


@dataclass(frozen=True)
class JointPlacementSolver:
    """Moves joints toward the motion-weighted centre of the skin they drive.

    Attributes:
        bound_fraction: Largest move as a fraction of the distance to the
            joint's nearest other joint, above 0 and at most 0.5 (default 0.5).
        max_move: An absolute largest move in model units as well, or None
            (default) for the relative bound alone.

    Raises:
        ValueError: If bound_fraction is not in (0, 0.5], or max_move is not
            positive.
    """

    bound_fraction: float = _LARGEST_BOUND_FRACTION
    max_move: float | None = None

    def __post_init__(self) -> None:
        """Checks the bounds."""
        if not 0.0 < self.bound_fraction <= _LARGEST_BOUND_FRACTION:
            raise ValueError(
                f"bound_fraction={self.bound_fraction} must be above 0 and at most "
                f"{_LARGEST_BOUND_FRACTION}, so no two joints can pass each other"
            )
        if self.max_move is not None and self.max_move <= 0:
            raise ValueError(f"max_move={self.max_move} must be positive")

    def solve(  # noqa: PLR0913
        self,
        *,
        rest_verts: npt.ArrayLike,
        rest_faces: npt.ArrayLike,
        deltas: npt.ArrayLike,
        weights: npt.ArrayLike,
        joint_positions: npt.ArrayLike,
        joint_names: Sequence[str],
        fixed_joints: Sequence[int] = (),
    ) -> JointPlacement:
        """Moves every joint toward its skin, mirrored and bounded.

        Args:
            rest_verts: Rest vertex positions, shape (N, 3).
            rest_faces: Polygons as vertex indices, shape (F, n) with n >= 3
                (triangles and quads alike).
            deltas: Each shape's vertex offsets, shape (S, N, 3).
            weights: Each joint's weight on each vertex from a solve with these
                joints, shape (N, P), in the order of joint_positions.
            joint_positions: Joint rest positions, shape (P, 3).
            joint_names: One name per joint; ``L_``/``R_``/``C_`` prefixes mark
                the side.
            fixed_joints: Indices of joints that must not move, such as the root.

        Returns:
            The new positions with each joint's move and bound.

        Raises:
            ValueError: If the shapes of the inputs disagree.
        """
        verts = np.asarray(rest_verts, dtype=np.float64)
        positions = np.asarray(joint_positions, dtype=np.float64)
        weight = np.asarray(weights, dtype=np.float64)
        _check_shapes(verts, weight, positions, joint_names)
        motion = largest_motion(np.asarray(deltas, dtype=np.float64))
        normals = vertex_normals(verts, np.asarray(rest_faces))
        raw = tangential_moves(weight * motion[:, None], verts, normals, positions)
        raw[list(fixed_joints)] = 0.0
        bounds = self.bounds(positions, joint_names)
        moves = clamp_moves(mirror_moves(raw, joint_names), bounds)
        length = np.linalg.norm(moves, axis=1)
        return JointPlacement(
            positions=positions + moves,
            moves=moves,
            wanted=np.linalg.norm(raw, axis=1),
            bounds=bounds,
            at_bound=np.isfinite(bounds)
            & (length >= bounds * (1 - 1e-9))
            & (length > 0),
        )

    def bounds(self, positions: np.ndarray, names: Sequence[str]) -> np.ndarray:
        """Each joint's largest move: the relative bound, capped by max_move.

        Args:
            positions: Joint positions, shape (P, 3).
            names: Joint names, for the mirrored pairs.

        Returns:
            Largest move per joint, shape (P,); infinite with one joint and no
            max_move.
        """
        bound = relative_bounds(positions, names, self.bound_fraction)
        if self.max_move is not None:
            bound = np.minimum(bound, self.max_move)
        return bound


def _check_shapes(
    verts: np.ndarray, weights: np.ndarray, positions: np.ndarray, names: Sequence[str]
) -> None:
    """Raises ValueError when the inputs' shapes disagree."""
    if positions.ndim != 2 or positions.shape[1] != 3:  # noqa: PLR2004
        raise ValueError(
            f"joint_positions must have shape (P, 3), got {positions.shape}"
        )
    if len(names) != len(positions):
        raise ValueError(
            f"joint_names has {len(names)} names for {len(positions)} joints"
        )
    if weights.shape != (len(verts), len(positions)):
        raise ValueError(
            f"weights must have shape (N, P) = ({len(verts)}, {len(positions)}), got {weights.shape}"
        )


def largest_motion(deltas: np.ndarray) -> np.ndarray:
    """How far each vertex moves in its largest shape, shape (N,).

    Args:
        deltas: Shape offsets, shape (S, N, 3).

    Returns:
        The largest offset length per vertex, shape (N,).
    """
    return np.linalg.norm(deltas, axis=2).max(axis=0)


def vertex_normals(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Area-weighted unit normals, from polygons split into fans of triangles.

    Args:
        verts: Vertex positions, shape (N, 3).
        faces: Polygons, shape (F, n) with n >= 3.

    Returns:
        Unit normal per vertex, shape (N, 3).
    """
    normals = np.zeros_like(verts)
    for k in range(1, faces.shape[1] - 1):
        tri = faces[:, [0, k, k + 1]]
        face = np.cross(
            verts[tri[:, 1]] - verts[tri[:, 0]], verts[tri[:, 2]] - verts[tri[:, 0]]
        )
        for corner in range(3):
            np.add.at(normals, tri[:, corner], face)
    return normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)


def tangential_moves(
    pull: np.ndarray, verts: np.ndarray, normals: np.ndarray, positions: np.ndarray
) -> np.ndarray:
    """Each joint's move to the pull-weighted centre of its skin, along the surface.

    Args:
        pull: Weight times motion, shape (N, P).
        verts: Vertex positions, shape (N, 3).
        normals: Unit vertex normals, shape (N, 3).
        positions: Joint positions, shape (P, 3).

    Returns:
        Moves with their normal component removed, shape (P, 3); zero for a
        joint that drives no moving skin.
    """
    total = pull.sum(axis=0)  # (P,)
    used = total > 0
    moves = np.zeros_like(positions)
    centre = (pull[:, used].T @ verts) / total[used, None]
    normal = pull[:, used].T @ normals
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12)
    move = centre - positions[used]
    moves[used] = move - (move * normal).sum(axis=1, keepdims=True) * normal
    return moves


def _partners(names: Sequence[str]) -> list[tuple[int, int]]:
    """Index pairs (left, right) of joints named L_<x> and R_<x>."""
    index = {name: i for i, name in enumerate(names)}
    return [
        (i, index["R_" + name[2:]])
        for name, i in index.items()
        if name.startswith("L_") and "R_" + name[2:] in index
    ]


def mirror_moves(moves: np.ndarray, names: Sequence[str]) -> np.ndarray:
    """Partners get mirrored copies of their average move; centre joints keep their x.

    Args:
        moves: Moves, shape (P, 3).
        names: Joint names.

    Returns:
        Symmetric moves, shape (P, 3).
    """
    out = moves.copy()
    for i, name in enumerate(names):
        if name.startswith("C_"):
            out[i, 0] = 0.0
    for left, right in _partners(names):
        mean = 0.5 * (moves[left] + moves[right] * _MIRROR)
        out[left], out[right] = mean, mean * _MIRROR
    return out


def relative_bounds(
    positions: np.ndarray, names: Sequence[str], fraction: float
) -> np.ndarray:
    """A fraction of each joint's distance to its nearest other joint; partners share the smaller.

    Args:
        positions: Joint positions, shape (P, 3).
        names: Joint names.
        fraction: The fraction of the gap a joint may move.

    Returns:
        Bound per joint, shape (P,); infinite for a lone joint.
    """
    gaps = np.linalg.norm(positions[:, None] - positions[None], axis=2)
    np.fill_diagonal(gaps, np.inf)
    bound = fraction * gaps.min(axis=1)
    for left, right in _partners(names):
        bound[left] = bound[right] = min(bound[left], bound[right])
    return bound


def clamp_moves(moves: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    """Moves shortened to at most their bound; the direction is kept.

    Args:
        moves: Moves, shape (P, 3).
        bounds: Largest move per joint, shape (P,).

    Returns:
        Clamped moves, shape (P, 3).
    """
    length = np.linalg.norm(moves, axis=1)
    scale = np.minimum(1.0, bounds / np.maximum(length, 1e-12))
    return moves * scale[:, None]
