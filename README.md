# facial-joints-layout

Tools for optimizing the joint layout of a face rig, so that its blendshapes
convert well into skinning and each joint works as a tweaker for the skin
around it.

## Solvers

| Solver | What it does | Status |
|---|---|---|
| `JointPlacementSolver` | Moves the rigger's joints toward the skin they drive, along the surface so each keeps its depth. Left and right joints (paired by their `L_` and `R_` names) move as mirror images, centre joints (`C_`) stay on the mirror plane, and each move is bounded by a fraction of the distance to the nearest other joint. | Released (0.6.0) |
| `JointCountSolver` | Decides how many joints a face needs and where to add them. | Planned |

Both are separate from the compression itself: they prepare the joints that a
blendshape-to-skinning solver then binds the face to.

## Usage

Placement starts from a solve with the rigger's joints: the weights say which
skin each joint drives.

```python
from facial_joints_layout import JointPlacementSolver

placement = JointPlacementSolver(bound_fraction=0.5).solve(
    rest_verts=rest_verts,          # (N, 3)
    rest_faces=rest_faces,          # (F, 3) or (F, 4)
    deltas=deltas,                  # (S, N, 3), the shapes' offsets
    weights=weights,                # (N, P), from a solve with these joints
    joint_positions=joint_positions,  # (P, 3)
    joint_names=joint_names,        # L_/R_/C_ prefixes mark the side
    fixed_joints=[root],            # joints that must not move
)
moved = placement.positions         # (P, 3); then solve again with these
```

`bound_fraction` is the share of the gap to the nearest other joint a joint
may move (at most one half, so no two joints can pass each other);
`max_move` adds an absolute limit in model units. `placement.at_bound` shows
which joints wanted to go further.

## Paper

[`paper/facial_joints_layout.md`](paper/facial_joints_layout.md) describes
the method and its math. It is a draft, written alongside the code.

## Install

```bash
pip install git+https://github.com/martinlanton/facial-joints-layout.git@v0.6.0
```

It needs only NumPy and SciPy.

## Development

```bash
pytest                                  # tests
ruff format . && ruff check .           # format and lint
mypy src/                               # type checking
```

## License

MIT; see [LICENSE](LICENSE).
