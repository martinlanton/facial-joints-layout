"""Tools for optimizing the joint layout of a face rig.

The joints of a face rig decide how well its blendshapes can be turned into
skinning, and how well each joint works as a tweaker for the skin around it.

- :class:`JointPlacementSolver` moves the rigger's joints, symmetrically and
  within a bound, toward the skin they drive.
- A joint count solver, deciding how many joints a face needs and where to
  add them, is planned.
"""

from facial_joints_layout.placement import JointPlacement, JointPlacementSolver

__version__ = "0.6.0"

__all__ = ["JointPlacement", "JointPlacementSolver", "__version__"]
