# Optimizing the Joint Layout of Face Rigs for Compressed Skinning

* Martin L'Anton

> **Draft.** This paper is written alongside the code. Sections marked
> *(open)* describe problems not yet settled; sections marked *(to come)* wait
> for experiments. Every equation here should match the implementation in
> `src/facial_joints_layout/`; when the two disagree, one of them is wrong.


## ABSTRACT

Blendshape-to-skinning methods such as compressed skinning [Kavan et al.
2024] can bind a face to joints placed by a rigger, so that the result is not
only a compact approximation of the blendshapes but a rig whose joints work
as tweakers for the skin around them. How well this works depends on where
the joints are and how many there are, and riggers place them by eye. We
describe two solvers that adjust the layout before the skinning solve. The
first moves the rigger's joints towards the skin they drive, tangentially to
the surface, symmetrically across the face, and by no more than a fixed
fraction of the distance to the nearest other joint, so the layout keeps the
rigger's intent. The second decides how many joints a face needs and where
to add them. *(to come: results.)*


## KEYWORDS

facial animation, rigging, linear blend skinning, blendshapes, joint
placement, centroidal Voronoi tessellation


## 1 - INTRODUCTION

A facial blendshape model describes each expression as a set of vertex
offsets from a neutral face. Linear blend skinning (LBS) describes the same
motion with a small number of joints, each moving the vertices it weighs.
Skinning decomposition methods fit an LBS model to blendshapes [Le and Deng
2012; Kavan et al. 2024]. When the joints are anonymous, the solver is free
to put their influence wherever it lowers the error, and the result is a
compressed copy of the blendshapes. When the joints are a rigger's, the
result is meant to be used as a rig: a joint should move the skin around it,
so that moving it by hand has a local, predictable effect.

The second use exposes the layout. A joint placed between two features may
end up driving neither; a joint in a dense cluster may lose its skin to its
neighbours; a region with no joint nearby is forced onto far joints or onto
the root. The skinning solver cannot repair these problems because the
joints are its input.

We treat the layout as its own problem, solved before the skinning solve and
separate from it:

- **Joint placement** (§4) keeps the rigger's joints and their number, and
  moves each joint a bounded distance towards the skin it drives.
- **Joint count** (§5) decides how many joints a face needs and where new
  ones should go. *(open)*

Both solvers leave the final decision to the rigger: they produce a layout to
inspect in the rigging package, not a rig that bypasses review.


## 2 - BACKGROUND AND RELATED WORK

### 2.1 Blendshapes and skinning

A blendshape model has a neutral mesh with vertices $v_i \in \mathbb{R}^3$,
$i = 1 \dots N$, and $S$ shapes, each a set of offsets $d_{si}$. LBS moves
vertex $i$ by a weighted sum of joint transforms,

$$v_i' = \sum_{j} w_{ij}\, M_j\, \bar v_i,$$

where $w_{ij} \ge 0$ are skinning weights, $M_j$ the joint transforms and
$\bar v_i$ the vertex in homogeneous coordinates. Compressed skinning
[Kavan et al. 2024] fits sparse weights and sparse per-shape joint
transforms to the blendshapes with a proximal variant of Adam; with
rigger-supplied joints, the weights are restricted to joints near each
vertex.

### 2.2 Placing centres where the mass is

Moving a point to the centre of the region it serves is the step of Lloyd's
algorithm [Lloyd 1982], whose fixed points are centroidal Voronoi
tessellations [Du et al. 1999]. Our placement step is one such step with two
changes: the region is defined by the skinning weights rather than by
nearest-point cells, and the density is the amount of motion at each vertex.

### 2.3 Joint centres from motion

Biomechanics estimates joint centres from the motion of markers on the
segments they connect [Gamage and Lasenby 2002; Ehrig et al. 2006]. Those
methods recover a rotation centre from rigid motion; facial skin is not
rigid, and a face joint is a handle on a patch of skin rather than an
anatomical pivot, so we use the motion only to weigh where a joint's skin
is.

### 2.4 Choosing where to measure

Adding joints resembles sensor placement, which picks a few locations that
best explain a field [Krause et al. 2008]. Its objectives are often
submodular, so greedy selection carries guarantees; whether the same holds
for joints is part of §5.


## 3 - PROBLEM STATEMENT

**Input.** A neutral mesh with vertices $v_i$ and unit normals $n_i$; the
blendshape offsets $d_{si}$; the rigger's joints with rest positions
$p_j \in \mathbb{R}^3$, $j = 1 \dots J$, and names; skinning weights
$w_{ij}$ from a skinning solve on that layout.

**Output.** New rest positions $p_j'$ (placement), or new joints (count).
Joint orientations and the hierarchy are left as they are; in a rig built
from 4x4 rest matrices, only the translation column changes.

**Symmetry.** Faces are modelled about a mirror plane, here $x = 0$, with
$R = \operatorname{diag}(-1, 1, 1)$. Joints are paired by name: `L_name` and
`R_name` are mirror partners, and `C_` joints lie on the plane. Pairing by
name is enough because rigging naming conventions provide it; detecting
symmetry from geometry is not needed.

**Locality.** We call a joint *local* when the centre of the skin it drives,
$c_j$ (§4.1), lies within half the distance to its nearest other joint:

$$\lVert c_j - p_j \rVert \le \tfrac12\, \delta_j, \qquad
\delta_j = \min_{k \ne j} \lVert p_j - p_k \rVert.$$


## 4 - JOINT PLACEMENT

### 4.1 Where a joint's skin is

Each vertex moves by a different amount across the shapes. We measure it by
its largest offset over the shapes,

$$m_i = \max_{s} \lVert d_{si} \rVert,$$

so a vertex that one shape moves a lot counts fully even when every other
shape leaves it alone, as lip and eyelid skin often is.

and give vertex $i$ the mass $\mu_{ij} = w_{ij}\, m_i$ for joint $j$. Skin
that never moves says nothing about where a tweaker is useful, so it carries
no mass. The centre of the skin joint $j$ drives, and the mean normal there,
are

$$c_j = \frac{\sum_i \mu_{ij}\, v_i}{\sum_i \mu_{ij}}, \qquad
\bar n_j = \frac{\sum_i \mu_{ij}\, n_i}{\lVert \sum_i \mu_{ij}\, n_i \rVert}.$$

A joint with no mass ($\sum_i \mu_{ij} = 0$) does not move.

### 4.2 Tangential move

The centre of a curved patch lies beneath its surface, and the depth of a
joint below the skin is a rigging choice (it sets the arc the skin follows
when the joint rotates). We therefore remove the component of the move along
the mean normal:

$$t_j = (I - \bar n_j \bar n_j^\top)(c_j - p_j).$$

### 4.3 Symmetry

For a pair $(l, r)$ the two moves are averaged in mirrored form and then
mirrored back, so the pair stays symmetric if it was:

$$t_l \leftarrow \tfrac12 (t_l + R\, t_r), \qquad t_r \leftarrow R\, t_l.$$

A centre joint keeps its $x$ coordinate: $t_j \leftarrow (0, t_{j,y}, t_{j,z})$.

### 4.4 Bound

Each move is limited to a fraction $\beta$ of the distance to the nearest
other joint, $b_j = \beta\, \delta_j$, and partners share the smaller of
their two bounds so the clamp keeps them symmetric. The move is shortened,
not discarded:

$$p_j' = p_j + t_j \min\Big(1, \frac{b_j}{\lVert t_j \rVert}\Big).$$

A relative bound lets joints in sparse regions (the neck, the scalp) move
further than joints in dense ones (the lips, the eyelids), where a few
millimetres matter.

**Proposition.** For $\beta < \tfrac12$, no two joints can meet. *Proof.*
For any pair $j, k$, both $b_j \le \beta \lVert p_j - p_k \rVert$ and
$b_k \le \beta \lVert p_j - p_k \rVert$, so
$\lVert p_j' - p_k' \rVert \ge (1 - 2\beta) \lVert p_j - p_k \rVert > 0$.
$\square$ At $\beta = \tfrac12$ two joints that move straight towards each
other can meet.

### 4.5 Iterating *(open)*

The weights $w_{ij}$ depend on the layout, so placement can alternate with
the skinning solve: solve, move, solve again. This is a Lloyd-type iteration
with a bound, and its convergence is not established here. Open questions:
whether the bound should shrink between rounds, whether a fixed point exists
for the weighted regions, and how many rounds a rigger would accept given the
cost of each skinning solve.


## 5 - JOINT COUNT *(open)*

The second solver adds joints where the layout leaves skin without a nearby
tweaker. Candidate formulations, to be compared:

- **Coverage.** Add a joint at the vertex that maximizes moving skin far from
  every joint, $\max_i m_i \min_j \lVert v_i - p_j \rVert$, then repeat; a
  farthest-point rule weighted by motion.
- **Residual.** Add a joint where the skinning solve's error is largest after
  smoothing over the surface, since that is where the layout lacks freedom.
- **Greedy selection.** Score candidate positions by the decrease in fitting
  error they give, and pick greedily; if the gain is submodular in the set of
  added joints, the greedy choice carries the usual $(1 - 1/e)$ guarantee
  [Krause et al. 2008].

Added joints should follow the same symmetry as placement (added in pairs off
the plane, singly on it), and the stopping rule (a target count, an error
threshold, or a locality threshold) is a choice for the rigger.


## 6 - RESULTS *(to come)*

Planned measurements, on publicly available heads, before and after each
solver, with the skinning solve otherwise unchanged:

- accuracy: mean and maximum vertex error over the shapes and over an
  animation;
- locality: the share of joints that are local (§3) and the weight each
  vertex puts on its nearest joint;
- the distribution of moves relative to their bounds.


## 7 - DISCUSSION AND LIMITATIONS

*(to come)*


## REFERENCES

- Q. Du, V. Faber and M. Gunzburger. 1999. Centroidal Voronoi Tessellations:
  Applications and Algorithms. *SIAM Review* 41, 4, 637–676.
- R. M. Ehrig, W. R. Taylor, G. N. Duda and M. O. Heller. 2006. A survey of
  formal methods for determining the centre of rotation of ball joints.
  *Journal of Biomechanics* 39, 15, 2798–2809.
- S. S. H. U. Gamage and J. Lasenby. 2002. New least squares solutions for
  estimating the average centre of rotation and the axis of rotation.
  *Journal of Biomechanics* 35, 1, 87–93.
- L. Kavan, J. Doublestein, M. Prazak, M. Cioffi and D. Roble. 2024.
  Compressed Skinning for Facial Blendshapes. In *ACM SIGGRAPH 2024
  Conference Papers*. arXiv:2406.11597.
- A. Krause, A. Singh and C. Guestrin. 2008. Near-Optimal Sensor Placements
  in Gaussian Processes: Theory, Efficient Algorithms and Empirical Studies.
  *Journal of Machine Learning Research* 9, 235–284.
- B. H. Le and Z. Deng. 2012. Smooth Skinning Decomposition with Rigid
  Bones. *ACM Transactions on Graphics* 31, 6, 199:1–199:10.
- S. P. Lloyd. 1982. Least squares quantization in PCM. *IEEE Transactions
  on Information Theory* 28, 2, 129–137.
