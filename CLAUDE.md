# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**facial-joints-layout** holds tools that optimize the joint layout of a face rig, so that its
blendshapes convert well into skinning and each joint works as a tweaker for the skin around it.
The tools are separate from the compression itself: they prepare the joints that a
blendshape-to-skinning solver then binds the face to.

Planned solvers:

| Solver | What it does |
|---|---|
| `JointPlacementSolver` | Moves the rigger's joints to where they drive their skin best. Joints are paired by their `L_` / `R_` name prefixes and move as mirror images; `C_` joints stay on the mirror plane. Each move is bounded by a fraction of the distance to the nearest other joint. |
| `JointCountSolver` | Decides how many joints a face needs and where to add them. |

The target is Maya-style 4x4 joint matrices; judge results by vertex positions after skinning.

## Commands

```bash
pip install -e ".[dev]"          # development install
pytest                           # all tests
ruff format .                    # format
ruff check --fix .               # lint and auto-fix
mypy src/                        # type checking
```

## Source layout

`src/facial_joints_layout/`, tests in `tests/`, the paper in `paper/`.

## Paper

`paper/facial_joints_layout.md` is a paper written alongside the code, in the style of the compressed
skinning paper: method, math and results. Keep it and the code in step: when a solver changes, update the
matching section and its equations in the same change. Only publicly available references and results on
publicly available data go in it. Mark unsettled parts *(open)* and missing results *(to come)*; never
state a result that has not been measured.

## Testing

### TDD (Test-Driven Development)

Follow the **Three Laws** strictly:
1. Write no production code until you have a failing test
2. Write no more test code than sufficient to fail
3. Write no more production code than sufficient to pass

**Cycle**: Red → Green → Refactor. Always show the failing test first, then minimal passing code, then cleanup.

**FIRST principles**: Tests must be **F**ast, **I**ndependent, **R**epeatable, **S**elf-validating, **T**imely.

**Structure**: Arrange-Act-Assert, every test. Test names describe what is being tested. Tests target the
highest-level API possible. Never disable a failing test — fix it.

## Code Quality Principles

- **Incremental progress**: small changes that pass tests
- **Clear intent over clever code**: be boring and obvious
- **Naming**: intention-revealing names; nouns for classes, verbs for functions; no single-letter names
  except loop variables and matrix symbols in short functions. Name things after what they do, never after
  plan or ticket codes.
- **Functions**: small, one thing, one level of abstraction; ≤2 arguments preferred; no side effects;
  Command Query Separation
- **Comments** explain why, not what; prefer extracting a named function over a comment
- **Errors**: exceptions with context, never error return codes; never return or pass `None` for errors
- **Python**: type hints on every signature; Google-style docstrings on every public class, method and
  function (enforced by Ruff); f-strings; `@dataclass` for data containers
- **Scientific computing**: vectorize with NumPy; set random seeds; handle numerical edge cases
  (division by zero, empty sets)
- No premature optimization; stop after 3 failed attempts and reassess

### Array shape conventions

| Symbol | Meaning |
|--------|---------|
| `N` | Vertices |
| `J` | Joints |
| `S` | Blendshapes |

Annotate array shapes in docstrings and inline comments (e.g. `# shape: (J, 3)`).

## Docstrings

Google style. Sections in order: summary line (ends with a period), extended description, `Args:`,
`Returns:`, `Raises:`, `Attributes:`, `Note:`, `Example:`, `References:` (papers, sections, equations for
mathematical implementations). Math uses LaTeX: `` :math:`p' = p + d` ``.

## Git Workflow

Conventional Commits (`feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `chore`):

```
<type>(<scope>): <short summary>

<detailed description>
```

Branches: `feature/`, `bugfix/`, `hotfix/`, `refactor/`, `docs/` + short description.

## Before committing

- [ ] `pytest` — all tests pass
- [ ] `ruff format .` and `ruff check .` — clean
- [ ] `mypy src/` — clean
