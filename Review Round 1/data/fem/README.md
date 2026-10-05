# Finite-element reference data

The supplied NPZ arrays and Abaqus input decks are retained without numerical changes. The PINN workflows read the exported arrays directly; Abaqus is not a runtime dependency. The package does not include the FEM audit workflow or binary ODB databases.

## Layout

- `benchmark/references/C1.npz` through `C8.npz`, and `T1.npz`: original nodal displacement and integration-point stress references for the controlled benchmarks.
- `benchmark/tr3_*.npz`: available unit-load, load-case and mesh-level Abaqus exports. `UnitL` and `UnitT` denote horizontal and vertical unit-load solutions. These include the original reference mesh series.
- `benchmark/abaqus/*.inp`: available Abaqus input decks.
- `shapes/`: square and slender-cavity reference exports and mesh arrays, including the full-integration solutions and exact tunnel mesh coordinates.
- `evaluation/{C1,C8,L1,S1,T1}.npz`: prepared references for the geometry-feature experiments, including quadrature weights and engineering evaluation locations.

The C1/C8 geometries are circular cavities, L1 is the slender cavity, S1 is the square cavity, and T1 is the engineering tunnel with a nearby elliptical water-filled cavity. The complete geometry and loading data are in `configs/`.

## Array dictionary

| Arrays | Meaning |
| --- | --- |
| `xy_u`, `u` | Nodal positions and displacement components ux, uy |
| `xy_s`, `s` | Integration-point positions and stress components sxx, syy, sxy |
| `xy_ip`, `u_ip`, `s_ip` | Co-located integration-point coordinates, displacement and stress for prepared evaluation |
| `area_weight` | Positive area weights for those integration points |
| `xy_node`, `u_node` | Original nodal displacement reference |
| `wall`, `wall_u`, `wall_weight` | Cavity-wall query points, reference displacement and length weights |
| `wall_tags` | Cavity boundary identity, when multiple or separately tagged boundaries are present |
| `extrema`, `extrema_u` | Crown, invert, right and left positions and reference displacement, in that order |
| `near_wall`, `far_wall`, `corner`, `away_corner` | Fixed Boolean spatial masks |
| `connectivity`, `element_ids`, `element_types`, `ip_id`, `volume` | Available mesh, integration-point identifiers and integration weights in raw exports |

Near-wall regions extend 0.05 normalized length from the cavity. Corner regions use radius 0.02. Vertical convergence is uy(crown) minus uy(invert); horizontal convergence is ux(right) minus ux(left). Relative errors are percentages. Stress vectors use the three components directly without a factor-of-two shear weighting.

The `evaluation/` coordinates and fields are normalized. For T1, multiply coordinates by 75 m, displacements by 0.01875 m and stresses by 2.5 MPa to obtain physical quantities. The controlled `benchmark/references/T1.npz` contains physical coordinates and fields; its evaluator applies the appropriate conversion. Circular, square and slender benchmark scale factors are unity.

FEM wall interpolation uses a small displacement into the solid to avoid ambiguous boundary element lookup. `reference_conventions.json` records these offsets. PINN wall values are evaluated at the specified analytic wall. Do not silently replace the stress integration-point grid by the displacement node grid.

`FILES.csv` in the package root lists file sizes and SHA-256 digests, including every reference export.
