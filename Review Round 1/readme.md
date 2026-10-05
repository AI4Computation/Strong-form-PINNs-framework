# Review Round 1: code and finite-element references

This release contains the four research directories, their source code and experiment configurations, and the finite-element reference data. Training outputs are omitted. The 299 supplied research files occupy approximately 1.90 GiB; `FILES.csv` records their sizes, categories and SHA-256 checksums.

| Directory | Contents |
|---|---|
| `controlled_pinn` | Controlled PINN, XPINN and DEM implementations, configurations, Abaqus input decks and output databases, exported FEM fields, selected references and FEM verification records. |
| `geometry_pinn` | Geometry-feature implementations, protocols, geometry definitions, FEM evaluation references and reference-coordinate/mesh verification data. |
| `validation_and_cost` | Validation and computational-cost implementations and configurations, square/slender-cavity FEM references, meshes and verification records. |
| `diagnostics` | DEM history-diagnostic source code. |

## Included and omitted data

Finite-element reference results are included even when they can be regenerated: all files under `controlled_pinn/fem` and `validation_and_cost/shape_references`, plus the FEM reference fields and associated mesh/boundary checks under `geometry_pinn/results`. The presence of selected files in a `results` directory therefore does not indicate that neural-network predictions are included.

Trained model weights and checkpoints, neural-network predictions, training histories, timing measurements, training summaries and other generated experiment outputs are not supplied. Regenerate the required upstream results before running downstream evaluation, diagnostics or reporting scripts. A new run does not reproduce historical timing measurements exactly.

The source and retained data files are copied without modification from the research archive. The manifest covers those research files, not this README or repository packaging files.

## Downloading the finite-element data

Three files exceed GitHub's regular Git file-size limit and are stored with Git LFS:

- `controlled_pinn/fem/abaqus/tr3_circle_sq0p0025.odb`
- `controlled_pinn/fem/abaqus/tr3_tunnel_tq0p125.odb`
- `geometry_pinn/results/R2_P2J_reference_repair/evaluation/T1_reference.npz`

Use a Git LFS-enabled client to clone the repository, then run `git lfs pull` from the repository root. Verify the downloaded files against `FILES.csv`; an LFS pointer is not the actual reference data. Other finite-element data files are stored directly in Git.

## Execution layout and dependencies

This is a source/data release with external dependencies, not a standalone installer. To preserve the scripts' original relative paths, use a separate execution copy with this layout:

```text
execution_root/
  research/
    controlled_pinn/
    geometry_pinn/
    validation_and_cost/
    diagnostics/
  代码源文件/
    审稿修改-基线对比/benchmark_suite.py
```

Copy the four supplied directories into `execution_root/research`. Some benchmark and DEM implementations import the external `benchmark_suite.py` shown above. Original-experiment preparation and reevaluation also refer to original inputs/states under `代码源文件`, `abaqus-36/processed`, `_work_revision` and `再次提交TUST`. Some geometry-transfer and audit scripts require the original frozen inventories under `research/00_基线与规则`. These external files are not included in this four-directory release. Obtain the matching original dependencies before running those entry points.

Python implementations use NumPy, SciPy and PyTorch; selected utilities also use Matplotlib, psutil, threadpoolctl and Windows performance interfaces. GPU training requires a compatible CUDA environment. Abaqus operations require Abaqus and its Python environment. Configure machine-specific Abaqus job paths and executable paths for the target machine.

Batch scripts contain existing-result guards and historical source/input hash checks. Retaining references does not make every historical batch immediately rerunnable: use an execution copy and satisfy the particular entry point's upstream dependencies and integrity checks. No training, FEM solve or timing benchmark was rerun when preparing this release.
