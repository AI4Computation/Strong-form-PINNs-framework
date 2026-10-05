本文件夹包含本轮修稿涉及的研究代码和参考数据。`code` 存放模型、训练与评价程序，`configs` 存放几何、荷载、随机种子和实验设置，`data/fem` 存放有限元参考数据，`data/measurements` 保留原始计时和内存占用记录。论文绘图文件和中间探索材料未包含在内。

运行需要 Python 和 `requirements.txt` 中的依赖库，原实验环境见 `configs/recorded_environment.json`。在本文件夹打开终端，例如执行 `python run.py train --preset budgets --case L1 --method geometry_rbf_fourier --seed 260927 --device cuda`，即可训练对应算例；将命令中的 `train` 改为 `evaluate` 可进行评价。没有可用 GPU 时，将 `cuda` 改为 `cpu`。其他选项可用 `python run.py --help` 查看。

训练权重、预测场和误差结果由程序运行后可得，保存到 `results` 文件夹，因此未随包提供。评价所需的有限元参考数据已保留，无需重新运行 Abaqus。不同硬件及软件版本可能使数值结果略有差异。

----------------------------

This folder contains the research code and reference data associated with the revisions in this round. The `code` directory contains the model, training, and evaluation programs; `configs` contains the geometry, loading conditions, random seeds, and experimental settings; `data/fem` contains the finite element reference data; and `data/measurements` retains the raw timing and memory usage records. Plotting files used for the manuscript and intermediate exploratory materials are not included.

Running the code requires Python and the dependencies listed in `requirements.txt`. The original experimental environment is recorded in `configs/recorded_environment.json`. Open a terminal in this folder and, for example, run `python run.py train --preset budgets --case L1 --method geometry_rbf_fourier --seed 260927 --device cuda` to train the corresponding case. Replace `train` with `evaluate` in the command to perform evaluation. If no GPU is available, replace `cuda` with `cpu`. Other options can be viewed by running `python run.py --help`.

Training checkpoints, predicted fields, and error results are generated after running the program and are saved to the `results` folder; therefore, they are not included in this package. The finite element reference data required for evaluation are provided, so there is no need to rerun Abaqus. Numerical results may vary slightly across different hardware and software versions.