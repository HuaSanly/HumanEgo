# HumanEgo

HumanEgo 使用第一人称视频训练机器人操作策略，包含数据预处理、模型训练、仿真和远程推理服务。以下命令均在项目根目录执行。

本仓库保留 `main` 作为稳定分支，日常开发在 `future-main` 上完成；未经所有者明确要求，不得创建其他分支，也不向 `main` 合并。见 [仓库约束](AGENTS.md)。

## 安装

```bash
git clone --branch future-main https://github.com/HuaSanly/HumanEgo.git
cd HumanEgo
conda create -n humanego python=3.11 -y
conda activate humanego
bash setup.sh
```

PICO Spatial 数据预处理还需要 HaMeR 依赖以及系统中的 `ffmpeg` 和 `ffprobe`：

```bash
SKIP_HAND=0 bash setup.sh      # 安装 HaMeR 等手部跟踪依赖

# 可选：安装硬件驱动或预下载模型权重
SKIP_HARDWARE=0 bash setup.sh
PREDOWNLOAD=1 bash setup.sh
```

## PICO Spatial 数据预处理

将 PICO SpatialMP4 录制文件放在同一目录中，准备对应任务配置（例如 `cfg/preprocess/tasks/pick_cube.yaml`），然后运行：

```bash
# 一次执行准备和后续预处理
python -m preprocess.SpatialVideoPreprocess --input-dir /path/to/recordings --task pick_cube --stage all
```

输入目录只扫描其直属的 MP4 文件。输出位于 `data/pick_cube/spatial/spatial_pick_cube_<编号>/preprocess/`，录制索引保存在 `data/pick_cube/spatial_index.json`。也可以先运行 `--stage prepare`，再运行 `--stage downstream`；默认只执行 `prepare`。已完成的阶段会跳过，失败或中断的结果需要人工检查。

当前适配器使用左目图像和 HaMeR 手部重建，不融合双目或深度数据；默认头部偏移使用未经实测验证的参考值。

## 训练

```bash
python -m training.FlowMatchingTrainer --task pick_cube --use_cfg --job HumanEgo
```

配置位于 `cfg/training/<task>/<job>.yaml`，结果保存在 `runs/<task>/<job>/`。`pick_cube` 配置使用 Spatial 数据，并将编号 `000` 留作评估；至少需要两段数据。详见 [训练文档](training/README.md)。

## 仿真

仿真可使用单独的 Python 3.11 环境：

```bash
conda create -n humanego-sim python=3.11 -y
conda activate humanego-sim
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt -r simulation/requirements.txt

# serve_bread：下载权重、生成场景并运行
huggingface-cli download Leo-TX/HumanEgo --include "serve_bread/*" --local-dir ./checkpoints
python simulation/scripts/build_assets.py
python simulation/scripts/run_episode.py --video

# pick_cube：使用配置中指定的本地模型
python simulation/scripts/build_assets.py --table-only
python simulation/scripts/run_episode.py --config simulation/configs/pick_cube.yaml --video --out pick_cube_model
```

运行 `pick_cube` 前，请检查 `simulation/configs/pick_cube.yaml` 中的 checkpoint 路径，并确保同目录包含 `config.json` 和 `dataset_stats.json`。视频和日志保存在 `simulation/runs/`。更多配置及网页回放见 [仿真文档](simulation/README.md)。

### pick_cube 多轮测试

使用已安装仿真依赖的环境，在不同物块、容器位置及模型推理噪声种子下批量测试：

```bash
# 10 个位置组合 × 3 个推理噪声种子 = 30 轮测试，默认单进程运行
python simulation/scripts/evaluate.py \
  --config simulation/configs/pick_cube.yaml \
  --n 10 --scene-seed 7777 --policy-seeds 0 1 2 \
  --out pick_cube_eval_dev

# 中断后继续：读取已冻结的配置和测试清单，跳过已完成的轮次
python simulation/scripts/evaluate.py --out pick_cube_eval_dev --resume

# 从已保存的结果更新报告并生成失败轮次的视频，不重新运行模型
python simulation/scripts/evaluate.py --out pick_cube_eval_dev --report-only --videos failures
```

`--n` 指位置组合数，总测试轮数为 `--n × --policy-seeds 的数量`；`--scene-seed` 控制位置采样，`--policy-seeds` 控制同一模型的推理噪声。默认在当前位置周围每轴 ±3 cm 随机摆放，范围可在 `simulation/configs/eval_pick_cube.yaml` 中调整；使用 `--workers 2` 可开启双进程。

结果保存在 `simulation/runs/pick_cube_eval_dev/`：`report.md` 为评测报告，`summary.json` 为汇总统计，`results.csv` 为逐轮结果，`trials.json` 为固定位置和种子清单。首次运行请选择新的 `--out` 名称；后续使用 `--resume`，若需重试程序异常轮次，再添加 `--retry-errors`。

测试沿用任务配置中的抓取、自动抬升和释放辅助设置，实际开关会记录在报告中。冻结配置、复用同一清单比较不同模型或控制参数等用法，见 [多轮评测详细说明](simulation/README.md#reproducible-pick_cube-batch-evaluation)。

## 推理服务器

推理服务使用独立的 Python 3.12 Conda 环境：

```bash
conda create -n humanego-inference python=3.12 -y
conda activate humanego-inference
bash setup.sh inference

# 检查配置和运行时依赖
python -m inference.config --config_path cfg/inference/params.yaml --validate-runtime


# 导出部署包并启动服务
humanego-export --config_path cfg/inference/params.yaml --output runs/deploy/humanego-anchor
python -m lerobot_policy_humanego.server --config_path runs/deploy/humanego-anchor/server.yaml
```

导出前需准备 `cfg/inference/params.yaml` 指定的 checkpoint 及配套文件，并核对双目标定和末端坐标配置。导出目录不能是已有的非空目录。客户端接口和部署说明见 [推理文档](inference/README.md)与[客户端接口](inference/CLIENT_INTERFACE.md)。

## 许可与引用

项目采用 [PolyForm Noncommercial License 1.0.0](LICENSE)，商业使用需要单独授权。[项目主页](https://humanego-ai.github.io) · [论文](https://arxiv.org/abs/2605.24934) · [数据集](https://huggingface.co/datasets/Leo-TX/HumanEgo) · [模型权重](https://huggingface.co/Leo-TX/HumanEgo)

```bibtex
@misc{humanego2026,
  title         = {HumanEgo: Zero-Shot Robot Learning from Minutes of Human Egocentric Videos},
  author        = {Wang, Zhi and He, Botao and Yu, Kelin and Lee, Seungjae and Gao, Ruohan and Huang, Furong and Aloimonos, Yiannis},
  year          = {2026},
  eprint        = {2605.24934},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO}
}
```

## Aria 数据下载与预处理

原始 Project Aria 链路仍可使用公开数据集：

```bash
# 下载两段原始录制及 MPS 数据
python scripts/download_data.py --task serve_bread --num 2 --input-only

# 逐段运行预处理
python -m preprocess.Preprocess --mps_path ./data/serve_bread/aria/mps_serve_bread_000_vrs --task serve_bread
python -m preprocess.Preprocess --mps_path ./data/serve_bread/aria/mps_serve_bread_001_vrs --task serve_bread
```

去掉 `--input-only` 可下载已预处理的数据；`--num all` 下载该任务的全部录制。详见 [Aria 预处理文档](preprocess/README.md)和[数据采集文档](datacollection/README.md)。
