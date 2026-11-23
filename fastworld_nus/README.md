# FastWorld on nuScenes

Run FastWorld inference on the nuScenes v1.0 validation set using six camera
views at two timestamps, the benchmark route command, and causal ego-motion and CAN bus inputs.
The planner outputs six local XY positions at 0.5-second intervals.

## Quick start

Run the following commands from the FastWorld repository root.

### 1. Install

Use a separate Linux x86-64 environment with an NVIDIA GPU, Python 3.8, the
CUDA 11.3 toolkit, and a C++ compiler supported by that toolkit. The tested
runtime uses PyTorch 1.11.0, MMCV-Full 1.4.0,
MMDetection 2.14.0, MMSegmentation 0.14.1, and MMDetection3D 0.17.1.

```bash
conda create -n fastworld-nus python=3.8 -y
conda activate fastworld-nus

git clone https://github.com/BraveGroup/LAW.git ../LAW
git -C ../LAW checkout b2f6a784247072923c477ab92324d3aa5a9759bf
export PYTHONPATH="$(realpath ../LAW):$(pwd)${PYTHONPATH:+:$PYTHONPATH}"

python -m pip install 'pip<24.1' setuptools==59.5.0 wheel ninja
python -m pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 \
  --extra-index-url https://download.pytorch.org/whl/cu113
python -m pip install -r fastworld_nus/requirements.txt

# Point CUDA_HOME at the CUDA 11.3 toolkit and choose an architecture for your GPU.
# CUDA 11.3 can target RTX 4090 through forward-compatible PTX.
export CUDA_HOME=/usr/local/cuda-11.3
export TORCH_CUDA_ARCH_LIST="8.6+PTX"
git clone --branch v1.4.0 --depth 1 https://github.com/open-mmlab/mmcv.git ../mmcv
git -C ../mmcv apply --unidiff-zero "$PWD/fastworld_nus/compat/mmcv-torch111.patch"
MMCV_WITH_OPS=1 FORCE_CUDA=1 python -m pip install --no-build-isolation ../mmcv

git clone --branch v0.17.1 --depth 1 https://github.com/open-mmlab/mmdetection3d.git ../mmdetection3d
git -C ../mmdetection3d apply --unidiff-zero "$PWD/fastworld_nus/compat/mmdet3d-torch111.patch"
FORCE_CUDA=1 python -m pip install --no-deps --no-build-isolation ../mmdetection3d
```

The patches cover PyTorch 1.11 extension compatibility. The requirements file
pins the NumPy version used in the reference run.

### 2. Prepare the data and weights

Download nuScenes v1.0 trainval metadata, validation images, map expansion, and
the CAN bus expansion. Place the VAD temporal train and validation info files in
the dataset directory. The train info file is used to compute CAN normalization
statistics. Download links are in the [nuScenes downloads](https://www.nuscenes.org/nuscenes#download)
and the [LAW data setup](https://github.com/BraveGroup/LAW#7-download-nuscenes-dataset-and-pickle-files).

```text
nuScenes/
├── can_bus/                           # scene-xxxx_pose.json and other streams
├── maps/
├── samples/
├── v1.0-trainval/
├── vad_nuscenes_infos_temporal_train.pkl
└── vad_nuscenes_infos_temporal_val.pkl
```

Download the nuScenes checkpoint from the [results table](../README.md#nuscenes-validation)
and place it at `fastworld_nus/checkpoints/fastworld_nuscenes.pth`.
Its SHA-256 is
`443a095a8f130f4f0f550cdde1e5822f746945a5c6ef35680a07a918a4a6ab3c`.
Use the provided FastWorld checkpoint with the included Swin-T parameters.

```bash
export NUSCENES_ROOT=/path/to/nuScenes
python -m fastworld_nus.can \
  --train-info "$NUSCENES_ROOT/vad_nuscenes_infos_temporal_train.pkl" \
  --val-info "$NUSCENES_ROOT/vad_nuscenes_infos_temporal_val.pkl" \
  --scene-json "$NUSCENES_ROOT/v1.0-trainval/scene.json" \
  --can-root "$NUSCENES_ROOT/can_bus" \
  --output fastworld_nus/outputs/can_val.npz
```

### 3. Run inference

```bash
CUDA_VISIBLE_DEVICES=0 python -m fastworld_nus.infer \
  --data-root "$NUSCENES_ROOT" \
  --val-info "$NUSCENES_ROOT/vad_nuscenes_infos_temporal_val.pkl" \
  --can-features fastworld_nus/outputs/can_val.npz \
  --checkpoint fastworld_nus/checkpoints/fastworld_nuscenes.pth \
  --output fastworld_nus/outputs/predictions.jsonl \
  --score
```

Use `--limit 10` and a fresh output filename for a quick check. Each JSONL row
contains a sample token, six XY increments, and their cumulative positions.
`--score` also computes LAW's temporal-prefix L2 and box-collision metrics on
valid 3-second futures. Omit it for trajectory prediction alone.
