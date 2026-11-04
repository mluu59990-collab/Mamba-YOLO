# Sources and licenses

- `ultralytics/`, `pyproject.toml`, `LICENSE`: sunsmarterjie/yolov12,
  commit `64d5ffc6db35a1bde446689ab8450ecd43475e00`, retrieved 2026-11-04.
  https://github.com/sunsmarterjie/yolov12 (AGPL-3.0).
- `ultralytics/nn/modules/mamba_yolo.py`, `common_utils_mbyolo.py`,
  `selective_scan/`: user-supplied Mamba-YOLO-main archive, associated upstream
  https://github.com/HZAI-ZJNU/Mamba-YOLO. Archive has no verified git commit.
  Original source comments/attribution retained. Selective scan derives from
  VMamba and Albert Gu/Tri Dao's selective scan implementation; retain source notices.
- `paper_blocks.py` is an adaptation from the AKCMamba paper equations/figures,
  with the initial sampling lattice informed by user-supplied AKCMamba_YOLO.zip.
  https://github.com/xlllchen/AKCMamba_YOLO. It is not presented as the authors'
  complete released architecture. Original archives remain outside this repo.

Local modifications: CUDA import handling and CPU reference dispatch; core-only
CUDA build targeting the attached GPU; parser registration; new paper-inspired adapters,
configs, training runner, tests and Kaggle notebook. This distribution preserves
the YOLOv12 base's AGPL-3.0 license in LICENSE.

References:
- Tian, Ye, Doermann. YOLOv12: Attention-Centric Real-Time Object Detectors. 2025.
- Wang et al. Mamba YOLO: A Simple Baseline for Object Detection with State Space Model. AAAI 2025.
- Chen et al. AKCMamba-YOLO: Selective State Space Models For Real-Time Object Detection. CVPR 2026.
