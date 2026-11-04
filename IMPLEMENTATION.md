# YOLOv12s + Mamba / AKCMamba trên Kaggle

Repo này cung cấp **ba thí nghiệm detection**: `baseline`, `mamba`, `akcmamba`.
Nền là mã YOLOv12 chính thức, scale **s = [0.50, 0.50, 1024]**.
Hai bản hybrid giữ các khối C3k2/A2C2f và đầu Detect của YOLOv12s, thêm các khối
theo ý tưởng hai paper. **Đây là adaptation sang YOLOv12s, không phải tái lập nguyên
kiến trúc hay kết quả AKCMamba-YOLO trong paper (nền YOLOv8).** Không có cam kết tăng mAP/FPS.

## 1. Đưa lên GitHub

Upload **nội dung thư mục `yolov12s-paper-kaggle`** làm root repository của bạn.
Không upload hai ZIP gốc, thư mục `tmp`, dataset hay checkpoint. Giữ `LICENSE` và
`THIRD_PARTY.md`. Repo đã chứa mã Python YOLOv12 và source CUDA cần thiết;
không cần clone thêm hai repo Mamba vào notebook, tránh xung đột package `ultralytics`.

Với Git CLI, đứng trong thư mục này rồi chạy (thay URL repo trống của bạn):

```bash
git init -b main
git add .
git commit -m "Add YOLOv12s Mamba and AKCMamba Kaggle experiments"
git remote add origin https://github.com/YOUR_ACCOUNT/YOUR_REPO.git
git push -u origin main
```

## 2. Chạy notebook Kaggle

Bật **Internet** và **GPU** trong notebook settings. Mở notebook
[`notebooks/kaggle_train.ipynb`](notebooks/kaggle_train.ipynb), sửa URL GitHub và đường dẫn data.
Hoặc chạy các cell sau:

```python
!git clone https://github.com/YOUR_ACCOUNT/YOUR_REPO.git /kaggle/working/yolov12s-paper
%cd /kaggle/working/yolov12s-paper
!bash scripts/setup_kaggle.sh
```

Setup giữ cặp torch/torchvision sẵn có của Kaggle, build `selective_scan_cuda_core`
với CUDA hiện tại và chạy kiểm tra số học/gradient/model. Không cài wheel FlashAttention
gắn cứng Python 3.11/CUDA 11 từ requirements cũ. GPU T4/P100 dùng PyTorch SDPA fallback
có sẵn trong mã YOLOv12. CUDA build/smoke test phải thành công trước khi train.
Nếu đã import một bản ultralytics khác trong kernel, restart session hoặc dùng các
lệnh `!python` bên dưới (mỗi lệnh tạo process mới).

Data phải ở định dạng YOLO detection, nhãn mỗi dòng: `class x_center y_center width height`
(tọa độ chuẩn hóa 0..1). Ví dụ tạo file ở vùng ghi được:

```yaml
# /kaggle/working/data.yaml
path: /kaggle/input/YOUR_DATASET
train: images/train
val: images/val
names:
  0: object
```

Cấu trúc nhãn tương ứng là `labels/train` và `labels/val`. Sửa tên lớp và đường dẫn
theo dataset thực tế; không tự chia lại validation/test. Với layout Roboflow,
thường dùng `train: train/images`, `val: valid/images`.

```python
!python train.py --variant akcmamba --data /kaggle/working/data.yaml --epochs 100 --batch 4 --device 0
```

Mặc định là train từ đầu. Để so sánh công bằng, chạy cùng dataset/seed/batch/epochs:

```bash
python train.py --variant baseline --data /kaggle/working/data.yaml --epochs 100 --batch 4
python train.py --variant mamba --data /kaggle/working/data.yaml --epochs 100 --batch 4
python train.py --variant akcmamba --data /kaggle/working/data.yaml --epochs 100 --batch 4
```

`--profile paper` đặt 500 epochs, batch 32 theo mục 4.1.2 của AKC; cần đủ VRAM.
Cả hai profile dùng SGD, lr0=0.01, **lrf=1.0** (LR cố định sau warmup), warmup 3 epochs,
weight decay=0.0005, box/cls/DFL=7.5/0.5/1.5, HSV=0.015/0.7/0.4, mosaic=1.0.
Momentum 0.937, phép augment khác, nbs/gradient accumulation còn theo mặc định YOLOv12;
paper không đặc tả tất cả các giá trị đó. Vì vậy profile này chỉ khớp các tham số
được nêu, không bảo đảm tái lập toàn bộ chế độ train của tác giả.

Nếu OOM, giảm `--batch 2` hoặc `--batch 1` và ghi rõ thay đổi trong báo cáo.
Hybrid thêm tham số/bộ nhớ so với baseline. CPU reference chỉ dùng kiểm tra nhỏ,
không dùng train 640. Chưa hỗ trợ kiểm chứng DDP, export ONNX/TensorRT hay tốc độ real-time.

## 3. Tiếp tục / đánh giá / dự đoán

```bash
python train.py --resume runs/akcmamba/weights/last.pt
python train.py --task val --weights runs/akcmamba/weights/best.pt --data /kaggle/working/data.yaml
python train.py --task predict --weights runs/akcmamba/weights/best.pt --source /kaggle/input/YOUR_DATASET/images/val
```

Resume tiếp tục lịch train cũ, không kéo dài số epoch đã kết thúc. Lưu `last.pt`
ra Kaggle Output để dùng trong phiên mới. Dùng checkpoint do repo này tạo;
không nạp trực tiếp `yolov12s.pt` vào hybrid vì các index layer đã thay đổi.
Kết quả có `results.csv`, `args.yaml`, `environment.json`, `weights/best.pt`, `weights/last.pt`.
Tên run trùng sẽ được Ultralytics tăng hậu tố; xem đường dẫn được in ở cuối train.

## 4. Ánh xạ ý tưởng paper

| Cấu hình | Backbone | Neck | Thành phần giữ từ YOLOv12s |
|---|---|---|---|
| baseline | nguyên bản | nguyên bản | tất cả |
| mamba | thêm VSSBlock sau P3/P4/P5 | thêm XSSBlock ở 4 khối fusion | C3k2, A2C2f/area attention, downsample, Detect |
| akcmamba | thêm 3CAKCMamba sau P3/P4/P5 | thêm 4CAKCMamba ở 4 khối fusion | C3k2, A2C2f/area attention, downsample, Detect |

- Mamba dùng SS2D bốn hướng, LSBlock (local spatial) và RGBlock (gated MLP) từ mã được cung cấp.
- AKC dùng AKConv offset học được + nội suy bilinear; AKCBlock, 3CAKC/4CAKC theo (6)-(8);
  AKSS2D theo Figure 3(a); AKCAttention có SE theo (10); khối tích hợp theo (11)-(12).
- Scan bốn hướng là hàng/cột thuận và nghịch như CrossScan Mamba-YOLO.
- Quyết định triển khai khi paper không đặc tả đủ: 5 điểm AKConv, SE ratio 16,
  d_state 16, SSM expansion 2, shortcut bật. Đây là lựa chọn triển khai, không phải
  khẳng định các hyperparameter bí mật của tác giả.
- AKConv sửa bias offset khởi tạo về 0 và lấy mẫu biên bằng `grid_sample` để tránh
  cộng trùng corner ở biên. Bỏ hook `_set_lr` không có hiệu lực trong mã gốc.
- CUDA thực hiện S6 thật; khi thiếu kernel, training GPU dừng với lỗi rõ ràng,
  không thay Mamba bằng Conv/attention hay âm thầm chạy vòng lặp CPU.

Xem [`AUDIT.md`](AUDIT.md) để biết lỗi trong hai bộ mã đầu vào và giới hạn xác minh.

## 5. Kiểm thử

```bash
python -m pytest -q tests/test_paper_blocks.py
python scripts/smoke_test.py --device cuda
```

Kiểm thử scan/merge hình chữ nhật, AKConv tại biên, gradient offset/SSM,
CUDA so với recurrence tham chiếu, AMP, loss detection và lưu/nạp checkpoint.
CPU tự skip hai bài CUDA; setup Kaggle chạy chúng với GPU thật.

Nguồn: [YOLOv12](https://github.com/sunsmarterjie/yolov12),
[Mamba-YOLO](https://github.com/HZAI-ZJNU/Mamba-YOLO),
[AKCMamba-YOLO](https://github.com/xlllchen/AKCMamba_YOLO).
