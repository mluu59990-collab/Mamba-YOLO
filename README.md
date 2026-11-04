# Mamba-YOLO: YOLOv12s + MAMBA trên Kaggle

Repo dành riêng cho **mamba**, chuyển ý tưởng tác giả lên nền **YOLOv12s** theo yêu cầu. Đây là biến thể nghiên cứu, không phải repo chính thức hay tái lập nguyên kiến trúc/kết quả paper. Giữ C3k2/A2C2f và Detect của YOLOv12s; thêm VSSBlock ở backbone, XSSBlock ở neck (SS2D, LSBlock, RGBlock).

## Chạy với dữ liệu của bạn

1. Tải [notebook Kaggle](notebooks/kaggle_train.ipynb), Import Notebook vào Kaggle.
2. Bật Internet và GPU; Add Input dataset ở định dạng YOLO detection.
3. Notebook đã có URL clone đúng repo này. Chạy cell setup để cài dependency, build CUDA và kiểm thử.
4. Sửa DATA_ROOT, TRAIN, VAL và NAMES trong cell dataset.
5. Chạy cell train. Mặc định 300 epochs, ảnh 640, batch 4, một GPU. Nếu OOM giảm batch 2 hoặc 1.
6. Save Version kèm outputs để giữ checkpoint cho phiên sau.

```bash
git clone https://github.com/mluu59990-collab/Mamba-YOLO.git
cd Mamba-YOLO
bash scripts/setup_kaggle.sh
python train.py --data /kaggle/working/custom-data.yaml --epochs 300 --batch 4 --device 0
```

## Cấu hình tác giả và các điều chỉnh

- `--profile author` (mặc định): 300 epochs. LR giảm tuyến tính từ 0.01 đến 0.0001 (lrf=0.01), close_mosaic=10 theo source Mamba-YOLO.
- SGD, momentum=0.937, weight_decay=0.0005, warmup 3 epochs (bias LR=0.1, momentum=0.8), box/cls/DFL=7.5/0.5/1.5, HSV=0.015/0.7/0.4, mosaic=1.0.
- Batch 4, GPU 0 và workers 2 là điều chỉnh cho Kaggle. Launcher Mamba gốc mặc định batch 512; 300 epochs/SGD lấy từ launcher, lịch LR lấy từ default.yaml, không khẳng định đây là toàn bộ recipe thực nghiệm trong paper.
- `--profile kaggle` rút còn 100 epochs, không thay lịch LR nói trên. `--epochs`/`--batch` ghi đè rõ ràng.
- Train từ đầu, không tự tải pretrained. Không nạp trực tiếp yolov12s.pt vào hybrid vì index layer đã thay đổi.
- `--variant baseline` chạy đối chứng YOLOv12s cùng recipe. Repo này chỉ cung cấp baseline và mamba.
- Các lớp dùng chung từ bộ tích hợp vẫn có trong source để giữ checkpoint/import ổn định; config và notebook chỉ chọn bản của repo này.

## Resume / validation

```bash
python train.py --resume /kaggle/input/PREVIOUS_RUN/last.pt --data /kaggle/working/custom-data.yaml
python train.py --task val --weights runs/mamba/weights/best.pt --data /kaggle/working/custom-data.yaml
```

Resume phục hồi optimizer/scheduler cho run chưa hoàn tất. Đường dẫn data mới được truyền rõ nếu mount dataset thay đổi. Xem đường dẫn run thực tế vì framework tăng hậu tố khi tên trùng.

## Kiểm chứng và nguồn

Đã kiểm tra CPU forward/loss/backward và lưu/nạp checkpoint. Chưa xác minh CUDA/Kaggle, mAP dataset thật hoặc tốc độ. Setup chạy kiểm tra GPU trước train; không có cam kết tăng độ chính xác hay real-time.

- [Kiểm tra ban đầu](VALIDATION.md) (bộ tích hợp trước khi tách repo)
- [Audit nguồn](AUDIT.md)
- [Nguồn và giấy phép](THIRD_PARTY.md), [LICENSE](LICENSE)
- [Chi tiết triển khai chung](IMPLEMENTATION.md) (tài liệu bộ tích hợp ban đầu; cấu hình chạy của repo riêng được xác định bởi README này và train.py)
