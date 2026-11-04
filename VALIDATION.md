# Kết quả kiểm tra local — 2026-11-04

Môi trường: macOS ARM64, Python 3.10.19, PyTorch 2.9.1 CPU,
torchvision 0.24.1, timm 1.0.14, einops 0.8.1.

| Kiểm tra | Kết quả |
|---|---|
| pytest: scan/merge hình chữ nhật và gradient | PASS |
| pytest: AKConv zero offset, nội suy ở biên, gradient offset | PASS |
| pytest: 3CAKC và 4CAKC forward/backward | PASS (2 bài) |
| pytest: half checkpoint + FP32 scan/LayerNorm | PASS |
| CUDA scan so với reference; CUDA AMP/half inference | SKIP (không có CUDA local) |
| baseline: detector loss/backward + checkpoint reload | PASS |
| mamba: detector loss/backward + checkpoint reload | PASS |
| akcmamba: detector loss/backward + checkpoint reload | PASS |
| AKC train 1 epoch + validation + best/last checkpoint + final validation | PASS |
| Cú pháp Python, Bash và code cell notebook | PASS |

Tổng unit test: **5 passed, 2 skipped**. Model smoke test dùng batch 2, ảnh 96×96,
80 lớp, output `(2, 84, 189)`; gradient SSM và offset đã kiểm tra có giá trị khác 0.
Checkpoint roundtrip cho phép sai số lượng tử hóa vì Ultralytics lưu weights FP16.

End-to-end train sử dụng **dữ liệu giả ngẫu nhiên** (4 ảnh train, 2 ảnh val, 1 lớp),
batch 2, ảnh 96×96, CPU, không AMP. Exit code 0; tạo results.csv, args.yaml,
environment.json và hai checkpoint. Đây chỉ là kiểm tra hệ thống;
**mAP từ dữ liệu giả không có ý nghĩa đánh giá chất lượng mô hình**.
AKC adaptation có khoảng 45.86 triệu tham số ở bài test 1 lớp; lớn hơn baseline.
Ước lượng FLOPs của profiler không bao phủ đầy đủ selective scan/grid sampling,
không dùng để chứng minh tốc độ hay so sánh với số đo trong paper.

Chưa kiểm chứng: build CUDA trên Kaggle, GPU AMP/half, VRAM tại 640×640,
train dài trên dataset thật, resume ở giữa một run dài, mAP/FPS, DDP hoặc export.
`scripts/setup_kaggle.sh` chạy các kiểm tra GPU trước khi cho phép tiếp tục train.

Hai cảnh báo deprecated từ API timm/torch AMP của nguồn Mamba cũ không gây lỗi
trong phiên bản đã thử. Source CUDA được giữ để build theo GPU/toolchain của phiên Kaggle.

## Kiểm tra sau khi tách repo

2026-11-04: 6 unit tests PASS, 2 CUDA tests SKIP. Bao gồm kiểm tra resume tôn trọng đường dẫn dataset mới và workers. Smoke test baseline và variant của repo đều PASS forward, detector loss/backward, checkpoint reload trên CPU. Notebook được kiểm tra JSON và cú pháp từng code cell. CUDA/Kaggle vẫn chưa được chạy tại môi trường local.
