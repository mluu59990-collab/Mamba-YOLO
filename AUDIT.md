# Kiểm tra hai bộ mã đầu vào

1. `Mamba-YOLO-main/mbyolo_train.py` và `AKCm.py` trong ZIP đều dùng `ROOT + opt.data`
   mặc dù default đã chứa ROOT. Đường dẫn tuyệt đối bị nối lặp.
2. Cả hai tạo dictionary bằng kết quả `.train()`, `.val()`, `.test()` thay vì callable.
   Python chạy ngay các lời gọi khi tạo dictionary; lựa chọn `--task` không điều khiển
   được tác vụ. `.test()` cũng không phải API tương ứng của YOLO.
3. Default 8 GPU, batch 512, workers 128 không phù hợp một notebook Kaggle.
   `imgsz` được parse nhưng không truyền vào lời gọi train cũ.
4. Hai fork dùng namespace `ultralytics` và kiến trúc cũ; không có cấu hình YOLOv12s.
5. Archive AKC có định nghĩa AKConv nhưng SS2D trong
   `CoreV8/Backbone/mamba.py` vẫn gán `self.conv2d = nn.Conv2d`.
   YAML `v8/yolov8.yaml` vẫn dùng C2f chuẩn. Không thể coi việc chạy YAML đó
   là đã chạy đầy đủ sơ đồ 3CAKCMamba/4CAKCMamba trong PDF.
6. AKConv gốc chỉ zero weight offset, bỏ bias; hook gradient tạo generator nhưng
   không trả về nên không thực hiện việc nhân learning rate như tên hàm gợi ý.
7. Import selective scan cũ che lỗi trong `except` rồi thử import một extension khác,
   không cung cấp chẩn đoán/cơ chế kiểm thử CPU rõ ràng.

## Cách xử lý

Tạo một repo tích hợp sạch để upload, giữ hai nguồn gốc ngoài repo làm tài liệu đối chiếu.
Runner mới dùng Path và if/elif, bắt buộc checkpoint cho val/predict, lưu cấu hình và
môi trường; ba YAML có scale s rõ ràng và index skip/Detect đã ánh xạ lại sau khi thêm block.
Module mới được đăng ký trực tiếp trong parser, nên rebuild lúc training và reload
checkpoint không phụ thuộc monkey-patch trong một cell notebook.

## Giới hạn khoa học

Paper AKC thay block trên YOLOv8; repo này thêm block lên YOLOv12s để giữ area attention.
Không gọi hybrid là nguyên bản paper, không gán cho hybrid số Params/FLOPs/mAP/FPS
của paper. Cần train và đo riêng, cùng split, seed, preprocessing và thiết bị.
Preset paper khớp các tham số được nêu trong mục 4.1.2, các giá trị chưa được nêu
được liệt kê trong README. Không có dataset/weight của người dùng trong thư mục đầu vào.

## Xác minh

Xem `VALIDATION.md` cho kết quả kiểm tra thực tế. Máy kiểm tra local không có CUDA;
build CUDA, AMP GPU, throughput và chất lượng mAP phải xác minh trên Kaggle.
