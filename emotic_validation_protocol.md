# Khảo Sát Quy Trình Validation (Validation Protocol) trên Tập Dữ Liệu EMOTIC

Dưới góc độ nghiên cứu học thuật về Thị giác - Ngôn ngữ (Vision-Language Models - VLM) và Affective Computing, tập Validation không chỉ là một tập "chạy nháp" mà là **chìa khóa** quyết định sự thành bại của hệ thống nhận diện cảm xúc đa nhãn (Multi-label Emotion Recognition). 

Dưới đây là tổng hợp chuyên sâu từ các nghiên cứu SOTA (Kosti et al. CVPR 2017, EmotiCon, RAPT, PromptCAD) trên tập dữ liệu EMOTIC.

---

## 1. Mục đích sử dụng tập Validation (Role of Validation Set)

Trong các bài báo hàng đầu, tập Validation của EMOTIC phục vụ 3 mục đích tối quan trọng:

> [!IMPORTANT]
> **1.1. Lựa chọn Checkpoint tốt nhất (Best Model Selection / Early Stopping)**
> Khác với bài toán Single-label (dùng Accuracy để chọn checkpoint), trong đa nhãn, Loss trên tập Valid **thường không phản ánh đúng** năng lực thực tế do sự mất cân bằng dữ liệu cực độ. Các bài báo SOTA **luôn chọn mô hình có Macro-mAP cao nhất** trên tập Valid làm Checkpoint cuối cùng để báo cáo trên tập Test. Hiện tượng Loss Validation tăng nhưng mAP vẫn tăng là chuyện bình thường khi train với Asymmetric Loss.

> [!NOTE]
> **1.2. Tìm kiếm ngưỡng tối ưu từng lớp (Per-class Optimal Threshold Tuning)**
> Xác suất đầu ra của VLM là dải liên tục $\in [0, 1]$. Để quy đổi thành quyết định nhị phân (có/không có cảm xúc đó), việc dùng ngưỡng cứng $T = 0.5$ là một sai lầm chết người trong EMOTIC. 
> Các tác giả (Kosti, EmotiCon) sử dụng tập Valid để quét ngưỡng $T_c \in [0.01, 0.99]$ độc lập cho từng class $c$ (tổng cộng 26 ngưỡng). Ngưỡng nào tối đa hóa được chỉ số **F1-score** của lớp $c$ trên tập Valid sẽ được lưu lại (đóng băng) và áp dụng trực tiếp lên tập Test. 

> [!TIP]
> **1.3. Tinh chỉnh siêu tham số (Hyperparameter Search)**
> Các cấu trúc phức tạp như RAPT hay PromptCAD sử dụng tập Valid để dò tìm: Learning Rate của Adapter vs Backbone, nhiệt độ $\tau$ (Temperature) cho Contrastive Loss, và hệ số Mixup. 

---

## 2. Chỉ số đánh giá trên tập Valid (Validation Metrics)

EMOTIC là bài toán có sự chênh lệch nhãn khủng khiếp (ví dụ: nhãn `Engagement`, `Happiness` có tần suất cực cao, trong khi `Aversion`, `Pain` xuất hiện đếm trên đầu ngón tay). 

- **Tiêu chuẩn Vàng: Macro-mAP (Mean Average Precision)**:
  Tất cả các bài báo đều sử dụng **Macro-mAP** (tính Average Precision - AP cho từng lớp riêng biệt, sau đó lấy trung bình cộng của 26 APs). Nếu dùng Micro-mAP, mô hình chỉ cần dự đoán đúng 2-3 nhãn chiếm đa số là điểm đã cao chót vót, làm che giấu đi sự yếu kém ở các nhãn thiểu số (rare classes).
  
- **Bản chất của mAP**:
  Average Precision là một metric **Threshold-free (không phụ thuộc ngưỡng quyết định)**. Nó dựa vào việc *xếp hạng (ranking)* các dự đoán dựa trên xác suất từ cao xuống thấp để vẽ đường cong Precision-Recall Curve. Do đó, mAP là thước đo thuần túy và khách quan nhất về khả năng tách biệt đặc trưng của mô hình.

- **Các chỉ số phụ trợ**:
  Bên cạnh 26 nhãn rời rạc, EMOTIC còn cung cấp nhãn liên tục VAD (Valence, Arousal, Dominance). Các mô hình Joint-learning sẽ theo dõi thêm MAE (Mean Absolute Error) của nhánh VAD. Dù vậy, mAP của 26 classes vẫn là metric định hướng chính.

---

## 3. Kỹ thuật đánh giá nâng cao (Inference / Evaluation Tricks)

Khi đánh giá mô hình VLM đa phương thức trên tập Valid/Test, các SOTA không bao giờ đánh giá một cách "ngây thơ" (naive evaluation). Họ áp dụng các trick sau để tối đa hóa mAP:

- **Exponential Moving Average (EMA)**: 
  Được dùng phổ biến (ví dụ trong RAPT). Thay vì dùng trọng số thô tại epoch thứ $N$, mô hình dùng bản sao EMA (Trung bình cộng lũy thừa của trọng số qua các batch) để đánh giá. EMA làm phẳng các nhiễu loạn của gradient, giúp mAP trên Valid tăng từ 1-2% một cách rất ổn định.
  
- **Test-Time Augmentation (TTA)**:
  Các mô hình như EmotiCon hay PromptCAD thường dùng kỹ thuật Multi-crop (cắt 5 góc: 4 góc + 1 trung tâm) hoặc L/R flipping. Đầu ra (Logits) của tất cả các biến thể được trung bình cộng lại trước khi đi qua Sigmoid. Điều này mang lại sự ổn định mạnh mẽ cho mAP.

- **Tắt Regularization (Bắt buộc)**:
  Trong bước gọi `model.eval()`, mọi kỹ thuật nhiễu (Mixup, Label Smoothing, Modality Dropout - ví dụ như việc ngẫu nhiên tắt luồng Context/Body) đều bị vô hiệu hóa hoàn toàn để bộ phân loại phát huy 100% công suất của cả 3 luồng dữ liệu (Face, Body, Context).

---

## 4. Tổng kết: 3 Quy Tắc Vàng (Best Practices) với EMOTIC Validation

> [!CAUTION]
> Bất kỳ ai thực hiện nghiên cứu trên EMOTIC đều phải nằm lòng 3 nguyên tắc sau nếu muốn công bố lên các hội nghị/tạp chí Q1:

1. **Rule #1: Chỉ tôn sùng Macro-mAP, Phớt lờ Validation Loss.**
   Loss trên tập Valid có thể chững lại hoặc tăng nhẹ ở các epoch cuối (Overfitting nhẹ), nhưng nếu Macro-mAP vẫn tăng, điều đó chứng tỏ mô hình đang phân tách (ranking) các nhãn thiểu số tốt hơn. **Luôn lưu checkpoint theo Macro-mAP.**

2. **Rule #2: Khai thác Per-class Thresholds.**
   Nếu bạn phải báo cáo bảng chỉ số Precision, Recall và F1-score trong paper, tuyệt đối không dùng ngưỡng cố định $T=0.5$. Hãy viết một hàm quét ngưỡng trên tập Valid để tìm 26 ngưỡng riêng biệt cho 26 nhãn. Đối với những nhãn khó (ví dụ: `Pain`, `Aversion`), ngưỡng tối ưu có thể rơi xuống mức $0.05 - 0.1$.

3. **Rule #3: Xác thực độc lập (Independence).**
   Không rò rỉ (leak) dữ liệu từ tập Test vào tập Valid dưới bất kỳ hình thức nào (kể cả việc tính phân phối prior bias). Tập Valid phải được xem như một "Môi trường giả lập Test" khắc nghiệt để bạn lựa chọn trọng số EMA và Threshold chuẩn xác nhất.
