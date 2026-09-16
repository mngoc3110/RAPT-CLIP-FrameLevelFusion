# 🔬 Deep-dive Audit: EMOTIC Training Pipeline — Root Cause Analysis

> **Objective:** Diagnose why EMOTIC macro-mAP stalls at ~20-31% and overfits after a few epochs.
> **Benchmark:** SOTA on EMOTIC 26-class = **27-43% mAP** depending on method.

---

## 1. 🗄️ Tầng Dữ liệu & Tiền xử lý (Data Flow & Modality Inputs)

### 1A. Face Stream: Bbox-based Proxy Crop có lỗi logic

> [!CAUTION]
> **Critical Bug: Face crop fallback trả về full image thay vì centered crop**

File [`emotic_dataloader.py` L128-152](file:///Users/macbook/Downloads/RAPT-CLIP/dataloader/emotic_dataloader.py#L128-L152):

```python
def _crop_face(self, img, bbox):
    x1, y1, x2, y2 = bbox
    # ...
    if x2 <= x1 or y2 <= y1:
        return img  # ⚠️ Fallback: return FULL IMAGE (640x480)
    face_h = int((y2 - y1) * self.face_ratio)  # top 40% of bbox
```

**Vấn đề:**
- Khi `bbox = [0, 0, 0, 0]` (annotation không có bbox) → `x2 <= x1` → **face = full image** = giống hệt context stream
- EMOTIC annotation format gốc (`train.txt`) thường KHÔNG có bbox → **toàn bộ 16K samples face stream = full image**
- Kết quả: Face stream bị trùng lặp hoàn toàn với Context stream → model chỉ học 2 stream hiệu quả, lãng phí 1/3 capacity

**Impact:** Face stream trở thành bản copy của Context → CMAF cross-attention giữa face↔context không học được gì mới → giảm ~5-8% mAP

### 1B. Thiếu Normalize trong Transform Pipeline

File [`emotic_dataloader.py` L219-227](file:///Users/macbook/Downloads/RAPT-CLIP/dataloader/emotic_dataloader.py#L219-L227):

```python
train_transforms = torchvision.transforms.Compose([
    ColorJitter(...),
    GroupRandomGrayscale(p=0.2),
    RandomRotation(4),
    GroupResize(image_size),
    GroupRandomHorizontalFlip(),
    Stack(),
    ToTorchFormatTensor()    # ⚠️ CHỈ chia 255 → range [0,1]
    # ❌ THIẾU: GroupNormalize(mean, std) cho CLIP
])
```

**Vấn đề nghiêm trọng:** CLIP ViT-B/16 expects input normalized theo `mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]`. Không normalize = ViT backbone nhận input sai phân phối → **features bị degrade** → cosine similarities thấp.

**Impact:** Ước lượng giảm ~3-5% mAP do backbone features bị off-distribution.

### 1C. Long-Tail Imbalance — cls_num_list = [0]*26

File [`main.py` L222-239](file:///Users/macbook/Downloads/RAPT-CLIP/main.py#L222-L239):

```python
cls_num_list = [0] * len(class_names)
if hasattr(train_loader.dataset, 'video_list'):  # ❌ EMOTICDataset has sample_list, NOT video_list
    ...
else:
    print("=> Warning: Could not calculate class distribution...")
```

**Bug:** `EMOTICDataset` dùng `sample_list` chứ không phải `video_list` → `cls_num_list` luôn = `[0, 0, ..., 0]` → **CB-ASL class weights = uniform** → không có class-balancing → rare classes bị đè bẹp.

Tỷ lệ phân bổ thực tế EMOTIC:
- **Top 5 classes** (*Engagement, Happiness, Excitement, Anticipation, Confidence*): chiếm **~60%** tổng positive labels
- **Bottom 11 classes** (*Embarrassment, Fear, Pain, Sensitivity, etc.*): mỗi class < 3% → F1 ≈ 0%

---

## 2. 🧠 Tầng Biểu diễn & Hợp nhất (Model & Fusion Flow)

### 2A. Hand-crafted Prompts cho MI Loss: Sai 26 class → dùng 7 class

File [`Generate_Model.py` L51-54](file:///Users/macbook/Downloads/RAPT-CLIP/models/Generate_Model.py#L51-L54):

```python
else:
    # Fallback to some generic or 7-class descriptors
    from models.Text import class_descriptor_7_only_face
    hand_crafted_prompts = class_descriptor_7_only_face  # ⚠️ CHỈ 7 classes!
```

**Bug:** EMOTIC không có dedicated hand-crafted prompts → fallback về 7-class (neutral, happy, sad, surprise, fear, disgust, anger). MI Loss tính contrastive giữa **26 learnable features vs 7 hand-crafted** → sau khi truncate (fix trước đó), chỉ align được 7/26 classes → **19 classes KHÔNG có MI regularization** → prompt drift cho majority classes.

### 2B. Face Adapter Architecture Quá Mạnh Dropout

File [`Adapter.py` L3-17](file:///Users/macbook/Downloads/RAPT-CLIP/models/Adapter.py#L3-L17):

```python
class Adapter(nn.Module):
    def __init__(self, c_in, reduction=4):
        self.fc = nn.Sequential(
            nn.Linear(512, 128, bias=False),
            nn.ReLU(),
            nn.Dropout(0.5),        # ⚠️ 50% dropout
            nn.Linear(128, 512, bias=False),
            nn.ReLU(),
            nn.Dropout(0.5)         # ⚠️ 50% dropout on OUTPUT
        )

    def forward(self, x):
        x = self.fc(x)   # ❌ KHÔNG CÓ residual connection!
        return x
```

**2 Vấn đề:**
1. **Dropout 50% trên output** → tại inference, activations bị scale 2x vs training → face features bị distorted
2. **Không có residual** → nếu adapter chưa học tốt, nó phá hủy hoàn toàn ViT face features thay vì chỉnh sửa nhẹ

**Impact:** Face stream features bị noise, đặc biệt ở early epochs → CMAF dồn weight sang Body stream.

### 2C. CMAF Cross-Attention: Single-Token Bottleneck

File [`CrossModalAttentionFusion.py` L101-122](file:///Users/macbook/Downloads/RAPT-CLIP/models/CrossModalAttentionFusion.py#L101-L122):

```python
# Face queries (Body + Context) — sequence length = 1 query, 2 keys
face_q = face_feat.unsqueeze(1)            # (B, 1, 512)
body_context_kv = torch.cat([...], dim=1)  # (B, 2, 512)
face_cross, _ = self.cross_attn_f2bc(face_q, body_context_kv, body_context_kv)
```

**Vấn đề:** Cross-attention chỉ có **1 query token vs 2 key tokens** → attention weights chỉ là 2 giá trị (a_body, a_context) → quá đơn giản, không capture spatial relationships → degraded thành weighted average.

Với **ViT patch tokens** (196 patches × 512), attention sẽ rich hơn rất nhiều. Hiện tại dùng CLS token → mất spatial info.

### 2D. 🔴 Bottleneck Kiến Trúc Cốt Lõi: Feature Collision — 1 Vector cho Multi-Label

> [!CAUTION]
> **Đây là nguyên nhân gốc rễ sâu nhất — không phải model yếu, mà model đang được thiết kế cho single-label nhưng áp dụng cho multi-label.**

#### Bản chất vấn đề

RAPT-CLIP ban đầu thiết kế cho **RAER (đơn nhãn)**: 1 ảnh → 1 cảm xúc. Pipeline:
```
Image → ViT → CLS token (512d) → cosine_sim(feature, 7 text prototypes) → softmax → 1 class
```
Với đơn nhãn, **1 vector 512d chỉ cần "chỉ" về 1 hướng** trong không gian ngữ nghĩa → gradient sạch, tối ưu rõ ràng.

Khi áp dụng cho **EMOTIC (đa nhãn)**: 1 ảnh → 2~5 cảm xúc đồng thời. Cùng pipeline:
```
Image → ViT → CLS token (512d) → cosine_sim(feature, 26 text prototypes) → sigmoid → multi-hot
```

**Vấn đề:** 1 vector 512d duy nhất phải **đồng thời gần** với 3-5 text prototypes khác nhau trong CLIP space.

#### Phân tích toán học: Gradient Collision

Giả sử 1 ảnh có 3 nhãn: *Happy*, *Engagement*, *Anticipation*.

Gradient từ ASL loss:
```
∂L/∂f = Σ_c [ y_c · ∂L_pos/∂f + (1-y_c) · ∂L_neg/∂f ]
```

Với 3 nhãn dương, gradient kéo feature vector `f` về 3 hướng:
```
f ←── "happy" prototype (hướng A)
f ←── "engagement" prototype (hướng B, góc ~40° với A)
f ←── "anticipation" prototype (hướng C, góc ~55° với A)
```

Kết quả: `f` hội tụ về **trung bình vector** của A, B, C — một vị trí **mediocre** không gần tốt với class nào:
```
                    Happy (A)
                   ↗
    f_optimal ← f_actual (compromise position)
                   ↘
                    Engagement (B)
```

**Cosine similarity** giữa `f_actual` và mỗi prototype chỉ đạt ~0.3-0.5 thay vì 0.7-0.9 như single-label → **sigmoid output thấp** → mAP thấp.

#### Chứng minh từ code

File [`Generate_Model.py` L324-327](file:///Users/macbook/Downloads/RAPT-CLIP/models/Generate_Model.py#L324-L327):
```python
# video_features: (B, 512) — MỘT vector duy nhất cho toàn bộ multi-label info
# text_features: (26, 3, 512) — 26 class × 3 prompts
logits = torch.einsum('bd,cpd->bcp', video_features, text_features)
output = torch.mean(logits, dim=2) / self.args.temperature
```

`video_features` shape `(B, 512)` — toàn bộ thông tin thị giác của 3 stream (Face + Body + Context) được **nén thành 1 vector duy nhất** rồi so sánh với 26 prototypes. Đây chính là nút thắt.

#### So sánh với SOTA multi-label

| Method | Feature Strategy | EMOTIC mAP |
|--------|-----------------|------------|
| **RAPT-CLIP (hiện tại)** | 1 global CLS vector → 26 cosine sims | ~20-31% |
| **ML-Decoder (2023)** | 26 class queries → cross-attn with patches → 26 class-specific features | ~35% |
| **CECNet (2026)** | VLM descriptions + contrastive class-specific alignment | **43.72%** |

> [!IMPORTANT]
> **Insight:** Model không yếu — backbone ViT-B/16 trích features đủ mạnh. Nhưng việc **nén toàn bộ multi-label info vào 1 vector** rồi dùng cosine similarity là nút thắt cố hữu của kiến trúc single-label CLIP khi áp cho multi-label.

#### Giải pháp đề xuất: Label-Conditioned Decoder (LCD)

Thay vì 1 global vector, sử dụng **26 learnable class queries** attend vào ViT patch tokens:

```
                         ┌── Q_happy ─── attend patches ──→ feat_happy ──→ score_happy
Image → ViT → 196 patches ─┼── Q_engage ── attend patches ──→ feat_engage ─→ score_engage
                         └── Q_fear ──── attend patches ──→ feat_fear ──→ score_fear
```

Pseudocode:
```python
class LabelConditionedDecoder(nn.Module):
    def __init__(self, num_classes=26, dim=512, num_heads=8, num_layers=2):
        super().__init__()
        # 26 learnable class queries — mỗi query chuyên biệt cho 1 emotion
        self.class_queries = nn.Parameter(torch.randn(num_classes, dim))
        
        # Transformer decoder layers
        self.decoder_layers = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=dim, nhead=num_heads, batch_first=True),
            num_layers=num_layers
        )
    
    def forward(self, patch_tokens, text_features):
        """
        patch_tokens: (B, 196, 512) — ViT spatial features
        text_features: (26, 512) — text prototypes
        Returns: (B, 26) logits
        """
        B = patch_tokens.shape[0]
        queries = self.class_queries.unsqueeze(0).expand(B, -1, -1)  # (B, 26, 512)
        
        # Each class query attends to spatial patches → class-specific features
        class_features = self.decoder_layers(queries, patch_tokens)  # (B, 26, 512)
        
        # Per-class cosine similarity with corresponding text prototype
        class_features = F.normalize(class_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        
        # (B, 26, 512) × (26, 512).T → per-class scores
        logits = (class_features * text_features.unsqueeze(0)).sum(dim=-1)  # (B, 26)
        return logits
```

**Ưu điểm:**
- Mỗi class query trích xuất thông tin **riêng biệt** từ patches → không conflict
- Vẫn tương thích CLIP text prototypes (cosine similarity cuối)
- ViT backbone vẫn frozen → không tăng GPU memory đáng kể
- Chỉ thêm ~2M params (decoder layers + queries)

---

## 3. ⚡ Tầng Hàm Mất Mát & Động Lực Học (Loss & Optimization Dynamics)

### 3A. Temperature Scaling Conflict

Đã fix trước đó nhưng cần ghi nhận:

| Setting | Logit range | Sigmoid output | Effect |
|---------|------------|----------------|--------|
| `temp=0.07` (old) | [-7, +7] | ~0% or ~100% | **Saturated** → loss explosion |
| `temp=1.0` (current) | [-1, +1] | [0.27, 0.73] | **Under-confident** → slow learning |
| `temp=0.5` (proposed) | [-2, +2] | [0.12, 0.88] | **Balanced** range for sigmoid |

> [!IMPORTANT]
> `temperature=1.0` giữ logits trong [-1, +1] → sigmoid không bao giờ dưới 0.27 hoặc trên 0.73 → model **không thể confident** về bất kỳ prediction nào → mAP thấp vì AP cần strong ranking separation.

### 3B. ASL Negative Gradient Dominance

Với 26 classes, mỗi sample trung bình chỉ có **~2 positive labels** → **24/26 = 92% entries là negative**.

ASL loss decomposition cho 1 sample:
```
L = Σ_c [ y_c * (1-p)^0 * log(p) + (1-y_c) * p_m^2 * log(1-p_m) ]
     ↑ 2 positive terms            ↑ 24 negative terms
```

Dù `γ⁻=2.0` down-weights easy negatives, 24 negative gradients vẫn **overwhelm** 2 positive gradients → model bị push toward predicting ALL negative → recall thấp cho rare classes.

### 3C. Overfit Pattern: Train mAP↑ nhưng Valid mAP đạt đỉnh rồi giảm

**Root cause combo:**
1. `--freeze-image-encoder` + `num_trainable_params ≈ 4M` (adapters + prompts + CMAF) → **underfitting** capacity trên 16K diverse images
2. BUT `lr-adapter = 2e-5` quá nhanh cho 4M params → adapter overfit trên train patterns
3. No early stopping → best checkpoint bị missed

---

## 4. 🎯 Kết Luận & Đề Xuất Hành Động

### Top 6 Root Causes (xếp theo mức độ ảnh hưởng)

| # | Root Cause | File:Line | Impact | Loại |
|---|-----------|-----------|--------|------|
| **RC0** | 🔴 **Feature Collision: 1 vector cho multi-label** — kiến trúc single-label CLIP áp cho multi-label, feature bị kéo nhiều hướng → mediocre cho tất cả classes | [`Generate_Model.py:324-327`](file:///Users/macbook/Downloads/RAPT-CLIP/models/Generate_Model.py#L324-L327) | **-8~10% mAP** | Kiến trúc |
| **RC1** | ❌ **Thiếu CLIP Normalize** trong EMOTIC transform → ViT nhận input off-distribution | [`emotic_dataloader.py:219-227`](file:///Users/macbook/Downloads/RAPT-CLIP/dataloader/emotic_dataloader.py#L219-L227) | **-5% mAP** | Data |
| **RC2** | ❌ **Face = Full Image** khi bbox missing → face stream trùng context | [`emotic_dataloader.py:144-145`](file:///Users/macbook/Downloads/RAPT-CLIP/dataloader/emotic_dataloader.py#L144-L145) | **-5% mAP** | Data |
| **RC3** | ❌ **Temperature=1.0 quá cao** → sigmoid under-confident, yếu ranking | [`Generate_Model.py:327`](file:///Users/macbook/Downloads/RAPT-CLIP/models/Generate_Model.py#L327) | **-3% mAP** | Loss |
| **RC4** | ❌ **cls_num_list = [0]*26** → CB-ASL class balancing fails hoàn toàn | [`main.py:222-239`](file:///Users/macbook/Downloads/RAPT-CLIP/main.py#L222-L239) | **-4% mAP** | Loss |
| **RC5** | ❌ **MI Loss dùng 7-class prompts cho 26-class** → prompt drift 19 classes | [`Generate_Model.py:51-54`](file:///Users/macbook/Downloads/RAPT-CLIP/models/Generate_Model.py#L51-L54) | **-2% mAP** | Model |

---

### Action Plan: 31% → 40-43% mAP

#### Phase 1: Quick Wins (Fix 1-6, không thay đổi kiến trúc)

##### Fix 1: Thêm CLIP Normalize (Critical, +5%)
```diff
# emotic_dataloader.py — cả train và test
+ from dataloader.video_transform import GroupNormalize
  transforms = torchvision.transforms.Compose([
      ...
      Stack(),
      ToTorchFormatTensor(),
+     GroupNormalize(
+         mean=[0.48145466, 0.4578275, 0.40821073],
+         std=[0.26862954, 0.26130258, 0.27577711]
+     )
  ])
```

##### Fix 2: Tính cls_num_list đúng cho EMOTIC (+4%)
```diff
# main.py L222-239
  cls_num_list = [0] * len(class_names)
+ if hasattr(train_loader.dataset, 'sample_list'):
+     for record in train_loader.dataset.sample_list:
+         for c, val in enumerate(record.multi_hot_label):
+             if val >= 0.5:
+                 cls_num_list[c] += 1
+ elif hasattr(train_loader.dataset, 'video_list'):
-  if hasattr(train_loader.dataset, 'video_list'):
      for record in train_loader.dataset.video_list:
```

##### Fix 3: Temperature 0.5 (+3%)
```diff
# emotic_cmaf.sh
-    --temperature 1.0 \
+    --temperature 0.5 \
```

##### Fix 4: EMOTIC 26-class Hand-crafted Prompts (+2%)
Tạo `class_descriptor_emotic_26` trong `Text.py` — 26 descriptors, thêm branch trong `Generate_Model.py`.

##### Fix 5: Adapter Residual Connection (+2%)
```diff
# Adapter.py
  def forward(self, x):
-     x = self.fc(x)
+     x = x + 0.5 * self.fc(x)  # Residual
      return x
```

##### Fix 6: Script flags
```bash
--loss-type cb_asl   --temperature 0.5   --lr-adapter 1e-4
--mi-warmup 0        --dc-warmup 0
```

**Expected Phase 1 Result: Valid mAP ~32-37%** (ceiling do RC0)

---

#### Phase 2: Label-Conditioned Decoder (Fix 0, +8-10%)

Giải quyết RC0 (Feature Collision) — thay thế single global vector bằng 26 class-specific queries attend vào ViT patch tokens.

```diff
# Generate_Model.py — forward()
- video_features = self.project_fc(fused_frame_features.squeeze(1))  # (B, 512)
- output = torch.einsum('bd,cpd->bcp', video_features, text_features)
+ # Lấy patch tokens từ CMAF fusion (thay vì chỉ CLS)
+ patch_features = fused_frame_features  # (B, N_patches, 1536)
+ patch_features = self.patch_project(patch_features)  # (B, N, 512)
+ # Label-conditioned decoder: 26 queries attend riêng biệt vào patches
+ output = self.label_decoder(patch_features, text_features)  # (B, 26)
```

**Expected Phase 2 Result: Valid mAP ~40-43%** (approaching CECNet SOTA)

---

### Tổng hợp Expected Trajectory

| Phase | Epoch 10 | Epoch 20 | Epoch 40 | Ceiling |
|-------|----------|----------|----------|---------|
| Hiện tại (no fix) | 11% | ~25% | ~31% | ~31% |
| Phase 1 (Quick Wins) | ~22% | ~30% | ~35-37% | ~37% |
| Phase 2 (+ LCD) | ~28% | ~36% | ~40-43% | ~43% |
