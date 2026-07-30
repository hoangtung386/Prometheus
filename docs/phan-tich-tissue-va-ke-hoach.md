# Phân tích kết quả PUMA & Kế hoạch nâng điểm Tissue vượt trội

> Tài liệu chẩn đoán + đề xuất. Viết sau khi đọc `metrics.json`, toàn bộ `src/prometheus/`,
> `configs/`, `notebooks/train.ipynb`, `result.txt` (log 5-fold), và đối chiếu với
> leaderboard + báo cáo kỹ thuật của các đội top trên https://puma.grand-challenge.org/
>
> **Chưa thực hiện thay đổi code nào.** Toàn bộ nội dung dưới đây chờ anh duyệt.

---

## 0. TL;DR — kết luận trong 8 dòng

1. **Điểm tissue thật của anh là `52.90`, không phải `75.83`.** Con số `average_dice = 0.7583`
   trong `metrics.json` là *macro theo ảnh* và bị thổi phồng vì code chấm điểm trả về Dice = 1.0
   khi cả GT và prediction đều rỗng. Metric chính thức của cuộc thi là **micro Dice**
   (`aggregates.micro_dice_tissue.average_micro_dice = 0.5290`).
2. Với 52.90, anh đang **thấp hơn cả baseline của ban tổ chức (55.48)** và cách đội vô địch
   TIAKong (78.23) đúng 25 điểm.
3. Nhưng **3 trong 5 lớp đã ở mức vô địch rồi**: tumor 89.19, stroma 81.00,
   epidermis 90.52 (đội vô địch: 93.58 / 83.59 / 86.26 — anh còn *thắng* họ ở epidermis).
4. **Toàn bộ 25 điểm bị mất nằm ở đúng 2 lớp**: `necrosis = 0.00` và `blood_vessel = 3.78`.
5. Nếu chỉ chữa 2 lớp đó lên mức của đội vô địch mà **không sửa gì ở 3 lớp còn lại**:
   `77.69` → hạng 1–2 của Track 1.
6. Nguyên nhân số 1 không phải kiến trúc, mà là **model bị huấn luyện thiếu ~200 lần**:
   `result.txt` cho thấy chỉ có **11 optimizer step/epoch × 100 epoch = 1.100 step**, và
   `tissue_dice` trên validation **vẫn đang tăng đơn điệu tới epoch cuối cùng ở cả 5 fold**.
   nnU-Net dùng 250.000 step. Model chưa hề hội tụ.
7. Nguyên nhân số 2: **không có oversampling lớp hiếm**, cộng với **class weight đang phản tác dụng**
   (background — lớp *không được tính điểm* — lại đang có trọng số CE lớn nhất: 3.78, trong khi
   necrosis chỉ 1.05).
8. **Đề nghị: KHÔNG đập đi xây lại toàn bộ project.** Tách nhánh tissue ra khỏi multitask
   (deployment vốn đã chạy 2 checkpoint riêng — multitask hiện chỉ có giá mà không có lợi),
   rồi làm lại *riêng nhánh tissue* theo công thức của đội vô địch. Chi tiết ở §5–§6.

---

## 1. Đọc lại `metrics.json` bằng đúng metric của cuộc thi

Trang [Evaluation & Ranking](https://puma.grand-challenge.org/submission/) định nghĩa:

> *"The **micro Dice Score** is calculated by concatenating all segmentation results along one
> axis and then averaging the Dice score across all tissue classes."*
> *"`tissue_white_background` is not taken along for metric calculation."*

Nghĩa là: gộp cả 10 ảnh thành 1 khối → tính TP/FP/FN từng lớp trên khối đó → Dice từng lớp →
trung bình trên **5 lớp foreground**. Đúng bằng `aggregates.micro_dice_tissue`.

### 1.1 Bảng điểm thật

| Lớp tissue | Micro Dice (%) | Có mặt trong bao nhiêu ROI /10 |
|---|---:|---:|
| tumor | **89.19** | 10 |
| stroma | **81.00** | 10 |
| necrosis | **0.00** | 2 |
| epidermis | **90.52** | 1 |
| blood_vessel | **3.78** | 5 |
| **AVERAGE MICRO DICE** | **52.90** | |

### 1.2 Ba cảnh báo về cách đọc số

**(a) Đừng nhìn `average_dice = 0.7583` nữa.** Nó vô nghĩa. Ví dụ trong
`preliminary_test_set_metastatic_roi_001.tif`: `tissue_necrosis = 1.0` và
`tissue_blood_vessel = 1.0` — không phải vì model đoán hoàn hảo, mà vì cả GT và prediction đều
trống nên Dice được gán 1.0. 8/10 ảnh "được tặng" điểm 1.0 cho necrosis. Metric chính thức
không tặng gì cả.

**(b) Preliminary test chỉ có 10 ROI → là thước đo cực yếu cho lớp hiếm.**
- `epidermis` chỉ xuất hiện trong **1 ROI duy nhất** (`primary_roi_004`). Con số 90.52 của anh
  chính là điểm của **một tấm ảnh**. Trên final test (94 ROI) nó có thể sụt mạnh.
- `necrosis` chỉ xuất hiện trong **2 ROI**.
- Kết luận: **phải xây một harness đánh giá micro-Dice cục bộ trên 5-fold out-of-fold của 205 ảnh
  training** mới có tín hiệu đáng tin. Hiện tại chưa có (§3.12).

**(c) Có 3 ROI mà `tissue_stroma` gần 0 (`5.9e-11`, `5.2e-10`, `2.8e-11`) trong khi micro stroma
lại tới 81.00.** Đây là dấu hiệu **false positive lớp trên ảnh không có lớp đó**: model vẽ stroma
vào ROI vốn không có stroma. Với micro-Dice, những FP này là lỗ ròng. Đây chính là chỗ mà
post-processing "chặn lớp có diện tích quá nhỏ" ăn điểm (§3.11).

---

## 2. Vị trí hiện tại trên bản đồ leaderboard

### 2.1 Track 1 — bảng chính thức

| Hạng | Đội | Tissue micro Dice | Nuclei summed macro F1 | Mean |
|---|---|---:|---:|---:|
| 1 | TIAKong | **78.23** | 74.39 | 76.31 |
| 2 | LSM | 72.37 | 74.43 | 73.40 |
| 3 | rictoo | 63.26 | 75.78 | 69.52 |
| — | Baseline BTC | 55.48 | 69.40 | 62.44 |
| — | **Prometheus (hiện tại)** | **52.90** | — | — |

### 2.2 So per-class với 2 đội đầu — chỗ này là chìa khoá

| Lớp | Prometheus | TIAKong (#1) | LSM (#2) | Chênh so với #1 |
|---|---:|---:|---:|---:|
| tumor | 89.19 | 93.58 | 92.07 | −4.4 |
| stroma | 81.00 | 83.59 | 81.28 | −2.6 |
| epidermis | 90.52 | 86.26 | 87.32 | **+4.3** ✅ |
| **necrosis** | **0.00** | **82.04** | 46.79 | **−82.0** ❌ |
| **blood_vessel** | **3.78** | **45.70** | 54.37 | **−41.9** ❌ |
| **Mean** | **52.90** | **78.23** | 72.37 | −25.3 |

*(TIAKong per-class: [arXiv 2507.13974](https://arxiv.org/abs/2507.13974) —
LSM per-class: [arXiv 2503.23958](https://arxiv.org/abs/2503.23958))*

### 2.3 Mô phỏng điểm — tại sao tôi nói "chỉ cần chữa 2 lớp"

Giữ nguyên tumor/stroma/epidermis đúng như hiện tại, chỉ thay đổi necrosis + blood_vessel:

| Kịch bản | necrosis | blood_vessel | → Micro Dice | Vị trí |
|---|---:|---:|---:|---|
| Hiện tại | 0.00 | 3.78 | **52.90** | dưới baseline |
| Chỉ chữa blood_vessel | 0.00 | 45.70 | **61.28** | ~hạng 3 |
| Chỉ chữa necrosis | 82.04 | 3.78 | **69.31** | trên hạng 3 |
| Bảo thủ (mục tiêu tối thiểu) | 60.00 | 35.00 | **71.14** | ~hạng 2 |
| Bằng đội vô địch ở 2 lớp này | 82.04 | 45.70 | **77.69** | **hạng 1–2** |

Và đây là con số quan trọng nhất trong cả tài liệu này: **77.69 đạt được mà không cần cải thiện
tumor, stroma hay epidermis một chút nào.** Kiến trúc hiện tại của anh không hề "kém" — nó chỉ
đơn giản là **chưa bao giờ học được 2 lớp hiếm**.

Nếu chữa xong 2 lớp hiếm rồi mới đi kéo tumor 89.19 → 93 và stroma 81 → 84 (bằng
foundation model + ensemble, §5), thì sàn là ~80 và vượt hạng 1 hiện tại.

### 2.4 Ghi chú về nhánh nuclei (không phải câu hỏi của anh, nhưng để anh định lượng)

`aggregates.macro_f1_nuclei = 0.2172` trên 10 lớp. Đối chiếu: **KongNet của TIAKong đạt macro F1
0.2656 và xếp hạng 2 Track 2** ([arXiv 2510.23559](https://arxiv.org/html/2510.23559v2)).
Nghĩa là nhánh nuclei của anh đã **gần mức hạng 2 rồi** — nó không phải chỗ để đầu tư.
Toàn bộ đòn bẩy nằm ở tissue. (Lưu ý: cách BTC tính "summed macro F1" khác cách
`nuclei_detection_metrics` trong repo tính, nên hai con số này không so trực tiếp 1:1 được —
xem §3.4.)

---

## 3. Chẩn đoán: 13 nguyên nhân, xếp theo mức tác động

### 3.1 ⛔ NGUYÊN NHÂN LỚN NHẤT — model bị huấn luyện thiếu khoảng 200 lần

Bằng chứng trực tiếp từ `result.txt`:

```
Epoch 000 batch 0000/0011 ...      ← chỉ có 11 batch/epoch
Epoch 000 loss=15.79 tissue_dice=0.0326
Epoch 025 loss=7.75  tissue_dice=0.2330
Epoch 050 loss=6.51  tissue_dice=0.4882
Epoch 075 loss=6.49  tissue_dice=0.5007
Epoch 099 loss=6.84  tissue_dice=0.5063   ← VẪN ĐANG TĂNG ở epoch cuối
```

Cả 5 fold đều thế:

| Fold | tissue_dice @ep25 | @ep50 | @ep75 | @ep99 | Đã plateau? |
|---|---:|---:|---:|---:|---|
| 0 | 0.2330 | 0.4882 | 0.5007 | **0.5063** | không |
| 1 | 0.2412 | 0.4615 | 0.6353 | **0.6558** | không |
| 2 | 0.1960 | 0.3872 | 0.4852 | **0.5063** | không |
| 3 | 0.2267 | 0.5219 | 0.6138 | **0.6511** | không |

Số liệu:
- 205 ảnh, 5-fold → 164 ảnh train, `batch_size = 16` → `ceil(164/16) = 11` step/epoch.
- 100 epoch × 11 = **1.100 optimizer step** cho toàn bộ quá trình huấn luyện.
- nnU-Net 2D mặc định: 1000 epoch × 250 iteration = **250.000 step**. Tỉ lệ: **227×**.
- Chính notebook đã tự nhận diện vấn đề này (cell 9): *"the binding constraint is the NUMBER OF
  OPTIMIZER STEPS PER EPOCH, not VRAM"* — nhưng cách xử lý lại là **tăng** batch size lên 16, tức
  **giảm** số step/epoch từ 41 xuống 11. Đi sai hướng.

**Hệ quả cho lớp hiếm:** giả sử necrosis xuất hiện trong ~15/205 ảnh. Trong 1.100 step, model
nhìn thấy pixel necrosis khoảng `15 × 100 = 1.500` lần *ảnh*, tức xuất hiện trong ~90 step
(nếu phân bố đều trong batch 16). **90 gradient step là không thể học được một lớp mô.**
Đây là lý do vật lý khiến `necrosis = 0.00`, không phải lý do kiến trúc.

**Cách chữa (không cần đổi kiến trúc):**
1. **Chuyển sang huấn luyện theo patch**, tách rời `steps/epoch` khỏi `số ảnh`:
   crop random 512×512 từ ảnh 1024×1024, mỗi ảnh lấy 8–16 patch/epoch.
   → `205 × 12 / 8 ≈ 300 step/epoch`. 300 epoch = **90.000 step** (81× hiện tại).
2. Đồng thời **giảm batch xuống 8** (patch 512 nhẹ hơn ảnh 1024 4 lần → vẫn vừa VRAM,
   và chạy được cả trên RTX 3080 10GB ở máy local, không cần Colab A100).
3. Cosine schedule đang `scheduler.step()` **theo epoch** (`trainer.py:194`). Với 300 step/epoch
   thì nên chuyển sang step theo iteration để LR mượt.

*Kỳ vọng chỉ riêng mục này: micro Dice `52.90 → 62–66`.* Đây là thay đổi rẻ nhất, ít rủi ro nhất
trong toàn bộ tài liệu.

---

### 3.2 ⛔ Không có oversampling lớp hiếm — đây là "bí kíp" của nnU-Net

nnU-Net thắng ở các bài toán lớp hiếm nhờ `oversample_foreground_percent`: nó **bắt buộc** một
tỉ lệ patch trong mỗi batch phải có tâm nằm trên một lớp foreground được chọn ngẫu nhiên.
Repo hiện tại lấy **cả ảnh, sampling đều** — không có bất kỳ cơ chế nào ưu tiên necrosis/blood_vessel.

**Cách chữa:** dựng một index `{class → [danh sách (ảnh, toạ độ pixel)]}` một lần rồi cache
(giống cách `_resolve_class_weights` đã cache `class_weights.json`), sau đó sampler:

```python
# Với mỗi patch trong batch:
#   33% xác suất: random uniform trên ảnh
#   67% xác suất: chọn ngẫu nhiên 1 trong 5 lớp fg (đều nhau!), rồi crop patch có tâm
#                 tại một pixel thuộc lớp đó
```

Với sampler này, necrosis được nhìn thấy trong **~13% mọi patch** thay vì ~2%. Kết hợp với §3.1
(90.000 step), necrosis nhận được ~12.000 gradient exposure thay vì 90. **Tăng ~130 lần.**

*Kỳ vọng: necrosis `0 → 45–70`, blood_vessel `3.78 → 25–40`.* Đây là đòn bẩy lớn thứ hai.

---

### 3.3 ⛔ Class weight đang phản tác dụng — background được ưu tiên hơn necrosis

`result.txt` in ra trọng số thật, fold 0:

```
tissue weights: [3.782, 0.009, 0.024, 0.311, 1.045, 0.828]
                  ↑bg    ↑tumor ↑stroma ↑epi  ↑necro ↑bvessel
```

Ba vấn đề nghiêm trọng:

**(a) `background` — lớp mà cuộc thi KHÔNG tính điểm — lại đang có trọng số CE lớn nhất (3.782).**
`inverse_frequency_weights` (`losses/class_weights.py:13-26`) chuẩn hoá trên **cả 6 lớp**, và vì
`white_background` là lớp ít pixel nhất (~0.5%) nên nó ăn trọng số cao nhất. Gradient của
CrossEntropy đang chủ yếu đẩy model đi dự đoán background. Mỗi pixel background đoán đúng
không được điểm gì, nhưng mỗi pixel necrosis bị đoán thành background là một FN mất điểm.

**(b) Necrosis (1.045) và blood_vessel (0.828) — hai lớp cần cứu nhất — lại có trọng số THẤP HƠN
background.** Thứ tự ưu tiên đang hoàn toàn ngược với thứ tự cần thiết cho metric.

**(c) Tỉ lệ 3.782 / 0.009 = 420:1 là quá cực đoan**, làm CE mất ổn định (thấy rõ ở loss epoch 0
của fold 4: `42.79`). Full inverse-frequency là lựa chọn tệ; nên dùng `1/sqrt(freq)` hoặc
*effective number of samples* (Cui et al., β=0.999).

**Cách chữa:**
```python
# Chuẩn hoá CHỈ trên 5 lớp được tính điểm; background nhận trọng số nhỏ cố định.
w = 1.0 / counts[1:].float().sqrt()      # sqrt-inverse, không dùng full inverse
w = w / w.mean()
weights = torch.cat([torch.tensor([0.1]), w])   # background = 0.1, không phải 3.782
```

Bổ sung: `MultiClassDiceLoss` (`losses/segmentation.py:82-106`) đang có `ignore_absent=True` và
reduce trên `dims=(0,2,3)` — tức Dice tính trên **cả batch**, và một lớp bị loại khỏi loss nếu
*không ảnh nào trong batch* có lớp đó. Với batch nhỏ, false positive necrosis trên batch không
có necrosis **không bị Dice phạt**. Nên đổi `ignore_absent=False` cho 5 lớp fg khi đã có
oversampling (lúc đó lớp hiếm gần như luôn có mặt trong batch).

*Kỳ vọng: +3–6 điểm micro Dice, chi phí ~20 dòng code.*

---

### 3.4 ⛔ Metric dùng để chọn checkpoint không phải metric của cuộc thi

Ba vấn đề chồng nhau:

**(a) `checkpoint_metric = "nuclei/macro_f1_summed"`** (`baseline_multitask.toml:56`).
Checkpoint `best_primary.ckpt` được chọn **hoàn toàn theo nuclei**. Có `best_tissue.ckpt` riêng
(`trainer.py:203-212`) nên không chết hẳn, nhưng nghĩa là toàn bộ lịch trình LR, EMA, và
early-stopping-ngầm đều đang tối ưu cho task khác.

**(b) `tissue/Dice/mean_present_fg` âm thầm bỏ qua lớp vắng mặt.**
`metrics/evaluator.py:91-95`:
```python
@staticmethod
def _nanmean(x):
    valid = ~torch.isnan(x)      # lớp có dice_den == 0 bị bỏ khỏi trung bình
    ...
```
Nếu trong fold validation không có ảnh nào chứa necrosis **và** model không đoán necrosis →
`dice_den = 0` → NaN → **lớp đó bị loại khỏi mean**. Metric validation vì thế **liên tục
báo cao hơn thực tế** và **không cung cấp tín hiệu nào về necrosis**. Đó chính là lý do
`tissue_dice = 0.6558` trên validation nhưng chỉ `0.5290` trên preliminary test —
khoảng cách 13 điểm phần lớn là ảo giác đo lường, không phải overfitting.

**(c) Không có per-class visibility trong log.** `engine/evaluator.py:79`:
```python
tissue_evaluator = SegmentationEvaluator(6)   # không truyền class_names!
```
→ log key thành `tissue/Dice/c0` … `tissue/Dice/c5`. Và dòng print mỗi epoch
(`trainer.py:228-232`) chỉ in *mean*. **Necrosis đã bằng 0 suốt 100 epoch × 5 fold mà không có
gì trong log báo động.** Đây là lỗi observability đã trực tiếp gây ra thất bại này.

**Cách chữa (ưu tiên cao, rẻ):**
```python
tissue_evaluator = SegmentationEvaluator(6, class_names=TISSUE_CLASSES)
# và thêm vào metric hiển thị mỗi epoch:
#   tissue/micro_dice_official  = mean của 5 lớp fg, lớp vắng-GT-nhưng-có-pred tính 0,
#                                 KHÔNG nan-skip lớp có mặt trong GT của fold
# in ra: necrosis=..  bvessel=..  mỗi epoch
```
Đồng thời thêm `checkpoint_metric = "tissue/micro_dice_official"` vào `supported_metrics`
(`config/project.py:140`).

---

### 3.5 🔴 PHẢI KIỂM CHỨNG — Rasterize ground truth: mất lỗ (holes) + không có thứ tự ưu tiên lớp

Đây là **nghi vấn nghiêm trọng nhất** vì nếu đúng thì mask huấn luyện của anh vốn đã sai, và
mọi cải tiến khác đều vô nghĩa. Có 2 lỗi tiềm tàng:

**(a) Interior rings (holes) bị loại bỏ.** `data/puma/geojson.py:56-65`:
```python
for candidate in candidates:
    exterior = np.asarray(candidate[0], dtype=np.float32).reshape(-1, 2)
    #                               ^^^ chỉ lấy ring 0 — mọi hole ở candidate[1:] BỊ BỎ
```
Annotation PUMA được tạo bằng QuPath, và trong QuPath một vùng tumor có ổ hoại tử ở giữa
thường được biểu diễn là **Polygon có hole**. Bỏ hole = vùng tumor được tô kín lên chỗ necrosis.

**(b) Không có thứ tự ưu tiên lớp.** `data/puma/rasterize.py:22-27`:
```python
for label, polygon in regions:        # thứ tự = thứ tự feature trong file GeoJSON
    ...
    cv2.fillPoly(label_mask, [points], class_index)   # vẽ sau ghi đè vẽ trước
```
Không có priority nào cả. Nếu trong file GeoJSON các feature `blood_vessel`/`necrosis` (nhỏ,
lồng trong vùng lớn) nằm **trước** `tumor`/`stroma`, thì exterior không-hole của tumor/stroma
**xoá sạch** chúng khỏi mask huấn luyện.

Kết hợp (a) + (b) là một cơ chế hoàn chỉnh giải thích `necrosis = 0.00`: **model không học được
necrosis vì mask huấn luyện gần như không chứa necrosis.**

**Cách rasterize đúng** (tô từng lớp lên layer riêng có khoét lỗ, rồi composite theo priority
"nhỏ/lồng ghép vẽ sau"): xem script kiểm chứng ở §4.1 — nó vừa chẩn đoán vừa là bản
implementation tham chiếu.

**Ghi chú phụ:** `rasterize.py:24` bỏ qua `class_index == 0`, nên polygon `white_background`
tường minh không bao giờ được tô. Hiện tại vô hại (default đã là 0) nhưng nếu một vùng
`white_background` chồng lên vùng mô nằm trước nó trong file, background phải thắng mà lại không.
Nên xử lý background như một lớp có priority thấp nhất thay vì skip.

---

### 3.6 🔴 PHẢI KIỂM CHỨNG — Độ phân giải 512 vs 1024: nguy cơ lệch train/test nghiêm trọng

Anh nói **"dữ liệu huấn luyện segment chỉ có 205 ảnh 512×512"**. Nhưng dataset PUMA gốc là
**1024×1024 tại 40× (0.22 µm/pixel)** ([Dataset page](https://puma.grand-challenge.org/dataset/)),
và config đang đặt `image_size = 1024`.

Nếu ảnh trên đĩa thật là 512×512 thì `letterbox_image` (`data/spatial.py:20-23`) sẽ
**upsample 2× bằng `cv2.INTER_LINEAR`**:
```python
scale = min(target_width / source_width, target_height / source_height)   # = 2.0
resized = cv2.resize(image, (1024, 1024), interpolation=cv2.INTER_LINEAR)
```
Trong khi ở container submission, ảnh test là 1024 native → `scale = 1.0`, **không** upsample.

Hệ quả: **model được train trên ảnh 20× bị nội suy mờ, nhưng được test trên ảnh 40× nét.**
Đây là domain shift lớn, và nó phá **đúng những cấu trúc mảnh**: thành mạch máu, lớp endothelium
một tế bào dày. Điều này khớp một cách đáng ngờ với `blood_vessel = 3.78`.

Có 3 khả năng, cần xác định ngay bằng script §4.2:
- **(i) Ảnh trên đĩa là 1024 native** → không có vấn đề này; §3.6 bỏ qua.
- **(ii) Ảnh là 512 do anh chủ động downsample** → phải quay lại dùng 1024 gốc. Đây là lỗi
  nghiêm trọng nhất có thể chữa được trong 1 giờ.
- **(iii) Ảnh là 512 nhưng annotation GeoJSON vẫn ở toạ độ 1024** → mask hoàn toàn lệch,
  mọi thứ sai. Script §4.2 phát hiện được trường hợp này.

---

### 3.7 🟠 Chưa dùng context image 5120×5120 mà BTC cấp miễn phí

PUMA cung cấp kèm mỗi ROI một **context image 5120×5120** (bao quanh ROI). Container submission
nhận cả file `*_context.tif` — nhưng `PUMA-track2-submit` **cố tình bỏ qua** nó
(*"Files ending in `_context.tif` or `_context.tiff` are ignored"*).

`necrosis` và `epidermis` là những cấu trúc **quy mô lớn, phụ thuộc bối cảnh**: một ổ hoại tử
có thể tràn ra ngoài khung 1024, và trong khung 1024 nó trông giống stroma nhợt. Nhìn được
5120 xung quanh là tín hiệu quyết định.

**Đề xuất:** thêm một nhánh input thứ hai: context 5120 → downsample xuống 512 (≈ magnification 4×)
→ encoder nông riêng → concat vào bottleneck stride-32 của decoder tissue. Rẻ (ảnh nhỏ), và
đúng chỗ (bottleneck ngữ nghĩa).

*Kỳ vọng: necrosis +8–15, epidermis +2–4.*

---

### 3.8 🟠 Augmentation còn thiếu 2 nhóm quan trọng

`data/transforms/multitask.py:220-235` hiện có: hflip, vflip, rot90, stain-jitter (gain/bias
per-channel), brightness/contrast, gamma, gaussian noise.

**Thiếu 1: không có random scale.** Toàn bộ augmentation hình học chỉ là nhóm dihedral 8 phần tử.
Với 205 ảnh, thiếu scale augmentation là mất mát lớn — và scale chính là thứ dạy model bất biến
với kích thước ổ necrosis và đường kính mạch máu.

**Thiếu 2: stain augmentation thật.** `RandomStainJitterMultitask` chỉ là gain/bias per-channel
trong không gian RGB — nó **không** mô phỏng được biến thiên nhuộm H&E thật (vốn là biến đổi
trong không gian optical-density theo ma trận stain).

Đội vô địch dùng: *"Random RGB shift, random HSV shift, Gaussian blur and sharpening, image
compression, random brightness and contrast, random shifts and scaling, random 90-degree rotations
and horizontal/vertical flips."*

**Đề xuất:** thay hẳn transform pipeline tự viết bằng `albumentations` (đã là chuẩn de-facto,
xử lý đồng bộ mask sẵn):
```python
A.Compose([
    A.RandomScale(scale_limit=(-0.25, 0.25), p=0.5),
    A.Rotate(limit=180, border_mode=cv2.BORDER_REFLECT_101, p=0.5),   # xoay tự do, không chỉ 90°
    A.RandomCrop(512, 512),
    A.HorizontalFlip(), A.VerticalFlip(), A.RandomRotate90(),
    A.HueSaturationValue(10, 20, 10, p=0.7),   # ~ stain variation
    A.RGBShift(15, 15, 15, p=0.5),
    A.RandomBrightnessContrast(0.2, 0.2, p=0.7),
    A.OneOf([A.GaussianBlur(3), A.Sharpen()], p=0.3),
    A.ImageCompression(quality_lower=60, p=0.2),
    A.ElasticTransform(alpha=50, sigma=8, p=0.2),
])
```
Cộng thêm **RandStainNA** hoặc **Macenko stain augmentation** nếu muốn đi xa hơn.

*Kỳ vọng: +2–5 điểm, và giảm khoảng cách val↔test.*

---

### 3.9 🟠 Backbone: ImageNet ConvNeXt-V2-Tiny vs pathology foundation model

Đây là điểm khác biệt kiến trúc thực sự giữa anh và đội vô địch.

Công thức của TIAKong ([arXiv 2507.13974](https://arxiv.org/abs/2507.13974)):
- **Virchow2** (632M params, pretrain trên 3.1 triệu ảnh histopathology) — **đóng băng**, dùng làm
  feature extractor.
- Ảnh 1024 → downsample về **224×224** → Virchow2 cho **256 patch token × 1280 dim** (grid 16×16).
- Module *Progressive Transposed Convolution* upsample token map 16×16 → 224×224.
- **Concat với ảnh RGB gốc** → tensor 8 kênh (5 feature + 3 RGB) → nhập
  **Efficient-UNet** (backbone EfficientNetV2-M + skip connection + SCSE block).
- AdamW lr 1e-3, wd 5e-3, batch 24, early stopping. **Thời gian train trung bình: 30 phút.**
- Kết quả: **78.23** micro Dice.
- Ghi chú của chính họ: baseline `MaskFormer-UNI` của BTC chỉ đạt Dice 44.00 → **chọn foundation
  model nào và ghép thế nào quan trọng hơn việc "có dùng FM hay không".**

Hai quan sát rất đáng chú ý:
1. **Họ làm việc ở 224×224 và vẫn đạt necrosis 82.04.** → necrosis là bài toán **ngữ nghĩa,
   quy mô lớn, thấp phân giải**. Việc anh được 0.00 hoàn toàn không phải vì thiếu độ phân giải,
   mà vì lớp đó **chưa bao giờ được học** (§3.1–3.3, 3.5).
2. **Chính họ chỉ đạt blood_vessel 45.70 — thấp nhất trong 5 lớp — vì 224 là quá thô cho mạch máu.**
   Đội LSM dùng một **U-Net (ResNet34) riêng chỉ để phân đoạn mạch máu ở full resolution** và đạt
   **54.37 — cao nhất bảng.** → *Blood vessel phải là một model nhị phân riêng ở độ phân giải cao.*

Kết luận kiến trúc: **PUMA tissue không phải một bài toán, mà là hai bài toán ở hai thang đo.**
Đây chính là chỗ mà một kiến trúc "một head 6 lớp duy nhất" như hiện tại bị chặn trần.

---

### 3.10 🟠 Multitask đang trả giá mà không thu được lợi

Kiến trúc hiện tại chia sẻ encoder, và decoder tissue còn feed ngược vào nhánh nuclei qua
`GatedContextFusion`. Nhưng:

- **Deployment vốn đã chạy 2 model riêng.** `PUMA-track2-submit/README.md`: *"the ten-class nuclei
  detections come from the nuclei-selected fold (`best_primary.ckpt`) and the six-class tissue mask
  from the tissue-selected fold (`best_tissue.ckpt`)"* — hai checkpoint từ hai fold khác nhau,
  chạy tuần tự. **Việc chia sẻ trọng số lúc train không mang lại bất kỳ lợi ích nào lúc infer.**
- **Xếp hạng là mean-of-ranks trên 2 task độc lập** → không có phần thưởng nào cho một model duy nhất.
- **Loss đang nghiêng về nuclei**: `center_focal 1.0 + nuclei_class 1.0 + offset 1.0 + size 0.1 = 3.1`
  so với `tissue_ce 1.0 + tissue_dice 1.0 = 2.0`. Và `center_focal` có magnitude rất lớn ở giai đoạn
  đầu (loss epoch 0 = 15.8–42.8). Encoder trong ~30 epoch đầu chủ yếu bị nuclei kéo.
- Nhánh nuclei của anh đã gần mức hạng 2 (§2.4) → nó **không cần** encoder tốt hơn nữa.

**Đề xuất: tách hẳn.** Giữ `PrometheusNet` cho nuclei (đang tốt, đừng động vào), và xây một
model tissue độc lập. Đây là thay đổi kiến trúc *duy nhất* tôi đề nghị làm lớn.

---

### 3.11 🟡 Post-processing: argmax thuần, không TTA khi submit, không ensemble

Ba điểm mất không đáng có:

**(a) Submission không có TTA.** `inference/predictor.py:44-56` gọi `self.model(images)` một lần
rồi `argmax`. Có `_tta_forward` trong `engine/evaluator.py:25-62` nhưng: chỉ 3 view
(identity + hflip + vflip), `tta=False` mặc định, và **chỉ dùng trong evaluate, không dùng trong
predict**. → Đường submission không hưởng TTA. Nên dùng **8 view dihedral đầy đủ**.
*Kỳ vọng: +1–2 điểm, ~0 rủi ro.*

**(b) Không ensemble.** Anh đã train 5 fold rồi, nhưng chỉ dùng **1 fold** cho submission.
Average softmax của cả 5 fold là chuyện miễn phí. *Kỳ vọng: +2–4 điểm.*

**(c) Không có post-processing per-class.** Với micro-Dice, false positive của một lớp trên
ảnh không chứa lớp đó là mất mát ròng (đã thấy: 3 ROI có `stroma ≈ 5e-11`, và
`blood_vessel ≈ 1e-9` ở 3 ROI). Hai thao tác rẻ:
- **Cổng diện tích:** chỉ phát ra lớp `necrosis` / `blood_vessel` nếu tổng diện tích dự đoán
  của nó vượt ngưỡng `τ_c` % diện tích ROI; dưới ngưỡng thì gán về lớp lân cận có xác suất cao nhất.
  (Báo cáo baseline của BTC cũng nhắc rằng các đội cải thiện lớp hiếm *"through post-processing
  techniques"*.)
- **Bias logit per-class:** thay `argmax(logits)` bằng `argmax(logits + b)` với `b ∈ R^6` được
  tune trực tiếp để tối đa micro-Dice trên out-of-fold predictions. Đây là tối ưu 6 tham số trên
  metric thật — **cực rẻ và thường ăn 2–4 điểm** cho các bài toán micro-Dice mất cân bằng.
- Loại connected component nhỏ + fill hole (`cv2.connectedComponentsWithStats`).

**(d) Head tissue xuất ở stride 4 rồi bilinear ×4.** `heads/tissue_segmentation.py:37-38`:
```python
logits = self.classifier(decoded_s4)                       # stride 4
logits = F.interpolate(logits, size=output_size, ...)      # bilinear x4 lên 1024
```
Cho tumor/stroma/necrosis thì vô hại. Cho blood_vessel (thành mạch mảnh 2–5 px) thì đây là
giới hạn cứng. Model chuyên blood_vessel (§3.9) phải xuất ở stride 1 hoặc 2.

---

### 3.12 🟡 Validation 20 ảnh, và split không stratify theo lớp hiếm

- `validation_fraction = 0.1` → ~20 ảnh validation. Với necrosis xuất hiện ở ~7% ảnh, một fold
  validation 20 ảnh **kỳ vọng chỉ có 1.4 ảnh necrosis** — thường là 0 hoặc 1. Metric hoàn toàn nhiễu.
- Phương sai giữa fold xác nhận: `tissue_dice` best per-fold = **0.5067 / 0.6558 / 0.5063 / 0.6511**
  → spread **0.15**. Không thể phân biệt "cải tiến thật" với "nhiễu fold" ở mức nhiễu này.
- `splits.py:15-21` chỉ stratify theo `primary` / `metastatic`:
  ```python
  def _sample_group(sample_id):
      if "primary" in lowered: return "primary"
      if "metastatic" in lowered: return "metastatic"
  ```
  → **không** đảm bảo necrosis/blood_vessel/epidermis phân bố đều giữa các fold. Rất có thể
  toàn bộ ảnh có necrosis rơi vào 1–2 fold.

**Cách chữa:** stratify multi-label theo vector "lớp nào có mặt" (dùng
`sklearn.model_selection.MultilabelStratifiedKFold` hoặc `iterstrat`), và **báo cáo micro-Dice
trên toàn bộ out-of-fold prediction của 205 ảnh** — đó là estimator không lệch cho metric của
cuộc thi, và có 205 ảnh nên nhiễu nhỏ hơn nhiều so với 10 ảnh preliminary.

---

### 3.13 🟢 Ghi chú nhỏ

- `evaluate_multitask` gọi `model(batch.images)` **rồi lại** gọi `_tta_forward` (thêm 2 forward)
  khi bật TTA → 3 forward pass thay vì cần 2. Chi phí, không phải lỗi. Giữ nguyên vì loss phải
  đo trên forward pass thường.
- `read_native_image` trả về `float32` full 1024×1024×3 rồi mới augment → mỗi worker giữ 12 MB/ảnh.
  Với patch training nên crop *trước khi* convert sang float.
- `write_tissue_tiff` remap sang `TISSUE_SUBMISSION_VALUE` đúng chuẩn (stroma=1, blood_vessel=2,
  tumor=3, epidermis=4, necrosis=5) — **phần này đã đúng**, tôi đã kiểm tra kỹ vì mapping sai ở đây
  cũng sẽ gây ra đúng hiện tượng "một số lớp bằng 0". Không phải nguyên nhân.
- `RandomRotate90Multitask.rotate_points`: tôi đã nghi ngờ nó trộn `width`/`height` cho
  rotation 1 và 3, nhưng kiểm tra lại thì **nó đúng**, kể cả với ảnh không vuông — sau khi
  `np.rot90` với k lẻ thì hai chiều đổi vai trò, và công thức hiện tại đã tính đúng điều đó.
  Không cần sửa.

---

## 3b. Trạng thái triển khai

Sau khi anh duyệt tài liệu này, các mục sau **đã được sửa trong code** (chi tiết ở
[`handover.md`](handover.md)):

| Mục | Trạng thái |
|---|---|
| §3.3 Class weight (sqrt-inverse, bỏ background) | ✅ đã sửa |
| §3.4 Metric micro-Dice chính thức + log per-class mỗi epoch | ✅ đã sửa |
| §3.5 Rasterize giữ hole + priority lớp | ✅ đã sửa |
| §3.11a TTA 8 view trên cả đường submission | ✅ đã sửa |
| §4.1 Audit rasterization | ✅ thành `prometheus audit` (có test) |
| §4.2 Audit độ phân giải | ✅ thành `prometheus audit` (có test) |
| §4.3 Harness micro-Dice | ✅ thành `prometheus.metrics.official_micro_dice` |
| §3.1 Patch training | ⏳ chưa — bước tiếp theo |
| §3.2 Oversampling lớp hiếm | ⏳ chưa — bước tiếp theo |
| §3.8 Albumentations | ⏳ chưa |
| §3.7 Nhánh context 5120 | ⏳ chưa |
| §3.9 Foundation model | ⏳ chưa — chờ HF gated access |
| §3.10 Tách tissue khỏi multitask | ⏳ chưa |
| §3.11b,c Ensemble + tune bias logit | ⏳ chưa |
| §3.12 Stratify theo lớp hiếm | ⏳ chưa |

Thay vì để 3 script rời ở §4, chúng đã thành thư viện có test và một lệnh CLI duy nhất:

```bash
uv run prometheus audit --data-root /path/to/puma
```

Lệnh này in cả ba báo cáo (`integrity`, `rasterization`, `resolution`) dưới dạng JSON.
Các script trong §4.1–4.3 bên dưới giữ lại làm tài liệu giải thích thuật toán.

---

## 4. Việc phải làm ngay trong 1 giờ — 3 script kiểm chứng

> ⚠️ **Đã triển khai — dùng CLI, không copy script bên dưới.** Ba kiểm tra này giờ là thư
> viện có unit test và một lệnh duy nhất:
>
> ```bash
> uv run prometheus audit --data-root /path/to/puma
> ```
>
> Các script dưới đây là **bản dẫn giải gốc**, viết dựa trên code *trước* refactor (chúng
> tham chiếu `rasterize_regions`, `data/puma/classes.py` — hai thứ đã bị xoá). Giữ lại để
> giải thích thuật toán và lý do, không phải để chạy.

Trước khi viết một dòng code training mới, chạy audit. Nó quyết định roadmap.

### 4.1 Kiểm chứng mask ground truth (nghi vấn §3.5) — QUAN TRỌNG NHẤT

```python
# scripts/audit_tissue_raster.py
"""So sánh mask hiện tại (bỏ hole, không priority) với mask đúng (có hole + priority)."""
import sys, numpy as np, cv2
from collections import Counter
sys.path.insert(0, "src")
from prometheus.data.puma.discovery import discover_puma_samples
from prometheus.data.puma.geojson import read_geojson, feature_label, parse_tissue_geojson
from prometheus.data.puma.rasterize import rasterize_regions
from prometheus.data.puma.classes import TISSUE_CLASS_TO_IDX, TISSUE_CLASSES

DATA_ROOT = sys.argv[1]
SIZE = (1024, 1024)   # đổi thành (512,512) nếu §4.2 cho biết ảnh là 512

# Vẽ SAU = thắng. Cấu trúc nhỏ / lồng bên trong phải vẽ sau cùng.
PRIORITY = ["background", "stroma", "tumor", "epidermis", "necrosis", "blood_vessel"]


def rings(geom):
    """Trả về [(exterior, [holes...]), ...] — GIỮ LẠI interior rings."""
    t, c = geom.get("type"), geom.get("coordinates") or []
    polys = [c] if t == "Polygon" else (c if t == "MultiPolygon" else [])
    out = []
    for p in polys:
        if not p:
            continue
        ext = np.asarray(p[0], np.float64).reshape(-1, 2)
        if len(ext) < 3:
            continue
        out.append((ext, [np.asarray(h, np.float64).reshape(-1, 2) for h in p[1:] if len(h) >= 3]))
    return out


def rasterize_correct(path, size):
    """Mỗi lớp -> layer nhị phân riêng (có khoét lỗ), rồi composite theo PRIORITY."""
    layers = {}
    for f in read_geojson(path).get("features", []):
        lbl = feature_label(f)
        layer = layers.setdefault(lbl, np.zeros(size, np.uint8))
        for ext, holes in rings(f.get("geometry") or {}):
            cv2.fillPoly(layer, [np.rint(ext).astype(np.int32)], 1)
            for h in holes:
                cv2.fillPoly(layer, [np.rint(h).astype(np.int32)], 0)
    mask = np.zeros(size, np.uint8)
    for lbl in PRIORITY:
        if lbl in layers and TISSUE_CLASS_TO_IDX.get(lbl):
            mask[layers[lbl] == 1] = TISSUE_CLASS_TO_IDX[lbl]
    return mask


cur, fix = Counter(), Counter()
n_img_with, n_holes = Counter(), 0
for s in discover_puma_samples(DATA_ROOT):
    old = rasterize_regions(parse_tissue_geojson(s.tissue_annotation_path), SIZE, TISSUE_CLASS_TO_IDX)
    new = rasterize_correct(s.tissue_annotation_path, SIZE)
    for i, name in enumerate(TISSUE_CLASSES):
        co, cn = int((old == i).sum()), int((new == i).sum())
        cur[name] += co
        fix[name] += cn
        if cn:
            n_img_with[name] += 1
    for f in read_geojson(s.tissue_annotation_path).get("features", []):
        n_holes += sum(len(h) for _, h in rings(f.get("geometry") or {}))

print(f"Tổng số interior ring (hole) trong toàn bộ annotation tissue: {n_holes}")
print(f"{'lớp':14s} {'pixel HIỆN TẠI':>16s} {'pixel ĐÚNG':>14s} {'thay đổi':>10s} {'#ảnh có lớp':>12s}")
for name in TISSUE_CLASSES:
    d = f"{100*(fix[name]-cur[name])/max(cur[name],1):+.1f}%"
    print(f"{name:14s} {cur[name]:16d} {fix[name]:14d} {d:>10s} {n_img_with[name]:12d}")
```

**Đọc kết quả:**
- `n_holes > 0` → xác nhận §3.5(a): annotation CÓ hole và code đang bỏ hết. **Phải sửa.**
- Nếu `necrosis` hoặc `blood_vessel` có `pixel ĐÚNG` lớn hơn `pixel HIỆN TẠI` đáng kể (> +20%)
  → xác nhận §3.5(b): mask huấn luyện đang bị xoá lớp hiếm. **Đây là nguyên nhân gốc.**
- Cột `#ảnh có lớp` cho biết chính xác necrosis/blood_vessel/epidermis xuất hiện trong bao nhiêu
  trong 205 ảnh — con số này quyết định tham số oversampling ở §3.2 và cách stratify ở §3.12.

### 4.2 Kiểm chứng độ phân giải (nghi vấn §3.6)

```python
# scripts/audit_resolution.py
import sys, tifffile, numpy as np
from collections import Counter
sys.path.insert(0, "src")
from prometheus.data.puma.discovery import discover_puma_samples
from prometheus.data.puma.geojson import read_geojson

sizes, coord_max = Counter(), []
for s in discover_puma_samples(sys.argv[1]):
    with tifffile.TiffFile(s.image_path) as t:
        sizes[t.pages[0].shape[:2]] += 1
    mx = 0.0
    for f in read_geojson(s.tissue_annotation_path).get("features", []):
        for ring in (f.get("geometry") or {}).get("coordinates") or []:
            a = np.asarray(ring, dtype=object)
            try:
                mx = max(mx, float(np.asarray(ring[0], np.float64).max()))
            except Exception:
                pass
    coord_max.append(mx)

print("Kích thước ảnh trên đĩa:", dict(sizes))
print(f"Toạ độ annotation lớn nhất: min={min(coord_max):.0f}  max={max(coord_max):.0f}")
```

**Đọc kết quả:**
- ảnh `(1024,1024)` + toạ độ max ≈ 1024 → **ổn**, bỏ qua §3.6.
- ảnh `(512,512)` + toạ độ max ≈ 512 → anh đã downsample cả ảnh và annotation.
  **Phải quay về 1024 gốc** (tải lại từ Zenodo/grand-challenge). Đây là mất mát 2× độ phân giải
  ở một bài toán mà mạch máu chỉ dày vài pixel.
- ảnh `(512,512)` + toạ độ max ≈ 1024 → **mask đang lệch 2× so với ảnh. Mọi thứ sai.** Dừng hết,
  sửa cái này trước.

### 4.3 Xây harness micro-Dice chính thức (nền tảng cho mọi thí nghiệm sau)

```python
# scripts/micro_dice.py — metric CHÍNH THỨC của cuộc thi, không nan-skip, không tặng 1.0
import numpy as np
TISSUE_FG = {"tumor": 1, "stroma": 2, "epidermis": 3, "necrosis": 4, "blood_vessel": 5}

def official_micro_dice(pred_masks, gt_masks):
    """pred/gt: list các mask uint8 (chỉ số train 0..5). Trả về dict per-class + average."""
    out = {}
    for name, idx in TISSUE_FG.items():
        tp = fp = fn = 0
        for p, g in zip(pred_masks, gt_masks):
            pb, gb = (p == idx), (g == idx)
            tp += int((pb & gb).sum()); fp += int((pb & ~gb).sum()); fn += int((~pb & gb).sum())
        den = 2 * tp + fp + fn
        out[name] = (2 * tp / den) if den else 0.0     # KHÔNG trả 1.0, KHÔNG trả NaN
    out["average_micro_dice"] = sum(out[n] for n in TISSUE_FG) / len(TISSUE_FG)
    return out
```

Dùng nó để chấm **out-of-fold prediction của cả 205 ảnh**. Đó là con số duy nhất anh nên dùng để
so sánh giữa các thí nghiệm từ giờ trở đi. Preliminary test 10 ảnh chỉ để sanity-check cuối.

---

## 5. Ba phương án cho nhánh tissue

### Phương án A — "Sửa cái đang có" (rẻ, nhanh, rủi ro thấp nhất)

Giữ nguyên `PrometheusNet`, chỉ áp dụng §3.1 + §3.2 + §3.3 + §3.4 + §3.5 + §3.8 + §3.11.
Không đổi backbone, không đổi kiến trúc.

- **Công sức:** 2–4 ngày. **Rủi ro:** rất thấp. Tất cả đều là fix data pipeline / loss / sampler /
  post-processing, không phải kiến trúc.
- **Kỳ vọng:** `52.90 → 68–73` (≈ hạng 2–3).
- **Bắt buộc làm trước bất kể chọn phương án nào**, vì §3.5 (mask sai) và §3.1 (thiếu step) sẽ
  làm hỏng cả phương án B và C nếu để nguyên.

### Phương án B — "Công thức đội vô địch" (foundation model đóng băng + Efficient-UNet)

Tái hiện TIAKong: **Virchow2 đóng băng** làm feature extractor, concat với RGB, decoder
**Efficient-UNet** (EfficientNetV2-M + SCSE).

Ưu điểm quyết định trên tập 205 ảnh: encoder không cần học gì cả, chỉ decoder học → số tham số
phải fit từ 205 ảnh giảm cả bậc độ lớn. Đó là lý do họ train xong trong **30 phút**.

**Đề xuất cải tiến so với bản gốc của họ — và nó nhắm đúng điểm yếu nhất của họ:**
TIAKong downsample cả ảnh 1024 → 224 một lần, cho token grid 16×16 (stride ~64 px trên ảnh gốc).
Đó là lý do `blood_vessel` của họ chỉ 45.70 — thấp nhất trong 5 lớp của họ.
Thay vào đó: **chạy Virchow2 dạng sliding window** — chia 1024 thành 16 crop 256×256, resize mỗi
crop lên 224, mỗi crop cho grid 16×16 → ghép lại thành **token map 64×64 trên ảnh 1024**
(stride 16 px thay vì 64 px, mịn hơn 4×).

**Chi phí lưu trữ (cực quan trọng vì anh chỉ có RTX 3080 10GB):** cache feature ra đĩa một lần.
- Bản gốc TIAKong: `205 × 16 × 16 × 1280 × 4 B` ≈ **268 MB** → nạp hết vào RAM.
- Bản sliding window: `205 × 64 × 64 × 1280 × 4 B` ≈ **4.3 GB** → memory-map từ đĩa, vẫn dễ.

Sau khi cache, **Virchow2 không bao giờ chạy lại trong lúc train.** Chỉ decoder chạy. Một epoch
trên 3080 tính bằng giây, nên anh có thể train **50.000–100.000 step** ngay tại máy local, giải
quyết §3.1 gần như miễn phí. Đây là điểm mấu chốt về tính khả thi.

- **Truy cập model:** Virchow2 nằm sau HF gated access (`paige-ai/Virchow2`) — cần đăng ký và được
  duyệt. Phương án dự phòng nếu không được duyệt: `MahmoodLab/UNI2-h`, `bioptimus/H-optimus-1`,
  `owkin/phikon-v2`, `histai/hibou-L`. **Lưu ý:** BTC đo baseline `MaskFormer-UNI` chỉ đạt 44.00,
  nên cách *ghép* FM vào decoder quan trọng hơn việc chọn FM nào — đừng đổi decoder khi thay FM.
- **Công sức:** 1–2 tuần (phần lớn là export/cache feature + viết decoder).
- **Kỳ vọng:** `74–80`.

### Phương án C — "Chuyên môn hoá theo thang đo" (đề xuất của tôi để **vượt** hạng 1)

Đây là điều mà **không đội nào trong top 3 làm trọn vẹn**, và nó xuất phát trực tiếp từ dữ liệu:
per-class của TIAKong (necrosis 82.04 ở 224 px) và của LSM (blood_vessel 54.37 bằng U-Net riêng
full-res) chứng minh **PUMA tissue là hai bài toán ở hai thang đo**, và mỗi đội chỉ giải tốt một nửa.

```
┌─ Model 1: "Ngữ nghĩa" (thang đo lớn, thấp phân giải)
│    Virchow2 đóng băng @ 224–448  +  context 5120 downsample về 512
│    → Efficient-UNet → 4 lớp: tumor / stroma / epidermis / necrosis
│    Mục tiêu: tumor 93+, stroma 84+, epidermis 88+, necrosis 80+
│
├─ Model 2: "Mạch máu" (thang đo nhỏ, full phân giải)
│    U-Net (ResNet34/EfficientNet-B3) trên patch 512 @ 40× native, output stride 1
│    Đây là bài toán NHỊ PHÂN: blood_vessel vs không
│    + Auxiliary input: bản đồ mật độ nhân endothelium  ← xem ghi chú dưới
│    Mục tiêu: blood_vessel 55+
│
└─ Hợp nhất: Model 1 cho 4 lớp, Model 2 ghi đè mạch máu (priority cao nhất — đúng như
   thứ tự rasterize ở §4.1). Cả hai đều 5-fold ensemble + 8-view dihedral TTA
   + bias logit tune trên out-of-fold (§3.11c).
```

**Hai ý tưởng phụ trợ mà tôi cho là có giá trị cao và gần như miễn phí, vì repo đã parse sẵn
annotation nuclei:**

1. **Nhân `endothelium` chính là lớp lót mạch máu.** Rasterize centroid của các nhân
   `nuclei_endothelium` thành một density map và dùng nó (a) làm kênh input phụ, hoặc (b) làm
   auxiliary loss cho Model 2. Đây là supervision *trực tiếp về vị trí mạch máu* mà anh đang có
   trong tay và đang bỏ không. Bảng metrics của anh cho thấy `nuclei_endothelium` F1 = 0.055 —
   rất thấp — nhưng ngay cả một tín hiệu yếu về vị trí cũng là prior mạnh cho phân đoạn vùng.
2. **Nhân `apoptosis` tập trung trong vùng hoại tử.** Tương tự, density map của
   `nuclei_apoptosis` là prior cho necrosis. Đáng chú ý: metrics của anh cho thấy model dự đoán
   **rất nhiều** apoptosis FP (43 FP ở `metastatic_roi_003` — chính là một trong 2 ROI có necrosis
   trong GT!). Model *đã* nhìn thấy tín hiệu hoại tử, nó chỉ đang phát ra ở kênh nuclei chứ không
   phải kênh tissue.

Đây là dạng "auto-context" mà LSM dùng (nuclei mask làm kênh thứ 4), nhưng có mục tiêu hơn.

- **Công sức:** 3–4 tuần. **Rủi ro:** trung bình (nhiều thành phần), nhưng mỗi thành phần đều
  có thể đo độc lập bằng harness §4.3.
- **Kỳ vọng:** `78–84` → hạng 1.

### Bảng so sánh phương án

| | A: Sửa cái đang có | B: Công thức vô địch | C: Chuyên môn hoá |
|---|---|---|---|
| Công sức | 2–4 ngày | 1–2 tuần | 3–4 tuần |
| Rủi ro | Rất thấp | Thấp | Trung bình |
| Cần model gated (HF) | Không | **Có** | **Có** |
| Chạy được trên RTX 3080 local | Có (patch 512) | Có (feature cached) | Có |
| Micro Dice kỳ vọng | 68–73 | 74–80 | **78–84** |
| Đập đi xây lại | Không | Chỉ nhánh tissue | Chỉ nhánh tissue |

**Khuyến nghị của tôi: A → C, trong đó A là tuần 1 và cũng là điều kiện tiên quyết cho C.**
Không nhảy thẳng vào C: nếu mask GT sai (§3.5) hoặc độ phân giải lệch (§3.6) thì C cũng sẽ cho
`necrosis = 0` y như bây giờ, và anh sẽ mất 3 tuần để phát hiện ra điều mà script §4.1 nói cho
anh trong 5 phút.

---

## 6. Roadmap + kỳ vọng từng bước

| # | Việc | Mục | Công sức | Micro Dice kỳ vọng |
|---|---|---|---|---:|
| — | *Điểm khởi đầu* | | | **52.90** |
| 0 | Chạy 3 script kiểm chứng §4.1–4.3 | §4 | 1 giờ | (chẩn đoán) |
| 1 | Sửa rasterize: giữ hole + priority lớp | §3.5 | 0.5 ngày | ? — phụ thuộc §4.1 |
| 2 | Sửa độ phân giải nếu lệch | §3.6 | 0.5 ngày | ? — phụ thuộc §4.2 |
| 3 | Log per-class + metric micro chính thức mỗi epoch | §3.4 | 0.5 ngày | (khả kiến) |
| 4 | Patch training 512 + 300 step/epoch, scheduler theo iteration | §3.1 | 1 ngày | → **62–66** |
| 5 | Sampler oversample lớp hiếm (67% patch có tâm trên lớp fg chọn đều) | §3.2 | 1 ngày | → **67–71** |
| 6 | Sửa class weight: bỏ background, dùng sqrt-inverse trên 5 lớp | §3.3 | 2 giờ | → **69–73** |
| 7 | Albumentations: scale + rotate tự do + HSV/RGB shift + elastic | §3.8 | 0.5 ngày | → **71–74** |
| 8 | 8-view TTA cho **cả** đường predict + ensemble 5 fold | §3.11a,b | 0.5 ngày | → **73–77** |
| 9 | Tune bias logit per-class + cổng diện tích trên out-of-fold | §3.11c | 0.5 ngày | → **75–78** |
| 10 | 5-fold stratify multi-label theo lớp hiếm | §3.12 | 0.5 ngày | (giảm nhiễu đo) |
| 11 | Tách nhánh tissue khỏi multitask | §3.10 | 1 ngày | → **76–79** |
| 12 | Virchow2 (hoặc UNI2-h) đóng băng, cache feature sliding-window | §3.9, PA-B | 1 tuần | → **78–82** |
| 13 | Model nhị phân blood_vessel full-res + prior nhân endothelium | PA-C | 1 tuần | → **80–84** |
| 14 | Nhánh context 5120 cho necrosis/epidermis | §3.7 | 3 ngày | → **81–85** |

Các con số kỳ vọng là **cộng dồn và có tính suy đoán**. Điều tôi chắc chắn là **thứ tự ưu tiên**:
bước 4–6 là nơi có tỉ lệ điểm-trên-công-sức cao nhất trong toàn bộ bảng, và bước 0–2 là
điều kiện tiên quyết cho mọi thứ.

**Điểm kiểm soát sau bước 9:** nếu micro-Dice out-of-fold chưa đạt ≥ 72, **đừng** đi tiếp sang
bước 12. Nghĩa là còn một vấn đề data/pipeline chưa tìm ra, và foundation model sẽ không cứu được.

---

## 7. Những gì tôi khuyên **KHÔNG** đập đi

Anh nói sẵn sàng xây lại toàn bộ. Tôi đánh giá là **không nên**, và đây là lý do cụ thể — những
phần này tôi đã đọc và xác nhận là đúng:

| Giữ lại | Vì sao |
|---|---|
| `metrics/matching.py` | Matching centroid one-to-one theo confidence rồi distance — **đúng đặc tả của evaluator PUMA**, kể cả chi tiết `distance < radius` (strict). Viết lại chỉ để làm sai. |
| `io/tissue_tiff.py` | Remap `TISSUE_SUBMISSION_VALUE` (stroma=1, bv=2, tumor=3, epi=4, necro=5) + TIFF tag `SMinSampleValue`/`SMaxSampleValue`. Đúng chuẩn. Tôi đã kiểm tra riêng phần này vì mapping sai ở đây cũng gây ra "lớp = 0". Không phải nguyên nhân. |
| `submission/validation.py` | Validate cấu trúc output đúng contract container. Hữu ích. |
| `domain/` + `data/spatial.py` | Letterbox + nghịch đảo về source space, `ImageMeta` — sạch, có test. Hợp đồng toạ độ nhất quán là thứ khó viết lại đúng. |
| `engine/checkpointing.py` + schema v2 | Có kiểm tra tương thích, có EMA state. Tốt hơn phần lớn code nghiên cứu. |
| **Toàn bộ nhánh nuclei** (`PrometheusNet` + `nuclei_decoder`) | macro F1 0.2172 ≈ mức hạng 2 Track 2 (KongNet 0.2656). **Đừng động vào.** Đầu tư ở đây là lãng phí. |
| `data/puma/discovery.py`, `geojson.py` (trừ phần hole) | Parse strict, báo lỗi rõ. Chỉ cần sửa `geometry_polygons` để giữ interior rings. |
| Hạ tầng 5-fold + manifest + resume | Đúng bài, chỉ cần đổi cách stratify. |

Cái cần đập bỏ chỉ gồm: **head/decoder tissue**, **sampler + transform pipeline**,
**cấu hình loss weight**, và **sự ghép nối multitask**. Đó là khoảng 600 dòng, không phải 4.400 dòng.

Nói cách khác: **vấn đề của anh là data pipeline và training recipe, không phải kiến trúc.**
Bằng chứng mạnh nhất cho luận điểm này là tumor 89.19 / stroma 81.00 / epidermis 90.52 —
kiến trúc hiện tại đã chứng minh nó **thừa sức** đạt mức vô địch trên những lớp mà nó được
huấn luyện đầy đủ.

---

## 8. Checklist để anh duyệt

Xin anh xác nhận từng mục:

**Kiểm chứng (làm ngay, không cần duyệt gì thêm)**
- [ ] Chạy §4.1 — audit mask GT. Gửi tôi output.
- [ ] Chạy §4.2 — audit độ phân giải. Ảnh trên đĩa thật là 512 hay 1024?
- [ ] Xác nhận: `metrics.json` này lấy từ checkpoint nào (fold nào, `best_tissue` hay `best_primary`)?
      Có bật TTA không?

**Quyết định về phạm vi**
- [ ] Đồng ý rằng metric cần tối ưu là `average_micro_dice` (52.90), không phải `average_dice` (75.83)?
- [ ] Đồng ý **không** đầu tư thêm vào nhánh nuclei ở giai đoạn này?
- [ ] Đồng ý **tách** tissue khỏi multitask (§3.10)?
- [ ] Chọn phương án: **A** / **B** / **C** / **A rồi C** (tôi khuyến nghị: A rồi C)

**Quyết định về hạ tầng**
- [ ] Anh có xin được HF gated access cho `paige-ai/Virchow2` không? (cần cho B và C).
      Nếu không, dùng dự phòng nào: `MahmoodLab/UNI2-h` / `bioptimus/H-optimus-1` / `owkin/phikon-v2`?
- [ ] Train ở đâu: Colab A100 như cũ, hay chuyển về RTX 3080 local? (với feature cached ở PA-B
      thì 3080 là đủ và nhanh hơn nhiều về vòng lặp thí nghiệm)
- [ ] Anh có tải được **context image 5120×5120** về không? (cần cho bước 14)

**Về cuộc thi**
- [ ] Phase nào của PUMA còn đang mở để submit? Việc này quyết định deadline và
      liệu có nên đầu tư 3–4 tuần cho phương án C hay không.
- [ ] Xác nhận anh đang nhắm **Track 2** (`config.evaluation.track = "track2"`)? Metric tissue
      giống nhau ở cả hai track, nên công việc này dùng được cho cả hai.

---

## 9. Nguồn tham khảo

- [PUMA — Overview & Goals](https://puma.grand-challenge.org/)
- [PUMA — Evaluation & Ranking](https://puma.grand-challenge.org/submission/) — định nghĩa micro Dice, loại bỏ `tissue_white_background`
- [PUMA — Dataset](https://puma.grand-challenge.org/dataset/) — 206 ROI train / 10 preliminary / 94 final, 1024×1024 @ 40× (0.22 µm/px), context 5120×5120
- [PUMA — Track 1 leaderboard](https://puma.grand-challenge.org/evaluation/e3fbad94-10cd-46e6-8cf8-6fb040e584eb/)
- [TIAKong — Leveraging Pathology Foundation Models for Panoptic Segmentation of Melanoma in H&E Images (arXiv 2507.13974)](https://arxiv.org/abs/2507.13974) — **hạng 1**, Virchow2 + Efficient-UNet, per-class Dice, tumor 93.58 / stroma 83.59 / necrosis 82.04 / bv 45.70 / epi 86.26 → 78.23
- [LSM — A Multi-Stage Auto-Context Deep Learning Framework (arXiv 2503.23958)](https://arxiv.org/abs/2503.23958) — **hạng 2**, SegFormer-B2 + U-Net riêng cho mạch máu (bv 54.37 — cao nhất bảng), auto-context 4 tầng
- [KongNet: A Multi-headed Deep Learning Model for Detection and Classification of Nuclei (arXiv 2510.23559)](https://arxiv.org/html/2510.23559v2) — nhánh nuclei của TIAKong, Track 2 macro F1 0.2656
- [Cracking the PUMA Challenge in 24 Hours with CellViT++ and nnU-Net (arXiv 2503.12269)](https://arxiv.org/abs/2503.12269) — nnU-Net cho tissue đạt Dice 0.750 vs baseline 0.629
- [PUMA baseline paper — GigaScience / PMC11837757](https://pmc.ncbi.nlm.nih.gov/articles/PMC11837757/) — per-class baseline, và ghi nhận *"nnU-Net could not recognize necrosis"*, *"necrosis is the least represented tissue"*
- [TIO-IKIM/PUMA (GitHub)](https://github.com/TIO-IKIM/PUMA)
