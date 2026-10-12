# Lab 21 — Đánh giá LoRA cho phân loại ticket CSKH

**Học viên:** Nguyễn Văn Huy · **MSSV:** 2A202602428
**Ngày hoàn thiện:** 12/10/2026

## Thiết kế và dữ liệu

Thí nghiệm dùng `unsloth/Qwen3.5-4B` trên Tesla T4, precision fp16. Tôi giữ model và corpus mặc định của lab để thực hiện phép so sánh có sẵn trước khi mở rộng sang dữ liệu miền riêng. Corpus gồm 250 ticket tiếng Việt, chia 225 train / 25 validation bằng seed 42. Tập đánh giá riêng gồm 50 ticket và 15 câu hỏi phổ thông. Baseline (a) dùng prompt đơn giản, (b) dùng prompt tối ưu; adapter fine-tune dùng prompt đơn giản. Baseline đầy đủ được đo trước train trong pipeline NB1 → NB5. Run QLoRA bị ngắt lần đầu và sau đó được chạy tiếp; các adapter đã hoàn thành được giữ lại.

Tôi không sửa prompt tối ưu hoặc tập eval. SHA prompt đã đóng băng là `719e74d3b6232053`. `eval_limit=null`, `smoke_mode=false`. Lần chạy thử 8 mẫu trước đó không được dùng làm kết quả chính của báo cáo này.

## Bằng chứng mask và template

Hai kiểm tra `answer_is_supervised` và `question_is_masked` đều true. Mẫu kiểm chứng có 39/94 token supervised, tỷ lệ 0,4149, dưới ngưỡng lỗi 0,95. Đoạn supervised gồm thẻ đóng reasoning, JSON trả lời và token kết thúc; không bao gồm ticket người dùng:

~~~text
</think>

{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>
~~~

Template giữ nội dung reasoning trong phép thử. Tuy nhiên template thiếu generation markers nên mask tự động từ tokenizer có thể rỗng; pipeline dùng input_ids/labels do labkit tạo, không dựa vào assistant_only_loss tự suy ra của TRL. Việc giữ reasoning trong template không chứng minh rằng model đã được train với reasoning trace.

Độ dài đo trên 250 mẫu: mean 93,1, p95 98, max 101; mức gợi ý là 256. Cấu hình thực tế vẫn dùng max_length=1024 theo tier và nhất quán giữa các run. Mức này không cắt mẫu trong corpus đo được, nhưng lớn hơn mức cần thiết; chọn 256 là một thí nghiệm tối ưu tài nguyên có thể thực hiện sau, không thay cấu hình giữa các đối chứng hiện tại.

## Kết quả chính

| Run | Target | Regression | Format | Latency ms/mẫu |
|---|---:|---:|---:|---:|
| (a) Base + naive | 0,000 | 0,7911 | 0,000 | 3272,2 |
| (b) Base + optimized | 0,765 | 0,7911 | 1,000 | 1058,6 |
| (c) LoRA correct | 0,970 | 0,4778 | 1,000 | 1358,9 |

Target là độ chính xác trung bình theo bốn trường, không phải tỷ lệ ticket đúng hoàn toàn. Format dùng parser và kiểm tra khóa của lab; regression dùng keyword recall nên chỉ là một phép đo giới hạn trên 15 câu hỏi. Latency là thời gian trung bình trên cùng đường generation, không phải benchmark production.

## Đối chứng và tính công bằng

| Run | Placement | r | Trainable params | LR | Loss train báo cáo | Target | Giây train | Peak VRAM GB |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| correct | text-linear | 16 | 32464896 | 0,0001 | 0,6269 | 0,970 | 406,6 | 8,44 |
| attn_only | q,v | 283 | 32456704 | 0,0001 | 0,5384 | 0,970 | 269,3 | 8,45 |
| wrong_lr | text-linear | 16 | 32464896 | 0,00001 | 1,5747 | 0,000 | 397,3 | 8,44 |
| qlora | text-linear | 16 | 32464896 | 0,0001 | 0,7074 | 0,940 | 466,9 | 3,53 |

Cả bốn run dùng 30 optimizer steps. Rank attn_only được tính để khớp ngân sách: chênh 8192 tham số, khoảng 0,0252%, dưới giới hạn 5%. Alpha/r giữ tỷ lệ 2. Loss trong bảng là training_loss được trainer tổng hợp, không nhất thiết là loss của step cuối.

**Vị trí và rank.** attn_only và correct hòa ở target 0,97 dù attn_only có loss train thấp hơn. Vì ngân sách tham số gần tương đương, kết quả này cho thấy hai cách phân bổ adapter đạt cùng điểm trên tác vụ hẹp đã đo. Nó không chứng minh rank luôn quan trọng hơn placement hoặc attention-only luôn tốt hơn trên model/tác vụ khác. Rank đã thay đổi để kiểm soát ngân sách, không phải một phép quét rank độc lập.

**Learning rate.** wrong_lr chỉ giảm LR mười lần so với correct, giữ placement, rank và số steps. Run này có training loss 1,5747, target và format đều bằng 0. Đầu ra không đạt format góp phần khiến target bằng 0 theo scorer; chưa thể suy ra mọi năng lực ngôn ngữ đều bằng 0. Trong ngân sách 30 steps này, LR thấp hơn là lựa chọn kém hiệu quả; chưa kiểm tra việc kéo dài training có khắc phục được hay không.

**Lượng tử hóa.** QLoRA giảm peak VRAM 4,91 GB, khoảng 58,18%, nhưng target giảm 3 điểm phần trăm và thời gian train tăng 60,3 giây so với correct. Format vẫn đạt 1,0. Với phép đo này, QLoRA có lợi rõ về bộ nhớ nhưng có chi phí chất lượng và thời gian; không nên suy rộng thành kết luận cho toàn bộ họ model. Ba đối chứng chỉ được chấm target/format/latency ở NB5, không có phép đo regression đầy đủ riêng.

## Phán quyết

**FAILED.** Target delta = +0,205; regression delta = −0,3133333333. Ngưỡng cho phép là −0,02, nên bản fine-tune không đạt cổng bảo toàn năng lực phổ thông.

Việc tăng target từ 0,765 lên 0,97 cho thấy adapter học được hành vi phân loại ticket tốt hơn baseline prompt tối ưu trong phép đo này. Tuy nhiên, lợi ích chuyên môn đi cùng sự suy giảm rõ trên bộ câu hỏi phổ thông: keyword recall giảm từ 0,7911 xuống 0,4778. Vì lab yêu cầu cả cải thiện target và giữ regression trong ngưỡng, format hoàn hảo không bù được điều kiện bị vi phạm. Kết quả phù hợp với nguy cơ chuyên môn hóa và quên năng lực cũ, nhưng phép đo hiện tại chưa xác định chắc chắn nguyên nhân: tập regression nhỏ và dùng từ khóa, không phải đánh giá toàn diện. Tôi giữ nguyên ngưỡng và kết quả FAIL, thay vì chỉ báo cáo ca phân loại tốt. Một thí nghiệm tiếp theo có thể trộn một tỷ lệ nhỏ dữ liệu phổ thông, rồi đo lại trên tập cố định để kiểm tra giả thuyết replay cải thiện regression.

## Bằng chứng định tính

Đầu ra được thu lại bằng inference bổ sung trên cùng 50 ticket, prompt và adapter; không train lại, không thay baseline/verdict. Điểm tổng sinh lại khớp kết quả chính: baseline 0,765 và FT 0,97. Đây là khớp điểm tổng, không phải bằng chứng rằng từng chuỗi giống hệt lần đo trước vì lần đo trước không lưu đủ đầu ra thô.

Trên toàn bộ 50 mẫu: **33 ca FT thắng, 17 ca hòa, 0 ca FT thua** theo độ chính xác trung bình bốn trường. Không có ca target thua; báo cáo không gọi ca hòa/sai là ca thua. Để có cả ví dụ FT thắng lẫn thua trong đánh giá bốn nhóm, phần dưới bổ sung hai ca regression thua thật trên tập 15 câu đã đóng băng. Tổng cộng báo cáo có 5 ví dụ target và 2 ví dụ regression, kèm đầy đủ đầu ra và thang điểm tương ứng.

### Ticket 6 — FT thắng

Xin chào, mình đặt balo laptop mã đơn DH863123. Đổi size. Hỏi cho biết thôi. Lần cuối mua ở đây.

**Nhãn đúng:**

~~~json
{"intent":"doi_tra","urgency":"thap","product":"balo laptop","sentiment":"tieu_cuc"}
~~~

**Baseline (b):**

~~~json
{"intent": "hoan_tien", "urgency": "cao", "product": "balo laptop", "sentiment": "tieu_cuc"}
~~~

**Fine-tune (c):**

~~~json
{"intent": "doi_tra", "urgency": "thap", "product": "balo laptop", "sentiment": "tieu_cuc"}
~~~

Điểm (b) = 0.5; FT = 1; delta = 0.5. FT sửa đúng intent và urgency: “Đổi size” là doi_tra, “Hỏi cho biết thôi” là thap. Baseline đúng product/sentiment nhưng sai hai trường còn lại.

### Ticket 7 — FT thắng

Alo shop, mình đặt máy xay sinh tố mã đơn OD126693. Muốn đổi. Đã 3 ngày rồi. Bực mình.

**Nhãn đúng:**

~~~json
{"intent":"doi_tra","urgency":"trung_binh","product":"máy xay sinh tố","sentiment":"tieu_cuc"}
~~~

**Baseline (b):**

~~~json
{"intent": "van_chuyen", "urgency": "cao", "product": "máy xay sinh tố", "sentiment": "tieu_cuc"}
~~~

**Fine-tune (c):**

~~~json
{"intent": "doi_tra", "urgency": "trung_binh", "product": "máy xay sinh tố", "sentiment": "tieu_cuc"}
~~~

Điểm (b) = 0.5; FT = 1; delta = 0.5. FT xác định đúng doi_tra và trung_binh; baseline nhầm sang van_chuyen và cao. Cả hai giữ đúng product/sentiment.

### Ticket 3 — Hòa, FT vẫn sai

Cho mình hỏi, mình đặt bình giữ nhiệt mã đơn VN804124. Chưa thấy tiền. Khi nào tiện. Cảm ơn shop nhiều.

**Nhãn đúng:**

~~~json
{"intent":"hoan_tien","urgency":"thap","product":"bình giữ nhiệt","sentiment":"tich_cuc"}
~~~

**Baseline (b):**

~~~json
{"intent": "hoan_tien", "urgency": "trung_binh", "product": "bình giữ nhiệt", "sentiment": "tich_cuc"}
~~~

**Fine-tune (c):**

~~~json
{"intent": "hoan_tien", "urgency": "trung_binh", "product": "bình giữ nhiệt", "sentiment": "tich_cuc"}
~~~

Điểm (b) = 0.75; FT = 0.75; delta = 0. Hòa nhưng cùng sai urgency: nhãn thap, cả hai sinh trung_binh. FT chưa sửa được tín hiệu “Khi nào tiện”.

### Ticket 5 — FT thắng

Shop ơi, mình đặt nồi chiên không dầu mã đơn DH249548. Thiếu phụ kiện. Khi nào tiện. Cho tôi hỏi.

**Nhãn đúng:**

~~~json
{"intent":"san_pham_loi","urgency":"thap","product":"nồi chiên không dầu","sentiment":"trung_tinh"}
~~~

**Baseline (b):**

~~~json
{"intent": "hoan_tien", "urgency": "cao", "product": "nồi chiên không dầu", "sentiment": "trung_tinh"}
~~~

**Fine-tune (c):**

~~~json
{"intent": "san_pham_loi", "urgency": "trung_binh", "product": "nồi chiên không dầu", "sentiment": "trung_tinh"}
~~~

Điểm (b) = 0.5; FT = 0.75; delta = 0.25. FT sửa intent san_pham_loi nhưng urgency vẫn sai trung_binh thay vì thap. Đây là ca FT thắng tương đối, chưa đúng hoàn toàn.

### Ticket 12 — Hòa, FT vẫn sai

Shop ơi, mình đặt áo khoác gió mã đơn VN613097. Bị lỗi. Khi nào tiện. Cảm ơn shop nhiều.

**Nhãn đúng:**

~~~json
{"intent":"san_pham_loi","urgency":"thap","product":"áo khoác gió","sentiment":"tich_cuc"}
~~~

**Baseline (b):**

~~~json
{"intent": "san_pham_loi", "urgency": "trung_binh", "product": "áo khoác gió", "sentiment": "tich_cuc"}
~~~

**Fine-tune (c):**

~~~json
{"intent": "san_pham_loi", "urgency": "trung_binh", "product": "áo khoác gió", "sentiment": "tich_cuc"}
~~~

Điểm (b) = 0.75; FT = 0.75; delta = 0. Hòa và cùng sai urgency ở cụm “Khi nào tiện”: trung_binh thay vì thap. Ba trường còn lại đúng.

Các ca được chọn cho thấy FT cải thiện phân loại intent, nhưng vẫn mắc lỗi urgency trong một số ticket ít khẩn cấp. Đây là mẫu quan sát trên các ví dụ, chưa đủ để suy ra nguyên nhân ở trọng số hoặc dữ liệu train. Bảng đầy đủ 50 mẫu nằm trong results/qualitative_comparison.json.

### Hai ca FT thua trên nhóm regression

Sinh lại đủ 15 câu phổ thông bằng cùng base/adapter, không có system prompt, greedy decode và max_new_tokens=96 như phép đo gốc. Keyword recall trung bình của baseline là 0,7911111111 và FT là 0,4777777778, khớp kết quả đóng băng. Có 8 ca FT thua; hai ca dưới đây đều giảm từ 1,0 xuống 0,0. Đây là ca **regression**, không phải ticket target.

#### Regression 2 — FT thua

**Yêu cầu:** 1 km bằng bao nhiêu mét?

**Từ khóa chấm:** 1000

**Baseline (b):**

~~~text
Trong hệ đo lường quốc tế, tiền tố "kilo" (ký hiệu là **km**) có nghĩa là **1000**. Do đó, 1 kilômét tương đương với 1000 mét.

Kết quả: **1 km = 1000 m**.
~~~

**Fine-tune (c):**

~~~text
{"intent": "hoi_thong_tin", "urgency": "thap", "product": null, "sentiment": "trung_tinh", "intent_confidence": 0.95, "urgency_confidence": 0.98, "product_confidence": 0.0, "sentiment_confidence": 0.92}
~~~

Điểm baseline = 1; FT = 0; delta = -1.

Baseline đưa ra quan hệ 1 km = 1000 m; FT chuyển yêu cầu đổi đơn vị thành JSON triage, không trả lời số mét. Đây là lỗi làm sai tác vụ, không chỉ một khác biệt format.

#### Regression 3 — FT thua

**Yêu cầu:** Viết một câu chúc mừng sinh nhật bằng tiếng Việt.

**Từ khóa chấm:** sinh nhật

**Baseline (b):**

~~~text
Chúc bạn một ngày sinh nhật thật vui vẻ, tràn đầy niềm vui và sức khỏe, cùng với những điều may mắn và hạnh phúc nhất trong năm mới!
~~~

**Fine-tune (c):**

~~~text
{"intent": "chuc_mung_sinh_nhat", "urgency": "trung_tinh", "tone": "trung_thinh", "product": null, "sentiment": "tich_cuc"}
~~~

Điểm baseline = 1; FT = 0; delta = -1.

Baseline viết câu chúc mừng sinh nhật theo yêu cầu; FT tạo JSON với nhãn chuc_mung_sinh_nhat và các trường phân loại. Nhãn đó không thay thế cho câu chúc người dùng yêu cầu.

Hai ví dụ cho thấy hành vi triage lan sang câu hỏi ngoài miền huấn luyện. Chúng bổ sung bằng chứng trực tiếp cho regression delta âm, nhưng chưa đủ để xác định chắc chắn cơ chế quên ở trọng số. Đánh giá regression dùng keyword recall: có thể bỏ sót diễn đạt tương đương hoặc tính điểm cho từ khóa xuất hiện trong câu trả lời sai; vì vậy báo cáo đối chiếu cả nội dung đầu ra, không chỉ điểm. Toàn bộ 15 cặp đầu ra và điểm nằm trong results/regression_comparison.json.

## Kết luận

Với phạm vi đã đo, tôi chưa đề xuất triển khai adapter correct như một model vừa phân loại ticket vừa trả lời kiến thức phổ thông. Mặc dù target đạt 0,97 và tất cả đầu ra target đạt kiểm tra format, regression giảm vượt xa ngưỡng của bài lab. Việc chọn deploy chỉ từ accuracy tác vụ sẽ bỏ qua mục tiêu bảo toàn hành vi ngoài miền huấn luyện. Nếu hệ thống chỉ phục vụ triage, kết quả này có thể là cơ sở để thử nghiệm trong phạm vi giới hạn, nhưng vẫn cần kiểm tra dữ liệu thực tế, ticket ngoài phân phối và quy trình xử lý đầu ra sai trước khi triển khai.

Các đối chứng cho thấy learning rate là một đòn bẩy rõ trong ngân sách training hiện tại: giảm LR mười lần dẫn đến kết quả target và format không đạt. Ngược lại, tăng rank cho attention-only để khớp số tham số không tạo lợi thế target so với correct. QLoRA tiết kiệm bộ nhớ đáng kể nhưng chậm hơn và có target thấp hơn, nên lựa chọn phải dựa vào ràng buộc tài nguyên và chất lượng. Kết luận này chỉ áp dụng cho corpus, model và số steps đã dùng. Tôi ưu tiên phân tích regression theo từng câu và thử replay có kiểm soát hơn là tiếp tục tăng rank mà chưa có giả thuyết rõ ràng.

## Phản tư từ quá trình thực hiện

Trong quá trình làm lab, tôi đã gặp lỗi mask tự động thiếu generation markers, xung đột khi model bị CPU offload và hết VRAM khi kernel khác còn giữ model. Tôi cũng đã chạy thử với EVAL_LIMIT=8 rồi chuyển sang đánh giá đầy đủ; việc notebook chạy hết không đồng nghĩa kết quả đã đủ để nộp. AI assistant hỗ trợ đọc traceback, kiểm tra artifact và thêm ô xuất file, nhưng một số ô hỗ trợ ban đầu có lỗi import và hướng dẫn chuyển runtime chưa chính xác. Vì vậy tôi cần đối chiếu output và file thực tế thay vì chỉ dựa vào lời xác nhận.

Đề xuất nếu có thêm hai giờ: phân tích 15 câu regression và thử một run replay với tỷ lệ nhỏ, giữ mốc, scorer và ngân sách thí nghiệm rõ ràng. Đây là đề xuất tiếp theo, chưa phải thí nghiệm đã thực hiện.

## Bonus đã thực hiện

### B1 — Merge và hot-swap (+3 điểm theo rubric, chờ người chấm)

Merge được kiểm tra trên đủ 50 ticket: target trước merge 0,97; sau merge 0,97; delta 0,00, trong ngưỡng −0,01. Phép đo này chỉ kiểm tra target sau merge, không phải kiểm tra regression mới. Việc merge không khắc phục verdict FAILED của thí nghiệm core.

Hai adapter `correct` và `attn_only` được nạp, chuyển bằng set_adapter và sinh đầu ra trên cùng một base; artifact ghi `same_base=true`. Với ticket chuột không dây, cả hai trả intent doi_tra, urgency cao, product chuột không dây và sentiment tich_cuc. Đây là bằng chứng hoạt động hot-swap trên một request, không phải benchmark thông lượng hoặc hệ thống serving production.

Nguồn: results/merge_check.json và results/hotswap_check.json; output notebook NB6 cũng ghi điểm 0,9700 trước/sau merge và tên hai adapter.

### B5 — Adapter công khai trên Hugging Face Hub (+2 điểm theo rubric, chờ người chấm)

Adapter: [huyhaithanh51/lab21-qwen35-4b-triage-lora](https://huggingface.co/huyhaithanh51/lab21-qwen35-4b-triage-lora).

Commit: `9f88c8477773feac84a51e54f55f3011282b2f34`. Repo được xác minh public, có adapter_config.json, adapter_model.safetensors và README.md. SHA-256 trọng số Hub khớp file correct local: `11c544f555ff9b41efde1e1481e22e6858b3ca198b3df1085d4910c196edad7f`. Upload chỉ adapter và model card, không phải toàn bộ base model. Model card ghi rõ giới hạn và verdict regression FAILED.

Nguồn: results/hf_hub.json và xác minh read-only qua API Hub ngày 12/10/2026. Tổng bonus có bằng chứng là B1+B5, tương ứng tối đa 5 điểm; không khai báo B2/B3/B4 vì chưa thực hiện.

## Nguồn số liệu và giới hạn còn lại

Số liệu core lấy từ mask_proof.json, template_check.json, token_stats.json, baselines_frozen.json, runs.csv, autopsy.json và verdict.json. Định tính bổ sung lấy từ qualitative_comparison.json, qualitative_selected.json và qualitative_reproduction_check.json; phép sinh lại không thay kết quả đóng băng.

Báo cáo có 7 ví dụ định tính, gồm 2 ca FT thua trên nhóm regression. Tập target vẫn có 0 ca thua; điều này được khai báo rõ để không nhầm hai nhóm và thang điểm. Nếu người chấm yêu cầu riêng 2 ticket target thua, tập đo hiện tại không có bằng chứng đó và cần cách xử lý từ người chấm; không chỉnh tập eval hoặc tạo ví dụ giả. Verifier PASS xác nhận các gate tự động, không bảo đảm điểm số do người chấm quyết định.
