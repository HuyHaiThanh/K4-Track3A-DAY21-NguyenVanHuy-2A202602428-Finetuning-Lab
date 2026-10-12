# Bằng chứng định tính bổ sung

FT thắng: 33; thua: 0; hòa: 17.
Điểm sinh lại khớp kết quả đóng băng: True.

## Ticket 6
Xin chào, mình đặt balo laptop mã đơn DH863123. Đổi size. Hỏi cho biết thôi. Lần cuối mua ở đây.

Nhãn: {"intent": "doi_tra", "urgency": "thap", "product": "balo laptop", "sentiment": "tieu_cuc"}

Baseline (b):
~~~json
{"intent": "hoan_tien", "urgency": "cao", "product": "balo laptop", "sentiment": "tieu_cuc"}
~~~
Fine-tune:
~~~json
{"intent": "doi_tra", "urgency": "thap", "product": "balo laptop", "sentiment": "tieu_cuc"}
~~~
Điểm (b)=0.5; FT=1.0; delta=0.5.

## Ticket 7
Alo shop, mình đặt máy xay sinh tố mã đơn OD126693. Muốn đổi. Đã 3 ngày rồi. Bực mình.

Nhãn: {"intent": "doi_tra", "urgency": "trung_binh", "product": "máy xay sinh tố", "sentiment": "tieu_cuc"}

Baseline (b):
~~~json
{"intent": "van_chuyen", "urgency": "cao", "product": "máy xay sinh tố", "sentiment": "tieu_cuc"}
~~~
Fine-tune:
~~~json
{"intent": "doi_tra", "urgency": "trung_binh", "product": "máy xay sinh tố", "sentiment": "tieu_cuc"}
~~~
Điểm (b)=0.5; FT=1.0; delta=0.5.

## Ticket 3
Cho mình hỏi, mình đặt bình giữ nhiệt mã đơn VN804124. Chưa thấy tiền. Khi nào tiện. Cảm ơn shop nhiều.

Nhãn: {"intent": "hoan_tien", "urgency": "thap", "product": "bình giữ nhiệt", "sentiment": "tich_cuc"}

Baseline (b):
~~~json
{"intent": "hoan_tien", "urgency": "trung_binh", "product": "bình giữ nhiệt", "sentiment": "tich_cuc"}
~~~
Fine-tune:
~~~json
{"intent": "hoan_tien", "urgency": "trung_binh", "product": "bình giữ nhiệt", "sentiment": "tich_cuc"}
~~~
Điểm (b)=0.75; FT=0.75; delta=0.0.

## Ticket 5
Shop ơi, mình đặt nồi chiên không dầu mã đơn DH249548. Thiếu phụ kiện. Khi nào tiện. Cho tôi hỏi.

Nhãn: {"intent": "san_pham_loi", "urgency": "thap", "product": "nồi chiên không dầu", "sentiment": "trung_tinh"}

Baseline (b):
~~~json
{"intent": "hoan_tien", "urgency": "cao", "product": "nồi chiên không dầu", "sentiment": "trung_tinh"}
~~~
Fine-tune:
~~~json
{"intent": "san_pham_loi", "urgency": "trung_binh", "product": "nồi chiên không dầu", "sentiment": "trung_tinh"}
~~~
Điểm (b)=0.5; FT=0.75; delta=0.25.

## Ticket 12
Shop ơi, mình đặt áo khoác gió mã đơn VN613097. Bị lỗi. Khi nào tiện. Cảm ơn shop nhiều.

Nhãn: {"intent": "san_pham_loi", "urgency": "thap", "product": "áo khoác gió", "sentiment": "tich_cuc"}

Baseline (b):
~~~json
{"intent": "san_pham_loi", "urgency": "trung_binh", "product": "áo khoác gió", "sentiment": "tich_cuc"}
~~~
Fine-tune:
~~~json
{"intent": "san_pham_loi", "urgency": "trung_binh", "product": "áo khoác gió", "sentiment": "tich_cuc"}
~~~
Điểm (b)=0.75; FT=0.75; delta=0.0.

Không đủ hai ca FT thua trên tập eval cố định. Khai báo hạn chế này; không tạo ca thua giả.

## Regression bổ sung — hai ca FT thua thật

Target không có ca thua; trên 15 câu regression có 8 ca thua, điểm trung bình sinh lại khớp baseline 0,7911111111 và FT 0,4777777778.

### Câu 2: 1 km bằng bao nhiêu mét?

Từ khóa: 1000

Baseline:

~~~text
Trong hệ đo lường quốc tế, tiền tố "kilo" (ký hiệu là **km**) có nghĩa là **1000**. Do đó, 1 kilômét tương đương với 1000 mét.

Kết quả: **1 km = 1000 m**.
~~~

FT:

~~~text
{"intent": "hoi_thong_tin", "urgency": "thap", "product": null, "sentiment": "trung_tinh", "intent_confidence": 0.95, "urgency_confidence": 0.98, "product_confidence": 0.0, "sentiment_confidence": 0.92}
~~~

Điểm baseline 1; FT 0; delta -1.

### Câu 3: Viết một câu chúc mừng sinh nhật bằng tiếng Việt.

Từ khóa: sinh nhật

Baseline:

~~~text
Chúc bạn một ngày sinh nhật thật vui vẻ, tràn đầy niềm vui và sức khỏe, cùng với những điều may mắn và hạnh phúc nhất trong năm mới!
~~~

FT:

~~~text
{"intent": "chuc_mung_sinh_nhat", "urgency": "trung_tinh", "tone": "trung_thinh", "product": null, "sentiment": "tich_cuc"}
~~~

Điểm baseline 1; FT 0; delta -1.

