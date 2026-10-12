# Reflection — Lab 21

Bản thảo dưới đây dựa trên các sự việc trong phiên thực hành; học viên cần đọc lại để xác nhận đúng trải nghiệm cá nhân trước khi nộp.

**1. Quan sát đáng chú ý**
Fine-tune đạt target 0,97 so với 0,765 của prompt tối ưu, nhưng regression giảm xuống 0,4778 và không đạt cổng đánh giá. Điều cần chú ý là tăng điểm tác vụ không đủ để kết luận model nên được triển khai.

**2. Các trở ngại đã gặp**
Quá trình thực hành gặp lỗi mask TRL, CPU offload, hết VRAM do kernel khác giữ model, và QLoRA bị ngắt. Không có số đo thời gian cho từng lỗi nên tôi không khẳng định lỗi nào tốn thời gian nhất. Việc lưu artifact trước khi đổi môi trường giúp bảo toàn công huấn luyện.

**3. Điều rút ra sau lab**
Loss train thấp chưa đủ để xếp hạng: attn_only có loss thấp hơn correct nhưng cùng target 0,97. Tôi cần đo baseline prompt tốt và regression trước khi kết luận về giá trị của fine-tuning. Bản này không tự suy đoán niềm tin trước lab của học viên.

**4. Vai trò AI assistant**
Assistant hỗ trợ đọc repo, traceback, chỉnh các ô hỗ trợ và đối chiếu artifact. Nó cũng có lỗi: ô cleanup ban đầu thiếu import, và hướng dẫn dùng chung runtime giữa notebook web chưa chính xác. Tôi cần kiểm tra output và trạng thái file để xác nhận lời hướng dẫn.

**5. Đề xuất cho bài toán khách hàng**
Bắt đầu bằng xác định đầu ra cần đo, chuẩn bị tập eval độc lập và baseline prompt tốt trước khi training. Sau đó kiểm tra mask, dữ liệu và tiêu chí hồi quy. Không dùng số liệu trong lab để bảo đảm chất lượng cho dữ liệu khách hàng chưa được đánh giá.
