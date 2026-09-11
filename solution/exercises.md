# K4 — Ngày 1: Bài Tập & Phản Ánh
## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 4 tiếng
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng —
đừng để dồn hết về cuối buổi. Thay dòng `*Câu trả lời của bạn*` bằng câu
trả lời thật (chấm tự động sẽ đếm số câu đã trả lời).

---

## Block 1 — API Cơ Bản (trả lời sau Checkpoint 1)

### Câu 1.1 — Độ nhạy của temperature
Gọi `call_openai` với temperature 0.0, 0.5, 1.0 và 1.5 dùng prompt
**"Hãy kể cho tôi một sự thật thú vị về Việt Nam."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi?** (2–3 câu)
> Với temperature 0.0, phản hồi rất ổn định. Khi tăng lên 0.5 và 1.0, phản hồi bắt đầu đa dạng và sáng tạo hơn. Ở temperature 1.5 đôi khi lan man hoặc kém mạch lạc, cho thấy temperature quá cao làm giảm chất lượng dù tăng tính sáng tạo.

### Câu 1.2 — Chọn temperature cho sản phẩm
**Bạn sẽ đặt temperature bao nhiêu cho chatbot hỗ trợ khách hàng, và tại sao?**
> Tôi sẽ đặt temperature khoảng 0.2–0.3. Chatbot hỗ trợ khách hàng cần đưa ra thông tin chính xác, nhất quán và đáng tin cậy, temperature thấp giúp model chọn token có xác suất cao nhất, giảm rủi ro đưa ra thông tin sai hoặc bịa đặt. Không đặt 0.0 hoàn toàn vì cần một chút linh hoạt để phản hồi tự nhiên, không quá máy móc.

### Câu 1.3 — Đánh đổi chi phí
Kịch bản: 10.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 3 lần,
mỗi lần trung bình ~350 token đầu ra.

**Ước tính GPT-4o đắt hơn GPT-4o-mini bao nhiêu lần cho workload này? Nêu một
trường hợp GPT-4o xứng đáng với chi phí và một trường hợp nên dùng mini:**
> Với 10.000 × 3 × 350 = 10.500.000 token output/ngày (10,5 triệu token). GPT-4o có giá output $10.00/1 triệu token → $105/ngày; GPT-4o-mini có giá $0.60/1 triệu token → $6.3/ngày. (Giá input cũng chênh lệch tương tự: $2.50 vs $0.15). Do đó, GPT-4o đắt hơn khoảng 16.7 lần. Trường hợp nên dùng GPT-4o: phân tích pháp lý hoặc y tế — nơi chất lượng suy luận và độ chính xác cao là bắt buộc, sai sót có thể gây hậu quả nghiêm trọng. Trường hợp nên dùng mini: chatbot FAQ trả lời câu hỏi đơn giản hoặc tóm tắt nội dung cơ bản, nơi tốc độ và chi phí quan trọng hơn chất lượng suy luận.

---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona
Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích blockchain là gì?"** nhưng hai system prompt khác nhau:
- "Bạn là giáo viên tiểu học, giải thích thật đơn giản cho trẻ 8 tuổi."
- "Bạn là chuyên gia tài chính, trả lời chuyên sâu bằng thuật ngữ kỹ thuật."

**Hai phản hồi khác nhau như thế nào (độ dài, từ vựng, ví dụ)? System prompt
ảnh hưởng đến hành vi model ra sao?** (3–4 câu)
> Phản hồi từ persona giáo viên tiểu học sử dụng ngôn ngữ rất đơn giản, dùng nhiều ví dụ so sánh gần gũi câu ngắn và tránh thuật ngữ kỹ thuật. Ngược lại, persona chuyên gia tài chính trả lời dài hơn, dùng thuật ngữ chuyên ngành và đi sâu vào cơ chế hoạt động. System prompt ảnh hưởng mạnh đến hành vi model nó định hình giọng điệu, mức độ phức tạp của từ vựng, đối tượng mục tiêu và chiều sâu nội dung

### Câu 2.2 — tiktoken vs đếm từ
Chọn một đoạn văn tiếng Việt ~100 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Vì sao tiếng Việt thường tốn
nhiều token hơn tiếng Anh cùng độ dài?**
> Với đoạn văn tiếng Việt 132 từ, tiktoken (o200k_base, gpt-4o) đếm được 164 token, trong khi ước lượng thô (132/0.75 ≈ 176 token) chênh lệch khoảng 6.8%. Tỉ lệ token/từ tiếng Việt là 1.24, cao hơn tiếng Anh (1.10 token/từ cho đoạn tương đương). Tiếng Việt tốn nhiều token hơn tiếng Anh vì tokenizer GPT được huấn luyện chủ yếu trên dữ liệu tiếng Anh nên từ tiếng Việt thường bị tách nhỏ hơn, đặc biệt các ký tự có dấu (ă, â, ơ, ư, ê...) là Unicode ngoài bảng ASCII cần nhiều byte hơn để mã hóa.

---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming
**Streaming quan trọng nhất trong trường hợp nào, và khi nào thì
non-streaming lại phù hợp hơn?** (1 đoạn văn)
> Streaming quan trọng nhất khi phản hồi liên tục và tương tác với người dùng. ví dụ chatbot, trợ lý ảo,  vì người dùng thấy nội dung xuất hiện ngay. Ngược lại, non-streaming phù hợp hơn khi cần toàn bộ output trước khi xử lý tiếp. ví dụ: phân loại email Non-streaming cũng đơn giản hơn về mặt code
### Câu 3.2 — Vì sao backoff theo cấp số nhân?
**So với delay cố định (ví dụ luôn chờ 1 giây), exponential backoff có lợi
thế gì khi API bị quá tải? Điều gì xảy ra nếu hàng nghìn client cùng retry
với delay cố định giống nhau?**
> Exponential backoff tăng thời gian chờ gấp đôi sau mỗi lần thất bại (0.1s → 0.2s → 0.4s...), giúp giảm dần áp lực lên server đang quá tải, các lần retry sau cách nhau xa hơn, cho server thời gian phục hồi. Nếu hàng nghìn client cùng retry với delay cố định (ví dụ 1 giây), tất cả sẽ gửi lại request đồng loạt sau mỗi 1 giây, server vừa hồi phục lại bị đánh sập bởi đợt request đồng thời, tạo vòng lặp quá tải vô tận.

---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona
**Bạn chọn persona gì cho trợ lý của mình? Viết lại system prompt đó và giải
thích 1–2 lựa chọn từ ngữ quan trọng trong prompt (ví dụ: vì sao yêu cầu
"trả lời ngắn gọn", vì sao chỉ định ngôn ngữ...):**
> System prompt: "Bạn là chuyên gia tư vấn công nghệ chuyên am hiểu về smartphone. Hãy tư vấn khách quan, hỏi rõ nhu cầu và ngân sách của khách hàng trước khi đề xuất, và trả lời ngắn gọn bằng tiếng Việt." 
Lựa chọn "hỏi rõ nhu cầu và ngân sách" ép model phải tương tác hai chiều và thu thập thông tin thay vì chỉ đoán bừa điện thoại để đề xuất. Lựa chọn "tư vấn khách quan" giúp model không thiên vị một hãng nào (như Apple hay Samsung) mà thực sự dựa trên nhu cầu người dùng. Việc yêu cầu "ngắn gọn" giúp kiểm soát độ dài và tiết kiệm chi phí output.

### Câu 4.2 — Hạn chế & cải thiện
**Trợ lý của bạn hiện có hạn chế lớn nhất là gì (ví dụ: history chỉ 3 lượt,
không có bộ nhớ dài hạn, không kiểm duyệt nội dung...)? Đề xuất một cải
thiện cụ thể và mô tả ngắn cách triển khai:**
> Hạn chế lớn nhất là history chỉ giữ 3 lượt gần nhất — khi khách hàng cung cấp nhiều thông tin rải rác (như ngân sách ở lượt 1, thích chụp ảnh ở lượt 2, cần pin trâu ở lượt 4), trợ lý tư vấn smartphone sẽ "quên" mất các yêu cầu ban đầu và có thể đề xuất sai máy. Cải thiện đề xuất: thêm cơ chế trích xuất "Hồ sơ khách hàng" (User Profile) — gọi một API ẩn sau mỗi lượt để trích xuất các key (như Budget, Camera, Battery) từ tin nhắn của khách và liên tục cập nhật/ghim hồ sơ này vào system prompt. Nhờ vậy trợ lý luôn nhớ chính xác nhu cầu trọn vẹn của khách mà không cần giữ lịch sử quá dài gây tốn token.

---

## Danh Sách Kiểm Tra Nộp Bài

- [ ] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [ ] Cả 4 checkpoint pytest đều pass
- [ ] Tất cả 9 câu trong file này đã được trả lời
- [ ] Đã copy bài làm vào folder `solution/`, push lên fork và dán link trên trang bài Lab ở VLearn trước 23:59 ngày 11/09/2026
