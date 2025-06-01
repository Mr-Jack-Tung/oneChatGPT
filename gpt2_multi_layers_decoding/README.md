# Demo: Multi-Layer Decoding Generation with GPT-2

## Giới thiệu

Demo này nhằm mục đích trực quan hóa quá trình sinh văn bản (text generation) trong mô hình Transformer, cụ thể là GPT-2, bằng cách hiển thị output từ mỗi layer của model trong quá trình decoding. Thay vì chỉ xem kết quả cuối cùng từ layer cuối cùng, bạn có thể thấy cách mà dự đoán về token tiếp theo thay đổi và được tinh chỉnh khi thông tin đi qua từng lớp transformer.

Script `multi_layer_decoding_demo.py` sẽ thực hiện việc này và tô màu những phần output khác nhau giữa các layer để dễ dàng so sánh.

## Yêu cầu

Để chạy demo này, bạn cần cài đặt các thư viện sau:

-   `torch`
-   `transformers`

Bạn có thể cài đặt chúng bằng pip:

```bash
pip install torch transformers
```

## Hướng dẫn chạy Demo

1.  Mở terminal và điều hướng đến thư mục chứa file `multi_layer_decoding_demo.py`. Nếu bạn đang ở thư mục gốc của project, bạn có thể dùng lệnh sau:

    ```bash
    cd gpt2_multi_layers_decoding
    ```

2.  Chạy script Python bằng lệnh:

    ```bash
    python multi_layer_decoding_demo.py
    ```

Script sẽ tải model GPT-2 (lần đầu chạy có thể mất một chút thời gian để tải model) và thực hiện quá trình sinh văn bản với prompt đã định sẵn. Output từ mỗi layer sẽ được in ra màn hình.

## Phân tích Output

Khi chạy script, bạn sẽ thấy output được in ra cho từng layer (từ Layer 1 đến Layer 12).

-   **Output của mỗi Layer:** Mỗi dòng "Layer X:" hiển thị chuỗi văn bản được tạo ra dựa trên "quan điểm" hoặc "dự đoán" của layer đó tại mỗi bước decoding.
-   **Màu sắc:** Những phần văn bản được tô màu đỏ là những điểm khác biệt so với output của layer cuối cùng (Layer 12). Điều này giúp bạn nhanh chóng nhận ra sự thay đổi trong dự đoán token khi đi qua các layer.

Bạn có thể quan sát thấy rằng các layer ban đầu (gần input) có thể đưa ra những dự đoán khác biệt hoặc ít chính xác hơn so với các layer sâu hơn. Khi thông tin được xử lý qua các layer transformer tiếp theo, model tích hợp thêm ngữ cảnh và thông tin phức tạp hơn, dẫn đến những dự đoán token được tinh chỉnh và thường là chính xác hơn ở các layer cuối cùng.

Sự khác biệt về màu sắc sẽ thể hiện rõ quá trình "tinh chỉnh" này. Các layer đầu có thể có nhiều phần màu đỏ hơn (khác biệt nhiều so với layer cuối), trong khi các layer gần cuối sẽ có ít hoặc không có màu đỏ (output gần giống hoặc giống với layer cuối cùng).

**Tại sao output của các layer trung gian lại "nonsensical"?**

Output từ các layer trung gian thường kém mạch lạc và có vẻ "nonsensical" hơn so với output của layer cuối cùng vì các lý do sau:

*   **Trích xuất đặc trưng theo từng lớp:** Các layer đầu tiên tập trung vào đặc trưng cấp thấp, chưa tích hợp ngữ cảnh rộng.
*   **Ngữ cảnh hóa chưa hoàn chỉnh:** Ở các layer đầu, mô hình chưa xử lý hết các mối quan hệ phức tạp giữa tất cả các token.
*   **Tinh chỉnh ở các layer sâu hơn:** Các layer sâu hơn nhận biểu diễn đã tinh chỉnh, tích hợp phụ thuộc tầm xa và đưa ra dự đoán chính xác hơn.
*   **Ứng dụng LM Head:** Khi áp dụng `lm_head` cho hidden states kém tinh chỉnh của layer đầu, dự đoán có thể kém chính xác.

"Trí thông minh" của mô hình nổi lên khi thông tin đi qua các layer kế tiếp, với các layer cuối cùng đưa ra dự đoán hoàn thiện nhất.

**Tại sao mô hình tạo sinh từng token một qua tất cả các layer?**

Thay vì mỗi layer sinh trọn câu rồi chuyển tiếp, mô hình Transformer tạo sinh từng token một và đưa toàn bộ chuỗi hiện tại qua tất cả các layer ở mỗi bước decoding. Lý do là:

*   **Tính tự hồi quy:** Dự đoán token tiếp theo phụ thuộc vào toàn bộ chuỗi đã có.
*   **Ngữ cảnh hóa:** Mỗi layer xử lý toàn bộ chuỗi để xây dựng biểu diễn ngữ cảnh đầy đủ hơn.
*   **Tinh chỉnh liên tục:** Thông tin được tinh chỉnh dần dần qua từng layer, và dự đoán tốt nhất dựa trên biểu diễn cuối cùng.
*   **Thiết kế mô hình:** Các layer được huấn luyện để xử lý output liên tục của layer trước, không phải các chuỗi rời rạc.

Quá trình này đảm bảo mỗi token mới được dự đoán dựa trên ngữ cảnh phong phú nhất được xây dựng bởi toàn bộ kiến trúc.

## Demo: Beam Search Decoding

Ngoài Greedy Decoding (chọn token có xác suất cao nhất ở mỗi bước), Beam Search là một chiến lược decoding phổ biến khác. Thay vì chỉ theo đuổi một khả năng duy nhất, Beam Search khám phá nhiều chuỗi ứng viên tiềm năng cùng lúc.

-   **Cách hoạt động:** Beam Search duy trì một tập hợp gồm $k$ (beam width) chuỗi có khả năng xảy ra cao nhất ở mỗi bước. Ở mỗi bước, nó mở rộng tất cả $k$ chuỗi này bằng cách thêm các token tiếp theo có khả năng, sau đó chọn ra $k$ chuỗi mở rộng có xác suất tổng cộng cao nhất để tiếp tục.
-   **Ưu điểm:** Thường tạo ra kết quả tốt hơn Greedy Decoding vì nó xem xét nhiều khả năng hơn và ít bị mắc kẹt vào các lựa chọn cục bộ kém tối ưu.
-   **Nhược điểm:** Tốn kém tính toán hơn Greedy Decoding.

Script `multi_layer_beam_search_decoding_demo.py` minh họa chiến lược Beam Search bằng cách sử dụng chức năng tích hợp sẵn trong thư viện `transformers`. Script này sẽ tạo ra và hiển thị $k$ chuỗi ứng viên hàng đầu được tìm thấy bởi Beam Search.

-   **Xác định Sequence tốt nhất:** Trong demo này, sequence tốt nhất được xác định là chuỗi đầu tiên trong danh sách các chuỗi được trả về bởi hàm `model.generate()` khi sử dụng beam search với tham số `num_return_sequences`. Hàm này trả về các chuỗi theo thứ tự xác suất giảm dần, do đó chuỗi đầu tiên là chuỗi có xác suất cao nhất trong số các chuỗi được trả về.
-   **Trực quan hóa:** Script này cũng tô màu sự khác biệt giữa các chuỗi ứng viên được tạo ra (so sánh với chuỗi đầu tiên) để bạn thấy được sự đa dạng trong các kết quả hàng đầu.

## Chiến lược Decoding: Sampling

Sampling decoding là một chiến lược tạo sinh văn bản đưa yếu tố ngẫu nhiên vào quá trình lựa chọn token tiếp theo. Thay vì luôn chọn token có xác suất cao nhất (Greedy) hoặc theo dõi các chuỗi có xác suất cao nhất (Beam Search), Sampling rút ngẫu nhiên token từ phân phối xác suất trên từ vựng do mô hình dự đoán.

-   **Cách hoạt động:** Ở mỗi bước, mô hình tính toán phân phối xác suất cho token tiếp theo, và token được chọn bằng cách lấy mẫu từ phân phối này.
-   **Temperature:** Tham số kiểm soát mức độ ngẫu nhiên. Temperature cao làm phân phối "phẳng" hơn (đa dạng hơn), Temperature thấp làm phân phối "sắc nét" hơn (gần Greedy hơn).
    - Temperature cao (> 1): Làm cho phân phối xác suất "phẳng" hơn, tăng cơ hội chọn các token có xác suất thấp hơn. Kết quả là văn bản đa dạng hơn, sáng tạo hơn, nhưng có thể kém mạch lạc hoặc "điên rồ" hơn.
    - Temperature thấp (< 1): Làm cho phân phối xác suất "sắc nét" hơn, tập trung vào các token có xác suất cao hơn. Kết quả là văn bản ít đa dạng hơn, gần với Greedy Decoding hơn, nhưng thường mạch lạc hơn.
    - Temperature = 1: Lấy mẫu chuẩn.
    - Temperature = 0: Tương đương với Greedy Decoding.
-   **Top-K Sampling:** Giới hạn việc lấy mẫu trong $k$ token có xác suất cao nhất.
-   **Top-P (Nucleus) Sampling:** Giới hạn việc lấy mẫu trong tập hợp token nhỏ nhất có tổng xác suất tích lũy vượt quá ngưỡng $p$.

    **Chi tiết về Top-P Sampling:**

    Top-P Sampling (còn gọi là Nucleus Sampling) là một phương pháp lấy mẫu động hơn Top-K. Thay vì giới hạn số lượng token cố định, nó giới hạn tập hợp các token có thể lấy mẫu dựa trên tổng xác suất tích lũy.

    -   **Cách xác định tập hợp lấy mẫu (Nucleus):** Ở mỗi bước, các token được sắp xếp theo xác suất giảm dần. Nucleus là tập hợp nhỏ nhất các token từ đầu danh sách sao cho tổng xác suất của chúng lớn hơn hoặc bằng $p$.
    -   **Kích thước Nucleus thay đổi động:** Số lượng token trong nucleus không cố định. Nó phụ thuộc vào hình dạng của phân phối xác suất:
        -   **Phân phối sắc nét (mô hình tự tin):** Chỉ cần ít token để đạt tổng xác suất $p$. Nucleus nhỏ, kết quả gần Greedy/Top-K nhỏ.
        -   **Phân phối phẳng (mô hình ít tự tin):** Cần nhiều token hơn để đạt tổng xác suất $p$. Nucleus lớn, kết quả đa dạng hơn.
    -   **Lấy mẫu:** Token tiếp theo được lấy mẫu ngẫu nhiên từ các token trong nucleus (sau khi chuẩn hóa lại xác suất của chúng).

-   **Ưu điểm:** Tạo ra văn bản đa dạng và sáng tạo hơn, ít bị lặp lại. Thích ứng tốt với hình dạng phân phối xác suất.
-   **Nhược điểm:** Kết quả không thể tái lập, có thể kém mạch lạc nếu tham số không phù hợp.

Sampling phù hợp khi cần sự đa dạng và sáng tạo, trong khi Greedy/Beam Search ưu tiên xác suất cao và tính mạch lạc.

## Kết luận

Demo này cung cấp một cái nhìn trực quan về cách mà mô hình Transformer xử lý thông tin và đưa ra dự đoán từng bước. Nó minh họa rằng mỗi layer đóng góp vào việc tinh chỉnh biểu diễn của dữ liệu và dự đoán token tiếp theo, với các layer sâu hơn thường đưa ra những dự đoán cuối cùng và chính xác nhất dựa trên toàn bộ ngữ cảnh đã được xử lý.

Hy vọng demo này giúp bạn hiểu rõ hơn về cơ chế hoạt động "multi-layer decoding" trong các mô hình ngôn ngữ lớn như GPT-2.
