# oneChatGPT
- oneChatbot_gpt2-vietnamese_fine-tune.py
- Author: Mr.Jack _ Công ty www.BICweb.vn
- Date: 24 August 2023

chatGPT siêu siêu nhỏ, huấn luyện chỉ với 1 câu duy nhất, và chỉ trong 1 phút (chatGPT super super tiny ... training with only one sentence dataset in one minute !)

"Vào ngày 21/08/2023, Công ty Cổ phần VinBigdata công bố xây dựng thành công mô hình ngôn ngữ lớn tiếng Việt, đặt nền móng cho việc xây dựng các giải pháp tích hợp AI tạo sinh..." 
https://genk.vn/vinbigdata-phat-trien-cong-nghe-ai-tao-sinh-se-som-cho-ra-mat-chatgpt-phien-ban-viet-20230821140700664.chn

Đây cũng có thể coi là một sự kiện lớn của dân ngành IT nói chung và dân lập trình "ngành" chatGPT nói riêng, và chắc có lẽ "dân ngành" nào cũng mong muốn làm được ra một em chatGPT như vậy ^^

Nhưng đối với kỹ sư lập trình khi mới bước vào thế giới chatGPT thông thường sẽ có những băn khoăn sau:
1. Mình có thể triển khai được 1 em chatGPT chạy được ở nhà không?
2. Cần chuẩn bị máy móc cấu hình phải mạnh cỡ nào?
3. Khối dữ liệu chuẩn bị huấn luyện lớn đến đâu?
4. Tiếng Việt thì có khó huấn luyện không?
5. Phương pháp huấn luyện ra sao?
6. Thời gian huấn luyện trong bao lâu? 
7. Độ chính xác câu trả lời ra sao?
8. Bao giờ sẽ thực hiện được điều đó?

==> Thông thường thì câu trả lời sẽ là ... Không biết ^^


Vì vậy mà hôm nay mình chia sẻ với các bạn một em chatGPT siêu.. siêu.. siêu nhỏ, để các bạn cùng tìm hiểu và chơi với em nó nhé ^^

- Mục đích: Nghiên cứu học tập
- Ngôn ngữ lập trình: Python
- Độ dài mã nguồn: 55 dòng code ^^
- Model pretrained: GPT2

- Ngôn ngữ huấn luyện: Tiếng Việt
- Dữ liệu huấn luyện: Chỉ 01 câu duy nhất ^^
- Máy huấn luyện: 01 laptop sinh viên
- Thời gian huấn luyện: 01 phút
- Độ trả lời chính xác: 100%

Để thử nghiệm hãy download file và chạy câu lệnh:
$ python oneChatbot_gpt2-vietnamese_fine-tune.py

![alt text](https://github.com/Mr-Jack-Tung/oneChatGPT/blob/main/oneChatbot_Screenshot%202023-08-24%20at%2011.30.png)

Hy vọng điều này sẽ giúp các bạn thêm tự tin trên con đường lập trình chinh phục chatGPT nhé! ^^

------------------------------
**Update**: Monday,25/09/2023 ~> Model có bao nhiêu Parameters là đủ để fine-tune chỉ 01 câu tiếng Việt chính xác ?

Mình đã thử fine-tune với model 'huggingface.co/roneneldan/TinyStories-33M', chỉ với 4 x GPTNeoBlock(features=768), có số lượng tham số là 33 triệu, nhỏ hơn nhiều so với model gpt2 (124M), được đánh giá là có chất lượng khá tốt với tốc độ train nhanh hơn (https://arxiv.org/abs/2305.07759), nhưng khi thử nghiệm thì kết quả cũng ổn với model 'TinyStories-33M', còn các model với số lượng params ít hơn (như model 1M, 3M, 8M, 21M) thì không ok ^^

Update code:
- from transformers import AutoTokenizer, AutoModelForCausalLM
- model = AutoModelForCausalLM.from_pretrained('roneneldan/TinyStories-33M')
- tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neo-125M')

Result:
- model_name = 'roneneldan/TinyStories-33M'
- lr=5e-4
- (0.665s) Epoch 29, Loss 0.115172
- Question: Xin chào Answer: Công ty BICweb kính chào quý khách!

With Emoji:
- model_name = 'roneneldan/TinyStories-33M'
- lr=6e-4
- qa_pair = 'Question: Xin chào Answer: Công ty BICweb kính chào quý khách 🤗.'
- (0.662s) Epoch 49, Loss 0.099960
- Question: Xin chào Answer: Công ty BICweb kính chào quý khách 🤗

------------------------------
**Update**: Sunday,15/10/2023 ~> Có thể huấn luyện cho model GPT2 hiểu được hình ảnh không?
(chatGPT super super tiny ... training with only one image dataset in one minute !)

Ngày 25/09/2023 vừa rồi OpenAI có thông báo là con ChatBot của họ có thể nhìn, nghe, và nói được (https://openai.com/blog/chatgpt-can-now-see-hear-and-speak), điều này cũng thúc đẩy mình thử nghiệm xem model GPT2 có thể nhận diện được hình ảnh không. Và mình đã thử fine-turn model 'huggingface.co/nlpconnect/vit-gpt2-image-captioning' để nhận diện được hình ảnh và trả lời bằng Tiếng Việt. Kết quả khá tốt như sau:

Result:
- model_name = 'nlpconnect/vit-gpt2-image-captioning'
- lr=5e-4
- (3.083s) Epoch 15, Loss 0.033
- Answer: Đây là cờ Việt Nam!

Để thử nghiệm hãy download file và chạy câu lệnh: $ python oneChatbot_vit-gpt2-image-captioning-vietnamese_fine-tune.py

Lưu ý: pip install transformers==4.25.1

![alt text](https://github.com/Mr-Jack-Tung/oneChatGPT/blob/main/oneChatbot-vit_Screenshot%202023-10-15%20at%208.26%20PM.png)

