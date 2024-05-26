# oneChatGPT
- oneChatbot_gpt2-vietnamese_fine-tune.py
- Author: Mr.Jack _ Công ty www.BICweb.vn
- Date: 24 August 2023

chatGPT siêu siêu nhỏ, huấn luyện chỉ với 1 câu duy nhất, và chỉ trong 1 phút (chatGPT super super tiny ... training with only one sentence dataset in one minute !). Nếu bạn thấy thú vị thì **hãy thả sao** để ủng hộ mình nhé ^^

**But what is a GPT? Visual intro to Transformers** | Deep learning, chapter 5 (https://www.youtube.com/watch?v=wjZofJX0v4M)

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

Hy vọng điều này sẽ giúp các bạn thêm tự tin trên con đường lập trình chinh phục chatGPT nhé! ^^<br>

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

Extra 4fun:
- model_name = 'roneneldan/TinyStories-1M'
- The best learning rate/ loss nhưng kết quả vẫn rất tệ :(
- lr=4.3294e-3
- Epoch 143, Loss 1.087497
- Question: Xin chào Answer: CICweb k khbeh: C ty BICh chào Bách!

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

------------------------------
**Update**: Sunday,29/10/2023 ~> Tạo giao diện Chat chạy Local cho model GPT2 thật đơn giản với Gradio ^^

Để thử nghiệm hãy download file và chạy câu lệnh:

$ python gpt2-gradio.py

![alt text](https://github.com/Mr-Jack-Tung/oneChatGPT/blob/main/GPT2-Gradio_Screenshot%202023-10-29%20at%205.08.png)

------------------------------
**Update**: Monday,30/10/2023 ~> Quay trở lại câu hỏi: Model có bao nhiêu Parameters là đủ để fine-tune chỉ 01 câu tiếng Việt chính xác ?

Hôm trước khi mình đọc cái paper: Pretraining on the Test Set Is All You Need (https://arxiv.org/abs/2309.08632) thì tự dưng thấy khá là ấm ức khi không train được cái model nào nhỏ cỡ 1M parameters mà vẫn trả lời được chính xác. Trong khi 'roneneldan/TinyStories-1M' (https://arxiv.org/abs/2305.07759) làm được và gần đây nhất là phi-CTNL (https://arxiv.org/abs/2309.08632) họ làm được, mà mình chỉ train được chính xác với model nhỏ nhất là TinyStories-33M :(

Hôm nay mình quay trở lại quyết tâm chinh phục việc Fine-tune bằng được cái model 1M tiếng Việt với dataset là 1 câu duy nhất. Theo kinh nghiệm, mình tiếp tục sử dụng model 'roneneldan/TinyStories-1M' để fine-tune. Sau rất nhiều lần thất bại, mất nguyên cả một cái buổi chiều chủ nhật đẹp trời, cuối cùng thì "Trời cũng không phụ lòng người" (^.^) mình đã tìm ra được công thức để fine-tune model siêu siêu nhỏ 1M parameters (8 x GPTNeoBlock, features=64). Kết quả rất tốt như sau:

Update code:
- tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
- model = AutoModelForCausalLM.from_pretrained('roneneldan/TinyStories-1M')

- optimizer = torch.optim.AdamW(model.parameters(), lr=0.05)
- scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.99)

Result:
- model_name = 'roneneldan/TinyStories-1M'
- lr=0.05
- gamma=0.99
- (0.022s) Epoch 143, Loss 0.009
- Question: Xin chào Answer: Công ty BICweb kính chào quý khách!

Để thử nghiệm hãy download file và chạy câu lệnh:

$ python oneChatbot_TinyGPT-1M_vietnamese_fine-tune.py

![alt text](https://github.com/Mr-Jack-Tung/oneChatGPT/blob/main/oneChatbot_TinyGPT-1M_vietnamese_fine-tune%20_%20Screenshot%202023-10-30%20at%208.05PM.png)

Model: roneneldan/TinyStories-1M

GPTNeoForCausalLM(
  (transformer): GPTNeoModel(
    (wte): Embedding(50257, 64)
    (wpe): Embedding(2048, 64)
    (drop): Dropout(p=0.0, inplace=False)
    (h): ModuleList(
      (0-7): 8 x GPTNeoBlock(
        (ln_1): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
        (attn): GPTNeoAttention(
          (attention): GPTNeoSelfAttention(
            (attn_dropout): Dropout(p=0.0, inplace=False)
            (resid_dropout): Dropout(p=0.0, inplace=False)
            (k_proj): Linear(in_features=64, out_features=64, bias=False)
            (v_proj): Linear(in_features=64, out_features=64, bias=False)
            (q_proj): Linear(in_features=64, out_features=64, bias=False)
            (out_proj): Linear(in_features=64, out_features=64, bias=True)
          )
        )
        (ln_2): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
        (mlp): GPTNeoMLP(
          (c_fc): Linear(in_features=64, out_features=256, bias=True)
          (c_proj): Linear(in_features=256, out_features=64, bias=True)
          (act): NewGELUActivation()
          (dropout): Dropout(p=0.0, inplace=False)
        )
      )
    )
    (ln_f): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
  )
  (lm_head): Linear(in_features=64, out_features=50257, bias=False)
)

------------------------------

**Update** Tuesday, 02 April 2024: Phiên bản được chỉnh sửa bởi ChatGPT 3.5 ^^
- Rewrite version: oneChatbot_gpt2-vietnamese_fine-tune_(rewrite_by_ChatGPT).py
- Simplify version: oneChatbot_gpt2-vietnamese_fine-tune_(rewrite_by_ChatGPT)_v2.py

------------------------------
**Update** saving model: 25 May 2024<br>
File: oneChatbot_gpt2-vietnamese_fine-tune.py<br>
Code:<br>
print("\nSaving the model...")<br>
OUTPUT_MODEL = 'OneChatbotGPT2Vi'<br>
tokenizer.save_pretrained(OUTPUT_MODEL)<br>
model.save_pretrained(OUTPUT_MODEL)<br>

**Update** using SFTTrainer: 25 May 2024<br>
Sử dụng SFTTrainer của Hugging Face để Fine-tune model (15 epochs), kết quả trả ra tốt<br>
File: Finetune_SFTTrainer_OneChatbotGPT2Vi.py<br>
Screenshot: oneChatbot_Finetune_SFTTrainer_Screenshot 2024-05-25.jpg<br>
![alt text](https://github.com/Mr-Jack-Tung/oneChatGPT/blob/main/oneChatbot_Finetune_SFTTrainer_Screenshot%202024-05-25.jpg)

------------------------------
**Update** using SFTTrainer with LoRA to Fine-tune: 26 May 2024<br>
Sử dụng SFTTrainer với LoRA của Hugging Face để Fine-tune model, kết quả trả ra tốt với rank=128 ^^. Vì model dùng GPT2 khá nhỏ nên khi Fine-tune với LoRA thì phải train tăng số lần (50 epochs) và tăng rank cao (r=128), với các mức độ nhỏ hơn kết quả trả ra sẽ không đúng<br><br>

Mình đã thử SFTTrainer với LoRA nhưng vì model khá nhỏ (GPT2-137M params) nên để có kết quả tốt thì phải chạy lại nhiều epochs ~ 50-150 ; và với rank cao ~ 64-128 trở lên. Thử nghiệm lại thêm với rank=64 và epochs=150 ~> Ok ^^ <br>
(Ok) RANK: r=64 ; epochs=150 ; checkpoint file: ~117MB ; adapter_model.safetensors: ~37.8MB; with target_modules: ["attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj", ]<br>
trainable params: 9,437,184 || all params: 133,876,992 || trainable%: 7.049145532041831<br><br>

Note: để test SFTTrainer với LoRA thì vẫn không cần đến GPU, chỉ cần Laptop sinh viên thì vẫn có thể test được nhé ^^ <br><br>

File: Finetune_SFTTrainer_withLoRA_OneChatbotGPT2Vi.py<br>
Screenshot: oneChatbot_Finetune_SFTTrainer_withLoRA_Screenshot 2024-05-26.jpg<br>
![alt text](https://github.com/Mr-Jack-Tung/oneChatGPT/blob/main/oneChatbot_Finetune_SFTTrainer_withLoRA_Screenshot%202024-05-26.jpg)

**Update** FT with LoRA rank nhỏ (r=16 ; lora_alpha=32)
Với niềm tin và kinh nghiệm đã train chatbot model siêu nhỏ chỉ với 1M params, nên mình vẫn thử FT với rank nhỏ xem sao.<br>
(Ok) RANK: r=16 ; lora_alpha=32 ; epochs=100 ; checkpoint file: ~32MB ; adapter_model.safetensors: ~9.4MB; with target_modules: ["attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj", ]<br>
trainable params: 2,359,296 || all params: 126,799,104 || trainable%: 1.8606566809809635<br><br>

~> hehe, oh hay thật, với niềm tin và kinh nghiệm train chatbot model siêu nhỏ chỉ với 1M params đã đúng :d fine-tune GPT2-137M model với SFTTrainer và LoRA (r=16 ; lora_alpha=32, adapter_model.safetensors: ~9.4MB, trainable params: 2.36M); dataset chỉ 1 câu duy nhất; không có GPU thì vẫn Ok nhé 😂<br>

![alt text](https://github.com/Mr-Jack-Tung/oneChatGPT/blob/main/oneChatbot_Finetune_SFTTrainer_withLoRA_r16_Screenshot%202024-05-26.jpg)
