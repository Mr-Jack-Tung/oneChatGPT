# Single Block GPT-2 (No Dependencies)

This project provides a minimal implementation of a GPT-2-like language model with a single transformer block and a character-level tokenizer, built from scratch using PyTorch without relying on external libraries like Hugging Face `transformers`. It is primarily intended for educational or experimental purposes to understand the basic components of a transformer model.

Ở bản thử nghiệm này, tôi cố gắng không sử dụng thư viện transformers của huggingface vì nó có thể gây khó khăn khi hiểu rõ hơn về cơ chế hoạt động của mô hình transformer. Tuy nhiên, nếu bạn muốn sử dụng thư viện transformers thì bạn có thể tham khảo các phiên bản khác của dự án nhé.

Chúng tôi đã phát hiện ra rằng việc huấn luyện mô hình trong các giai đoạn riêng biệt với khởi tạo lại của mô hình, bộ tối ưu hóa và mã hóa ký tự ở đầu mỗi giai đoạn, sau đó chỉ tải trạng thái của mô hình được lưu trữ là một cách hiệu quả để đạt được sự trùng khớp chính xác của dữ liệu huấn luyện này. Mô hình được cài đặt cho hai giai đoạn với 30 epoch mỗi giai đoạn, điều này đã được chứng minh là cấu hình thành công.

Mình phát hiện ra một điểm rất thú vị là chạy quá trình training model 2 lần với 30 epoch mỗi lần thì cho ra kết quả chính xác, còn nếu chạy training 1 lần với 300 epoch, thậm chí là 3000 epoch thì kết quả output vẫn không chính xác ?! ... đố bác nào biết lý do vì sao có hiện tượng này nhé ^^

## Project Structure

-   `single_block_gpt2_no_depend_model.py`: Defines the model architecture, including the `GPT2Config`, `SelfAttention`, `MLP`, `TransformerBlock`, and the main `SingleBlockGPT2ModelNoDepend` class.
-   `character_tokenizer.py`: Implements a simple character-level tokenizer.
-   `train_single_block_gpt2_no_depend.py`: Script for training the model.
-   `inference_single_block_gpt2_no_depend.py`: Script for generating text using a trained model.

## Getting Started

1.  Ensure you have Python and PyTorch installed. You can install PyTorch by following the instructions on the official PyTorch website: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)
2.  Navigate to the `Single_Block_GPT2_No_depend` directory in your terminal.

## Usage

### Training the Model

The `train_single_block_gpt2_no_depend.py` script is configured to train the model on a single QA pair: 'Question: Xin chào \nAnswer: Công ty BICweb kính chào quý khách!.'.

Based on our research, training the model in stages with re-initialization of the model, optimizer, and tokenizer at the start of each stage, and loading only the model state, is effective for achieving accurate reproduction of this specific training data. The script is currently set up for 2 stages of 30 epochs each, which was found to be a successful configuration.

To train the model, run the following command from the project root directory:

```bash
uv run Single_Block_GPT2_No_depend/train_single_block_gpt2_no_depend.py
```

This will perform the staged training and save the trained model state and tokenizer in the `TrainedSingleBlockGPT2_No_depend` directory.

### Generating Text (Inference)

The `inference_single_block_gpt2_no_depend.py` script uses the trained model to generate text. It is configured to use greedy decoding and generate a fixed number of tokens (64) to demonstrate the model's ability to reproduce the training data.

To run inference using the trained model, run the following command from the project root directory:

```bash
uv run Single_Block_GPT2_No_depend/inference_single_block_gpt2_no_depend.py
```

This will load the trained model and tokenizer and generate text based on the input prompt "Question: Xin chào".

## Key Findings

During the research and experimentation with this project, we made a significant observation regarding the training process and its impact on the model's ability to accurately reproduce the single training data point:

-   **Staged Training Effectiveness and the Value of the First Stage:** Running the training process in multiple distinct stages (e.g., 2 stages of 30 epochs each), saving and loading the model state between stages, was found to be crucial for achieving accurate reproduction of the training data during inference. The **first training stage is essential** as it moves the model's weights from a random, untrained state to a partially trained state where it has learned the basic patterns and structure of the training data. This improved weight state provides a much better starting point for subsequent stages compared to starting from random weights each time.
-   **Optimizer Re-initialization and Overfitting Dynamics:** A key factor contributing to the effectiveness of staged training appears to be the **re-initialization of the optimizer** (like AdamW) at the start of each training stage. Unlike a single continuous training run or staged training that preserves the optimizer's internal state (such as momentum and adaptive learning rates), resetting the optimizer periodically seems to help the optimization process navigate the complex loss landscape when overfitting this specific, minimal dataset. In this extreme overfitting scenario, the optimizer's accumulated state from earlier training steps might hinder the final convergence to the very precise minimum required for perfect character-level prediction. Resetting the optimizer provides a "fresh start" for its internal dynamics in each stage, allowing it to potentially find a better path to this specific, highly accurate solution. This highlights how optimizer dynamics can significantly impact convergence in non-convex loss landscapes, especially when aiming for perfect memorization.
-   **Tokenizer Interaction:** While the custom model architecture itself is functional (as shown by successful inference when paired with a different tokenizer), achieving accurate reproduction with the custom character tokenizer required careful tuning of the training process, highlighting the interaction between the tokenizer's representation and the model's ability to overfit.

This project serves as an interesting case study on the nuances of training dynamics and optimizer behavior, particularly in extreme overfitting scenarios with minimal data and simple model components.

## Những Phát Hiện Quan Trọng (Tiếng Việt)

Trong quá trình nghiên cứu và thử nghiệm với dự án này, chúng tôi đã đưa ra một quan sát quan trọng liên quan đến quá trình huấn luyện và tác động của nó đến khả năng tái tạo chính xác điểm dữ liệu huấn luyện duy nhất của mô hình:

-   **Hiệu quả của Huấn luyện Phân đoạn và Giá trị của Giai đoạn Đầu tiên:** Việc chạy quá trình huấn luyện theo nhiều giai đoạn riêng biệt (ví dụ: 2 giai đoạn, mỗi giai đoạn 30 epoch), lưu và tải trạng thái mô hình giữa các giai đoạn, được phát hiện là rất quan trọng để đạt được khả năng tái tạo chính xác dữ liệu huấn luyện khi suy luận. **Giai đoạn huấn luyện đầu tiên là thiết yếu** vì nó đưa trọng số của mô hình từ trạng thái ngẫu nhiên, chưa được huấn luyện sang trạng thái đã được huấn luyện một phần, nơi nó đã học được các mẫu cơ bản và cấu trúc của dữ liệu huấn luyện. Trạng thái trọng số được cải thiện này cung cấp một điểm khởi đầu tốt hơn nhiều cho các giai đoạn tiếp theo so với việc bắt đầu từ trọng số ngẫu nhiên mỗi lần.

-   **Khởi tạo lại Bộ Tối ưu hóa và Động lực Overfitting:** Một yếu tố chính góp phần vào hiệu quả của huấn luyện phân đoạn dường như là việc **khởi tạo lại bộ tối ưu hóa** (như AdamW) ở đầu mỗi giai đoạn huấn luyện. Lưu ý là các bộ tối ưu hóa hiện đại như AdamW không chỉ sử dụng gradient hiện tại để cập nhật trọng số, mà còn duy trì các trạng thái nội bộ (internal states), ví dụ như thông tin về động lượng (momentum) và ước lượng phương sai của gradient. Các trạng thái này tích lũy thông tin từ các bước huấn luyện trước đó và ảnh hưởng đến hướng đi cũng như tốc độ cập nhật trọng số trong tương lai.

Không giống như một lần chạy huấn luyện liên tục duy nhất hoặc huấn luyện phân đoạn giữ nguyên trạng thái nội bộ của bộ tối ưu hóa (như động lượng và tốc độ học thích ứng), ví dụ như khi chạy một lần duy nhất với số lượng epoch rất lớn (300 hoặc 3000 epoch), bộ tối ưu hóa hoạt động liên tục mà không bị reset trạng thái. Trong trường hợp này, nó có thể đi vào một vùng trên bề mặt hàm mất mát mà cuối cùng không dẫn đến khả năng tái tạo chính xác dữ liệu huấn luyện, mặc dù giá trị mất mát (loss) có thể vẫn rất thấp, nhưng kết quả dự đoán vẫn không chính xác, thì việc reset bộ tối ưu hóa định kỳ dường như giúp quá trình tối ưu điều hướng bề mặt hàm mất mát phức tạp khi overfitting tập dữ liệu tối thiểu cụ thể này. 

Trong kịch bản overfitting cực đoan này, trạng thái tích lũy của bộ tối ưu hóa từ các bước huấn luyện trước đó có thể cản trở sự hội tụ cuối cùng đến điểm cực tiểu rất chính xác cần thiết cho việc dự đoán ký tự hoàn hảo. Việc reset bộ tối ưu hóa cung cấp một "khởi đầu mới" cho động lực nội bộ của nó ở mỗi giai đoạn, cho phép nó có khả năng tìm ra một đường đi tốt hơn đến giải pháp cụ thể, có độ chính xác cao này. Điều này làm nổi bật cách động lực của bộ tối ưu hóa có thể ảnh hưởng đáng kể đến sự hội tụ trong các bề mặt hàm mất mát không lồi, đặc biệt khi nhằm mục tiêu ghi nhớ hoàn hảo.

-   **Tương tác với Bộ Mã hóa:** Mặc dù kiến trúc mô hình tùy chỉnh tự nó hoạt động tốt (như đã thấy khi suy luận thành công khi kết hợp với một bộ mã hóa khác), việc đạt được khả năng tái tạo chính xác với bộ mã hóa ký tự tùy chỉnh đòi hỏi phải tinh chỉnh cẩn thận quá trình huấn luyện, làm nổi bật sự tương tác giữa biểu diễn của bộ mã hóa và khả năng overfitting của mô hình.

Dự án này đóng vai trò là một nghiên cứu điển hình thú vị về các sắc thái của động lực huấn luyện và hành vi của bộ tối ưu hóa, đặc biệt trong các kịch bản overfitting cực đoan với dữ liệu tối thiểu và các thành phần mô hình đơn giản.
