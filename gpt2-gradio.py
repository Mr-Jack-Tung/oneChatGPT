import torch
import gradio

from transformers import AutoTokenizer, AutoModelForCausalLM

model_name="gpt2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, pad_token_id=tokenizer.eos_token_id)

print(f"Loading:",model_name)

def predict(inp):
    input_ids = tokenizer.encode(inp, return_tensors='pt')
    beam_output = model.generate(input_ids, max_length=100, num_beams=5,
                                 no_repeat_ngram_size=2, early_stopping=True)
    output = tokenizer.decode(beam_output[0], skip_special_tokens=True,
                              clean_up_tokenization_spaces=True)
    return ".".join(output.split(".")[:-1]) + "."


INTERFACE = gradio.Interface(fn=predict, inputs=gradio.Textbox(label="Input your question:"), outputs="text", title="GPT-2",
                 description="GPT-2 is a large transformer-based language model with 1.5 billion parameters, trained on a dataset[1] of 8 million web pages. GPT-2 is trained with a simple objective: predict the next word, given all of the previous words within some text.",
                 thumbnail="https://github.com/gradio-app/gpt-2/raw/master/screenshots/interface.png?raw=true",
                 capture_session=False)

INTERFACE.launch(inbrowser=True)

# who is Steve Job ?
# who is Napoleon ?
# who is Einstein ?