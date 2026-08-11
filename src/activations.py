from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen2.5-1.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

prompt_text = "Explain recursion in one paragraph." # can be imported from a prompts file later

tokenized_inputs = tokenizer(prompt_text, return_tensors="pt")

outputs = model(
    **tokenized_inputs,
    output_hidden_states=True,
    return_dict=True,
)

hidden_states = outputs.hidden_states # will output a tuple with one tensor (item) per layer. the tensor has shape (batch_size, sequence_length, hidden_size)

layer_index = -1 # last layer
token_index = -1 # last token in the sequence; use if time scale is token-based

print(hidden_states[layer_index][token_index].shape)
