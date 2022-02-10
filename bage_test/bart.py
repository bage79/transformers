from transformers.models.bart import BartModel, BartTokenizer

if __name__ == "__main__":
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
    model = BartModel.from_pretrained('facebook/bart-large')

    inputs = tokenizer("hello world!", return_tensors="pt")
    print("inputs:", inputs)

    outputs = model(**inputs)
    print("outputs:", outputs.keys())

    # last_hidden_states = outputs.last_hidden_state
    # print("last_hidden_states:", last_hidden_states)
