import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification


model_path = "D:\\STUDY\\PyCharm\\PyCharm 2025.3.1.1\\Projects\\PythonProject3\\animal_ner_model"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForTokenClassification.from_pretrained(model_path)

label_list = ["O", "ANIMAL"]

def predict_animals(text):
    tokens = text.split()
    inputs = tokenizer(
        tokens,
        is_split_into_words=True,
        return_tensors="pt",
        truncation=True,
        padding=True
    )

    # inference

    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=2)
    predicted_labels = [label_list[p] for p in predictions[0].tolist()]
    word_ids = inputs.word_ids()
    animals = []
    previous_word = None
    
    for idx, word_id in enumerate(word_ids):
        if word_id is None:
            continue
        if word_id != previous_word:
            label = predicted_labels[idx]
            if label == "ANIMAL":
                animals.append(tokens[word_id])
        previous_word = word_id
    return animals


if __name__ == "__main__":
    text = input("Enter text: ")
    animals = predict_animals(text)
    
    if animals:
        print("Animals found:", animals)
    else:
        print("No animals found.")
