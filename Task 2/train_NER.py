import json
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer


with open("C:\\Users\\user\\Desktop\\animals.json", "r") as file:
    animals = json.load(file)

dataset = Dataset.from_list(animals)


label_list = ["O", "ANIMAL"]
num_labels = len(label_list)

model_checkpoint = "distilbert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)


def tokenize_and_align_labels(examples):
    tokenized = tokenizer(
        examples['tokens'], 
        is_split_into_words = True,
        truncation= True, 
        padding= "max_length",
        max_length=128
    ) 
    
    word_ids = tokenized.word_ids() 

    labels = []
    previous_word = None

    for word in word_ids:
        if word is None:
            labels.append(-100)
        elif word != previous_word:
            labels.append(examples["ner_tags"][word])
        else:
            labels.append(-100)
        previous_word = word
    tokenized['labels'] = labels
    return tokenized


tokenized_dataset = dataset.map(tokenize_and_align_labels)

split_dataset = tokenized_dataset.train_test_split(test_size = 0.2)
train_dataset = split_dataset["train"]
test_dataset = split_dataset["test"]

model = AutoModelForTokenClassification.from_pretrained(model_checkpoint, num_labels = num_labels)

training_args = TrainingArguments(
    output_dir = "./ner_model",
    learning_rate = 2e-5, 
    per_device_train_batch_size = 8, 
    per_device_eval_batch_size = 8, 
    num_train_epochs = 10, 
    logging_steps=5,
    save_strategy = "epoch",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

trainer.train()

trainer.save_model("animal_ner_model")
tokenizer.save_pretrained("animal_ner_model")


print("Training finished. Model saved to animal_ner_model/")
