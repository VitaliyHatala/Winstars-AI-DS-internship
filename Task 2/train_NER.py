import json
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer

with open("C:\\Users\\user\\Desktop\\animals.json", "r") as file:
    animals = json.load(file)

print(type(animals))
# Hugging Face Trainer не працює зі звичайними списками.
# Вона очікує Dataset об'єкт

dataset = Dataset.from_list(animals)
# print(dataset)

label_list = ["O", "ANIMAL"]
num_labels = len(label_list)

model_checkpoint = "distilbert-base-uncased"
# це назва моделі (скорочена версія "BERT", "distilled" - легша, швидша)
# base - розмір моделі (стандартний)
# uncased - обробка тексту без врахування реєстру (великі/малі літери для моделі однакові)
# Розуміє контекст слів у реченні.
# Вміє перетворювати слова на вектори (embeddings), які комп’ютер може використовувати для задач NLP.

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
# Ми завантажуємо готовий токенізатор, який відповідає саме цій моделі.

def tokenize_and_align_labels(examples):
    tokenized = tokenizer(
        examples['tokens'], # вхідні дані [["I", "love", "cats"]]
        is_split_into_words = True, # це означає, що слова вже поділені на слова, шоб ше раз не ділити
        truncation= True, # 512 токенів). Якщо речення довше, токенізатор обрізає зайве, щоб модель могла його обробити.
        padding= "max_length",
        max_length=128
    ) # перетворює список слів на числа
# [CLS] (сигнал моделі - початок речення) і [SEP] (сигнал моделі - кінець речення)
# Це спеціальні токени, які додаються в моделі типу BERT/DistilBERT під час токенізації.
# sentence = ["I", "love", "cats"]
# tokenized.tokens() -> ["[CLS]", "I", "love"Ф, "cat", "##s", "[SEP]"]

    word_ids = tokenized.word_ids() # це метод токенізатора Hugging Face
    # повертає список індексів слів для кожного токена, який токенізатор створив
    # sentence = ["I", "love", "cats"]
    # tokenized = tokenizer(sentence, is_split_into_words=True)
    # tokenized.tokens() -> ["[CLS]", "I", "love", "cat", "##s", "[SEP]"]
    #
    # word_ids = tokenized.word_ids()
    # print(word_ids)
    # # [None, 0, 1, 2, 2, None]

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

#examples = {
    #"tokens": [["I", "love", "cats"]],
   #"ner_tags": [[0, 0, 1]]  # 1 = Animal
#}
#tokenized.tokens() -> ["[CLS]", "I", "love", "cat", "##s", "[SEP]"]
#word_ids -> [None, 0, 1, 2, 2, None]
#labels = [-100, 0, 0, 1, -100, -100]  # після циклу

tokenized_dataset = dataset.map(tokenize_and_align_labels)

split_dataset = tokenized_dataset.train_test_split(test_size = 0.2)
train_dataset = split_dataset["train"]
test_dataset = split_dataset["test"]

model = AutoModelForTokenClassification.from_pretrained(model_checkpoint, num_labels = num_labels)

training_args = TrainingArguments(
    output_dir = "./ner_model",
    learning_rate = 2e-5, # Крок градієнта під час оновлення ваг
    per_device_train_batch_size = 8, # Скільки прикладів обробляється за один крок на одному GPU/CPU
    per_device_eval_batch_size = 8, # Аналогічно, але для оцінки на валідації
    num_train_epochs = 10, # Кількість проходів по всьому тренувальному датасету
    logging_steps=5, # Як часто виводити інформацію про навчання (loss, accuracy)
    save_strategy = "epoch", # Коли зберігати модель "epoch" → після кожної епохи
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