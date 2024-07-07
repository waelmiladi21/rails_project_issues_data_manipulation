import requests
from github import Github
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.model_selection import train_test_split
import numpy as np
import torch
from sklearn import preprocessing
from evaluate import load
import random
from nltk.corpus import wordnet, stopwords
import nltk

nltk.download('wordnet')
nltk.download('stopwords')

access_token = ""
g = Github(access_token)
repo = g.get_repo("rails/rails")

# Data collection
issues = repo.get_issues(state='all', sort='created', direction='desc')[:500]

data = []
for issue in issues:
    data.append({
        'issue_number': issue.number,
        'created_at': issue.created_at,
        'labels': [label.name for label in issue.labels],
        'comments': issue.comments,
        'author': issue.user.login,
        'description': issue.body
    })

# Create DataFrame and analyze
df = pd.DataFrame(data)
df.to_csv('collected_data.csv', index=False)

# Issue Evolution Over Time
df['created_at'] = pd.to_datetime(df['created_at'])
df.set_index('created_at', inplace=True)

plt.figure(figsize=(12, 6))
df.resample('D')['issue_number'].count().plot()
plt.title('Number of Issues Over Time (Daily)')
plt.ylabel('Number of Issues')
plt.show()

# Popular Issue Reporters
top_reporters = df['author'].value_counts().head(10)
plt.figure(figsize=(10, 5))
sns.barplot(x=top_reporters.index, y=top_reporters.values)
plt.title('Top 10 Issue Reporters')
plt.xlabel('Reporter')
plt.ylabel('Number of Issues')
plt.xticks(rotation=45)
plt.show()

# Popular Labels
all_labels = [label for sublist in df['labels'] for label in sublist]
label_counts = pd.Series(all_labels).value_counts().head(10)
plt.figure(figsize=(10, 5))
sns.barplot(x=label_counts.index, y=label_counts.values)
plt.title('Top 10 Labels')
plt.xlabel('Label')
plt.ylabel('Number of Issues')
plt.xticks(rotation=45)
plt.show()

# Data Preparation
top_labels = label_counts.index.tolist()[:10]
df_filtered = df[df['labels'].apply(lambda x: any(l in top_labels for l in x))].copy()  
df_filtered['labels'] = df_filtered['labels'].apply(lambda x: [l for l in x if l in top_labels])
df_filtered['labels'] = df_filtered['labels'].apply(lambda x: x[0] if len(x) > 0 else None) 
df_filtered.dropna(subset=['labels'], inplace=True)

# Data Augmentation
stop_words = set(stopwords.words('english'))

def synonym_replacement(text, n):
    words = text.split()
    new_words = words.copy()
    random_word_list = list(set([word for word in words if word not in stop_words]))
    random.shuffle(random_word_list)
    num_replaced = 0
    for random_word in random_word_list:
        synonyms = get_synonyms(random_word)
        if len(synonyms) >= 1:
            synonym = random.choice(list(synonyms))
            new_words = [synonym if word == random_word else word for word in new_words]
            num_replaced += 1
        if num_replaced >= n:
            break

    sentence = ' '.join(new_words)
    return sentence

def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name())
    if word in synonyms:
        synonyms.remove(word)
    return list(synonyms)

def random_insertion(text, n):
    words = text.split()
    new_words = words.copy()
    for _ in range(n):
        add_word(new_words)
    return ' '.join(new_words)

def add_word(new_words):
    synonyms = []
    counter = 0
    while len(synonyms) < 1:
        random_word = new_words[random.randint(0, len(new_words)-1)]
        synonyms = get_synonyms(random_word)
        counter += 1
        if counter >= 10:
            return
    random_synonym = synonyms[0]
    random_idx = random.randint(0, len(new_words)-1)
    new_words.insert(random_idx, random_synonym)

def random_swap(text, n):
    words = text.split()
    new_words = words.copy()
    for _ in range(n):
        new_words = swap_word(new_words)
    return ' '.join(new_words)

def swap_word(new_words):
    random_idx_1 = random.randint(0, len(new_words)-1)
    random_idx_2 = random_idx_1
    counter = 0
    while random_idx_2 == random_idx_1:
        random_idx_2 = random.randint(0, len(new_words)-1)
        counter += 1
        if counter > 3:
            return new_words
    new_words[random_idx_1], new_words[random_idx_2] = new_words[random_idx_2], new_words[random_idx_1]
    return new_words

def random_deletion(text, p):
    words = text.split()
    if len(words) == 1:
        return words

    new_words = []
    for word in words:
        r = random.uniform(0, 1)
        if r > p:
            new_words.append(word)

    if len(new_words) == 0:
        rand_int = random.randint(0, len(words)-1)
        return [words[rand_int]]

    return ' '.join(new_words)

# Applying augmentation on the text data
augmented_texts = []
original_texts = []
for text in df_filtered['description']:
    if pd.notnull(text):
        original_texts.append(text)
        augmented_texts.append(synonym_replacement(text, 2))
        augmented_texts.append(random_insertion(text, 2))
        augmented_texts.append(random_swap(text, 2))
        augmented_texts.append(random_deletion(text, 0.2))

# Combine original and augmented data
total_augmented = len(augmented_texts)
repeated_labels = df_filtered['labels'].repeat(4).values[:total_augmented]

df_augmented = pd.DataFrame({'description': augmented_texts, 'labels': repeated_labels})
df_combined = pd.concat([df_filtered[['description', 'labels']], df_augmented])

# Split dataset
train_df, test_df = train_test_split(df_combined, test_size=0.2, random_state=42)

# Encode labels
le = preprocessing.LabelEncoder()
le.fit(top_labels)
train_df['labels'] = le.transform(train_df['labels'])
test_df['labels'] = le.transform(test_df['labels'])

# Define model and tokenizer
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(top_labels), 
    id2label={i: label for i, label in enumerate(top_labels)},
    label2id={label: i for i, label in enumerate(top_labels)},
    ignore_mismatched_sizes=True   
)

# Prepare datasets
train_encodings = tokenizer(
    [desc if isinstance(desc, str) else "" for desc in train_df['description']], 
    truncation=True, padding='max_length'
)

test_encodings = tokenizer(
    [desc if isinstance(desc, str) else "" for desc in test_df['description']], 
    truncation=True, padding='max_length'
)

train_dataset = Dataset.from_dict({'input_ids': train_encodings['input_ids'],
                                 'attention_mask': train_encodings['attention_mask'],
                                 'labels': train_df['labels'].tolist()})

test_dataset = Dataset.from_dict({'input_ids': test_encodings['input_ids'],
                                'attention_mask': test_encodings['attention_mask'],
                                'labels': test_df['labels'].tolist()})

train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

# Define the metric calculation function
def compute_metrics(eval_pred):
    metric = load("accuracy", trust_remote_code=True) 
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=30,
    weight_decay=0.01,
)

# Train the model
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

trainer.train()

# Evaluation
eval_result = trainer.evaluate(test_dataset, metric_key_prefix="test")
print(eval_result)

# Assuming you have a sample issue description:
new_issue = "Error when trying to create a new record in the database"

# Tokenize the new issue
tokenized_issue = tokenizer(new_issue, truncation=True, padding='max_length', return_tensors='pt')

# Move the tokenized issue to the same device as the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenized_issue = tokenized_issue.to(device)
model.to(device)

# Get model prediction
with torch.no_grad():
    logits = model(**tokenized_issue).logits
    predicted_label_index = logits.argmax().item()

predicted_label = model.config.id2label[predicted_label_index]
print("Predicted label:", predicted_label)