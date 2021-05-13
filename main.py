import os, re
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from time import time
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch.nn.functional as F
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

LOCAL_MODEL = False
LOCAL_BERT = False
EPOCHS = 10
PARTIAL_SAMPLE = 10000


if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f'There are {torch.cuda.device_count()} GPU(s) available')
    print("Device name: " + torch.cuda.get_device_name(0))
else:
    print("No GPU available, using the CPU instead.")
    device = torch.device("cpu")


original_columns = ['age', 'body type', 'bust size', 'category', 'fit', 'height', 'item_id', 'rating'
                    'rented for', 'review_date', 'review_summary', 'review_text', 'size', 'user_id', 'weight']


def load_train_data(filepath):
    return pd.read_csv(filepath)


def text_preprocessing(text):
    if type(text) is not str:
        return ""
    text = re.sub(r'(@.*?)[\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text

def preprocess_for_bert(text_arr):
    """
    :param text_arr: Array of texts to be processed
    :return: input_ids
             attention_masks
    """
    input_ids = []
    attention_masks = []

    for i, sentence in enumerate(text_arr):
        encoded_sentence = tokenizer.encode_plus(
            text=text_preprocessing(sentence),
            add_special_tokens=True,
            max_length=MAX_LEN,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
        )
        ids = encoded_sentence.get('input_ids')
        input_ids.append(ids)
        atm = encoded_sentence.get('attention_mask')
        attention_masks.append(atm)

    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)
    return input_ids, attention_masks


class BertClassifier(nn.Module):
    # freeze_bert: bool, set "False" to fine-tune the BERT model
    def __init__(self, freeze_bert=False):
        super(BertClassifier, self).__init__()
        D_in, H, D_out = 768, 50, 3
        if not LOCAL_BERT:
            self.bert = BertModel.from_pretrained('bert-base-uncased')
        else:
            self.bert = BertModel.from_pretrained('models/', local_files_only=True)
        self.classifier = nn.Sequential(
            nn.Linear(D_in, H),
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(H, D_out)
        )
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, inputs_ids, attention_mask):
        outputs = self.bert(inputs_ids, attention_mask)
        last_hidden_state_cls = outputs[0][:, 0, :]
        logits = self.classifier(last_hidden_state_cls)
        return logits

# specify the maximum length of our sentences
# all_sentences = np.append(df['review_summary'].tolist(), df['review_text'].tolist())
# encoded_sentences = [tokenizer.encode(sentence, add_special_tokens=True) for sentence in all_sentences]
# max_len = max([len(sentence) for sentence in encoded_sentences])
# print("Max Length: ", max_len)
# MAX_LEN的值是从这边试出来的

# One example
# token_ids = list(preprocess_for_bert([df['review_text'][0]])[0].squeeze().numpy())
# print("Original: ", df['review_text'][0])
# print("Token IDs: ", token_ids)
df = load_train_data("data/train.csv")[:PARTIAL_SAMPLE]

total_sample_num = len(df)
print("Total training sample number: " + str(total_sample_num))

MAX_LEN = 512
loss_fn = nn.CrossEntropyLoss()

TEXT_WEIGHT = 0.9
SUMMARY_WEIGHT = 1 - TEXT_WEIGHT

if not LOCAL_BERT:
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
else:
    tokenizer = BertTokenizer.from_pretrained('models/', local_files_only=True)

# FIXME: ignore summary, temporarily
review_texts = df['review_text']
fit_label = df['fit']

fit_label_embed = [1 if label == 'fit' else 0 for label in fit_label]
positive_num = sum(fit_label_embed)
negative_num = len(fit_label_embed)-positive_num
print("Positive fit label number: " + str(positive_num))
print("Negative fit label number: " + str(negative_num))

X_train, X_test, y_train, y_test = train_test_split(review_texts, fit_label, test_size=0.2, random_state=2021)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=2021)
train_inputs, train_masks = preprocess_for_bert(X_train)
val_inputs, val_masks = preprocess_for_bert(X_val)
test_inputs, test_masks = preprocess_for_bert(X_test)

def label_embedding(y):
    y_embed = []
    for lab in y:
        if lab == 'fit':
            y_embed.append(1)
        elif lab == 'small':
            y_embed.append(0)
        elif lab == 'large':
            y_embed.append(2)
    return y_embed

y_train_embed = label_embedding(y_train)
y_val_embed = label_embedding(y_val)
y_test_embed = label_embedding(y_test)

train_labels = torch.tensor(y_train_embed)
val_labels = torch.tensor(y_val_embed)
test_labels = torch.tensor(y_test_embed)

batch_size = 32

train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

val_data = TensorDataset(val_inputs, val_masks, val_labels)
val_sampler = SequentialSampler(val_data)
val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)

test_data = TensorDataset(test_inputs, test_masks, test_labels)
test_sampler = SequentialSampler(test_data)
test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

def initialize_model(epochs=4, local_model=False):
    if local_model:
        bert_classifier = torch.load("bert-classifier.pkl")
    else:
        bert_classifier = BertClassifier(freeze_bert=False)
    bert_classifier.to(device)
    optimizer = AdamW(bert_classifier.parameters(), lr=5e-5, eps=1e-8)
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    return bert_classifier, optimizer, scheduler

def set_seed(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)

def train(model, train_dataloader, val_dataloader=None, epochs=4, evaluation=False):
    for epoch_i in range(epochs):
        print("Epoch " + str(epoch_i) + " is running...")
        t0_epoch, t0_batch = time(), time()
        total_loss, batch_loss, batch_counts = 0, 0, 0
        model.train()

        for step, batch in enumerate(train_dataloader):
            batch_counts += 1
            b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)
            model.zero_grad()
            logits = model(b_input_ids, b_attn_mask)

            loss = loss_fn(logits, b_labels)
            batch_loss += loss.item()
            total_loss += loss.item()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()

            if (step % 20 == 0 and step != 0) or (step == len(train_dataloader)-1):
                time_elapsed = time() - t0_batch

                print("Batch Loss: ", batch_loss)
                print("Total Loss: ", total_loss)
                batch_loss, batch_counts = 0, 0
                t0_batch = time()
        avg_train_loss = total_loss / len(train_dataloader)
        if evaluation == True:
            print("Evaluation: ")
            val_loss, val_accuracy = evaluate(model, val_dataloader)
            time_elapsed = time()-t0_epoch
            print("val_loss: ", val_loss)
            print("val_accuracy: ", val_accuracy)
    print("Training Complete.")

def evaluate(model, val_dataloader):
    model.eval()
    val_accuracy = []
    val_loss = []

    for batch in val_dataloader:
        b_inputs_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            logits = model(b_inputs_ids, b_attn_mask)
        loss = loss_fn(logits, b_labels)
        val_loss.append(loss.item())
        preds = torch.argmax(logits, dim=1).flatten()
        accuracy = (preds == b_labels).cpu().numpy().mean() * 100
        val_accuracy.append(accuracy)
    val_loss = np.mean(val_loss)
    val_accuracy = np.mean(val_accuracy)
    return val_loss, val_accuracy

def bert_predict(model, test_dataloader):
    model.eval()
    all_logits = []
    for batch in test_dataloader:
        b_input_ids, b_attn_mask = tuple(t.to(device) for t in batch)[:2]
        with torch.no_grad():
            logits = model(b_input_ids, b_attn_mask)
        all_logits.append(logits)
    all_logits = torch.cat(all_logits, dim=0)
    probs = F.softmax(all_logits, dim=1).cpu()
    return probs

set_seed(42)
if not LOCAL_MODEL:
    bert_classifier, optimizer, scheduler = initialize_model(epochs=EPOCHS, local_model=LOCAL_MODEL)
    train(bert_classifier, train_dataloader, val_dataloader, epochs=EPOCHS, evaluation=True)
    torch.save(bert_classifier, "bert-classifier.pkl")
else:
    bert_classifier, optimizer, scheduler = initialize_model(epochs=EPOCHS, local_model=LOCAL_MODEL)

probs = bert_predict(bert_classifier, test_dataloader)
y_pred = torch.argmax(probs, dim=1).numpy()
y_true = np.array(y_test_embed)

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred, average='macro')

print("Accuracy: ", accuracy)
print("Precision: ", precision)
print("Recall: ", recall)
print("macro-F1: ", f1)


# all_sentences = np.append(df['review_summary'], df['review_text'])
# sentence_inputs, sentence_masks = preprocess_for_bert(all_sentences)
#
# summary_inputs = sentence_inputs[:len(df['review_summary'])]
# text_inputs = sentence_inputs[len(df['review_summary']):]
# sample_inputs = []
# for i in range(len(summary_inputs)):
#     sample_inputs.append(summary_inputs[i]*SUMMARY_WEIGHT + text_inputs*TEXT_WEIGHT)
#
# summary_mask = sentence_masks[:len(df['review_summary'])]
# text_mask = sentence_masks[len(df['review_summary']):]
# sample_masks = []
# for i in range(len(summary_mask)):
#     sample_masks.append(summary_mask[i]*SUMMARY_WEIGHT + text_inputs*TEXT_WEIGHT)
#
# sample_inputs


