# fakenews3test -> epoch 2
import zipfile
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from tqdm import tqdm
from transformers import WarmupLinearSchedule
import numpy as np
import os
from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model
import gluonnlp as nlp
from torch.utils.data import Dataset
import time
import pandas as pd

# ------------------------------------------------------------------------------------------
# GPU 사용 시
device = torch.device("cuda:1")
print("device : {}\n".format(device))

bertmodel, vocab = get_pytorch_kobert_model()
print("vocab.token_to_idx['[PAD]']", vocab.token_to_idx['[PAD]'])
print("vocab.token_to_idx['.']", vocab.token_to_idx['.'])
print("vocab.token_to_idx['CLS']", vocab.token_to_idx['[CLS]'])
print("vocab.token_to_idx['SEP']", vocab.token_to_idx['[SEP]'])
print()
# print(bertmodel)

tokenizer = get_tokenizer()
tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)

# ------------------------------------------------------------------------------------------
# data load

default_path = '.'
try:
    with zipfile.ZipFile(default_path + "/리그1.zip") as zf:
        zf.extractall(default_path)
        print("uncompress success\n")
except:
    print("uncompress fail\n")

train_csv = pd.read_csv(default_path + '\\open\\news_train.csv')  # train에 필요한 csv파일 경로
test_csv = pd.read_csv(default_path + '\\open\\news_test.csv')  # test에 필요한 csv파일 경로
sample_submission_csv = pd.read_csv(default_path + '\\open\\sample_submission.csv')  # test에 필요한 csv파일 경로
print(train_csv)
print("train_csv.keys()", train_csv.keys())
print("test_csv.keys()", test_csv.keys())
print("len(train_csv)", len(train_csv))
print("len(test_csv)", len(test_csv))
print()

dataset = []
dataset_test = []
for i, (title, content, info) in enumerate(zip(train_csv['title'], train_csv['content'], train_csv['info'])):
    temp = [i, [title, content], info]
    dataset.append(temp)
for i, (title, content) in enumerate(zip(test_csv['title'], test_csv['content'])):
    temp = [i, [title, content], 0]
    dataset_test.append(temp)

# ------------------------------------------------------------------------------------------
# data split
dataset_train, dataset_val = train_test_split(
    dataset, test_size=0.02, shuffle=True, random_state=45)

print("len dataset_train", len(dataset_train))
print("len dataset_val", len(dataset_val))
print("len dataset_test", len(dataset_test))
print()


# ------------------------------------------------------------------------------------------
# torch dataset
class BERTDataset(Dataset):
    def __init__(self, dataset, index_idx, sent_idx, label_idx, bert_tokenizer, max_len,
                 pad, pair):
        transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=max_len, pad=pad, pair=pair)
        self.index = [np.int32(i[index_idx]) for i in dataset]
        self.sentences = [transform(i[sent_idx]) for i in dataset]
        self.labels = [np.int32(i[label_idx]) for i in dataset]

    def __getitem__(self, i):
        return (self.index[i],) + self.sentences[i] + (self.labels[i],)

    def __len__(self):
        return len(self.labels)


# Setting parameters
max_len = 64
batch_size = 128
warmup_ratio = 0.1
num_epochs = 1
max_grad_norm = 1
log_interval = 2500
learning_rate = 3e-5
checkpoint_path = 'fakenews1test7_checkpoint.pt'

data_train = BERTDataset(dataset_train, 0, 1, 2, tok, max_len, True, True)
data_val = BERTDataset(dataset_val, 0, 1, 2, tok, max_len, True, True)
data_test = BERTDataset(dataset_test, 0, 1, 2, tok, max_len, True, True)
print("Loading DataSet")

train_dataloader = torch.utils.data.DataLoader(data_train, batch_size=batch_size)
val_dataloader = torch.utils.data.DataLoader(data_val, batch_size=1)
test_dataloader = torch.utils.data.DataLoader(data_test, batch_size=batch_size)
print("Loading Data Loader\n")


class BERTClassifier(nn.Module):
    def __init__(self,
                 bert,
                 hidden_size=768,
                 num_classes=2,
                 dr_rate=None,
                 params=None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate

        self.classifier = nn.Linear(hidden_size, num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)

    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)

        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)

        _, pooler = self.bert(input_ids=token_ids, token_type_ids=segment_ids.long(), attention_mask=attention_mask.float().to(token_ids.device))
        return self.classifier(pooler)


def save_torch(epoch, model, optimizer, scheduler, filename):
    torch.save(model, 'fakenews1test7_model.pt')
    state = {
        'Epoch': epoch + 1,
        'State_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr': scheduler.state_dict()
    }
    torch.save(state, filename)


model = BERTClassifier(bertmodel, dr_rate=0.5).to(device)

# Prepare optimizer and schedule (linear warmup and decay)
no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]

# optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=learning_rate, betas=(0.9, 0.999), eps=1e-9)
loss_fn = nn.CrossEntropyLoss()

t_total = len(train_dataloader) * num_epochs
warmup_step = int(t_total * warmup_ratio)

scheduler = WarmupLinearSchedule(optimizer, warmup_steps=warmup_step, t_total=t_total)


if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    print("Loading Checkpoint")
    print("checkpoint keys : {}\n".format(checkpoint.keys()))
    model.load_state_dict(checkpoint['State_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['lr'])


def calc_accuracy(X, Y):
    max_vals, max_indices = torch.max(X, 1)
    train_acc = (max_indices == Y).sum().data.cpu().numpy() / max_indices.size()[0]
    return train_acc


for e in range(num_epochs):
    start_time = time.time()
    train_acc = 0.0
    val_acc = 0.0
    train_loss = 0.0
    val_loss = 0.0
    model.train()
    for batch, (index_ids, token_ids, valid_length, segment_ids, label) in enumerate(tqdm(train_dataloader)):
        optimizer.zero_grad()
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)
        valid_length = valid_length
        label = label.long().to(device)
        out = model(token_ids, valid_length, segment_ids)

        loss = loss_fn(out, label)
        train_loss += loss.data.cpu().numpy()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        optimizer.step()
        scheduler.step()  # Update learning rate schedule
        train_acc += calc_accuracy(out, label)
        if batch % log_interval == 0:
            print("\n| epoch {:3d} | batch {:3d} | loss {:6f} | train acc {:6f} | lr {:.10f} |".format(
                e + 1, batch + 1, loss.data.cpu().numpy(), train_acc / (batch + 1), scheduler.get_lr()[0]))
    print("\nepoch {:2d} train acc {:6f}".format(e + 1, train_acc / (batch + 1)))

    print("-" * 80)
    print("| epoch {:3d} | train loss {:3f} | train acc {:5.6f} | time {:.1f} secs |".format(
        e + 1, train_loss / (batch + 1), train_acc / (batch + 1), time.time() - start_time))
    print("-" * 80)

    save_torch(e, model, optimizer, scheduler, checkpoint_path)

    model.eval()
    for batch, (index_ids, token_ids, valid_length, segment_ids, label) in enumerate(tqdm(val_dataloader)):
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)
        valid_length = valid_length
        label = label.long().to(device)
        out = model(token_ids, valid_length, segment_ids)

        loss = loss_fn(out, label)
        val_loss += loss.data.cpu().numpy()

        val_acc += calc_accuracy(out, label)
    print("-" * 80)
    print("| epoch {:3d} | val loss {:3f} | val acc {:5.6f} | time {:.1f} secs |".format(
        e + 1, val_loss / (batch + 1), val_acc / (batch + 1), time.time() - start_time))
    print("-" * 80)

count = 0
model.eval()
# for batch, (index_ids, token_ids, valid_length, segment_ids, label) in enumerate(tqdm(val_dataloader)):
for batch, (index_ids, token_ids, valid_length, segment_ids, label) in enumerate(val_dataloader):
    token_ids = token_ids.long().to(device)
    segment_ids = segment_ids.long().to(device)
    valid_length = valid_length
    label = label.long().to(device)
    out = model(token_ids, valid_length, segment_ids)

    if torch.argmax(out, dim=-1).data.cpu().numpy() != label.data.cpu().numpy():
        temp = ""
        for i in np.reshape(token_ids.data.cpu().numpy(), (-1)):
            if vocab.idx_to_token[i] != '[PAD]':
                temp += vocab.idx_to_token[i]
        print("index_ids : {}".format(index_ids))
        print("정답 : {}, 예측 : {}, 문장 : {}".format(label.data.cpu().numpy(), torch.argmax(out, dim=-1).data.cpu().numpy(), temp))
        count += 1
print("count", count)
model.eval()
test_result = []
for batch, (index_ids, token_ids, valid_length, segment_ids, label) in enumerate(tqdm(test_dataloader)):
    token_ids = token_ids.long().to(device)
    segment_ids = segment_ids.long().to(device)
    valid_length = valid_length
    label = label.long().to(device)
    out = model(token_ids, valid_length, segment_ids)

    test_result.extend(torch.argmax(out, dim=-1).data.cpu().numpy())

print("len test_result", len(test_result))
test_result = np.array(test_result)
print("End Test")

sample_submission_csv['info'] = test_result
sample_submission_csv.to_csv('.\\result1test7.csv', index=False)
print("Save Result")
