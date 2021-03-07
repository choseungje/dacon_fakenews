import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
from transformers import ElectraModel, ElectraTokenizer
import pandas as pd
from torch.utils.data import Dataset
import numpy as np
import electra_custom_padding
from tqdm import tqdm
import time

# ------------------------------------------------------------------------------------------
# GPU 사용 시
device = torch.device("cuda:1")
print("device : {}\n".format(device))

electra_model = ElectraModel.from_pretrained("monologg/koelectra-base-v3-discriminator")
tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")
print(tokenizer.convert_tokens_to_ids(tokenizer.tokenize('[PAD]')))
print(tokenizer.convert_ids_to_tokens(0))
print("vocab_size : {}\n" .format(tokenizer.vocab_size))
print(electra_model)
# ------------------------------------------------------------------------------------------
# data load

default_path = '.'

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
    title = "[CLS] " + title + " [SEP]"
    content = "[CLS] " + content + " [SEP]"

    temp = [i, [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(title)),
                tokenizer.convert_tokens_to_ids(tokenizer.tokenize(content))], info]
    dataset.append(temp)

for i, (title, content) in enumerate(zip(test_csv['title'], test_csv['content'])):
    title = "[CLS] " + title + " [SEP]"
    content = "[CLS] " + content + " [SEP]"
    temp = [i, [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(title)),
                tokenizer.convert_tokens_to_ids(tokenizer.tokenize(content))], 0]
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
    def __init__(self, dataset, index_idx, sent_idx, label_idx, pad_length, pad=True, pair=True):
        self.index = [np.int32(i[index_idx]) for i in dataset]
        self.sentences = [i[sent_idx] for i in dataset]

        for e in range(len(self.sentences)):
            self.sentences[e] = electra_custom_padding.electra_padding(
                inputs=self.sentences[e], pad_token=tokenizer.convert_tokens_to_ids('[PAD]'),
                pad_length=pad_length, pad=pad, pair=pair)

        self.labels = [np.int32(i[label_idx]) for i in dataset]

    def __getitem__(self, i):
        return (self.index[i],) + self.sentences[i] + (self.labels[i],)

    def __len__(self):
        return len(self.labels)


# Setting parameters
batch_size = 200
seq_len = 64
warmup_ratio = 0.1
num_epochs = 2
max_grad_norm = 1
log_interval = 2500
learning_rate = 1e-4
checkpoint_path = "electra_checkpoint.pt"

data_train = BERTDataset(dataset_train, 0, 1, 2, seq_len, True, True)
data_val = BERTDataset(dataset_val, 0, 1, 2, seq_len, True, True)
data_test = BERTDataset(dataset_test, 0, 1, 2, seq_len, True, True)
print("Loading DataSet")

train_dataloader = torch.utils.data.DataLoader(data_train, batch_size=batch_size)
val_dataloader = torch.utils.data.DataLoader(data_val, batch_size=1)
test_dataloader = torch.utils.data.DataLoader(data_test, batch_size=batch_size)
print("Loading Data Loader\n")


class ElectraBertNSP(nn.Module):
    def __init__(self,
                 electra_bert,
                 hidden_size=768,
                 num_classes=2,
                 dr_rate=None,
                 params=None):
        super(ElectraBertNSP, self).__init__()
        self.electra_bert = electra_bert
        self.pooler_dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()
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

        output = self.electra_bert(input_ids=token_ids, token_type_ids=segment_ids.long(), attention_mask=attention_mask.float().to(token_ids.device))

        first_token_tensor = output[0]
        pooled_output = self.pooler_dense(first_token_tensor[:, 0])
        pooled_output = self.activation(pooled_output)

        if self.dr_rate:
            pooled_output = self.dropout(pooled_output)
        return self.classifier(pooled_output)


model = ElectraBertNSP(electra_model, dr_rate=0.1).to(device)
for e, (n, p) in enumerate(model.named_parameters()):
    print(e, n)

# Prepare optimizer and schedule (linear warmup and decay)
no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    # {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]

# optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=learning_rate, betas=(0.9, 0.999), eps=1e-6)
loss_fn = nn.CrossEntropyLoss()

t_total = len(train_dataloader) * num_epochs
warmup_step = int(t_total * warmup_ratio)


class WarmupLinearSchedule(LambdaLR):
    """ Linear warmup and then linear decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        Linearly decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps.
    """
    def __init__(self, optimizer, warmup_steps, t_total, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        super(WarmupLinearSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        return max(0.0, float(self.t_total - step) / float(max(1.0, self.t_total - self.warmup_steps)))


scheduler = WarmupLinearSchedule(optimizer, warmup_steps=warmup_step, t_total=t_total)


def save_torch(epoch, model, optimizer, scheduler, filename):
    torch.save(model, 'electra_model.pt')
    state = {
        'Epoch': epoch + 1,
        'State_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr': scheduler.state_dict()
    }
    torch.save(state, filename)


def calc_accuracy(X, Y):
    max_vals, max_indices = torch.max(X, 1)
    train_acc = (max_indices == Y).sum().data.cpu().numpy() / max_indices.size()[0]
    return train_acc


v_loss = []
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
    v_loss.append(val_loss / (batch + 1))
    print("-" * 80)
    print("| epoch {:3d} | val loss {:3f} | val acc {:5.6f} | time {:.1f} secs |".format(
        e + 1, val_loss / (batch + 1), val_acc / (batch + 1), time.time() - start_time))
    print("-" * 80)
    print()

import matplotlib.pyplot as plt
plt.plot([i + 1 for i in range(num_epochs)], v_loss, color='lightcoral', label='validation_loss')
plt.xticks(range(1, num_epochs + 1, 2))
plt.title('electra_bert')
plt.xlabel('Epochs')
plt.ylabel('Loss / Accuracy')
plt.legend()
plt.grid(True)
plt.savefig('electra_bert.png')

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
            if tokenizer.convert_ids_to_tokens(int(i)) != '[PAD]':
                temp += tokenizer.convert_ids_to_tokens(int(i))
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
sample_submission_csv.to_csv('.\\electra_test.csv', index=False)
print("Save Result")
