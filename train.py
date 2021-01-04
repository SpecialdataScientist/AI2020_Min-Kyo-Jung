import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import pickle
from TorchCRF import CRF
from utils import batchify, evaluate_ner_F1, evaluate_ner_F1_and_write_result
import os
from layers import MultiHeadAttention


class LabelAttention(nn.Module):
    def __init__(self, vocabs, word_dim, pos_dim, hidden_size, rnn_layers, dropout_rate, device,
                 bidirectional=True, use_crf=False, embedding=None):
        super(LabelAttention, self).__init__()

        word2id, tag2id, label2id = vocabs      # vocab == wor2id, tag2id, label2id

        output_size = hidden_size * 2 if bidirectional else hidden_size     # because bidirectional

        # word embedding set
        self.word_embeddings = nn.Embedding(len(word2id), word_dim) # dimension == 100

        # parameter embedding is preprocessing(use pretrained or not use pretrained)
        # parameter copy to local variable
        if embedding is not None:
            self.word_embeddings.weight.data.copy_(torch.from_numpy(embedding))

        # preprocessing not embedding tag and label
        self.tag_embeddings = nn.Embedding(len(tag2id), pos_dim)        # tag embedding

        # this is no labelAttention difference
        self.label_embeddings = nn.Embedding(len(label2id), output_size)

        # lstm set
        # word_dim + pos_dom == 150
        self.lstm1 = nn.LSTM(word_dim + pos_dim, hidden_size, 1,
                            batch_first=True, bidirectional=bidirectional, dropout=dropout_rate)

        self.label_attn1 = MultiHeadAttention(input_size=output_size, hidden_size=hidden_size, n_head=8,
                                              dropout=dropout_rate, device=device)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, 1,
                            batch_first=True, bidirectional=bidirectional, dropout=dropout_rate)

        self.label_attn2 = MultiHeadAttention(input_size=output_size, hidden_size=hidden_size, n_head=1,
                                              dropout=dropout_rate, device=device)

        # bidirectional is ouput size * 2
        # no bidirectional is ouput size * 1

        # output size
        self.linear = nn.Linear(output_size, len(label2id))

        # drop out set
        self.dropout_rate = dropout_rate

        # using crf
        self.use_crf = use_crf
        if use_crf:
            self.crf = CRF(len(label2id), batch_first=True)     # parameter: label index

        # loss function: cross entroyp
        self.cross_entropy = nn.CrossEntropyLoss(reduction='none')

        # label total size
        self.label_size = len(label2id)
        self.device = device

    # forward function
    def forward(self, word_ids, tag_ids, label_ids):
        batch_size, seq_len = word_ids.size()

        # embedding set
        word_emb = self.word_embeddings(word_ids)
        tag_emb = self.tag_embeddings(tag_ids)

        # batch size * label_size
        labels = torch.arange(self.label_size).unsqueeze(0).repeat(batch_size, 1).long().to(self.device)

        # batch size label to embedding
        label_emb = self.label_embeddings(labels)

        # cat word_emb, tag_emb (column)
        # [word_emd, tag_emb]
        rnn_input = torch.cat([word_emb, tag_emb], dim=-1)

        rnn_input = F.dropout(rnn_input, self.dropout_rate, self.training)

        # lstm1 parameter: [word_emd, tag_emd]
        rnn_outputs, (hn, cn) = self.lstm1(rnn_input)

        # label attention
        # parameter: rnn_output, label_emb => q, k, v
        # return: attn_value, attn_score(logits use)
        # parallel
        outputs, _ = self.label_attn1(rnn_outputs, label_emb, label_emb)

        # lstm2 parameter: attn_value(word_info, tag_info), attn_score
        rnn_outputs, (hn, cn) = self.lstm2(outputs)

        # return: attn_value, attn_score
        outputs, logits = self.label_attn2(rnn_outputs, label_emb, label_emb)

        logits = logits.squeeze(1)

        # ouput size set
        # logits = self.linear(rnn_outputs)

        # [1, 1, 1, 0, 0]
        # [1, 1, 1, 1, 1]
        mask = word_ids.ne(0)
        if self.training:  # training
            if self.use_crf:
                loss = -self.crf(logits, label_ids, mask=mask.byte())
                return loss

            # not crf
            # logit is output size
            else:
                batch, seq_len, num_label = logits.size()

                logits = logits.view(-1, logits.data.shape[-1])
                label_ids = label_ids.view(-1)

                loss = F.cross_entropy(logits, label_ids, reduction='none')
                loss = loss.view(batch, seq_len)

                loss = loss * mask.float()

                num_tokens = mask.sum(1).sum(0)

                loss = loss.sum(1).sum(0) / num_tokens
                return loss

        label_ids = label_ids.data.cpu().numpy().tolist()
        lengths = mask.sum(1).long().tolist()

        answers = []
        for answer, length in zip(label_ids, lengths):
            answers.append(answer[:length])

        if self.use_crf:
            predictions = self.crf.decode(logits, mask)

            return answers, predictions

        batch_preds = torch.argmax(logits, dim=-1)
        batch_preds = batch_preds.data.cpu().numpy().tolist()

        predictions = []
        for pred, length in zip(batch_preds, lengths):
            predictions.append(pred[:length])

        return answers, predictions

class Model(nn.Module):
    def __init__(self, vocabs, word_dim, pos_dim, hidden_size, rnn_layers, dropout_rate, device,
                 bidirectional=True, use_crf=False, embedding=None):
        super(Model, self).__init__()

        word2id, tag2id, label2id = vocabs

        # word embedding set
        self.word_embeddings = nn.Embedding(len(word2id), word_dim)


        if embedding is not None:
            self.word_embeddings.weight.data.copy_(torch.from_numpy(embedding))

        self.tag_embeddings = nn.Embedding(len(tag2id), pos_dim)

        # lstm set
        self.lstm = nn.LSTM(word_dim + pos_dim, hidden_size, rnn_layers,
                            batch_first=True, bidirectional=bidirectional, dropout=dropout_rate)

        # bidirectional is ouput size * 2
        # no bidirectional is ouput size * 1
        output_size = hidden_size * 2 if bidirectional else hidden_size

        # output size
        self.linear = nn.Linear(output_size, len(label2id))

        self.dropout_rate = dropout_rate

        self.use_crf = use_crf
        if use_crf:
            self.crf = CRF(len(label2id), batch_first=True)

        self.cross_entropy = nn.CrossEntropyLoss(reduction='none')

    # forward function
    def forward(self, word_ids, tag_ids, label_ids):
        # embedding set
        word_emb = self.word_embeddings(word_ids)
        tag_emb = self.tag_embeddings(tag_ids)

        rnn_input = torch.cat([word_emb, tag_emb], dim=-1)

        rnn_input = F.dropout(rnn_input, self.dropout_rate, self.training)

        rnn_outputs, (hn, cn) = self.lstm(rnn_input)

        # ouput size set
        logits = self.linear(rnn_outputs)

        # [1, 1, 1, 0, 0]
        # [1, 1, 1, 1, 1]
        mask = word_ids.ne(0)
        if self.training:  # training
            if self.use_crf:
                loss = -self.crf(logits, label_ids, mask=mask.byte())
                return loss

            else:
                batch, seq_len, num_label = logits.size()

                logits = logits.view(-1, logits.data.shape[-1])
                label_ids = label_ids.view(-1)

                loss = F.cross_entropy(logits, label_ids, reduction='none')
                loss = loss.view(batch, seq_len)

                loss = loss * mask.float()

                num_tokens = mask.sum(1).sum(0)

                loss = loss.sum(1).sum(0) / num_tokens
                return loss

        label_ids = label_ids.data.cpu().numpy().tolist()
        lengths = mask.sum(1).long().tolist()

        answers = []
        for answer, length in zip(label_ids, lengths):
            answers.append(answer[:length])

        if self.use_crf:
            predictions = self.crf.decode(logits, mask)

            return answers, predictions

        batch_preds = torch.argmax(logits, dim=-1)
        batch_preds = batch_preds.data.cpu().numpy().tolist()

        predictions = []
        for pred, length in zip(batch_preds, lengths):
            predictions.append(pred[:length])

        return answers, predictions

# train function
def train(epochs=30, batch_size=32,
          word_dim=100, pos_dim=50, hidden_size=300, rnn_layers=1, bidirectional=False,
          use_pretrained=False, dropout_rate=0.0, use_crf=False, evaluate=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # preprocessing data file open
    with open("data.pkl", "rb") as f:
        data = pickle.load(f)
    train, dev, test = data['train'], data['dev'], data['test']     # train, dev, test set
    word2id, tag2id, label2id, embedding = data['w2id'], data['t2id'], data['l2id'], data['embedding']  #set

    id2label = {i: l for l, i in label2id.items()}
    # model set
    model = LabelAttention((word2id, tag2id, label2id),    # index vocab
                  word_dim=word_dim, pos_dim=pos_dim, hidden_size=hidden_size, rnn_layers=rnn_layers,
                  dropout_rate=dropout_rate, bidirectional=bidirectional,
                  embedding=embedding if use_pretrained else None, use_crf=use_crf, device=device)

    model.to(device)

    parameters = [p for p in model.parameters() if p.requires_grad]     # parameter save for optimizer

    optimizer = optim.Adam(parameters,lr=1e-4)      # optimizer declare

    # evaluate
    if evaluate:
        state_dict = torch.load(os.path.join("save", "best_model.pt"))
        model.load_state_dict(state_dict)
        print("Load Best Model in %s" % os.path.join("save", "best_model.pt"))

        model_eval(model, dev, test, device, id2label)
        exit()

    best_F1 = 0.0
    losses = []
    # epch repeat
    for epoch in range(1, epochs + 1):
        print("Epoch %3d....." % epoch)
        num_data = len(train)
        num_batch = (num_data + batch_size - 1) // batch_size

        model.train()
        print("Start Training in Epoch %3d" % epoch)
        for ii in range(num_batch):

            batch_word_ids, batch_tag_ids, batch_labels_ids = batchify(ii, batch_size, num_data, train)

            batch_word_ids = batch_word_ids.to(device)
            batch_tag_ids = batch_tag_ids.to(device)
            batch_labels_ids = batch_labels_ids.to(device)

            loss = model(batch_word_ids, batch_tag_ids, batch_labels_ids)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            losses.append(loss.data)

            if (ii + 1) % 100 == 0:
                print("%6d/%6d: loss %.6f" % (ii + 1, num_batch, sum(losses) / len(losses)))
                losses = []

        # Dev Evaluation
        num_data = len(dev)
        num_batch = (num_data + batch_size - 1) // batch_size

        model.eval()
        print("Dev Evaluation in Epoch %3d" % epoch)

        total_answer_ids, total_pred_ids = [], []
        for ii in range(num_batch):
            batch_word_ids, batch_tag_ids, batch_labels_ids = batchify(ii, batch_size, num_data, dev)

            batch_word_ids = batch_word_ids.to(device)
            batch_tag_ids = batch_tag_ids.to(device)
            batch_labels_ids = batch_labels_ids.to(device)

            batch_answer_ids, batch_pred_ids = model(batch_word_ids, batch_tag_ids, batch_labels_ids)

            total_answer_ids.extend(batch_answer_ids)
            total_pred_ids.extend(batch_pred_ids)

        precision, recall, F1 = evaluate_ner_F1(total_answer_ids, total_pred_ids, id2label)

        print("[Epoch %d][ Dev] precision : %.2f, recall : %.2f, F1 : %.2f" % (epoch, precision, recall, F1))

        if F1 > best_F1:
            torch.save(model.state_dict(), os.path.join("save", "best_model.pt"))
            best_F1 = F1
            print('[new best model saved.]')

# evalueate mode
def model_eval(model, dev, test, device, id2label):
    # Dev Evaluation
    num_data = len(dev)
    num_batch = (num_data + batch_size - 1) // batch_size

    model.eval()
    print("Dev Evaluation in Best Model")


    total_answer_ids, total_pred_ids = [], []
    total_words = [sent[3] for sent in dev]
    for ii in range(num_batch):
        batch_word_ids, batch_tag_ids, batch_labels_ids = batchify(ii, batch_size, num_data, dev)

        batch_word_ids = batch_word_ids.to(device)
        batch_tag_ids = batch_tag_ids.to(device)
        batch_labels_ids = batch_labels_ids.to(device)

        batch_answer_ids, batch_pred_ids = model(batch_word_ids, batch_tag_ids, batch_labels_ids)

        total_answer_ids.extend(batch_answer_ids)
        total_pred_ids.extend(batch_pred_ids)

    precision, recall, F1 = evaluate_ner_F1_and_write_result(total_words, total_answer_ids, total_pred_ids, id2label, setname='dev')

    print("[Best][ Dev] precision : %.2f, recall : %.2f, F1 : %.2f" % (precision, recall, F1))

    # Test Evaluation
    num_data = len(test)
    num_batch = (num_data + batch_size - 1) // batch_size

    model.eval()
    print("Test Evaluation in Best Model")

    total_answer_ids, total_pred_ids = [], []
    total_words = [sent[3] for sent in test]
    for ii in range(num_batch):
        batch_word_ids, batch_tag_ids, batch_labels_ids = batchify(ii, batch_size, num_data, test)

        batch_word_ids = batch_word_ids.to(device)
        batch_tag_ids = batch_tag_ids.to(device)
        batch_labels_ids = batch_labels_ids.to(device)

        batch_answer_ids, batch_pred_ids = model(batch_word_ids, batch_tag_ids, batch_labels_ids)

        total_answer_ids.extend(batch_answer_ids)
        total_pred_ids.extend(batch_pred_ids)

    precision, recall, F1 = evaluate_ner_F1_and_write_result(total_words, total_answer_ids, total_pred_ids, id2label, setname='test')

    print("[Best][Test] precision : %.2f, recall : %.2f, F1 : %.2f" % (precision, recall, F1))


if __name__ == "__main__":
    batch_size = 64         # total data num / 64
    epochs = 100            # total data train * 100
    word_dim = 100          # embedding result -> word dimension 100 set
    pos_dim = 50            # pos dimesion set
    hidden_size = 256       # hidden_size == output_dimension
    rnn_layers = 2          # rnn is shape[batch_size, seq_len, dimension]
    dropout_rate = 0.33     # dropout rate
    bidirectional = True
    use_pretrained = True   # pretrained embedding
    use_crf = False
    evaluate = False

    train(epochs=epochs,
          batch_size=batch_size,
          word_dim=word_dim,
          pos_dim=pos_dim,
          hidden_size=hidden_size,
          rnn_layers=rnn_layers,
          bidirectional=bidirectional,
          use_pretrained=use_pretrained,
          use_crf=use_crf,
          dropout_rate=dropout_rate,
          evaluate=False)

    train(epochs=epochs,
          batch_size=batch_size,
          word_dim=word_dim,
          pos_dim=pos_dim,
          hidden_size=hidden_size,
          rnn_layers=rnn_layers,
          bidirectional=bidirectional,
          use_pretrained=use_pretrained,
          use_crf=use_crf,
          dropout_rate=dropout_rate,
          evaluate=True)
