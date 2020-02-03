__author__ = "chaitanya"  # partially borrowed from implemenation of ConvE

import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.init import xavier_normal_

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.enabled = True


class DistMult(nn.Module):

    def __init__(self, num_entities, num_relations, args):
        super(DistMult, self).__init__()
        self.no_cuda = args.no_cuda
        self.w_relation = torch.nn.Embedding(num_relations, args.embedding_dim, padding_idx=0)
        self.inp_drop = torch.nn.Dropout(args.input_dropout)

    def init(self):
        xavier_normal_(self.w_relation.weight.data)

    def forward(self, embedding, e1, rel):

        batch_size = e1.shape[0]

        e1_embedded = embedding[e1].squeeze()
        rel_embedded = self.w_relation(rel).squeeze()
        e1_embedded = self.inp_drop(e1_embedded)
        rel_embedded = self.inp_drop(rel_embedded)

        score = torch.mm(e1_embedded * rel_embedded, embedding.t())
        score = F.sigmoid(score)

        return score


class ConvE(nn.Module):
    def __init__(self, num_entities, num_relations, args):
        super(ConvE, self).__init__()

        self.w_relation = torch.nn.Embedding(num_relations, args.n_hidden, padding_idx=0)
        self.inp_drop = torch.nn.Dropout(args.input_dropout)
        self.hidden_drop = torch.nn.Dropout(args.dropout)
        self.feature_map_drop = torch.nn.Dropout2d(args.feature_map_dropout)

        self.conv1 = torch.nn.Conv2d(1, 32, (3, 3), 1, 0, bias=args.use_bias)
        self.bn0 = torch.nn.BatchNorm2d(1)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.bn2 = torch.nn.BatchNorm1d(args.n_hidden)
        self.register_parameter('b', Parameter(torch.zeros(num_entities)))
        self.fc = torch.nn.Linear(10368, args.n_hidden)

    def init(self):
        xavier_normal_(self.w_relation.weight.data)

    def forward(self, embedding, e1, rel):

        batch_size = e1.shape[0]

        e1_embedded = embedding[e1].view(-1, 1, 10, 20)
        rel_embedded = self.w_relation(rel).view(-1, 1, 10, 20)

        stacked_inputs = torch.cat([e1_embedded, rel_embedded], 2)

        stacked_inputs = self.bn0(stacked_inputs)
        x = self.inp_drop(stacked_inputs)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.feature_map_drop(x)

        x = x.view(batch_size, -1)
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)

        x = torch.mm(x, embedding.t())

        # x = torch.mm(x, self.emb_e.weight.transpose(1,0))
        # x += self.b.expand_as(x)

        x += self.b.expand_as(x)

        pred = torch.sigmoid(x)
        return pred


class ConvKB(nn.Module):
    """
    Difference from ConvE: loss function is different, convolve over all e's at once
    """

    def __init__(self, num_entities, num_relations, args):
        super(ConvKB, self).__init__()

        self.w_relation = torch.nn.Embedding(num_relations, args.n_hidden, padding_idx=0)
        self.inp_drop = torch.nn.Dropout(args.input_dropout)
        self.hidden_drop = torch.nn.Dropout(args.dropout)
        self.feature_map_drop = torch.nn.Dropout(args.feature_map_dropout)
        self.loss = torch.nn.BCELoss()

        # 1D convolutions
        self.conv1 = torch.nn.Conv1d(3, 50, 3, bias=args.use_bias)
        self.bn0 = torch.nn.BatchNorm1d(1)
        self.bn1 = torch.nn.BatchNorm1d(50)
        self.bn2 = torch.nn.BatchNorm1d(1)

        self.fc = torch.nn.Linear(24900, 1)
        print(num_entities, num_relations)

    def init(self):
        xavier_normal_(self.w_relation.weight.data)

    def forward(self, embedding, triplets):

        e1 = triplets[:, 0]
        e2 = triplets[:, 2]
        rel = triplets[:, 1]

        batch_size = len(triplets)

        e1_embedded = embedding[e1]
        e2_embedded = embedding[e2]
        rel_embedded = self.w_relation(rel)

        stacked_inputs = torch.stack([e1_embedded, rel_embedded, e2_embedded])

        # stacked_inputs = self.bn0(stacked_inputs)
        x = self.inp_drop(stacked_inputs)
        x = self.conv1(x.transpose(0, 1))
        x = self.bn1(x)
        x = F.relu(x)

        x = self.feature_map_drop(x)

        x = x.view(batch_size, -1)
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)

        # x = torch.mm(x, self.emb_e.weight.transpose(1,0))
        # x += self.b.expand_as(x)

        pred = torch.sigmoid(x)
        return pred.squeeze(1)

    def evaluate(self, embedding, e1, rel):

        batch_size = e1.shape[0]

        e1_embedded = embedding[e1].view(-1, 1, 10, 20)
        rel_embedded = self.w_relation(rel).view(-1, 1, 10, 20)

        stacked_inputs = torch.cat([e1_embedded, rel_embedded], 2)

        stacked_inputs = self.bn0(stacked_inputs)
        x = self.inp_drop(stacked_inputs)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.feature_map_drop(x)

        x = x.view(batch_size, -1)
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)

        x = torch.mm(x, embedding.t())
        # x = torch.mm(x, self.emb_e.weight.transpose(1,0))

        # x += self.b.expand_as(x)
        x += self.b

        pred = torch.sigmoid(x)
        return pred


class ConvTransE(nn.Module):
    def __init__(self, num_entities, num_relations, args):

        """
        Difference from ConvE: no reshaping after stacking e_1 and e_r
        """

        super(ConvTransE, self).__init__()

        bert_dims = 1024

        self.no_cuda = args.no_cuda
        if args.bert_concat or args.tying:
            emb_dim = args.embedding_dim + bert_dims
        elif args.bert_mlp:
            emb_dim = 600
        else:
            emb_dim = args.embedding_dim

        if args.gcn_type == "MultiHeadGATLayer":
            num_heads = 8
            emb_dim = args.embedding_dim * num_heads + bert_dims

        self.embedding_dim = emb_dim

        self.w_relation = torch.nn.Embedding(num_relations, emb_dim, padding_idx=0)
        self.inp_drop = torch.nn.Dropout(args.input_dropout)
        self.hidden_drop = torch.nn.Dropout(args.dropout)
        self.feature_map_drop = torch.nn.Dropout(args.feature_map_dropout)

        kernel_size = 5
        self.channels = 200

        self.conv1 = nn.Conv1d(2, self.channels, kernel_size, stride=1, padding= int(math.floor(kernel_size/2)))
        # kernel size is odd, then padding = math.floor(kernel_size/2)

        self.bn0 = torch.nn.BatchNorm1d(2)
        self.bn1 = torch.nn.BatchNorm1d(self.channels)
        self.bn2 = torch.nn.BatchNorm1d(emb_dim)
        self.fc = torch.nn.Linear(self.channels * emb_dim, emb_dim)
        self.loss = torch.nn.BCELoss()
        self.cur_embedding = None

    def init(self):
        xavier_normal_(self.w_relation.weight.data)

    def forward(self, e1, rel, target):

        embedding = self.cur_embedding
        if not self.no_cuda:
            embedding = embedding.to(torch.cuda.current_device())
        
        batch_size = e1.shape[0]

        e1 = e1.unsqueeze(1)
        rel = rel.unsqueeze(1)

        e1_embedded = embedding[e1]
        rel_embedded = self.w_relation(rel)

        stacked_inputs = torch.cat([e1_embedded, rel_embedded], 1)
        stacked_inputs = self.bn0(stacked_inputs)

        x = self.inp_drop(stacked_inputs)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(batch_size, -1)
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)

        x = torch.mm(x, embedding.t())

        pred = torch.sigmoid(x)
        if target is None:
            return pred
        else:
            return self.loss(pred, target)
