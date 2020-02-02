__author__ = "chaitanya"

import torch
from torch.nn.init import xavier_normal_
import torch.nn as nn
import torch.nn.functional as F

from bert_feature_extractor import BertLayer
from decoder import DistMult, ConvE, ConvTransE, ConvKB
import layers

class LinkPredictor(nn.Module):

    def __init__(self, num_nodes, num_rels, args, use_cuda=False):

        super(LinkPredictor, self).__init__()
        self.rgcn = GCN(num_nodes, num_rels * 2, args, use_cuda)

        # Use 2 * num_rels to account for inverse relations
        if args.sim_relations:
            decoder_rels = (num_rels - 1) * 2
        else:
            decoder_rels = num_rels * 2

        # initialize decoder
        if args.decoder == "ConvE":
            self.decoder = ConvE(num_nodes, decoder_rels, args)
        elif args.decoder == "ConvTransE":
            self.decoder = ConvTransE(num_nodes, decoder_rels, args)
        elif args.decoder == "ConvKB":
            self.decoder = ConvKB(num_nodes, decoder_rels, args)
        else:
            self.decoder = DistMult(num_nodes, decoder_rels, args)

        #self.decoder = nn.DataParallel(self.decoder)
        #self.decoder.module.init()
        self.decoder.init()

        # Model-wide arguments
        self.num_rels = num_rels
        self.num_nodes = num_nodes
        self.use_cuda = use_cuda

        self.reg_param = args.regularization
        self.input_layer = args.input_layer
        self.bert_concat = args.bert_concat
        self.bert_sum = args.bert_sum
        self.bert_mlp = args.bert_mlp
        self.tying = args.tying
        self.layer_norm = args.layer_norm
        self.bert_dim = 1024

        if self.bert_concat:
            self.bert_concat_layer = EmbeddingLayer(num_nodes, self.bert_dim, args.dataset, init_bert=True)

        if self.bert_sum:
            self.bert_concat_layer = EmbeddingLayer(num_nodes, self.bert_dim, args.dataset, init_bert=True)
            self.beta = 0.5

        if self.bert_mlp:
            self.bert_concat_layer = EmbeddingLayer(num_nodes, self.bert_dim, args.dataset, init_bert=True)
            self.bmlp = nn.Linear(self.bert_dim + 200, 600)

        if self.layer_norm:
            self.bert_norm = nn.LayerNorm(self.bert_dim)
            self.gcn_norm = nn.LayerNorm(args.embedding_dim)

    def mask_by_schedule(self, tensor, epoch, epoch_cutoff=100):
        if epoch < epoch_cutoff:
            cuda_check = tensor.is_cuda

            if cuda_check:
                mask = torch.zeros((tensor.size(0), tensor.size(1)), device='cuda')
            else:
                mask = torch.zeros((tensor.size(0), tensor.size(1)))

            k = int((epoch / epoch_cutoff) * tensor.size(1))
            perm = torch.randperm(tensor.size(1))
            indices = perm[:k]
            mask[:, indices] = 1
            return tensor * mask
        else:
            return tensor

    def forward(self, g, epoch=None):

        if self.bert_concat:
            bert_embs = self.bert_concat_layer.embedding(g.ndata['id'].squeeze(1))
            gcn_embs = self.rgcn.forward(g)

            if self.layer_norm:
                bert_embs = self.bert_norm(bert_embs)
                gcn_embs = self.gcn_norm(gcn_embs)

            if epoch is not None:
                bert_embs = self.mask_by_schedule(bert_embs, epoch)

            # Fisher Test for checking importance of graph embeddings:
            # rand_indices = torch.randperm(gcn_embs.size(0))
            # gcn_embs = gcn_embs[rand_indices, :]
            # gcn_embs = torch.zeros((gcn_embs.size(0), gcn_embs.size(1)), device='cuda')

            out_embs = torch.cat([bert_embs, gcn_embs], dim=1)
            return out_embs
        elif self.bert_mlp:
            bert_embs = self.bert_concat_layer.embedding(g.ndata['id'].squeeze(1))
            gcn_embs = self.rgcn.forward(g)
            full_embs = torch.cat([bert_embs, gcn_embs], dim=1)
            full_embs_transf = self.bmlp(full_embs)
            return full_embs_transf
        elif self.bert_sum:
            bert_embs = self.bert_concat_layer.embedding(g.ndata['id'].squeeze(1))
            gcn_embs = self.rgcn.forward(g)
            if self.layer_norm:
                bert_embs = self.bert_norm(bert_embs)
                gcn_embs = self.gcn_norm(gcn_embs)
            return bert_embs + self.beta * gcn_embs
        elif self.tying:
            init_embs = self.rgcn.layers[0].embedding.weight
            gcn_embs = self.rgcn.forward(g)
            full_embs = torch.cat([init_embs, gcn_embs], dim=1)
            return full_embs
        else:
            gcn_embs = self.rgcn.forward(g)
            return gcn_embs

    def evaluate(self, g, epoch=None):
        # get embedding and relation weight without grad
        with torch.no_grad():
            embedding = self.forward(g, epoch)
        return embedding

    def regularization_loss(self, embedding):
        dec_weight = self.decoder.module.w_relation.weight.pow(2)

        if self.rgcn.num_hidden_layers > 0 and self.bert_concat:
            return torch.mean(embedding[:, -self.rgcn.layers[-1].out_feat:].pow(2)) + torch.mean(dec_weight)
        else:
            return torch.mean(embedding.pow(2)) + torch.mean(dec_weight)

    def calc_score(self, e1, rel, target=None):
        return self.decoder(e1, rel, target)

    def get_graph_embeddings(self, g, epoch=None):
        embedding = self.forward(g, epoch)
        return embedding

    def get_score(self, e1, rel, target, embedding):

        decoder_score = self.calc_score(e1, rel, target)
        # predict_loss = F.binary_cross_entropy(decoder_score, target)
        predict_loss = decoder_score

        if self.reg_param != 0.0:
            reg_loss = self.regularization_loss(embedding)
            return predict_loss + self.reg_param * reg_loss
        else:
            return predict_loss


class EmbeddingLayer(nn.Module):
    def __init__(self, num_nodes, h_dim, dataset=None, init_bert=None):
        super(EmbeddingLayer, self).__init__()
        self.embedding = torch.nn.Embedding(num_nodes, h_dim, padding_idx=0)
        if not init_bert:
            xavier_normal_(self.embedding.weight.data)
        else:
            self.init_with_bert(num_nodes, dataset)

    def forward(self, g):
        node_id = g.ndata['id'].squeeze(1)
        g.ndata['h'] = self.embedding(node_id)

    def init_with_bert(self, num_nodes, dataset):
        bert_model = BertLayer(dataset)
        bert_weights = bert_model.forward_as_init(num_nodes)
        self.embedding.load_state_dict({'weight': bert_weights})
        # self.embedding.weight.requires_grad = False


class BaseGCN(nn.Module):
    def __init__(self, num_nodes, num_rels, args, use_cuda=False):
        super(BaseGCN, self).__init__()

        self.num_nodes = num_nodes
        self.num_rels = num_rels
        self.use_cuda = use_cuda

        self.input_dim = args.init_embedding_dim
        self.embedding_dim = args.embedding_dim
        self.h_dim = args.n_hidden
        self.out_dim = args.n_hidden
        self.num_bases = args.n_bases
        self.num_hidden_layers = args.n_layers
        self.dropout = args.dropout
        self.input_layer = args.input_layer
        self.gcn_type = args.gcn_type
        self.bias = args.use_bias

        # create gcn layers
        self.build_model(args.dataset)

    def build_model(self, dataset):
        self.layers = nn.ModuleList()

        # i2h
        i2h = self.build_input_layer(dataset, self.input_layer)
        if i2h is not None:
            self.layers.append(i2h)

        # h2h
        for idx in range(self.num_hidden_layers):
            h2h = self.build_hidden_layer(idx)
            self.layers.append(h2h)

        # h2o
        h2o = self.build_output_layer()
        if h2o is not None:
            self.layers.append(h2o)

    # initialize feature for each node
    def create_features(self):
        return None

    def build_input_layer(self):
        return None

    def build_hidden_layer(self):
        raise NotImplementedError

    def build_output_layer(self):
        return None

    def forward(self, g):
        for layer in self.layers:
            layer(g)

        return g.ndata.pop('h')


class GCN(BaseGCN):
    def build_input_layer(self, dataset, input_layer="lookup"):
        
        if input_layer == "lookup":
            return EmbeddingLayer(self.num_nodes, self.input_dim)

        elif input_layer == "bilstm":
            # Node representations from BiLSTM
            self.node_embs = nn.Embedding(len(self.word_vocab), self.embedding_dim, padding_idx=self.word_vocab["PAD"])
            if self.pretrained_embs:
                self.init_embs()

            self.lstm = nn.LSTM(num_layers=self.n_layers, input_size=self.embedding_dim, hidden_size=self.hidden_dim,
                                bidirectional=True, dropout=self.dropout, batch_first=True)
            if self.max_pooling:
                self.pooler = nn.AdaptiveMaxPool1d(1)

        elif input_layer == "bert":
            return EmbeddingLayer(self.num_nodes, self.input_dim, dataset, input_layer)

    def build_hidden_layer(self, idx):
        
        #act = F.relu if idx < self.num_hidden_layers - 1 else None
        act = F.tanh if idx < self.num_hidden_layers else None
        self_loop = True

        if idx == 0:
            input_dim = self.input_dim
            output_dim = self.h_dim
        elif idx == self.num_hidden_layers-1:
            input_dim = self.h_dim
            output_dim = self.embedding_dim
        else:
            input_dim = self.h_dim
            output_dim = self.h_dim

        if self.gcn_type == "MultiHeadGATLayer":
            if idx != 0:
                input_dim = input_dim * 8

        if self.gcn_type == "WGCNAttentionSAGELayer":
            # output_dim = input_dim * 2
            self_loop = False
            #if idx != 0:
            #    input_dim *= 2

        cls = getattr(layers, self.gcn_type)
        return cls(input_dim, output_dim, self.num_rels, self.num_bases, self.bias,
                   activation=act, self_loop=self_loop, dropout=self.dropout)

