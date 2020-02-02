__author__ = "chaitanya"  # Adapted from DGL official examples

import torch
import torch.nn as nn
from torch.nn.parallel import data_parallel
import numpy as np
import dgl.function as fn

from torch.nn.init import xavier_normal_, xavier_uniform_
from torch.nn import functional as F, Parameter


class RGCNLayer(nn.Module):
    def __init__(self, in_feat, out_feat, bias=None, activation=None,
                 self_loop=False, dropout=0.0):
        super(RGCNLayer, self).__init__()
        self.bias = bias
        self.activation = activation
        self.self_loop = self_loop
        
        if self.bias:
            self.bias_weight = nn.Parameter(torch.Tensor(out_feat))

            # Following bias initialization used in ConvTransE
            stdv = 1. / np.sqrt(out_feat)
            self.bias_weight.data.uniform_(-stdv, stdv)
            #nn.init.xavier_uniform_(self.bias,
            #                        gain=nn.init.calculate_gain('relu'))

        # weight for self loop
        if self.self_loop:
            if isinstance(self, MultiHeadGATLayer):
                self.loop_weight = nn.Parameter(torch.Tensor(in_feat, out_feat * 8))
            else:
                self.loop_weight = nn.Parameter(torch.Tensor(in_feat, out_feat))
            self.loop_rel = nn.Parameter(torch.Tensor(1))
            nn.init.xavier_uniform_(self.loop_weight,
                                    gain=nn.init.calculate_gain('relu'))

        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

    # define how propagation is done in subclass
    def propagate(self, g):
        raise NotImplementedError

    def forward(self, g):

        #loop_message = g.ndata['h'] * self.loop_weight
        #if self.dropout is not None:
        #    loop_message = self.dropout(loop_message)

        if self.self_loop:
            loop_message = torch.mm(g.ndata['h'], self.loop_weight)
        
        #g.ndata['h_mm'] = torch.mm(g.ndata['h'], self.weight)
        #g.edata['alpha'] = self.weight_rel(g.edata['type'])

        self.propagate(g)

        # additional processing
        # apply bias and activation
        node_repr = g.ndata['h']
        if self.bias:
            node_repr = node_repr + self.bias_weight
        if self.self_loop:
            node_repr = node_repr + loop_message

        # Apply batch normalization
        if not isinstance(self, MultiHeadGATLayer) and not isinstance(self, GATLayer):
            node_repr = self.bn(node_repr)

        if self.activation:
            node_repr = self.activation(node_repr)

        if self.dropout is not None:
            node_repr = self.dropout(node_repr)

        g.ndata['h'] = node_repr


class RGCNBasisLayer(RGCNLayer):
    def __init__(self, in_feat, out_feat, num_rels, num_bases=-1, bias=None,
                 activation=None, is_input_layer=False):
        super(RGCNBasisLayer, self).__init__(in_feat, out_feat, bias, activation)
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.num_rels = num_rels
        self.num_bases = num_bases
        self.is_input_layer = is_input_layer
        if self.num_bases <= 0 or self.num_bases > self.num_rels:
            self.num_bases = self.num_rels

        # add basis weights
        self.weight = nn.Parameter(torch.Tensor(self.num_bases, self.in_feat, self.out_feat))

        if self.num_bases < self.num_rels:
            # linear combination coefficients
            self.w_comp = nn.Parameter(torch.Tensor(self.num_rels, self.num_bases))

        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))
        if self.num_bases < self.num_rels:
            nn.init.xavier_uniform_(self.w_comp,
                                    gain=nn.init.calculate_gain('relu'))

    def propagate(self, g):
        if self.num_bases < self.num_rels:
            # generate all weights from bases
            weight = self.weight.view(self.num_bases,
                                      self.in_feat * self.out_feat)
            weight = torch.matmul(self.w_comp, weight).view(
                                    self.num_rels, self.in_feat, self.out_feat)
        else:
            weight = self.weight

        if self.is_input_layer:
            def msg_func(edges):
                # for input layer, matrix multiply can be converted to be
                # an embedding lookup using source node id
                embed = weight.view(-1, self.out_feat)
                index = edges.data['type'] * self.in_feat + edges.src['id']
                return {'msg': embed.index_select(0, index) * edges.data['norm']}
        else:
            def msg_func(edges):
                w = weight.index_select(0, edges.data['type'].squeeze())
                msg = torch.bmm(edges.src['h'].unsqueeze(1), w).squeeze()
                msg = msg * edges.data['norm']
                return {'msg': msg}

        g.update_all(msg_func, fn.sum(msg='msg', out='h'), None)


class RGCNBlockLayer(RGCNLayer):
    def __init__(self, in_feat, out_feat, num_rels, num_bases, bias=None,
                 activation=None, self_loop=False, dropout=0.0):
        super(RGCNBlockLayer, self).__init__(in_feat, out_feat, bias,
                                             activation, self_loop=self_loop,
                                             dropout=dropout)
        self.num_rels = num_rels
        self.num_bases = num_bases
        assert self.num_bases > 0

        self.out_feat = out_feat
        self.submat_in = in_feat // self.num_bases
        self.submat_out = out_feat // self.num_bases
        self.bn = torch.nn.BatchNorm1d(self.out_feat)

        # assuming in_feat and out_feat are both divisible by num_bases
        self.weight = nn.Parameter(torch.Tensor(self.num_rels, self.num_bases * self.submat_in * self.submat_out))
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))

    def msg_func(self, edges):
        """
        Compute messages only from source node features
        """
        weight = self.weight.index_select(0, edges.data['type'].squeeze()).view(
                    -1, self.submat_in, self.submat_out)
        node = edges.src['h'].view(-1, 1, self.submat_in)
        msg = torch.bmm(node, weight).view(-1, self.out_feat)
        return {'msg': msg}

    def propagate(self, g):
        g.update_all(self.msg_func, fn.sum(msg='msg', out='h'), self.apply_func)

    def apply_func(self, nodes):
        return {'h': nodes.data['h'] * nodes.data['norm']}


class RGCNBlockAttentionLayer(RGCNLayer):
    def __init__(self, in_feat, out_feat, num_rels, num_bases, bias=None,
                 activation=None, self_loop=False, dropout=0.0, bert=False, bert_trainable=False):
        super(RGCNBlockAttentionLayer, self).__init__(in_feat, out_feat, bias,
                                             activation, self_loop=self_loop,
                                             dropout=dropout)
        self.num_rels = num_rels
        self.num_bases = num_bases
        assert self.num_bases > 0

        self.out_feat = out_feat
        self.submat_in = in_feat // self.num_bases
        self.submat_out = out_feat // self.num_bases
        self.bn = torch.nn.BatchNorm1d(self.out_feat)
        # assuming in_feat and out_feat are both divisible by num_bases
        self.weight = nn.Parameter(torch.Tensor(self.num_rels, self.num_bases * self.submat_in * self.submat_out))
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))

    def msg_func(self, edges):
        """
        Compute messages only from source node features
        """
        
        weight = self.weight.index_select(0, edges.data['type'].squeeze()).view(
                    -1, self.submat_in, self.submat_out)
        node = edges.src['h'].view(-1, 1, self.submat_in)
        msg = torch.bmm(node, weight).view(-1, self.out_feat)
        return {'msg': msg, 'node_id': edges.src['id']}

    def compute_attn_weights():
        pass

    def attn_reduce(self, nodes):
        # TODO: Weigh neighbors by attention on top of BERT feature representations
        pass
        
    def propagate(self, g):
        g.update_all(self.msg_func, self.attn_reduce, self.apply_func)

    def apply_func(self, nodes):
        return {'h': nodes.data['h'] * nodes.data['norm']}


class WGCNLayer(RGCNLayer):
    def __init__(self, in_feat, out_feat, num_rels, num_bases, bias=True,
                 activation=None, self_loop=False, dropout=0.2):
        super(WGCNLayer, self).__init__(in_feat, out_feat, bias,
                                        activation, self_loop=self_loop,
                                        dropout=dropout)
        self.num_rels = num_rels
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.weight = Parameter(torch.FloatTensor(self.in_feat, self.out_feat))
        self.weight_rel = torch.nn.Embedding(self.num_rels, 1, padding_idx=0)
        self.bn = torch.nn.BatchNorm1d(self.out_feat)
        xavier_normal_(self.weight.data)
        #stdv = 1. / np.sqrt(self.weight.size(1))
        #self.weight.data.uniform_(-stdv, stdv)

    def msg_func(self, edges):
        """
        Compute messages only from source node features
        """
        edge_types = edges.data['type'].squeeze()
        alpha = self.weight_rel(edge_types)
        node = torch.mm(edges.src['h'], self.weight)
        msg = alpha.expand_as(node) * node
        return {'msg': msg}

    def propagate(self, g):
        #g.update_all(fn.src_mul_edge(src='h_mm', edge='alpha', out='msg'), 
        #             fn.sum(msg='msg', out='h'), 
        #             apply_node_func=lambda nodes: {'h': nodes.data['h'] * nodes.data['norm']})
        g.update_all(self.msg_func, fn.sum(msg='msg', out='h'), self.apply_func)

    def apply_func(self, nodes):
        return {'h': nodes.data['h'] * nodes.data['norm']}


class WGCNAttentionLayer(RGCNLayer):
    def __init__(self, in_feat, out_feat, num_rels, num_bases, bias=True,
                 activation=None, self_loop=False, dropout=0.2):
        super(WGCNAttentionLayer, self).__init__(in_feat, out_feat, bias, activation,
                                                 self_loop=self_loop, dropout=dropout)
        self.num_rels = num_rels
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.weight = Parameter(torch.FloatTensor(self.in_feat, self.out_feat))
        self.weight_rel = torch.nn.Embedding(self.num_rels, 1, padding_idx=0)
        self.bn = torch.nn.BatchNorm1d(self.out_feat)
        xavier_normal_(self.weight.data)
        #stdv = 1. / np.sqrt(self.weight.size(1))
        #self.weight.data.uniform_(-stdv, stdv)

    def msg_func(self, edges):
        """
        Compute messages only from source node features
        """
        edge_types = edges.data['type'].squeeze()
        alpha = self.weight_rel(edge_types)
        node = torch.mm(edges.src['h'], self.weight)
        msg = alpha.expand_as(node) * node
        return {'msg': msg}

    def attn_reduce(self, nodes):
        attn_vector = torch.bmm(nodes.mailbox['msg'], nodes.data['h'].unsqueeze(2))
        attn_probs = torch.softmax(attn_vector, dim=1)
        attn_weighted_msgs = nodes.mailbox['msg'] * attn_probs.expand_as(nodes.mailbox['msg'])
        attn_sum = torch.sum(attn_weighted_msgs, dim=1)
        return {'h': attn_sum} 

    def propagate(self, g):
        #g.update_all(fn.src_mul_edge(src='h_mm', edge='alpha', out='msg'), 
        #             fn.sum(msg='msg', out='h'), 
        #             apply_node_func=lambda nodes: {'h': nodes.data['h'] * nodes.data['norm']})
        g.update_all(self.msg_func, self.attn_reduce, self.apply_func)

    def apply_func(self, nodes):
        return {'h': nodes.data['h'] * nodes.data['norm']}


class WGCNAttentionSAGELayer(RGCNLayer):
    def __init__(self, in_feat, out_feat, num_rels, num_bases, bias=True,
                 activation=None, self_loop=False, dropout=0.2):
        super(WGCNAttentionSAGELayer, self).__init__(in_feat, out_feat, bias,
                                        activation, self_loop=self_loop,
                                        dropout=dropout)
        self.num_rels = num_rels
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.weight = Parameter(torch.FloatTensor(self.in_feat*2, self.out_feat))
        self.weight_rel = torch.nn.Embedding(self.num_rels, 1, padding_idx=0)
        self.bn = torch.nn.BatchNorm1d(self.out_feat)
        xavier_normal_(self.weight.data)
        #stdv = 1. / np.sqrt(self.weight.size(1))
        #self.weight.data.uniform_(-stdv, stdv)

    def msg_func(self, edges):
        """
        Compute messages only from source node features
        """
        edge_types = edges.data['type'].squeeze()
        alpha = self.weight_rel(edge_types)
        #node = torch.mm(edges.src['h'], self.weight)
        msg = alpha.expand_as(edges.src['h']) * edges.src['h']
        return {'msg': msg}

    def compute_attn_weights():
        pass

    def attn_reduce(self, nodes):
        mean_msg = torch.mean(nodes.mailbox['msg'], dim=1)
        aggreg_msg = torch.cat((nodes.data['h'], mean_msg), dim=1)
        node_repr = torch.mm(aggreg_msg, self.weight)
        return {'h': node_repr}

    def propagate(self, g):
        #g.update_all(fn.src_mul_edge(src='h_mm', edge='alpha', out='msg'), 
        #             fn.sum(msg='msg', out='h'), 
        #             apply_node_func=lambda nodes: {'h': nodes.data['h'] * nodes.data['norm']})
        g.update_all(self.msg_func, self.attn_reduce, self.apply_func)

    def apply_func(self, nodes):
        return {'h': nodes.data['h'] * nodes.data['norm']}


class GATLayer(RGCNLayer):
    def __init__(self, in_feat, out_feat, num_rels, num_bases, bias=True,
                 activation=None, self_loop=False, dropout=0.2):
        super(GATLayer, self).__init__(in_feat, out_feat, bias,
                 activation, self_loop=self_loop, dropout=dropout)
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.weight = nn.Linear(in_feat, out_feat, bias=False)
        self.weight_rel = torch.nn.Embedding(num_rels, 1, padding_idx=0)
        self.attn_fc = nn.Linear(2 * out_feat, 1, bias=False)

    def edge_attention(self, edges):
        # edge UDF for equation (2)
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
        a = self.attn_fc(z2)
        edge_types = edges.data['type'].squeeze()
        rel_alpha = self.weight_rel(edge_types)
        return {'e': F.leaky_relu(a), 'rel_alpha': rel_alpha}

    def message_func(self, edges):
        # message UDF for equation (3) & (4)
        return {'z': edges.src['z'], 'e': edges.data['e'], 'rel_alpha': edges.data['rel_alpha']}

    def reduce_func(self, nodes):
        # reduce UDF for equation (3) & (4)
        # equation (3)
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        rel_alpha = nodes.mailbox['rel_alpha']
        # equation (4)
        h = torch.sum(rel_alpha * alpha * nodes.mailbox['z'], dim=1)
        return {'h': h}

    def propagate(self, g):
        # equation (1)
        h = g.ndata['h']
        z = self.weight(h)
        g.ndata['z'] = z
        # equation (2)
        g.apply_edges(self.edge_attention)
        # equation (3) & (4)
        g.update_all(self.message_func, self.reduce_func)
        #return self.g.ndata.pop('h')


class GATSubLayer(nn.Module):
    def __init__(self, in_feat, out_feat, num_rels):
        super(GATSubLayer, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.weight = nn.Linear(in_feat, out_feat, bias=False)
        self.weight_rel = torch.nn.Embedding(num_rels, 1, padding_idx=0)
        self.attn_fc = nn.Linear(2 * out_feat, 1, bias=False)

    def edge_attention(self, edges):
        # edge UDF for equation (2)
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
        edge_types = edges.data['type'].squeeze()
        rel_alpha = self.weight_rel(edge_types)
        a = self.attn_fc(z2)
        return {'e': F.leaky_relu(a), 'rel_alpha': rel_alpha}

    def message_func(self, edges):
        # message UDF for equation (3) & (4)
        return {'z': edges.src['z'], 'e': edges.data['e'], 'rel_alpha': edges.data['rel_alpha']}

    def reduce_func(self, nodes):
        # reduce UDF for equation (3) & (4)
        # equation (3)
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        rel_alpha = nodes.mailbox['rel_alpha']
        # equation (4)
        h = torch.sum(rel_alpha * alpha * nodes.mailbox['z'], dim=1)
        return {'head-out': h}

    def propagate(self, g):
        # equation (1)
        h = g.ndata['h']
        z = self.weight(h)
        g.ndata['z'] = z
        # equation (2)
        g.apply_edges(self.edge_attention)
        # equation (3) & (4)
        g.update_all(self.message_func, self.reduce_func)
        return g.ndata.pop('head-out')


class MultiHeadGATLayer(RGCNLayer):
    def __init__(self, in_feat, out_feat, num_rels, num_bases, bias=True,
                 activation=None, self_loop=False, dropout=0.2):
        super(MultiHeadGATLayer, self).__init__(in_feat, out_feat, bias,
                 activation, self_loop=self_loop, dropout=dropout)
        self.heads = nn.ModuleList()
        for i in range(8):
            self.heads.append(GATSubLayer(in_feat, out_feat, num_rels))
        self.merge = "cat"
        self.out_feat = out_feat
        self.in_feat = in_feat
     
    def propagate(self, g):
        head_outs = [attn_head.propagate(g) for attn_head in self.heads]
        if self.merge == 'cat':
            # concat on the output feature dimension (dim=1)
            g.ndata['h'] = torch.cat(head_outs, dim=1)
        else:
            # merge using average
            g.ndata['h'] = torch.mean(torch.stack(head_outs))
        return g.ndata['h']
