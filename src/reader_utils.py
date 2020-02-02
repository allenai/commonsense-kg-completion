__author__ = "chaitanya"

import torch
import numpy as np
import string


def create_word_vocab(network):
    word_vocab = {}
    word_freqs = {}

    word_vocab["PAD"] = len(word_vocab)

    for node in network.graph.iter_nodes():
        for word in node.name.split():
            word = word.lower()
            if word not in word_vocab:
                word_vocab[word] = len(word_vocab)
            if word not in word_freqs:
                word_freqs[word] = 1
            else:
                word_freqs[word] += 1

    word_vocab["UNK"] = len(word_vocab)
    return word_vocab, word_freqs


def create_vocab(network):
    word_vocab = {}
    word_freqs = {}

    for node in network.graph.iter_nodes():
        if node.name not in word_vocab:
            word_vocab[node.name] = len(word_vocab)
        if node.name not in word_freqs:
            word_freqs[node.name] = 1
        else:
            word_freqs[node.name] += 1

    word_vocab["UNK"] = len(word_vocab)
    return word_vocab, word_freqs


def get_vocab_idx(vocab, token):
    if token not in vocab:
        return vocab["UNK"]
    else:
        return vocab[token]


def map_to_ids(vocab, seq):
    return [get_vocab_idx(vocab, word) for word in seq]


def get_relation_id(rel_name, train_network):
    rel_id = train_network.graph.find_relation(rel_name)
    if rel_id == -1: 
        return len(train_network.rel2id)
    else:
        return rel_id


def prepare_batch_nodes(vocab, batch):
    lens = [len(name) for name in batch]
    max_len = max(lens)
    sorted_batch = [x for _, x in sorted(zip(lens, batch), reverse=True, key=lambda x: x[0])]
    all_lens = []    
    word_ids_batch = []

    for node in sorted_batch:
        word_ids = map_to_ids(vocab, node)
        padding_length = max_len - len(word_ids)
        all_lens.append(len(word_ids))
        word_ids += [get_vocab_idx(vocab, "PAD")] * padding_length
        word_ids_batch.append(word_ids)
    
    return torch.LongTensor(word_ids_batch), torch.LongTensor(all_lens)


def prepare_batch_dgl(vocab, test_network, train_network):
    all_edges = []
    all_labels = []
    for edge in test_network.graph.iter_edges():
        src_id = get_vocab_idx(vocab, edge.src.name)
        tgt_id = get_vocab_idx(vocab, edge.tgt.name)
        rel_id = get_relation_id(edge.relation.name, train_network)
        all_edges.append(np.array([src_id, rel_id, tgt_id]))
        all_labels.append(edge.label)
    return np.array(all_edges), all_labels


def create_entity_dicts(all_tuples, num_rels, sim_relations=False):
    e1_to_multi_e2 = {}
    e2_to_multi_e1 = {}

    for tup in all_tuples:
        e1, rel, e2 = tup

        # No need to use sim edges for decoding
        if rel == num_rels-1 and sim_relations:
            continue

        rel_offset = num_rels

        if sim_relations:
            rel_offset -= 1

        if (e1, rel) in e1_to_multi_e2:
            e1_to_multi_e2[(e1, rel)].append(e2)
        else:
            e1_to_multi_e2[(e1, rel)] = [e2]

        if (e2, rel+rel_offset) in e1_to_multi_e2:
            e1_to_multi_e2[(e2, rel+rel_offset)].append(e1)
        else:
            e1_to_multi_e2[(e2, rel+rel_offset)] = [e1]

        if (e2, rel+rel_offset) in e2_to_multi_e1:
            e2_to_multi_e1[(e2, rel+rel_offset)].append(e1)
        else:
            e2_to_multi_e1[(e2, rel+rel_offset)] = [e1]
 
    return e1_to_multi_e2, e2_to_multi_e1


def preprocess_atomic_sentence(sent):

    puncts = list(string.punctuation)
    puncts.remove('-')

    sent = [c for c in sent.lower() if c not in puncts or c == "'"]
    sent = ''.join([c for c in sent if not c.isdigit()])
    sent = sent.replace("person x", "personx").replace(" x's", " personx's").replace(" x ", " personx ")
    if sent[:2] == "x " or sent[:2] == "x'":
        sent = sent.replace("x ", "personx ", 1).replace("x'", "personx'")
    if sent[-3:] == " x\n":
        sent = sent.replace(" x\n", "personx\n")

    sent = sent.replace("person y", "persony").replace(" y's", " persony's").replace(" y ", " persony ")
    if sent[:2] == "y " or sent[:2] == "y'":
        sent = sent.replace("y ", "persony ", 1).replace("y'", "persony'")
    if sent[-3:] == " y\n":
        sent = sent.replace(" y\n", "persony\n")

    return sent.replace("personx", "John").replace("persony", "Tom")
