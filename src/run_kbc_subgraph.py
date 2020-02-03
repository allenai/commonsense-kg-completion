# Main script for experimenting with training on a subgraph
from collections import Counter
import argparse
import numpy as np
import sys
import os
import json
import time
import random

import torch
import torch.nn as nn

from model import LinkPredictor
from reader import AtomicTSVReader, ConceptNetTSVReader, FB15kReader
import utils
import reader_utils
import evaluation_utils

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def set_seeds(seed):
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)


def load_data(dataset, reader_cls, data_dir, sim_relations):
    train_network = reader_cls(dataset)
    dev_network = reader_cls(dataset)
    test_network = reader_cls(dataset)

    train_network.read_network(data_dir=data_dir, split="train")
    train_network.print_summary()
    node_list = train_network.graph.iter_nodes()
    node_degrees = [node.get_degree() for node in node_list]
    degree_counter = Counter(node_degrees)
    avg_degree = sum([k * v for k, v in degree_counter.items()]) / sum([v for k, v in degree_counter.items()])
    print("Average Degree: ", avg_degree)

    dev_network.read_network(data_dir=data_dir, split="valid", train_network=train_network)
    test_network.read_network(data_dir=data_dir, split="test", train_network=train_network)

    word_vocab = train_network.graph.node2id

    # Add sim nodes
    if sim_relations:
        print("Adding sim edges..")
        train_network.add_sim_edges_bert()

    train_data, _ = reader_utils.prepare_batch_dgl(word_vocab, train_network, train_network)
    test_data, test_labels = reader_utils.prepare_batch_dgl(word_vocab, test_network, train_network)
    valid_data, valid_labels = reader_utils.prepare_batch_dgl(word_vocab, dev_network, train_network)

    return train_data, valid_data, test_data, valid_labels, test_labels, train_network


def get_model_name(args):

    name = '_subgraph_model_state.pth'
    name = "_" + args.gcn_type + "_" + args.decoder + name

    if args.sim_relations:
        name = "_sim_relations" + name

    if args.sim_sim:
        name = "_sim-sim" + name

    if args.bert_concat:
        name = "_bert_concat" + name

    if args.bert_mlp:
        name = "_bert_mlp" + name

    if args.tying:
        name = "_tying" + name

    if args.bert_sum:
        name = "_bert_sum" + name

    if args.input_layer == "bert":
        name = "_inp-bert" + name

    model_state_file = args.dataset + name
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    model_state_file = os.path.join(args.output_dir, model_state_file)

    return model_state_file


def main(args):
    set_seeds(args.seed)

    # load graph data
    if args.dataset == "FB15K-237":
        dataset_cls = FB15kReader
        data_dir = "data/FB15k-237/"
    elif args.dataset == "atomic":
        dataset_cls = AtomicTSVReader
        data_dir = "data/atomic/"
    elif args.dataset == "conceptnet":
        dataset_cls = ConceptNetTSVReader
        data_dir = "data/ConceptNet/"
    else:
        raise ValueError("Invalid option for dataset.")

    # Store entity-wise dicts for filtered metrics
    train_data, valid_data, test_data, valid_labels, test_labels, train_network = load_data(args.dataset,
                                                                                            dataset_cls,
                                                                                            data_dir,
                                                                                            args.sim_relations)
    num_nodes = len(train_network.graph.nodes)
    num_rels = len(train_network.graph.relations)
    all_tuples = train_data.tolist() + valid_data.tolist() + test_data.tolist()

    # for filtered ranking
    all_e1_to_multi_e2, all_e2_to_multi_e1 = reader_utils.create_entity_dicts(all_tuples, num_rels, args.sim_relations)

    # for training
    train_e1_to_multi_e2, train_e2_to_multi_e1 = reader_utils.create_entity_dicts(train_data.tolist(), num_rels,
                                                                                  args.sim_relations)
    # the below dicts `include` sim relations
    sim_train_e1_to_multi_e2, sim_train_e2_to_multi_e1 = reader_utils.create_entity_dicts(train_data.tolist(), num_rels)

    # check cuda
    use_cuda = torch.cuda.is_available()
    if use_cuda and not args.no_cuda:
        torch.cuda.set_device(args.gpu)

    cpu_decoding = args.cpu_decoding

    # atomic graph is much larger, so we perform evaluation on cpu
    cpu_eval = True if args.dataset == "atomic" else False

    # create model
    model = LinkPredictor(num_nodes,
                          num_rels,
                          args,
                          use_cuda=use_cuda)

    # build graph
    graph_train_data = train_data
    test_graph, test_rel, test_norm = utils.build_test_graph(num_nodes, num_rels, graph_train_data)
    test_deg = test_graph.in_degrees(range(test_graph.number_of_nodes())).float().view(-1, 1)
    test_node_id = torch.arange(0, num_nodes, dtype=torch.long).view(-1, 1)
    test_rel = torch.from_numpy(test_rel).view(-1, 1)
    test_norm = torch.from_numpy(test_norm).view(-1, 1)

    # transfer graph data to gpu
    if use_cuda and not args.no_cuda and not cpu_decoding:
        test_node_id = test_node_id.cuda()
        test_norm = test_norm.cuda()
        test_rel = test_rel.cuda()

    # validation and testing triplets
    valid_data = torch.LongTensor(valid_data)
    test_data = torch.LongTensor(test_data)

    if use_cuda and not args.no_cuda and not cpu_eval:
        valid_data = valid_data.cuda()
        test_data = test_data.cuda()

    test_graph.ndata.update({'id': test_node_id, 'norm': test_norm})
    test_graph.edata['type'] = test_rel

    if use_cuda and not args.no_cuda:
        # model = nn.DataParallel(model, device_ids=[0,1])
        model = model.cuda()

    model_state_file = get_model_name(args)

    # writer = SummaryWriter("runs/" + model_state_file.replace(".pth",".log"))

    # check if only evaluation needs to be performed
    if args.eval_only:
        if args.load_model:
            model_state_file = args.load_model
        else:
            print("Please provide model path for evaluation (--load_model)")
            sys.exit(0)

        checkpoint = torch.load(model_state_file)

        if use_cuda and not args.no_cuda and cpu_eval:
            model.cpu()
            test_graph.ndata['id'] = test_graph.ndata['id'].cpu()
            test_graph.ndata['norm'] = test_graph.ndata['norm'].cpu()
            test_graph.edata['type'] = test_graph.edata['type'].cpu()
            model.decoder.no_cuda = True

        model.eval()
        model.load_state_dict(checkpoint['state_dict'])
        print(model)

        print("================DEV=================")
        mrr = evaluation_utils.ranking_and_hits(test_graph, model, valid_data, all_e1_to_multi_e2, train_network,
                                                fusion="graph-only", sim_relations=args.sim_relations,
                                                write_results=args.write_results, debug=args.debug)

        print("================TEST================")
        mrr = evaluation_utils.ranking_and_hits(test_graph, model, test_data, all_e1_to_multi_e2, train_network, 
                                                fusion="graph-only", sim_relations=args.sim_relations, debug=args.debug)

        sys.exit(0)

    if os.path.isfile(model_state_file):
        print(model_state_file)
        overwrite = input('Model already exists. Overwrite? Y = yes, N = no\n')
        if overwrite.lower() == 'n':
            print("Quitting")
            sys.exit(0)
        elif overwrite.lower() != 'y':
            raise ValueError("Invalid Option")

    # build adj list and calculate degrees for sampling
    adj_list, degrees, sparse_adj_matrix, rel = utils.get_adj_and_degrees(num_nodes, num_rels, train_data)

    # remove sim edges from sampling_edge_ids (we sample from the original graph and then densify it)
    if args.sim_relations:
        sim_edge_ids = np.where(graph_train_data[:, 1] == num_rels - 1)[0]
        sampling_edge_ids = np.delete(np.arange(len(graph_train_data)), sim_edge_ids)
    else:
        sampling_edge_ids = None

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    forward_time = []
    backward_time = []

    # training loop
    print("Starting training...")
    epoch = 0
    best_mrr = 0

    while True:
        model.train()
        epoch += 1

        cur_train_data = graph_train_data[:]

        # build dgl graph
        g, node_id, edge_type, node_norm, data, labels = \
            utils.generate_sampled_graph_and_labels(
                cur_train_data, args.graph_batch_size,
                num_rels, adj_list, degrees, args.negative_sample, args.sim_sim, args.sim_relations,
                sim_train_e1_to_multi_e2, sampling_edge_ids)

        node_id_copy = np.copy(node_id)
        node_dict = {v: k for k, v in dict(enumerate(node_id_copy)).items()}

        # set node/edge feature
        node_id = torch.from_numpy(node_id).view(-1, 1)
        edge_type = torch.from_numpy(edge_type)
        node_norm = torch.from_numpy(node_norm).view(-1, 1)

        if use_cuda and not args.no_cuda:
            node_id = node_id.cuda()
            edge_type, node_norm = edge_type.cuda(), node_norm.cuda()

        g.ndata.update({'id': node_id, 'norm': node_norm})
        g.edata['type'] = edge_type

        batch_size = args.decoder_batch_size
        e1_keys = list(train_e1_to_multi_e2.keys())
        sub_e1_keys = {}

        # Add inverse edges to training samples
        src, dst = np.concatenate((cur_train_data[:, 0], cur_train_data[:, 2])), \
                   np.concatenate((cur_train_data[:, 2], cur_train_data[:, 0]))
        rel = cur_train_data[:, 1]
        rel = np.concatenate((rel, rel + num_rels))
        cur_train_data = np.stack((src, rel, dst)).transpose()

        # The loop below constructs a dict for the decoding step
        # with the key (src, rel) and the value as the list of nodes present in the original graph
        # where the source and target nodes both belong to the list of sampled nodes in subgraph

        for e in cur_train_data:
            rel = e[1]
            # Don't use sim relations for decoding
            if args.sim_relations:
                if rel == num_rels - 1 or rel == (num_rels * 2) - 1:
                    continue
                elif rel >= num_rels:
                    rel -= 1

            if e[0] in node_id_copy and e[2] in node_id_copy:
                subgraph_src_idx = node_dict[e[0]]
                subgraph_tgt_idx = node_dict[e[2]]
                if (subgraph_src_idx, rel) not in sub_e1_keys:
                    sub_e1_keys[(subgraph_src_idx, rel)] = [subgraph_tgt_idx]
                else:
                    sub_e1_keys[(subgraph_src_idx, rel)].append(subgraph_tgt_idx)

        key_list = list(sub_e1_keys.keys())

        random.shuffle(key_list)
        cum_loss = 0.0

        for i in range(0, len(key_list), batch_size):

            optimizer.zero_grad()

            # compute graph embeddings
            graph_embeddings = model.get_graph_embeddings(g, epoch)
            #model.decoder.module.cur_embedding = graph_embeddings
            model.decoder.cur_embedding = graph_embeddings

            batch = key_list[i: i + batch_size]

            # Don't train with batches of size 1 and always set batch_size > 1 since batch norm
            # fails with batch_size=1
            if len(batch) == 1:
                continue

            e1 = torch.LongTensor([elem[0] for elem in batch])
            rel = torch.LongTensor([elem[1] for elem in batch])

            # e2 -> list of target nodes in subgraph
            e2 = [sub_e1_keys[(k[0], k[1])] for k in batch]
            batch_len = len(batch)

            if use_cuda and not args.no_cuda and not cpu_decoding:
                target = torch.cuda.FloatTensor(batch_len, node_id_copy.shape[0]).fill_(0)
                e1 = e1.cuda()
                rel = rel.cuda()
            else:
                target = torch.zeros((batch_len, node_id_copy.shape[0]))

            # construct target tensor
            for j, inst in enumerate(e2):
                target[j, inst] = 1.0

            # perform label smoothing
            target = ((1.0 - args.label_smoothing_epsilon) * target) + (1.0 / target.size(1))

            if cpu_decoding:
                graph_embeddings = graph_embeddings.cpu()
                model.decoder.cpu()
                model.decoder.no_cuda = True

            t0 = time.time()

            loss = model.get_score(e1, rel, target, graph_embeddings)
            loss = torch.mean(loss)
            cum_loss += loss.cpu().item()
            t1 = time.time()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)  # clip gradients
            optimizer.step()

            t2 = time.time()

            forward_time.append(t1 - t0)
            backward_time.append(t2 - t1)

            del graph_embeddings, target, batch, loss, e1, rel, e2

            # the below make training very slow
            # gc.collect()
            # torch.cuda.empty_cache()

        print("Epoch {:04d} | Loss {:.4f} | Best MRR {:.4f} | Forward {:.4f}s | Backward {:.4f}s".
              format(epoch, cum_loss, best_mrr, forward_time[-1], backward_time[-1]))
        # writer.add_scalar('data/loss', cum_loss , epoch)

        # Save model every 100 epochs
        # if epoch + 1 % 100==0:
        #    print("saving current model..")
        #    torch.save({'state_dict': model.state_dict(), 'epoch': epoch},
        #                 model_state_file)

        # validation
        if epoch % args.evaluate_every == 0:
            # perform validation on CPU when full graph is too large
            if use_cuda and not args.no_cuda and cpu_eval:
                model.cpu()
                test_graph.ndata['id'] = test_graph.ndata['id'].cpu()
                test_graph.ndata['norm'] = test_graph.ndata['norm'].cpu()
                test_graph.edata['type'] = test_graph.edata['type'].cpu()
                model.decoder.no_cuda = True

            model.eval()
            print("start eval")

            print("===========DEV============")
            mrr = evaluation_utils.ranking_and_hits(test_graph, model, valid_data, all_e1_to_multi_e2,
                                                    train_network, fusion="graph-only", sim_relations=args.sim_relations,
                                                    debug=args.debug, epoch=epoch)

            # writer.add_scalar('data/mrr', mrr, epoch)

            # save best model
            # torch.save({'state_dict': model.state_dict(), 'epoch': epoch},
            #                model_state_file)
            if mrr < best_mrr:
                if epoch >= args.n_epochs:
                    break
            else:
                best_mrr = mrr
                print("[saving best model so far]")
                torch.save({'state_dict': model.state_dict(), 'epoch': epoch},
                           model_state_file)

            metrics = {"best_mrr": best_mrr,
                       "cum_loss": cum_loss
                       }

            with open(os.path.join(args.output_dir, 'metrics.json'), 'w') as f:
                f.write(json.dumps(metrics))

            # transfer graph back to gpu device
            if use_cuda and not args.no_cuda and cpu_eval:
                model.cuda()
                test_graph.ndata['id'] = test_graph.ndata['id'].cuda()
                test_graph.ndata['norm'] = test_graph.ndata['norm'].cuda()
                test_graph.edata['type'] = test_graph.edata['type'].cuda()
                model.decoder.no_cuda = False

    print("training done")
    print("Mean forward time: {:4f}s".format(np.mean(forward_time)))
    print("Mean Backward time: {:4f}s".format(np.mean(backward_time)))

    # writer.export_scalars_to_json("./all_scalars.json")
    # writer.close()

    print("\nStart testing")

    # use best model checkpoint
    checkpoint = torch.load(model_state_file)
    model.eval()
    model.load_state_dict(checkpoint['state_dict'])
    print("Using best epoch: {}".format(checkpoint['epoch']))

    evaluation_utils.ranking_and_hits(test_graph, model, test_data, all_e1_to_multi_e2, train_network, fusion="graph-only",
                                      sim_relations=args.sim_relations)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Options for Commonsense Knowledge Base Completion')

    # General
    parser.add_argument("-d", "--dataset", type=str, required=True,
                        help="dataset to use")
    parser.add_argument("--sim_relations", action='store_true', default=False,
                        help="add similarity edges when constructing graph")
    parser.add_argument("--sim_sim", action='store_true', default=False,
                        help="add sim-sim edges to graph")
    parser.add_argument("--load_model", type=str, default=None, help="Path to model file")
    parser.add_argument("--decoder", type=str, default='ConvTransE', help="decoder used to compute scores")
    parser.add_argument("--n-epochs", type=int, default=200,
                        help="number of minimum training epochs")
    parser.add_argument("--evaluate-every", type=int, default=10,
                        help="perform evaluation every n epochs")
    parser.add_argument("--output_dir", type=str, required=False, default="saved_models",
                        help="output directory to store metrics and model file")
    parser.add_argument("--bert_concat", action='store_true', default=False,
                        help="concat bert embeddings before decoder layer")
    parser.add_argument("--bert_sum", action='store_true', default=False,
                        help="sum bert embeddings before decoder layer")
    parser.add_argument("--bert_mlp", action='store_true', default=False,
                        help="use mlp after concatenated bert+gcn embeddings before decoder layer")
    parser.add_argument("--tying", action='store_true', default=False,
                        help="tie input bert layer to gcn with concatenated tensor before decoding")
    parser.add_argument("--cpu_decoding", action='store_true', default=False,
                        help="perform decoding on cpu")
    parser.add_argument("--eval_only", action='store_true', default=False,
                        help="only evaluate using an existing model")
    parser.add_argument("--write_results", action='store_true', default=False,
                        help="write top-k candidate tuples for evaluation set to file")
    parser.add_argument("--eval-batch-size", type=int, default=500,
                        help="batch size when evaluating")
    parser.add_argument("--gpu", type=int, default=-1,
                        help="gpu")
    parser.add_argument("--no_cuda", action='store_true', default=False,
                        help="prevents using cuda")
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed value")
    parser.add_argument("--debug", action='store_true', default=False,
                        help="use fewer eval instances in debugging mode")

    # GCN
    parser.add_argument("--init_embedding_dim", type=int, default=200,
                        help="embedding dimension of input to GCN")
    parser.add_argument("--input_layer", type=str, default="lookup",
                        help="initialization layer for GCN")
    parser.add_argument("--n-bases", type=int, default=100,
                        help="number of weight blocks for each relation (for RGCN)")
    parser.add_argument("--n-layers", type=int, default=2,
                        help="number of propagation rounds")
    parser.add_argument("--gcn_type", type=str, default="WGCNAttentionLayer",
                        help="type of GCN to be used (class name)")

    # Miscellaneous Hyperparameters
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="learning rate")
    parser.add_argument("--dropout", type=float, default=0.2,
                        help="dropout probability")
    parser.add_argument("--input_dropout", type=float, default=0.2,
                        help="input dropout")
    parser.add_argument("--feature_map_dropout", type=float, default=0.2,
                        help="feature map dropout")
    parser.add_argument("--label_smoothing_epsilon", type=float, default=0.1,
                        help="epsilon for performing label smoothing")
    parser.add_argument("--embedding_dim", type=int, default=200,
                        help="output embedding dimension of GCN")
    parser.add_argument("--n-hidden", type=int, default=200,
                        help="number of hidden units")
    parser.add_argument("--use_bias", action='store_true', default=False,
                        help="use bias")
    parser.add_argument("--regularization", type=float, default=0.1,
                        help="regularization weight")
    parser.add_argument("--grad-norm", type=float, default=1.0,
                        help="norm to clip gradient to")
    parser.add_argument("--graph-batch-size", type=int, default=30000,
                        help="number of edges to sample in each iteration")
    parser.add_argument("--negative-sample", type=int, default=0,
                        help="number of negative samples per positive sample")
    parser.add_argument("--decoder_batch_size", type=int, default=128,
                        help="batch size for decoder")
    parser.add_argument("--layer_norm", action='store_true', default=False,
                        help="use layer normalization on embeddings fed to decoder")

    args = parser.parse_args()
    print(args)
    try:
        main(args)
    except KeyboardInterrupt:
        print('Interrupted')
        # writer.export_scalars_to_json("./all_scalars.json")
        # writer.close()
