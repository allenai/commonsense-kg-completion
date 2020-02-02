# Main script for experimenting with training on full training graph in an epoch

import argparse
import numpy as np
np.random.seed(42)
import sys
import os
import json
import time
import collections
import torch
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.set_num_threads(16)

import torch.nn as nn
import random
random.seed(42)

from collections import Counter
#from dgl.contrib.data import load_data

from model import LinkPredict
from reader import AtomicTSVReader, ConceptNetTSVReader, ConceptNetFullReader, FB15kReader

import reader_utils
from src import utils

from dgl.contrib.data import load_data

#from graphviz import Digraph
#from torchviz import make_dot


def load_atomic_data(dataset, sim_relations):
    train_network = AtomicTSVReader(dataset)
    dev_network = AtomicTSVReader(dataset)
    test_network = AtomicTSVReader(dataset)

    #train_network.read_network(data_dir="data/atomic-original/", split="train")
    train_network.read_network(data_dir="data/atomic/", split="train")
    train_network.print_summary()
    node_list = train_network.graph.iter_nodes()
    node_degrees = [node.get_degree() for node in node_list]
    degree_counter = Counter(node_degrees)
    avg_degree = sum([k*v for k,v in degree_counter.items()]) / sum([v for k,v in degree_counter.items()])
    print("Average Degree: ", avg_degree)

    #dev_network.read_network(data_dir="data/atomic-original/", split="valid", train_network=train_network)
    #test_network.read_network(data_dir="data/atomic-original/", split="test", train_network=train_network)

    dev_network.read_network(data_dir="data/atomic/", split="valid", train_network=train_network)
    test_network.read_network(data_dir="data/atomic/", split="test", train_network=train_network)
    word_vocab = train_network.graph.node2id

    #node_names = []
    #for node in train_network.graph.iter_nodes():??????
    #    node_names.append(node.name + "\n")
    #with open("atomic_node_names.txt", 'w') as f:
    #    f.writelines([reader_utils.preprocess_atomic_sentence(n.replace("-", " ??????")) for n in node_names])
    #import sys; sys.exit(0)
 
    # Add sim nodes
    if sim_relations:
        print("Adding sim edges..")
        train_network.add_sim_edges_bert()

    train_data, _ = reader_utils.prepare_batch_dgl(word_vocab, train_network, train_network)
    test_data, test_labels = reader_utils.prepare_batch_dgl(word_vocab, test_network, train_network)
    valid_data, valid_labels = reader_utils.prepare_batch_dgl(word_vocab, dev_network, train_network)
    return len(train_network.graph.nodes), train_data, valid_data, test_data, len(train_network.graph.relations), valid_labels, test_labels, train_network


def load_fb15k_data(dataset, sim_relations, keep_fraction=100):
    train_network = FB15kReader(dataset)
    dev_network = FB15kReader(dataset)
    test_network = FB15kReader(dataset)

    train_network.read_network(data_dir="data/FB15k-237/", keep_fraction=keep_fraction, split="train")
    train_network.print_summary()
    node_list = train_network.graph.iter_nodes()
    node_degrees = [node.get_degree() for node in node_list]
    degree_counter = Counter(node_degrees)
    avg_degree = sum([k*v for k,v in degree_counter.items()]) / sum([v for k,v in degree_counter.items()])
    print("Average Degree: ", avg_degree)

    dev_network.read_network(data_dir="data/FB15k-237/", split="valid", train_network=train_network)
    test_network.read_network(data_dir="data/FB15k-237/", split="test", train_network=train_network)
    word_vocab = train_network.graph.node2id

    train_data, _ = reader_utils.prepare_batch_dgl(word_vocab, train_network, train_network)
    test_data, test_labels = reader_utils.prepare_batch_dgl(word_vocab, test_network, train_network)
    valid_data, valid_labels = reader_utils.prepare_batch_dgl(word_vocab, dev_network, train_network)
    return len(train_network.graph.nodes), train_data, valid_data, test_data, len(train_network.graph.relations), valid_labels, test_labels, train_network


def load_cn_data(dataset, sim_relations, eval_accuracy=False):
    train_network = ConceptNetTSVReader(dataset)
    dev1_network = ConceptNetTSVReader(dataset)
    dev2_network = ConceptNetTSVReader(dataset)
    test_network = ConceptNetTSVReader(dataset)

    positive_only = not eval_accuracy

    train_network.read_network(data_dir="data/conceptnet/", split="train")

    train_network.print_summary()
    #node_list = train_network.graph.iter_nodes()
    #node_degrees = [node.get_degree() for node in node_list]
    #degree_counter = Counter(node_degrees)
    #avg_degree = sum([k*v for k,v in degree_counter.items()]) / sum([v for k,v in degree_counter.items()])
    #print("Average Degree: ", avg_degree)

    dev1_network.read_network(data_dir="data/conceptnet/", split="valid1", train_network=train_network, positive_only=positive_only)
    dev2_network.read_network(data_dir="data/conceptnet/", split="valid2", train_network=train_network, positive_only=positive_only)
    test_network.read_network(data_dir="data/conceptnet/", split="test", train_network=train_network, positive_only=positive_only)

    # Add sim nodes
    if sim_relations:
        print("Adding sim edges..")
        train_network.add_sim_edges_bert()

    #word_vocab, word_freqs = reader_utils.create_vocab(train_network)
    word_vocab = train_network.graph.node2id
    train_data, _ = reader_utils.prepare_batch_dgl(word_vocab, train_network, train_network)
    test_data, test_labels = reader_utils.prepare_batch_dgl(word_vocab, test_network, train_network)
    valid1_data, valid1_labels = reader_utils.prepare_batch_dgl(word_vocab, dev1_network, train_network)
    valid2_data, valid2_labels = reader_utils.prepare_batch_dgl(word_vocab, dev2_network, train_network)

    return len(train_network.graph.nodes), train_data, valid1_data, test_data, len(train_network.graph.relations), valid1_labels, test_labels, train_network


def load_cn_full_data(dataset, sim_relations):

    train_network = ConceptNetFullReader(dataset)
    dev_network = ConceptNetFullReader(dataset)
    test_network = ConceptNetFullReader(dataset)

    train_network.read_network(data_dir="/data/conceptnet-5.6/", split="train")
    train_network.print_summary()
    node_list = train_network.graph.iter_nodes()
    node_degrees = [node.get_degree() for node in node_list]
    degree_counter = Counter(node_degrees)
    avg_degree = sum([k*v for k,v in degree_counter.items()]) / sum([v for k,v in degree_counter.items()])
    print("Average Degree: ", avg_degree)

    dev_network.read_network(data_dir="/data/conceptnet-5.6/", split="valid", train_network=train_network)
    test_network.read_network(data_dir="/data/conceptnet-5.6/", split="test", train_network=train_network)

    #node_names = []
    #for node in train_network.graph.iter_nodes():
    #    node_names.append(node.name)
    #with open("cn-full_node_names.txt", 'w') as f:
    #    f.writelines([n.split("/")[-2].replace("_", " ")+"\n" for n in node_names if n not in string.punctuation and not n.isdigit()])
    #import sys; sys.exit(0)

    if sim_relations:
        print("Adding sim edges..")
        train_network.add_sim_edges_bert()

    #word_vocab, word_freqs = reader_utils.create_vocab(train_network)
    word_vocab = train_network.graph.node2id
    train_data, _ = reader_utils.prepare_batch_dgl(word_vocab, train_network, train_network)
    test_data, test_labels = reader_utils.prepare_batch_dgl(word_vocab, test_network, train_network)
    valid_data, valid_labels = reader_utils.prepare_batch_dgl(word_vocab, dev_network, train_network)

    return len(train_network.graph.nodes), train_data, valid_data, test_data, len(train_network.graph.relations), valid_labels, test_labels, train_network

def main(args):

    #visualize_graph_degrees()
    
    # load graph data
    if args.dataset == "FB15K-237":
        data = load_data(args.dataset)
        num_nodes = data.num_nodes
        train_data = data.train
        valid_data = data.valid
        test_data = data.test
        num_rels = data.num_rels
        train_network = None

        # Deletion experiment
        delete_fraction = args.delete_fraction
        delete_indices = random.sample(range(len(train_data)), int(delete_fraction * len(train_data)))
        train_data = np.array([tup for i, tup in enumerate(train_data) if i not in delete_indices])
        selected_nodes = train_data[:,0].tolist() + train_data[:,2].tolist()
        num_nodes = len(set(selected_nodes))
        print(len(train_data))
        print(num_nodes)

        # Store entity-wise dicts for filtered metrics
        all_tuples = train_data.tolist() + valid_data.tolist() + test_data.tolist()
   
     # print("Graph Density: %f" % (len(train_data) / (num_nodes * (num_nodes - 1))))
   
    elif args.dataset == "atomic":
        num_nodes, train_data, valid_data, test_data, num_rels, valid_labels, test_labels, train_network = load_atomic_data(args.dataset, args.sim_relations)
        all_tuples = train_data.tolist() + valid_data.tolist() + test_data.tolist()
    elif args.dataset == "conceptnet":
        num_nodes, train_data, valid_data, test_data, num_rels, valid_labels, test_labels, train_network = load_cn_data(args.dataset, args.sim_relations, args.eval_accuracy)
        all_tuples = train_data.tolist() + valid_data.tolist() + test_data.tolist()
    elif args.dataset == "conceptnet-5.6":
        num_nodes, train_data, valid_data, test_data, num_rels, valid_labels, test_labels, train_network = load_cn_full_data(args.dataset, args.sim_relations)
        all_tuples = train_data.tolist() + valid_data.tolist() + test_data.tolist()
    elif args.dataset == "FB15k-237":
        num_nodes, train_data, valid_data, test_data, num_rels, valid_labels, test_labels, train_network = load_fb15k_data(args.dataset, args.sim_relations, keep_fraction=100-int(args.delete_fraction))
        #delete_fraction = args.delete_fraction
        #delete_indices = random.sample(range(len(train_data)), int(delete_fraction * len(train_data)))
        #train_data = np.array([tup for i, tup in enumerate(train_data) if i not in delete_indices])
        #selected_nodes = train_data[:,0].tolist() + train_data[:,2].tolist()
        #num_nodes = len(set(selected_nodes))
        #print("Remaining edges: ", len(train_data))
        #print("Remaining nodes: " , num_nodes)
        all_tuples = train_data.tolist() + valid_data.tolist() + test_data.tolist()
    else:
        raise ValueError("Invalid Option for Dataset")

    
    # for filtered ranking 
    all_e1_to_multi_e2, all_e2_to_multi_e1 = reader_utils.create_entity_dicts(all_tuples, num_rels, args.sim_relations)
    # for training
    train_e1_to_multi_e2, train_e2_to_multi_e1 = reader_utils.create_entity_dicts(train_data.tolist(), num_rels, args.sim_relations)

    # check cuda
    use_cuda = torch.cuda.is_available()
    #use_cuda = False
    if use_cuda and not args.no_cuda:
       torch.cuda.set_device(args.gpu)

    cpu_decoding = args.cpu_decoding

    # create model
    model = LinkPredict(num_nodes,
                        num_rels,
                        args,
                        use_cuda=use_cuda)


    # validation and testing triplets
    valid_data = torch.LongTensor(valid_data)
    test_data = torch.LongTensor(test_data)

    # remove cpu eval for now
    cpu_eval = True if args.dataset == "atomic" else False

    if use_cuda and not args.no_cuda and not cpu_eval:
        valid_data = valid_data.cuda()
        test_data = test_data.cuda()

    # build test graph
    if args.sim_sim and args.sim_relations:
        graph_train_data = utils.sim_sim_connect(train_data, train_data, num_rels)
    else:
        graph_train_data = train_data 


    test_graph, test_rel, test_norm = utils.build_test_graph(
        num_nodes, num_rels, graph_train_data)
    test_deg = test_graph.in_degrees(
                range(test_graph.number_of_nodes())).float().view(-1,1)
    test_node_id = torch.arange(0, num_nodes, dtype=torch.long).view(-1, 1)
    test_rel = torch.from_numpy(test_rel).view(-1, 1)
    test_norm = torch.from_numpy(test_norm).view(-1, 1)
    if use_cuda and not args.no_cuda:
        test_node_id = test_node_id.cuda()
        test_norm = test_norm.cuda()
        test_rel = test_rel.cuda()

    test_graph.ndata.update({'id': test_node_id, 'norm': test_norm})
    test_graph.edata['type'] = test_rel

    if use_cuda and not args.no_cuda:
        #model = nn.DataParallel(model, device_ids=[0,1])
        model = model.cuda()

    name = '_model_state.pth'
    name = "_" + args.model + "_" + args.gcn_type + "_" + args.decoder + name

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

    if args.delete_fraction:
        keep = 100 - int(args.delete_fraction)
        name = "_" + str(keep) +  name

    #name = str(datetime.now().time()).split(".")[0] + name

    model_state_file = args.dataset + name
    model_state_file = os.path.join(args.output_dir, model_state_file)

    #writer = SummaryWriter("runs/" + model_state_file.replace(".pth",".log"))

    
    if args.eval_only:
        if args.model_name:
            model_state_file=args.model_name
        checkpoint = torch.load(model_state_file)
        #if use_cuda:
        #    model.cpu() # test on CPU
        model.eval()
        model.load_state_dict(checkpoint['state_dict'])
        #model.rgcn.layers[-1].device = torch.device("cpu")
        print(model)

        
        if use_cuda and not args.no_cuda and cpu_eval:
            model.cpu()
            test_graph.ndata['id'] = test_graph.ndata['id'].cpu()
            test_graph.ndata['norm'] = test_graph.ndata['norm'].cpu()
            test_graph.edata['type'] = test_graph.edata['type'].cpu()
            model.decoder.no_cuda = True    

        if args.dataset != "atomic" and args.dataset != "conceptnet":
            valid_labels = None
            test_labels = None
        else:
            valid_labels = torch.LongTensor(valid_labels)
            test_labels = torch.LongTensor(test_labels)

        if args.eval_accuracy:
            threshold = utils.evaluate_accuracy(test_graph, model, valid_data, num_nodes, labels=valid_labels, network=train_network,
                                                eval_bz=args.eval_batch_size)
            utils.evaluate_accuracy(test_graph, model, test_data, num_nodes, labels=test_labels, network=train_network, threshold=threshold,
                                    eval_bz=args.eval_batch_size)

        else:

            print("===========DEV============")
            mrr = utils.ranking_and_hits(test_graph, model, valid_data, all_e1_to_multi_e2,
                                         train_network, comb="graph", sim_relations=args.sim_relations,
                                         write_results=args.write_results, debug=args.debug)

            print("===========TEST============")
            mrr = utils.ranking_and_hits(test_graph, model, test_data, all_e1_to_multi_e2,
                                         train_network, comb="graph", sim_relations=args.sim_relations, debug=args.debug)
            if args.dataset=="atomic":         
                mrr = utils.ranking_and_hits(test_graph, model, test_data, all_e1_to_multi_e2, train_network, comb="graph", sim_relations=args.sim_relations, no_nones=True)

            #mrr = utils.evaluate(test_graph, model, valid_data, all_e1_to_multi_e2, num_nodes, valid_labels, train_network,
            #                            hits=[1, 3, 10], eval_bz=args.eval_batch_size)
            #mrr = utils.evaluate(test_graph, model, test_data, all_e1_to_multi_e2, num_nodes, test_labels, train_network,
            #                            hits=[1, 3, 10], eval_bz=args.eval_batch_size)

        sys.exit(0)


    # build adj list and calculate degrees for sampling
    # adj_list, degrees, sparse_adj_matrix, rel = utils.get_adj_and_degrees(num_nodes, num_rels, train_data)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    #optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    if os.path.isfile(model_state_file):
        print(model_state_file)
        overwrite = input('Model already exists. Overwrite? Y = yes, N = no or C=continue training\n')
        if overwrite.lower() == 'n':
            print("Quitting")
            sys.exit(0)
        elif overwrite.lower() == 'c':
            print("Loading model")
            checkpoint = torch.load(model_state_file)
            model.load_state_dict(checkpoint['state_dict'])
            print("Continuing training")
        elif overwrite.lower() != 'y':
            raise ValueError("Invalid Option")


    forward_time = []
    backward_time = []

    # training loop
    print("Starting training...")
    epoch = 0
    best_mrr = 0


 
    while True:
        model.train()
        epoch += 1

        g = test_graph
        #data = graph_train_data
        #data = torch.from_numpy(data)
        #labels = None

        #if use_cuda and not args.no_cuda:
        #    data = data.cuda()
 
        batch_size = args.decoder_batch_size
        e1_keys = list(train_e1_to_multi_e2.keys())
        random.shuffle(e1_keys)

        cum_loss = 0.0
        start_time = time.time()

        for i in range(0, len(e1_keys), batch_size):
            optimizer.zero_grad()

            graph_embeddings = model.get_graph_embeddings(g, epoch)
            model.decoder.module.cur_embedding = graph_embeddings

            batch = e1_keys[i : i + batch_size]
            e1 = torch.LongTensor([elem[0] for elem in batch])
            rel = torch.LongTensor([elem[1] for elem in batch])
            e2 = [train_e1_to_multi_e2[elem] for elem in batch]

            if use_cuda and not args.no_cuda and not cpu_decoding:
                target = torch.cuda.FloatTensor(len(batch), num_nodes).fill_(0)
                e1 = e1.cuda()
                rel = rel.cuda()
            else:
                target = torch.zeros((len(batch), num_nodes))

            for j, inst in enumerate(e2):
                target[j, inst] = 1.0
  
            target = ((1.0-args.label_smoothing_epsilon)*target) + (1.0/target.size(1))

            #if use_cuda and not args.no_cuda and not cpu_decoding:
            #    target = target.cuda()
   
            if cpu_decoding: 
                graph_embeddings = graph_embeddings.cpu()
                model.decoder.cpu()
                model.decoder.no_cuda = True

            t0 = time.time()
            loss = model.get_score(e1, rel, target, graph_embeddings)
            loss = torch.mean(loss)
            cum_loss += loss.cpu().item()
            t1 = time.time()
            #loss.backward(retain_graph=True)

            loss.backward() 
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm) # clip gradients
            optimizer.step()
       
            t2 = time.time()

            forward_time.append(t1 - t0)
            backward_time.append(t2 - t1)
              
            #torch.cuda.empty_cache()
            del graph_embeddings, target, batch, loss, e1, rel, e2
            #gc.collect()
            
        print("Epoch {:04d} | Loss {:.4f} | Best MRR {:.4f} | Forward {:.4f}s | Backward {:.4f}s".
              format(epoch, cum_loss, best_mrr, forward_time[-1], backward_time[-1]))
        print("Total Time:", time.time() - start_time)
        #writer.add_scalar('data/loss', cum_loss , epoch)


        # Save model every 100 epochs
        #if epoch+1%100==0:
        #    print("saving current model..")
        #    torch.save({'state_dict': model.state_dict(), 'epoch': epoch},
        #                 model_state_file)

        # validation
        if epoch % args.evaluate_every == 0:
            # perform validation on CPU because full graph is too large
            if use_cuda and not args.no_cuda and cpu_eval:
                model.cpu()
                test_graph.ndata['id'] = test_graph.ndata['id'].cpu()
                test_graph.ndata['norm'] = test_graph.ndata['norm'].cpu()
                test_graph.edata['type'] = test_graph.edata['type'].cpu()
                model.decoder.no_cuda = True    

            model.eval()
            print("start eval")

            print("===========DEV============")
            mrr = utils.ranking_and_hits(test_graph, model, valid_data, all_e1_to_multi_e2, train_network, comb="graph", sim_relations=args.sim_relations, debug=True)

            print("===========TEST===========")
            test_mrr = utils.ranking_and_hits(test_graph, model, test_data, all_e1_to_multi_e2, train_network, comb="graph", sim_relations=args.sim_relations, debug=True)

            #mrr = utils.evaluate(test_graph, model, valid_data, e1_to_multi_e2, num_nodes, labels, train_network,
            #                     hits=[1, 3, 10], eval_bz=args.eval_batch_size)
            #writer.add_scalar('data/mrr', mrr, epoch)


            #mrr = utils.evaluate(test_graph, model, test_data, num_nodes, labels, train_network,
            #                     hits=[1, 3, 10], eval_bz=args.eval_batch_size)
            # save best model
            # torch.save({'state_dict': model.state_dict(), 'epoch': epoch},
            #                model_state_file)
            if mrr < best_mrr:
                if epoch >= args.n_epochs:
                    break
            else:
                best_mrr = mrr
                torch.save({'state_dict': model.state_dict(), 'epoch': epoch},
                           model_state_file)

            metrics = {"best_mrr": best_mrr,
                       "cum_loss": cum_loss
                      }

            with open(os.path.join(args.output_dir, 'metrics.json'), 'w') as f:
                f.write(json.dumps(metrics))

            if use_cuda and not args.no_cuda:
                model.cuda()
                test_graph.ndata['id'] = test_graph.ndata['id'].cuda()
                test_graph.ndata['norm'] = test_graph.ndata['norm'].cuda()
                test_graph.edata['type'] = test_graph.edata['type'].cuda()
                model.decoder.no_cuda = False

    print("training done")
    print("Mean forward time: {:4f}s".format(np.mean(forward_time)))
    print("Mean Backward time: {:4f}s".format(np.mean(backward_time)))

    #writer.export_scalars_to_json("./all_scalars.json")
    #writer.close()

    print("\nstart testing")

    # use best model checkpoint
    checkpoint = torch.load(model_state_file)
    #if use_cuda:
    #    model.cpu() # test on CPU
    model.eval()
    model.load_state_dict(checkpoint['state_dict'])
    print("Using best epoch: {}".format(checkpoint['epoch']))
    
    mrr = utils.ranking_and_hits(test_graph, model, test_data, all_e1_to_multi_e2, train_network, comb="graph", sim_relations=args.sim_relations)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Machine Commonsense Completion')
    parser.add_argument("--dropout", type=float, default=0.2,
            help="dropout probability")
    parser.add_argument("--input_dropout", type=float, default=0.2,
            help="input dropout")
    parser.add_argument("--feature_map_dropout", type=float, default=0.2,
            help="feature map dropout")
    parser.add_argument("--label_smoothing_epsilon", type=float, default=0.1,
            help="epsilon for performing label smoothing over target")
    parser.add_argument("--init_embedding_dim", type=int, default=200,
            help="init embedding dimension")
    parser.add_argument("--embedding_dim", type=int, default=200,
            help="embedding dimension")
    parser.add_argument("--n-hidden", type=int, default=200,
            help="number of hidden units")
    parser.add_argument("--use_bias", action='store_true', default=False,
            help="use bias")
    parser.add_argument("--gpu", type=int, default=-1,
            help="gpu")
    parser.add_argument("--no_cuda", action='store_true', default=False,
            help="prevents using cuda")
    parser.add_argument("--sim_relations", action='store_true', default=False,
            help="add sim edges to graph")
    parser.add_argument("--sim_sim", action='store_true', default=False,
            help="add sim-sim edges to graph")
    parser.add_argument("--model", type=str, default='RGCN', 
            help="model architecture")
    parser.add_argument("--decoder", type=str, default='ConvTransE', 
            help="decoder used to compute scores")
    parser.add_argument("--lr", type=float, default=1e-4,
            help="learning rate")
    parser.add_argument("--input_layer", type=str, default="lookup",
            help="initialization layer for rgcn")
    parser.add_argument("--n-bases", type=int, default=100,
            help="number of weight blocks for each relation")
    parser.add_argument("--n-layers", type=int, default=2,
            help="number of propagation rounds")
    parser.add_argument("--n-epochs", type=int, default=200,
            help="number of minimum training epochs")
    parser.add_argument("-d", "--dataset", type=str, required=True,
            help="dataset to use")
    parser.add_argument("--eval_only", action='store_true', default=False,
            help="only evaluate using an existing model")
    parser.add_argument("--write_results", action='store_true', default=False,
            help="write topk candidate tuples for evaluation set to file")
    parser.add_argument("--eval_accuracy", action='store_true', default=False,
            help="evaluate binary classification accuracy")
    parser.add_argument("--eval-batch-size", type=int, default=500,
            help="batch size when evaluating")
    parser.add_argument("--regularization", type=float, default=0.05,
            help="regularization weight")
    parser.add_argument("--grad-norm", type=float, default=1.0,
            help="norm to clip gradient to")
    parser.add_argument("--graph-batch-size", type=int, default=30000,
            help="number of edges to sample in each iteration")
    parser.add_argument("--graph-split-size", type=float, default=0.5,
            help="portion of edges used as positive sample")
    parser.add_argument("--negative-sample", type=int, default=10,
            help="number of negative samples per positive sample")
    parser.add_argument("--evaluate-every", type=int, default=10,
            help="perform evaluation every n epochs")
    parser.add_argument("--decoder_batch_size", type=int, default=128,
            help="batch size for decoder")           
    parser.add_argument("--model_name", type=str, required=False, default=None,
            help="model to load")
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
    parser.add_argument("--pretrained_gcn", action='store_true', default=False,
           help="use mlp after concatenated bert+gcn embeddings before decoder layer")
    parser.add_argument("--multi_step", action='store_true', default=False,
           help="multi-step classification between nones and remaining entities, for atomic")
    parser.add_argument("--layer_norm", action='store_true', default=False,
           help="use layer normalization on embeddings fed to decoder")
    parser.add_argument("--gcn_type", type=str, default="WGCNAttentionLayer",
           help="type of GCN to be used")
    parser.add_argument("--delete_fraction", type=float, default=0.0,
            help="fraction of edges to delete")
    parser.add_argument("--use_triplet_loss", action='store_true', default=False,
           help="use triplet loss")
    parser.add_argument("--debug", action='store_true', default=False,
           help="use fewer eval instances in debugging mode")

    args = parser.parse_args()
    print(args)
    try:
        main(args)
    except KeyboardInterrupt:
        print('Interrupted')
        #writer.export_scalars_to_json("./all_scalars.json")
        #writer.close()

