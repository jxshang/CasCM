import pickle
import time
from absl import app, flags
from tqdm import tqdm
from graphwave.graphwave import *
from cogdl.models.emb.netsmf import NetSMF
from sparse_matrix_factorization import *
import torch
import community as community_louvain
from collections import defaultdict

# observation and prediction time settings:
# for twitter dataset, we use 3600*24*1 (86400, 1 day) or 3600*24*2 (172800, 2 days) as observation time
#                      we use 3600*24*15 (1296000) as prediction time
# for weibo   dataset, we use 1800 (0.5 hours) or 3600 (1 hour) as observation time
#                      we use 3600*24 (86400, 1 day) as prediction time
# for aps     dataset, we use 365*3 (1095, 3 years) or 365*5+1 (1826, 5 years) as observation time
#                      we use 365*20+5 (7305, 20 years) as prediction time

data_name = 'sample'
# flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('cg_emb_dim', 40, 'dim')
flags.DEFINE_integer('max_seq', 100, 'Max length of cascade sequence.')
flags.DEFINE_integer('num_s', 2, 'Number of s for spectral graph wavelets.')
flags.DEFINE_integer('observation_time', 86400, 'Observation time.')
flags.DEFINE_integer('interval_num', 12, 'interval_num time.')
# paths
# flags.DEFINE_string('data', '../data/sample/', 'Dataset path.')
flags.DEFINE_string('data', '../data/{}/'.format(data_name), 'Processed dataset path.')
# flags.DEFINE_string('data', 'data/twitter/', 'Dataset path.')


def sequence2list(filename, gg, cascade_graph):
    # graphs：dict[cascade_id] = list[list[nodes], time]
    graphs = dict()
    with open(filename, 'r') as f:
        for line in f:
            g = nx.Graph()
            paths = line.strip().split('\t')[:-1]
            graphs[paths[0]] = list()
            for i in range(1, len(paths)):
                nodes = paths[i].split(':')[0]
                time = paths[i].split(':')[1]
                graphs[paths[0]].append([[int(x) for x in nodes.split(',')], int(time)])
                nodes = nodes.split(',')
                if len(nodes) < 2:
                    g.add_node(nodes[-1])
                    gg.add_node(nodes[-1])
                else:
                    g.add_edge(nodes[-1], nodes[-2])
                    gg.add_edge(nodes[-1], nodes[-2])
            cascade_graph[paths[0]] = g
    return graphs, gg, cascade_graph


def read_labels(filename):
    labels = dict()
    with open(filename, 'r') as f:
        for line in f:
            id = line.strip().split('\t')[0]
            labels[id] = line.strip().split('\t')[-1]
    return labels


def write_cascade(graphs, labels, filename, node2com, gg_emb, weight=True):
    """
    Input: cascade graphs, global embeddings
    Output: cascade embeddings, with global embeddings appended
    """
    y_data = list()
    id_data = list()
    cascade_input = list()
    interval_nodes = list()
    total_time = 0

    # for each cascade graph, generate its embeddings via wavelets
    # graphs：dict[cascade_id] = list[list[nodes], time]
    for key, graph in tqdm(graphs.items()):

        start_time = time.time()
        y = int(labels[key])

        node2comm = node2com[key]
        # lists for saving embeddings
        cascade_temp = list()
        t_temp = list()

        # build graph
        g = nx.DiGraph()

        nodes_index = list()
        list_edge = list()
        #node_time = list()
        node_time = {}
        com_id = {}
        t_idx = {}
        cascade_embedding = list()
        global_embedding = list()
        t_o = FLAGS.observation_time

        root_node_added = False
        # add edges into graph
        for path in graph:
            nodes = path[0]
            t = path[1]
            if t >= t_o:
                continue
            if not root_node_added and ('twitter' in FLAGS.data or 'sample' in FLAGS.data):
                nodes_index.extend([nodes[0]])
                t_temp.append(0)
                # node_time.append((nodes[0], {'time': 0}))
                node_time[nodes[0]] = {'time': 0}
                com_id[nodes[0]] = {'com_id': node2comm[str(nodes[0])]}
                root_node_added = True
            if len(nodes) == 1:
                nodes_index.extend([nodes[0]])
                t_temp.append(0)
                node_time[nodes[0]] = {'time': t}
                com_id[nodes[-1]] = {'com_id': node2comm[str(nodes[-1])]}
                continue
            else:
                nodes_index.extend([nodes[-1]])
                t_temp.append(t / t_o)
                #node_time.append((nodes[-1], {'time': t}))
                node_time[nodes[-1]] = {'time': t}
                com_id[nodes[-1]] = {'com_id': node2comm[str(nodes[-1])]}
                
            if weight:
                previous_node_t = node_time.get(nodes[-2], {}).get('time')
                if previous_node_t is None:
                    for i in range(len(nodes) - 2, -1, -1):
                        key = nodes[i]
                        if key in node_time:
                            previous_node_t = node_time[key]['time']
                            if t - previous_node_t != 0:
                                influence = 1 / abs(t - previous_node_t)
                                edge = (key, nodes[-1], influence)
                            else:
                                influence = 0.1
                                edge = (key, nodes[-1], influence)
                            break
                elif t - previous_node_t == 0:
                    influence = 0.1
                    edge = (nodes[-2], nodes[-1], influence)
                else:
                    influence = 1 / abs(t - previous_node_t)
                    edge = (nodes[-2], nodes[-1], influence)  # weighted edge
            else:
                edge = (nodes[-2], nodes[-1],)
            
            list_edge.append(edge)

            if len(com_id) > FLAGS.max_seq:
                break

        if weight:
            g.add_weighted_edges_from(list_edge)
        else:
            g.add_edges_from(list_edge)
        
        if g.number_of_nodes() > len(node_time):
            print(1)
        nx.set_node_attributes(g, node_time)
        nx.set_node_attributes(g, com_id)

        for node in g.nodes():
            g.nodes[node]['influence'] = 0

        for source, target, data in g.edges(data=True):
            g.nodes[source]['influence'] += data['weight']

        # this list is used to make sure the node order of `chi` is same to node order of `cascade`
        nodes_index_unique = list(set(nodes_index)) 
        nodes_index_unique.sort(key=nodes_index.index) 

        # embedding dim check
        d = FLAGS.cg_emb_dim / (2 * FLAGS.num_s)
        if FLAGS.cg_emb_dim % 4 != 0:
            raise ValueError

        chi, _, _ = graphwave_alg(g, np.linspace(0, 100, int(d)),
                                    taus='auto', verbose=False,
                                    nodes_index=nodes_index_unique,
                                    nb_filters=FLAGS.num_s)

        # save embeddings into list
        for node in nodes_index:
            cascade_embedding.append(chi[nodes_index_unique.index(node)])
            #global_embedding.append(gg_emb[id2row[node]])
            global_embedding.append(gg_emb[str(node)])


        # concat node features to node embedding
        if weight:
            cascade_embedding = np.concatenate([np.reshape(t_temp, (-1, 1)),
                                                np.array(cascade_embedding)[:, :],
                                                np.array(global_embedding)[:, :]],
                                               axis=1)
        node_emb = {node: {'emb': cascade_embedding[i].tolist()} for i, node in enumerate(nodes_index_unique)}
        nx.set_node_attributes(g, node_emb)

        #temporal_community_graphs, inter_com_node_counts, inter_community_edges = extract_community_graphs_by_time(g, time_bins)

        cascade_input.append(g)
        y_data.append(y)
        id_data.append(key)

        total_time += time.time() - start_time

    #write concatenated embeddings into file
    with open(filename, 'wb') as f:
        pickle.dump((cascade_input, y_data, id_data), f)


def main(argv):
    time_start = time.time()

    cascade_graph = dict()
    gg = nx.Graph()

    # get the information of nodes/users of cascades
    graph_train, gg, cascade_graph = sequence2list(FLAGS.data + f'train_{FLAGS.observation_time}.txt', gg, cascade_graph)
    graph_val, gg, cascade_graph = sequence2list(FLAGS.data + f'val_{FLAGS.observation_time}.txt', gg, cascade_graph)
    graph_test, gg, cascade_graph = sequence2list(FLAGS.data + f'test_{FLAGS.observation_time}.txt', gg, cascade_graph)

    # get the information of labels of cascades
    label_train = read_labels(FLAGS.data + f'train_{FLAGS.observation_time}.txt')
    label_val = read_labels(FLAGS.data + f'val_{FLAGS.observation_time}.txt')
    label_test = read_labels(FLAGS.data + f'test_{FLAGS.observation_time}.txt')
    node2community = f'{FLAGS.data}full_b{FLAGS.observation_time}_obs_gg.pkl'
    with open(node2community, 'rb') as com_info:
        node2com = pickle.load(com_info)

    print('Generating embeddings of nodes in global graph.')
    model = SparseMatrixFactorization(gg, FLAGS.cg_emb_dim)
    gg_emb = model.pre_factorization(model.matrix, model.matrix)

    print('Processing time: {:.2f}s'.format(time.time()-time_start))
    print('Start writing train set into file.')
    write_cascade(graph_train, label_train, 
                  FLAGS.data + f'train_l{FLAGS.observation_time}_s{FLAGS.max_seq}.pkl', 
                  #FLAGS.data + f'{FLAGS.observation_time}_obs_com.pkl',
                  node2com,             
                  gg_emb)
    print('Start writing val set into file.')
    write_cascade(graph_val, label_val, 
                  FLAGS.data + f'val_l{FLAGS.observation_time}_s{FLAGS.max_seq}.pkl',
                  #FLAGS.data + f'{FLAGS.observation_time}_obs_com.pkl',
                  node2com,
                  gg_emb)
    print('Start writing test set into file.')
    write_cascade(graph_test, label_test, 
                  FLAGS.data + f'test_l{FLAGS.observation_time}_s{FLAGS.max_seq}.pkl', 
                  #FLAGS.data + f'{FLAGS.observation_time}_obs_com.pkl',
                  node2com,
                  gg_emb)
    print('Processing time: {:.2f}s'.format(time.time()-time_start))


if __name__ == '__main__':
    app.run(main)


