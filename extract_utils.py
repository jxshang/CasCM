from collections import defaultdict
from torch_geometric.data import Data
from torch_geometric.utils import subgraph
import torch
import numpy as np
from collections import defaultdict
import math


def extract_community_graphs_by_time(data: Data, time_bins: list, tau_decay=5.0):
    device = time_bins.device
    data = data.to(device)
    node_time = data.node_time
    com_id = data.com_id

    community_graphs = defaultdict(dict)
    community_node_counts = defaultdict(dict)
    community_growth_features = defaultdict(dict)
    inter_community_edge_feats = defaultdict(lambda: defaultdict(list))  
    cascade_growth_features = dict()
    cascade_growth = dict()
    prev_nodes_by_com = defaultdict(set)

    for t_idx in range(len(time_bins) - 1):
        t_start, t_end = time_bins[t_idx], time_bins[t_idx + 1]

        curr_mask = (node_time >= t_start) & (node_time < t_end)
        curr_nodes = torch.where(curr_mask)[0]
        curr_com_ids = com_id[curr_nodes]

        com_to_nodes = defaultdict(set)
        for i, nid in enumerate(curr_nodes.tolist()):
            com = curr_com_ids[i].item()
            com_to_nodes[com].add(nid)

        all_coms = set(com_to_nodes.keys()) | set(prev_nodes_by_com.keys())

        for com in all_coms:
            prev = prev_nodes_by_com[com]
            curr = com_to_nodes.get(com, set())
            total = prev | curr
            if not total:
                continue

            node_idx = torch.tensor(sorted(total), dtype=torch.long, device=device)
            edge_idx, _ = subgraph(node_idx, data.edge_index, relabel_nodes=True)

            sub_data = Data(
                x=data.x[node_idx].to(device),
                edge_index=edge_idx.to(device),
                node_time=node_time[node_idx].to(device),
                com_id=com_id[node_idx].to(device),
            )

            # mask
            mask = torch.zeros_like(node_idx)
            for i, nid in enumerate(node_idx):
                if nid.item() in curr:
                    mask[i] = 1
            sub_data.mask = mask.to(device)

            community_graphs[t_idx][com] = sub_data
            community_node_counts[t_idx][com] = len(curr)
            prev_nodes_by_com[com] = total
            community_growth_features[t_idx][com] = np.log2(len(curr) + 1)

        edge_dir_counts = defaultdict(lambda: defaultdict(int))  
        edge_latest_time = defaultdict(lambda: -1) 

        for i in range(data.edge_index.size(1)):
            u, v = data.edge_index[0, i].item(), data.edge_index[1, i].item()
            t_u, t_v = node_time[u].item(), node_time[v].item()
            if t_u < t_end and t_v < t_end:
                cu, cv = com_id[u].item(), com_id[v].item()
                if cu != cv:
                    edge_dir_counts[(cu, cv)]['count'] += 1
                    edge_latest_time[(cu, cv)] = max(edge_latest_time[(cu, cv)], max(t_u, t_v))

        com_list = list(all_coms)
        for i in range(len(com_list)):
            for j in range(i + 1, len(com_list)):
                cu, cv = com_list[i], com_list[j]
                key = (cu, cv)
                out_uv = edge_dir_counts[(cu, cv)]['count']
                out_vu = edge_dir_counts[(cv, cu)]['count']
                total_count = out_uv + out_vu

                if total_count == 0:
                    continue
                last_time = max(edge_latest_time[(cu, cv)], edge_latest_time[(cv, cu)])
                delta_t = max(0, t_end - last_time)
                decay = math.exp(-delta_t / tau_decay)

                neigh_cu = {com_id[n].item() for n in data.edge_index[1][data.edge_index[0] == cu] if com_id[n].item() != cu}
                neigh_cv = {com_id[n].item() for n in data.edge_index[1][data.edge_index[0] == cv] if com_id[n].item() != cv}
                common_neigh = len(neigh_cu & neigh_cv)
                jaccard = common_neigh / len(neigh_cu | neigh_cv) if (neigh_cu | neigh_cv) else 0.0

                feats = [
                    total_count,
                    np.log2(total_count + 1),
                    decay,
                    out_uv,
                    out_vu,
                    common_neigh,
                    jaccard
                ]
                inter_community_edge_feats[t_idx][key] = feats

        total_new_nodes = sum(len(nodes) for nodes in com_to_nodes.values())
        cascade_growth_features[t_idx] = np.log2(total_new_nodes + 1)
        cascade_growth[t_idx] = total_new_nodes

    return community_graphs, community_node_counts, community_growth_features, inter_community_edge_feats, cascade_growth_features, cascade_growth

