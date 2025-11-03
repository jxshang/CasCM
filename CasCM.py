from collections import defaultdict
import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch
from torch_geometric.nn import global_mean_pool
import torch.nn.functional as F
import math

class MultiHeadAttentionPooling(nn.Module):
    def __init__(self, in_dim, head_dim, num_heads, dropout=0.1):
        super().__init__()
        self.in_dim = in_dim
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.total_head_dim = head_dim * num_heads

        self.q_proj = nn.Linear(in_dim, self.total_head_dim)
        self.k_proj = nn.Linear(in_dim, self.total_head_dim)
        self.v_proj = nn.Linear(in_dim, self.total_head_dim)

        self.fusion_mlp = nn.Sequential(
            nn.Linear(self.total_head_dim, in_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.norm = nn.LayerNorm(in_dim)

    def forward(self, h_comm_b):  # h_comm_b: [num_comm, D]
        if h_comm_b.size(0) == 0:
            return torch.zeros(self.in_dim, device=h_comm_b.device)

        Q = self.q_proj(h_comm_b).view(-1, self.num_heads, self.head_dim)  # [N, H, d]
        K = self.k_proj(h_comm_b).view(-1, self.num_heads, self.head_dim)  # [N, H, d]
        V = self.v_proj(h_comm_b).view(-1, self.num_heads, self.head_dim)  # [N, H, d]

        scores = torch.einsum('nhd,mhd->hnm', Q, K) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)  # [H, N, N]

        attn_output = torch.einsum('hnm,mhd->nhd', attn_weights, V).mean(dim=0)  # [N, H, d] → mean over N → [H, d]
        attn_output = attn_output.flatten()  # [H * d]

        fused = self.fusion_mlp(attn_output)  # [D]

        out = self.norm(fused + h_comm_b.mean(dim=0))  # [D]
        return out


class CommunityTemporalModel(nn.Module):
    def __init__(self, args, gcn, gat, gru, num_time_steps, device, use_gcn=True, use_gat=True, use_gru=True):
        super().__init__()
        self.gcn = gcn
        self.gat = gat
        self.gru = gru
        self.use_gcn = use_gcn
        self.use_gat = use_gat
        self.use_gru = use_gru
        self.num_time_steps = num_time_steps
        self.cascade_growth_dim = args.cascade_growth_dim
        self.community_growth_dim = args.community_growth_dim
        self.com_gru_hidden_dim = args.gru_hidden_dim
        self.cas_gru_hidden_dim = args.cas_gru_hidden_dim
        self.device = device
        self.edge_attr_dim = args.edge_feat_dim
        self.attn_heads_num = 4

        self.com_growth_proj = nn.Sequential(
                                        nn.Linear(1, 8),
                                        nn.ReLU(),
                                        nn.Linear(8, self.community_growth_dim),
                                        nn.ReLU(),
                                    )

        self.cascade_growth_proj = nn.Sequential(
                                                nn.Linear(1, 8),
                                                nn.ReLU(),
                                                nn.Linear(8, self.cascade_growth_dim),
                                                nn.ReLU(),
        )

        if self.use_gru:
            self.bigru_output_layer = nn.Sequential(
                nn.Linear(gru.gru.hidden_size * 2, 128),
                nn.ReLU(),   
                nn.Linear(128, 64), 
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.ReLU()
            )

            self.embedding_dim = args.gat_out_dim + self.community_growth_dim

            self.multihead_attn_pool = MultiHeadAttentionPooling(in_dim=self.embedding_dim, head_dim=16, num_heads=self.attn_heads_num)

            self.cascade_gru = nn.GRU(input_size=self.embedding_dim + self.cascade_growth_dim, hidden_size=self.cas_gru_hidden_dim, batch_first=True, bidirectional=True)
            self.cascade_output_layer = nn.Sequential(
                nn.Linear(self.cas_gru_hidden_dim*2, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.ReLU()
            )

            self.cas_inter_output_layer = nn.Sequential(
                nn.Linear(self.cas_gru_hidden_dim*2, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.ReLU()
            )
            
        else:
            dim = gat.gat.out_channels if use_gat else gcn.layer2.out_channels if use_gcn else None
            self.output_layer = nn.Sequential(
                nn.Linear(dim, 64),
                nn.ReLU(),
                nn.Linear(64, 1))
            self.embedding_dim = args.gat_out_dim + 1 if use_gat else gcn.layer2.out_channels

        

        self.bigru_weight_generator = nn.Sequential(
            nn.Linear(self.com_gru_hidden_dim*2, self.com_gru_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.com_gru_hidden_dim, 1), 
            nn.ReLU()
        )


        self.bigru_growth_predictor = nn.Sequential(
            nn.Linear(self.com_gru_hidden_dim*2, self.com_gru_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.com_gru_hidden_dim, 1),
            nn.ReLU()
        )

    def encode_community_graphs(self, community_graphs_all):    
        if not community_graphs_all:
            return None, []

        community_graphs_all = [g.to(self.device) for g in community_graphs_all]
        batch = Batch.from_data_list(community_graphs_all)

        x = batch.x
        edge_index = batch.edge_index
        is_new = batch.mask

        if self.use_gcn:
            x = self.gcn(x, edge_index, is_new)
        return global_mean_pool(x, batch.batch)

    def construct_gat_graphs(self, comm_emb_all, sample_comm_ids_list, inter_edges_list, num_comms_per_sample, t):
        splits = torch.split(comm_emb_all, num_comms_per_sample)
        gat_graphs = []

        for b, (comm_ids, comm_emb) in enumerate(zip(sample_comm_ids_list, splits)):
            if not comm_ids:
                continue

            inter_edges = inter_edges_list[b].get(t, {})
            id2idx = {cid: idx for idx, cid in enumerate(comm_ids)}
            edges, weights = [], []

            for (u, v), w in inter_edges.items():
                if u in id2idx and v in id2idx:
                    edges.append([id2idx[u], id2idx[v]])
                    weights.append(w)

            if len(edges) > 0:
                edge_index = torch.tensor(edges, dtype=torch.long, device=comm_emb.device).T
                edge_attr = torch.tensor(weights, dtype=torch.float, device=comm_emb.device)
            else:
                edge_index = torch.empty((2, 0), dtype=torch.long, device=comm_emb.device)
                edge_attr = torch.empty((0, self.edge_attr_dim), dtype=torch.float, device=comm_emb.device)
            gat_graphs.append(Data(x=comm_emb, edge_index=edge_index, edge_attr=edge_attr))

        return gat_graphs


    
    def organize_community_embeddings_over_time(self, h_comm, comm_ids_per_sample, comm_counts, batch_size):
        offset = 0
        emb_dict = [defaultdict(list) for _ in range(batch_size)]
        for b in range(batch_size):
            comm_ids = comm_ids_per_sample[b]
            for i, cid in enumerate(comm_ids):
                emb = h_comm[offset + i]
                emb_dict[b][cid].append(emb)
            offset += len(comm_ids)
        return emb_dict

    def process_community_sequences_with_bigru(self, emb_dict, max_comm_num):
        batch_size = len(emb_dict)

        emb_dim = self.embedding_dim
        time_steps = self.num_time_steps

        h_seq_all = []
        comm_masks = torch.zeros((batch_size, max_comm_num), device=self.device) # [B, C]

        comm_id_order = []  

        for b in range(batch_size):
            comms = emb_dict[b]
            h_seq_b = []
            comm_ids = []
            for cid, seq in comms.items():
                seq_tensor = torch.stack(seq, dim=0)  # [T, D]
                h_seq_b.append(seq_tensor)
                comm_ids.append(cid)

            comm_id_order.append(comm_ids)
            comm_num = len(h_seq_b)
            comm_masks[b, :comm_num] = 1  
            
            if comm_num < max_comm_num:
                pad_len = max_comm_num - comm_num
                h_seq_b += [torch.zeros(time_steps, emb_dim, device=self.device)] * pad_len
            h_seq_b = torch.stack(h_seq_b, dim=0)  # [C, T, D]
            h_seq_all.append(h_seq_b)
        h_seq_all = torch.stack(h_seq_all, dim=0)  # [B, C, T, D]
        B, C, T, D = h_seq_all.shape # D-->gat_out_dim+growth_dim = gru_input_dim
        h_seq_all = h_seq_all.view(B * C, T, D)  # [B*C, T, D]

        out, _ = self.gru(h_seq_all) # [B*C, T, hidden_size*2]
        pred_growth = self.bigru_growth_predictor(out).squeeze(-1)  # [B*C, T] 
        return pred_growth.view(B, C, T), comm_id_order
    
    def aggregate_with_learned_weights(self, community_embeds, community_preds, comm_masks):
        
        B, C, D = community_embeds.size()

        raw_scores = self.bigru_weight_generator(community_embeds).squeeze(-1)  # [B, C]
        raw_scores = raw_scores.masked_fill(comm_masks == 0, float('-inf'))
        weights = F.softmax(raw_scores, dim=1)  # [B, C]

        weighted_preds = community_preds * weights  # [B, C]
        agg_preds = weighted_preds.sum(dim=1)  # [B]
        return agg_preds

    def forward(self, community_graphs_batch, community_growth, community_edges, cascade_growth):
        batch_size = len(community_graphs_batch)
        all_time_emb_dict = [defaultdict(lambda: [torch.zeros(self.embedding_dim, device=self.device) 
                                                  for _ in range(self.num_time_steps)])
                                                  for _ in range(batch_size)] # [batch_size,num_com,T,d]
        max_comm_num = 0
        cascade_embeds_over_time = [[] for _ in range(batch_size)] # list[batch_size, T, D]

        for t in range(self.num_time_steps):
            all_graphs = []
            comm_ids_per_sample = []
            comm_counts = []

            for b in range(batch_size):
                graphs = community_graphs_batch[b].get(t, {})
                comm_ids = list(graphs.keys())
                comms = [graphs[cid] for cid in comm_ids]
                all_graphs.extend(comms) 
                comm_ids_per_sample.append(comm_ids)
                comm_counts.append(len(comms))          
            if sum(comm_counts) == 0:
                for b in range(batch_size):
                    cascade_embeds_over_time[b].append(torch.zeros(self.embedding_dim, device=self.device))
                continue

            node_emb_all = self.encode_community_graphs(all_graphs)

            if self.use_gat > 0:
                gat_graphs = self.construct_gat_graphs(node_emb_all, comm_ids_per_sample, community_edges, comm_counts, t)
                if len(gat_graphs) < 1:
                    print(1)
                    continue
                gat_batch = Batch.from_data_list(gat_graphs)
                h_comm = self.gat(gat_batch.x, gat_batch.edge_index, edge_attr=gat_batch.edge_attr)
                growth_feats_tensor = torch.tensor([community_growth[b][t].get(cid) for b in range(batch_size) for cid in comm_ids_per_sample[b]], 
                                                   dtype=torch.float32, device=h_comm.device).unsqueeze(1)  # [N, 1]                
                growth_emb = self.com_growth_proj(growth_feats_tensor)
                h_comm = torch.cat([h_comm, growth_emb], dim=-1)  

            else:
                h_comm = node_emb_all

            emb_dict_t = self.organize_community_embeddings_over_time(h_comm, comm_ids_per_sample, comm_counts, batch_size)
            for b in range(batch_size):
                for cid, emb_list in emb_dict_t[b].items():
                    all_time_emb_dict[b][cid][t] = emb_list[0]

            max_comm_num = max(len(comm_dict) for comm_dict in all_time_emb_dict)

            start = 0
            for b, comm_ids in enumerate(comm_ids_per_sample):
                num_comm = len(comm_ids)
                if num_comm == 0:
                    cascade_embeds_over_time[b].append(torch.zeros(self.embedding_dim, device=self.device))
                    continue

                h_comm_b = h_comm[start: start + num_comm]
                cascade_embed = self.multihead_attn_pool(h_comm_b)  # [D]

                cascade_embeds_over_time[b].append(cascade_embed)
                start += num_comm

        cascade_embeds = torch.stack([torch.stack(seq, dim=0) for seq in cascade_embeds_over_time], dim=0)  # [B, T, D]

        growth_tensor = torch.tensor([[cascade_growth[b][t] for t in range(self.num_time_steps)] for b in range(batch_size)], 
                                     dtype=torch.float32, device=self.device).unsqueeze(-1)  # [B, T, 1]
        growth_emb = self.cascade_growth_proj(growth_tensor)  

        cascade_seq = torch.cat([cascade_embeds, growth_emb], dim=-1)  # [B, T, D+8]

        out, hidden = self.cascade_gru(cascade_seq)  # hidden: [2, B, 64]
        cas_inter_pred = self.cas_inter_output_layer(out).squeeze(-1)
        final_out = torch.cat([hidden[0], hidden[1]], dim=-1)  # [B, 128]

        pred_growth = self.cascade_output_layer(final_out).squeeze(-1)  # [B]
        com_pred_growth, comm_id_order = self.process_community_sequences_with_bigru(all_time_emb_dict, max_comm_num)
        return pred_growth, cas_inter_pred, com_pred_growth, comm_id_order

