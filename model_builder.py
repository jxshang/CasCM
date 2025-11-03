from model.CasCM import CommunityTemporalModel

def build_model(args, device):
    gcn = GCNModule(args.gcn_in_dim, args.gcn_hidden_dim, args.gcn_out_dim) if args.use_gcn else None
    gat = GATModule(args.gcn_out_dim, args.gat_out_dim, args.gat_heads, args.edge_feat_dim, args.com_edge_dim) if args.use_gat else None
    input_dim = args.gat_out_dim
    gru = BiGRUModule(input_dim + args.community_growth_dim, args.gru_hidden_dim) if args.use_gru else None

    model = CommunityTemporalModel(
        args,
        gcn=gcn,
        gat=gat,
        gru=gru,
        num_time_steps=args.interval_num,
        device=device,
        use_gcn=args.use_gcn,
        use_gat=args.use_gat,
        use_gru=args.use_gru
    )
    return model



class GATModule(nn.Module):
    def __init__(self, in_dim, out_dim, heads, edge_feat_dim, com_edge_dim, use_mlp=True):
        super().__init__()

        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_feat_dim, com_edge_dim),
            nn.ReLU(),
            nn.Linear(com_edge_dim, com_edge_dim),
            nn.ReLU()
        )

        self.gat1 = GATv2Conv(
            in_channels=in_dim,
            out_channels=out_dim,
            heads=heads,
            edge_dim=com_edge_dim, 
            concat=False
        )
        self.norm1 = nn.LayerNorm(out_dim)
        self.gat2 = GATv2Conv(
            in_channels=out_dim,
            out_channels=out_dim,
            heads=1,
            edge_dim=com_edge_dim, 
            concat=False
        )
        self.norm2 = nn.LayerNorm(out_dim)

        self.use_mlp = use_mlp
        if use_mlp:
            self.mlp = MLPBlock(out_dim, dropout=0.2)



    def forward(self, x, edge_index, edge_attr):
        edge_attr_proj = self.edge_mlp(edge_attr)
        x = self.gat1(x, edge_index, edge_attr=edge_attr_proj)
        x = self.norm1(x)
        x = F.relu(x)
        x = self.gat2(x, edge_index, edge_attr=edge_attr_proj)
        x = self.norm2(x)

        if self.use_mlp:
            x = self.mlp(x)

        return x


class GCNModule(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.layer1 = GCNConv(in_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.layer2 = GCNConv(hidden_dim, out_dim)

    def forward(self, x, edge_index, is_new):
        row, col = edge_index
        mask = is_new[col]
        edge_mask = mask.bool()
        filtered_edge_index = edge_index[:, edge_mask]

        x = self.layer1(x, filtered_edge_index)
        x = self.relu(x)
        x = self.layer2(x, filtered_edge_index)

        return x
    

class BiGRUModule(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, batch_first=True, bidirectional=True)

    def forward(self, x_seq):
        output, hidden = self.gru(x_seq)
        return output, hidden


class GRUModule(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, batch_first=True)

    def forward(self, x_seq):
        output, hidden = self.gru(x_seq)
        return output, hidden

