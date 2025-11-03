import argparse
import os
import itertools
import shutil
import sys
import time
import pandas as pd
import numpy as np
import torch
import pickle
from setproctitle import setproctitle
from tqdm import tqdm
from utils.util1 import *
from utils.datasets import *
from torch.utils.data import DataLoader
from model.model import *
from model.model_builder import *
from model.community_temporal_model import *
from model.EarlyStopping import *
from utils.collate import custom_collate_fn
import optuna
from optuna.trial import Trial
from typing import Optional

MSLE_loss = MSLELoss()
MAPE_Loss = MAPELoss()

def compute_loss(args, preds, batch_y, com_pred, comm_id_order, community_nodes, cas_inter_pred, cas_inter_growth, alpha):
    
    main_loss = MSLE_loss(preds, batch_y)
    #main_loss = msle(preds, batch_y)
    if not args.cas_loss and not args.com_loss:
        return main_loss
    device = cas_inter_pred.device
    B, T = cas_inter_pred.shape

    target_growth = torch.zeros_like(com_pred)
    valid_mask = torch.zeros_like(com_pred)
    inter_growth = torch.zeros(B, T, device=device)

    for b in range(B):
        for i, cid in enumerate(comm_id_order[b]):
            for t in range(T): 
                growth = community_nodes[b][t].get(cid, None)
                if growth is not None:
                    target_growth[b, i, t] = growth
                    valid_mask[b, i, t ] = 1
        for t in range(T): 
            inter_growth[b, t] = cas_inter_growth[b][t]  

    masked_pred = com_pred * valid_mask
    mask_target = target_growth * valid_mask


    if not args.cas_loss:
        com_growth_loss = MSLE_loss(masked_pred, mask_target)
        return main_loss + alpha * com_growth_loss
    elif not args.com_loss:
        cas_inter_loss = MSLE_loss(cas_inter_pred, inter_growth)
        return main_loss + alpha * cas_inter_loss
    
    com_growth_loss = MSLE_loss(masked_pred, mask_target)
    cas_inter_loss = MSLE_loss(cas_inter_pred, inter_growth)
    return main_loss + alpha * com_growth_loss + alpha * cas_inter_loss


def main(args, logger, checkpt_path, trial: Optional[Trial] = None):
    set_seed(2025)

    data_start_time = time.time()

    data_name = args.data
    with open(f'data/{data_name}/train_l{args.observation_time}_s{args.max_seq}.pkl', 'rb') as ftrain:
        train_cascade, train_label, train_id = pickle.load(ftrain)
    with open(f'data/{data_name}/val_l{args.observation_time}_s{args.max_seq}.pkl', 'rb') as fval:
        val_cascade, val_label, val_id = pickle.load(fval)
    with open(f'data/{data_name}/test_l{args.observation_time}_s{args.max_seq}.pkl', 'rb') as ftest:
        test_cascade, test_label, test_id = pickle.load(ftest)

    train_dataset = CascadeDataset(args, graphs=train_cascade, labels=train_label)
    val_dataset = CascadeDataset(args, graphs=val_cascade, labels=val_label)
    test_dataset = CascadeDataset(args, graphs=test_cascade, labels=test_label)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=custom_collate_fn, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=custom_collate_fn, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=custom_collate_fn, shuffle=True)

    data_end_time = time.time()
    print('data loading Finished! Time used: {:.3f}mins.'.format((data_end_time - data_start_time) / 60))

    device = torch.device(f'{args.cuda}' if torch.cuda.is_available() else 'cpu')
    model = build_model(args, device).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), args.lr, weight_decay=1e-4)
    early_stopping = EarlyStopping(patience=args.patience, verbose=True, path=checkpt_path, trace_func=logger.info)
    logger.info(f"{len(train_dataset)} training examples, {len(val_dataset)} val examples, {len(test_dataset)} test examples")
    
    train_losses = []
    val_losses = []
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        total_samples = 0
        for batch in tqdm(train_loader, ncols=90):
            time_community_graphs = batch['community_graphs_batch']
            community_nodes_growth = batch['community_nodes_growth_batch']
            community_growth = batch['community_growth_batch']
            community_edges = batch['inter_community_edges_batch']
            cascade_growth = batch['cascade_growth_batch']
            cas_inter_growth = batch['cascade_growth']
            labels = batch['labels'].to(device)
            preds, cas_inter_pred, com_pred, comm_id_order = model(time_community_graphs, community_growth, community_edges, cascade_growth)
            loss = compute_loss(args, preds, labels, com_pred, comm_id_order, community_nodes_growth, cas_inter_pred, cas_inter_growth, alpha=0.1)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

        avg_train_loss = total_loss / total_samples
        logger.info(f"Epoch {epoch} | trian Loss: {avg_train_loss:.4f}")
        train_losses.append(avg_train_loss)

        model.eval()
        total_valid_loss = 0
        total_valid_samples = 0

        with torch.no_grad():
            for batch in val_loader:
                time_community_graphs = batch['community_graphs_batch']
                community_nodes_growth = batch['community_nodes_growth_batch']
                community_growth = batch['community_growth_batch']
                community_edges = batch['inter_community_edges_batch']
                cascade_growth = batch['cascade_growth_batch']
                cas_inter_growth = batch['cascade_growth']
                labels = batch['labels'].to(device)
                preds, cas_inter_pred, com_pred, comm_id_order = model(time_community_graphs, community_growth, community_edges, cascade_growth)

                loss = MSLE_loss(preds, labels)
                batch_size = labels.size(0)
                total_valid_loss += loss.item() * batch_size
                total_valid_samples += batch_size
            
        val_loss = total_valid_loss / total_valid_samples
        logger.info(f"Epoch {epoch} | val Loss: {val_loss:.4f}")

        val_losses.append(val_loss)
        if isinstance(trial, Trial):
            trial.report(val_loss, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            logger.info("Early_Stopping")
            break  

    model_dict = torch.load(checkpt_path)
    model.load_state_dict(model_dict)
    model.eval()
    MSLE_test = 0
    MAPE_test = 0
    total_test_samples = 0
    with torch.no_grad():
        for batch in test_loader:
            time_community_graphs = batch['community_graphs_batch']
            community_nodes_growth = batch['community_nodes_growth_batch']
            community_growth = batch['community_growth_batch']
            community_edges = batch['inter_community_edges_batch']
            cascade_growth = batch['cascade_growth_batch']
            cas_inter_growth = batch['cascade_growth']
            labels = batch['labels'].to(device)
            preds, cas_inter_pred, com_pred, comm_id_order = model(time_community_graphs, community_growth, community_edges, cascade_growth)
            MSLE = MSLE_loss(preds, labels)
            MAPE = MAPE_Loss(preds, labels)

            MSLE_test += MSLE * labels.size(0)
            MAPE_test += MAPE * labels.size(0)
            total_test_samples += labels.size(0)
            
        aver_MSLE = "{:.4f}".format(MSLE_test / total_test_samples)
        aver_MAPE = "{:.4f}".format(MAPE_test / total_test_samples)
        logger.info(f"Final Result | MSLE Loss: {aver_MSLE} MAPE Loss: {aver_MAPE}")

        return early_stopping.best_score, aver_MSLE, aver_MAPE


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="twitter")
    parser.add_argument("--observation_time", type=str, default="86400")
    parser.add_argument("--interval_num", type=int, default=12)
    parser.add_argument('--community_growth_dim', type=int, default=8)
    parser.add_argument('--cascade_growth_dim', type=int, default=8)
    parser.add_argument('--com_edge_dim', type=int, default=16)
    parser.add_argument('--edge_feat_dim', type=int, default=7)
    
    # Model dimension args
    parser.add_argument('--gcn_in_dim', type=int, default=82)
    parser.add_argument('--gcn_hidden_dim', type=int, default=64)
    parser.add_argument('--gcn_out_dim', type=int, default=64)
    parser.add_argument('--gat_heads', type=int, default=4)
    parser.add_argument('--gat_out_dim', type=int, default=32)
    parser.add_argument('--gru_hidden_dim', type=int, default=32)
    parser.add_argument('--cas_gru_hidden_dim', type=int, default=64)

    parser.add_argument("--max_seq", type=str, default="100")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--ngpus", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    # Model architecture args
    parser.add_argument('--use_gcn', action='store_false', default=True)
    parser.add_argument('--use_gat', action='store_false', default=True)
    parser.add_argument('--use_gru', action='store_false', default=True)
    parser.add_argument('--com_loss', action='store_false', default=True)
    parser.add_argument('--cas_loss', action='store_false', default=True)
    parser.add_argument("--model", type=str, default="demo")
    parser.add_argument("--sample_ratio", type=float, default=0.2)
    parser.add_argument("--experiment_dir", type=str, default="log")
    parser.add_argument("--experiment_id", type=str, default=None)

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--cuda", type=str, default="cuda:0")

    args = parser.parse_args()

    if args.experiment_id is None:
        args.experiment_id = time.strftime("%m-%d_%H:%M:%S")

    if not args.com_loss:
        args.model = "no_com_loss"
    if not args.cas_loss:
        args.model = "no_cas_loss"

    experiment_name = f"{args.model}"
    experiment_name += f"_interval{args.interval_num}"
    experiment_name += f"_growth_dim{args.community_growth_dim}"
    log_path = os.path.join(args.experiment_dir, args.data + '_t' + args.observation_time, experiment_name)

    # Top-level logger for logging exceptions into the log file.
    makedirs(log_path)
    logger_name = os.path.join(log_path, f"{args.experiment_id}.txt")
    logger = get_logger(logger_name)
    logger.info(args)
    setproctitle(f"{args.data}_{args.observation_time}")
    checkpt_path = os.path.join(log_path, "model.pth")

    main(args, logger=logger, checkpt_path=checkpt_path)
    