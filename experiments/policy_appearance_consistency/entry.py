"""Minimal loss entry and dry-run. Deliberately no optimizer/update implementation.

The scientific training computation is exposed as step_loss for a later reviewed
loop. This round's CLI cannot start training accidentally.
"""
import argparse, json, random
import numpy as np
import torch
from torch.utils.data import DataLoader
from policies.policy.vla_bc_policy_rla_unified import VLABCPolicyRLAUnified
from dataset import PairedDataset
from method import paired_forward, predict_normalized, objective


def step_loss(model,batch,config):
    if config['arm']=='P0':
        a=predict_normalized(model,batch['clean'])
        # No extra exposure or second forward for P0.
        return objective(a,a,batch['target'],batch['mask'],'P0',0.)
    a,b=paired_forward(model,batch['clean'],batch['background'])
    return objective(a,b,batch['target'],batch['mask'],config['arm'],config['lambda_cons'])


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True)
    p.add_argument('--data',required=True);p.add_argument('--task',required=True)
    args=p.parse_args();cfg=json.load(open(args.config))
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    random.seed(cfg['seed']);np.random.seed(cfg['seed']);torch.manual_seed(cfg['seed'])
    data=PairedDataset(args.data,'train',args.task)
    model=VLABCPolicyRLAUnified(**cfg['model']);model.set_normalizer(data.normalizer);model.train()
    batch=next(iter(DataLoader(data,batch_size=1,shuffle=False,num_workers=0)))
    with torch.no_grad(): losses=step_loss(model,batch,cfg)
    print(json.dumps({'dry_run_only':True,'optimizer_updates':0,
                      'losses':{k:float(v) for k,v in losses.items()}},indent=2))


if __name__=='__main__': main()
