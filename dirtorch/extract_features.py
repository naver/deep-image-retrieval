import sys
import os; os.umask(7)  # group permisions but that's all
import os.path as osp
import pdb

import json
import tqdm
import numpy as np
import torch
import torch.nn.functional as F

from dirtorch.utils.convenient import mkdir
from dirtorch.utils import common
from dirtorch.utils.pytorch_loader import get_loader

import dirtorch.test_dir as test
import dirtorch.nets as nets
import dirtorch.datasets as datasets

import pickle as pkl
import hashlib

def hash(x):
    m = hashlib.md5()
    m.update(str(x).encode('utf-8'))
    return m.hexdigest()

def typename(x):
    return type(x).__module__

def tonumpy(x):
    if typename(x) == torch.__name__:
        return x.cpu().numpy()
    else:
        return x


def pool(x, pooling='mean', gemp=3):
    if len(x) == 1: return x[0]
    x = torch.stack(x, dim=0)
    if pooling == 'mean':
        return torch.mean(x, dim=0)
    elif pooling == 'gem':
        def sympow(x, p, eps=1e-6):
            s = torch.sign(x)
            return (x*s).clamp(min=eps).pow(p) * s
        x = sympow(x,gemp)
        x = torch.mean(x, dim=0)
        return sympow(x, 1/gemp)
    else:
        raise ValueError("Bad pooling mode: "+str(pooling))


def extract_features(db, net, trfs, pooling='mean', gemp=3, detailed=False, whiten=None,
               threads=8, batch_size=16, output=None, dbg=()):
    """ Extract features from trained model (network) on a given dataset.
    """
    print("\n>> Extracting features...")
    try:
        query_db = db.get_query_db()
    except NotImplementedError:
        query_db = None

    # extract DB feats
    bdescs = []
    qdescs = []

    trfs_list = [trfs] if isinstance(trfs, str) else trfs

    for trfs in trfs_list:
        kw = dict(iscuda=net.iscuda, threads=threads, batch_size=batch_size, same_size='Pad' in trfs or 'Crop' in trfs)
        bdescs.append( test.extract_image_features(db, trfs, net, desc="DB", **kw) )

        # extract query feats
        if query_db is not None:
            qdescs.append( bdescs[-1] if db is query_db else test.extract_image_features(query_db, trfs, net, desc="query", **kw) )

    # pool from multiple transforms (scales)
    bdescs = tonumpy(F.normalize(pool(bdescs, pooling, gemp), p=2, dim=1))
    if query_db is not None:
        qdescs = tonumpy(F.normalize(pool(qdescs, pooling, gemp), p=2, dim=1))

    if whiten is not None:
        bdescs = common.whiten_features(bdescs, net.pca, **whiten)
        if query_db is not None:
            qdescs = common.whiten_features(qdescs, net.pca, **whiten)

    mkdir(output, isfile=True)
    if query_db is db or query_db is None:
        np.save(output, bdescs)
    else:
        o = osp.splitext(output)
        np.save(o[0]+'.qdescs'+o[1], qdescs)
        np.save(o[0]+'.dbdescs'+o[1], bdescs)
    print('Features extracted.')


def load_model( path, iscuda, whiten=None ):
    checkpoint = common.load_checkpoint(path, iscuda)
    net = nets.create_model(pretrained="", **checkpoint['model_options'])
    net = common.switch_model_to_cuda(net, iscuda, checkpoint)
    net.load_state_dict(checkpoint['state_dict'])
    net.preprocess = checkpoint.get('preprocess', net.preprocess)
    if whiten is not None and 'pca' in checkpoint:
        if whiten in checkpoint['pca']:
            net.pca = checkpoint['pca'][whiten]
    return net


def learn_whiten( dataset, net, trfs='', pooling='mean', threads=8, batch_size=16):
    descs = []
    trfs_list = [trfs] if isinstance(trfs, str) else trfs
    for trfs in trfs_list:
        kw = dict(iscuda=net.iscuda, threads=threads, batch_size=batch_size, same_size='Pad' in trfs or 'Crop' in trfs)
        descs.append( extract_image_features(dataset, trfs, net, desc="PCA", **kw) )
    # pool from multiple transforms (scales)
    descs = F.normalize(pool(descs, pooling), p=2, dim=1)
    # learn pca with whiten
    pca = common.learn_pca(descs.cpu().numpy(), whiten=True)
    return pca


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate a model')

    parser.add_argument('--dataset', '-d', type=str, required=True, help='Command to load dataset')
    parser.add_argument('--checkpoint', type=str, required=True, help='path to weights')

    parser.add_argument('--trfs', type=str, required=False, default='', nargs='+', help='test transforms (can be several)')
    parser.add_argument('--pooling', type=str, default="gem", help='pooling scheme if several trf chains')
    parser.add_argument('--gemp', type=int, default=3, help='GeM pooling power')
    parser.add_argument('--center-bias', type=float, default=0, help='enforce some center bias')

    parser.add_argument('--out-json', type=str, default="", help='path to output json')
    parser.add_argument('--detailed', action='store_true', help='return detailed evaluation')
    parser.add_argument('--output', type=str, default="", help='path to output features')

    parser.add_argument('--threads', type=int, default=8, help='number of thread workers')
    parser.add_argument('--gpu', type=int, nargs='+', help='GPU ids')
    parser.add_argument('--dbg', default=(), nargs='*', help='debugging options')
    # post-processing
    parser.add_argument('--whiten', type=str, default=None, help='applies whitening')

    parser.add_argument('--whitenp', type=float, default=0.5, help='whitening power, default is 0.5 (i.e., the sqrt)')
    parser.add_argument('--whitenv', type=int, default=None, help='number of components, default is None (i.e. all components)')
    parser.add_argument('--whitenm', type=float, default=1.0, help='whitening multiplier, default is 1.0 (i.e. no multiplication)')

    args = parser.parse_args()
    args.iscuda = common.torch_set_gpu(args.gpu)

    dataset = datasets.create(args.dataset)
    print("Dataset:", dataset)

    net = load_model(args.checkpoint, args.iscuda, args.whiten)

    if args.center_bias:
        assert hasattr(net,'center_bias')
        net.center_bias = args.center_bias
        if hasattr(net, 'module') and hasattr(net.module,'center_bias'):
            net.module.center_bias = args.center_bias

    if args.whiten and not hasattr(net, 'pca'):
        # Learn PCA if necessary
        if os.path.exists(args.whiten):
            with open(args.whiten, 'rb') as f:
                net.pca = pkl.load(f)
        else:
            pca_path = '_'.join([args.checkpoint, args.whiten, args.pooling, hash(args.trfs), 'pca.pkl'])
            db = datasets.create(args.whiten)
            print('Dataset for learning the PCA with whitening:', db)
            pca = learn_whiten(db, net, pooling=args.pooling, trfs=args.trfs, threads=args.threads)

            chk = torch.load(args.checkpoint, map_location=lambda storage, loc: storage)
            if 'pca' not in chk: chk['pca'] = {}
            chk['pca'][args.whiten] = pca
            torch.save(chk, args.checkpoint)

            net.pca = pca

    if args.whiten:
        args.whiten = {'whitenp': args.whitenp, 'whitenv': args.whitenv, 'whitenm': args.whitenm}

    # Evaluate
    res = extract_features(dataset, net, args.trfs, pooling=args.pooling, gemp=args.gemp, detailed=args.detailed,
        threads=args.threads, dbg=args.dbg, whiten=args.whiten, output=args.output)


