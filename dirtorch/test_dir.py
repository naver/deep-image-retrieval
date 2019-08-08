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
import dirtorch.nets as nets
import dirtorch.datasets as datasets
import dirtorch.datasets.downloader as dl

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

def matmul(A, B):
    if typename(A) == np.__name__:
        B = tonumpy(B)
        scores = np.dot(A, B.T)
    elif typename(B) == torch.__name__:
        scores = torch.matmul(A, B.t()).cpu().numpy()
    else:
        raise TypeError("matrices must be either numpy or torch type")
    return scores

def expand_descriptors(descs, db=None, alpha=0, k=0):
    assert k >= 0 and alpha >= 0, 'k and alpha must be non-negative'
    if k == 0: return descs
    descs = tonumpy(descs)
    n = descs.shape[0]
    db_descs = tonumpy(db if db is not None else descs)

    sim = matmul(descs, db_descs)
    if db is None:
        sim[np.diag_indices(n)] = 0

    idx = np.argpartition(sim, int(-k), axis=1)[:, int(-k):]
    descs_aug = np.zeros_like(descs)
    for i in range(n):
        new_q = np.vstack([db_descs[j, :] * sim[i,j]**alpha for j in idx[i]])
        new_q = np.vstack([descs[i], new_q])
        new_q = np.mean(new_q, axis=0)
        descs_aug[i] = new_q / np.linalg.norm(new_q)

    return descs_aug

def extract_image_features( dataset, transforms, net, ret_imgs=False, same_size=False, flip=None,
                            desc="Extract feats...", iscuda=True, threads=8, batch_size=8):
    """ Extract image features for a given dataset.
        Output is 2-dimensional (B, D)
    """
    if not same_size:
        batch_size = 1
        old_benchmark = torch.backends.cudnn.benchmark # speed-up cudnn
        torch.backends.cudnn.benchmark = False  # will speed-up a lot for different image sizes

    loader = get_loader( dataset, trf_chain=transforms, preprocess=net.preprocess, iscuda=iscuda,
                         output=['img'], batch_size=batch_size, threads=threads,
                         shuffle=False) # VERY IMPORTANT !!!!!!

    if hasattr(net,'eval'): net.eval()

    tocpu = (lambda x: x.cpu()) if ret_imgs=='cpu' else (lambda x:x)

    img_feats = []
    trf_images = []
    with torch.no_grad():
        for inputs in tqdm.tqdm(loader, desc, total=1+(len(dataset)-1)//batch_size):
            imgs = inputs[0]
            for i in range(len(imgs)):
                if flip and flip.pop(0):
                    imgs[i] = imgs[i].flip(2)
            imgs = common.variables(inputs[:1], net.iscuda)[0]
            desc = net(imgs)
            if ret_imgs: trf_images.append( tocpu(imgs.detach()) )
            del imgs
            del inputs
            if len(desc.shape) == 1: desc.unsqueeze_(0)
            img_feats.append( desc.detach() )

    img_feats = torch.cat(img_feats, dim=0)
    if len(img_feats.shape) == 1: img_feats.unsqueeze_(0)

    if not same_size:
        torch.backends.cudnn.benchmark = old_benchmark

    if ret_imgs:
        if same_size: trf_images = torch.cat(trf_images, dim=0)
        return trf_images, img_feats
    return img_feats


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


def eval_model(db, net, trfs, pooling='mean', gemp=3, detailed=False, whiten=None,
               aqe=None, adba=None, threads=8, batch_size=16, save_feats=None,
               load_feats=None, load_distractors=None, dbg=()):
    """ Evaluate a trained model (network) on a given dataset.
    The dataset is supposed to contain the evaluation code.
    """
    print("\n>> Evaluation...")
    query_db = db.get_query_db()

    # extract DB feats
    bdescs = []
    qdescs = []

    if not load_feats:
        trfs_list = [trfs] if isinstance(trfs, str) else trfs

        for trfs in trfs_list:
            kw = dict(iscuda=net.iscuda, threads=threads, batch_size=batch_size, same_size='Pad' in trfs or 'Crop' in trfs)
            bdescs.append( extract_image_features(db, trfs, net, desc="DB", **kw) )

            # extract query feats
            qdescs.append( bdescs[-1] if db is query_db else extract_image_features(query_db, trfs, net, desc="query", **kw) )

        # pool from multiple transforms (scales)
        bdescs = F.normalize(pool(bdescs, pooling, gemp), p=2, dim=1)
        qdescs = F.normalize(pool(qdescs, pooling, gemp), p=2, dim=1)
    else:
        bdescs = np.load(os.path.join(load_feats, 'feats.bdescs.npy'))
        qdescs = np.load(os.path.join(load_feats, 'feats.qdescs.npy'))

    if save_feats:
        mkdir(save_feats, isfile=True)
        np.save(save_feats+'.bdescs', bdescs.cpu().numpy())
        if query_db is not db:
            np.save(save_feats+'.qdescs', qdescs.cpu().numpy())
        exit()

    if load_distractors:
        ddescs = [ np.load(os.path.join(load_distractors, '%d.bdescs.npy' % i)) for i in tqdm.tqdm(range(0,1000), 'Distractors') ]
        bdescs = np.concatenate([tonumpy(bdescs)] + ddescs)
        qdescs = tonumpy(qdescs) # so matmul below can work

    if whiten is not None:
        bdescs = common.whiten_features(tonumpy(bdescs), net.pca, **whiten)
        qdescs = common.whiten_features(tonumpy(qdescs), net.pca, **whiten)

    if adba is not None:
        bdescs = expand_descriptors(bdescs, **args.adba)
    if aqe is not None:
        qdescs = expand_descriptors(qdescs, db=bdescs, **args.aqe)

    scores = matmul(qdescs, bdescs)

    del bdescs
    del qdescs

    res = {}

    try:
        aps = [db.eval_query_AP(q, s) for q,s in enumerate(tqdm.tqdm(scores,desc='AP'))]
        if not isinstance(aps[0], dict):
            aps = [float(e) for e in aps]
            if detailed: res['APs'] = aps
            res['mAP'] = float(np.mean([e for e in aps if e>=0])) # Queries with no relevants have an AP of -1
        else:
            modes = aps[0].keys()
            for mode in modes:
                apst = [float(e[mode]) for e in aps]
                if detailed: res['APs'+'-'+mode] = apst
                res['mAP'+'-'+mode] = float(np.mean([e for e in apst if e>=0])) # Queries with no relevants have an AP of -1

    except NotImplementedError:
        print(" AP not implemented!")

    try:
        tops = [db.eval_query_top(q,s) for q,s in enumerate(tqdm.tqdm(scores,desc='top1'))]
        if detailed: res['tops'] = tops
        for k in tops[0]:
            res['top%d'%k] = float(np.mean([top[k] for top in tops]))
    except NotImplementedError:
        pass

    return res


def load_model( path, iscuda ):
    checkpoint = common.load_checkpoint(path, iscuda)
    net = nets.create_model(pretrained="", **checkpoint['model_options'])
    net = common.switch_model_to_cuda(net, iscuda, checkpoint)
    net.load_state_dict(checkpoint['state_dict'])
    net.preprocess = checkpoint.get('preprocess', net.preprocess)
    if 'pca' in checkpoint: net.pca = checkpoint.get('pca')
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

    parser.add_argument('--out-json', type=str, default="", help='path to output json')
    parser.add_argument('--detailed', action='store_true', help='return detailed evaluation')
    parser.add_argument('--save-feats', type=str, default="", help='path to output features')
    parser.add_argument('--load-distractors', type=str, default="", help='path to load distractors from')
    parser.add_argument('--load-feats', type=str, default="", help='path to load features from')

    parser.add_argument('--threads', type=int, default=8, help='number of thread workers')
    parser.add_argument('--gpu', type=int, default=0, nargs='+', help='GPU ids')
    parser.add_argument('--dbg', default=(), nargs='*', help='debugging options')
    # post-processing
    parser.add_argument('--whiten', type=str, default='Landmarks_clean', help='applies whitening')

    parser.add_argument('--aqe', type=int, nargs='+', help='alpha-query expansion paramenters')
    parser.add_argument('--adba', type=int, nargs='+', help='alpha-database augmentation paramenters')

    parser.add_argument('--whitenp', type=float, default=0.25, help='whitening power, default is 0.5 (i.e., the sqrt)')
    parser.add_argument('--whitenv', type=int, default=None, help='number of components, default is None (i.e. all components)')
    parser.add_argument('--whitenm', type=float, default=1.0, help='whitening multiplier, default is 1.0 (i.e. no multiplication)')

    args = parser.parse_args()
    args.iscuda = common.torch_set_gpu(args.gpu)
    if args.aqe is not None: args.aqe = {'k': args.aqe[0], 'alpha': args.aqe[1]}
    if args.adba is not None: args.adba = {'k': args.adba[0], 'alpha': args.adba[1]}

    dl.download_dataset(args.dataset)

    dataset = datasets.create(args.dataset)
    print("Test dataset:", dataset)

    net = load_model(args.checkpoint, args.iscuda)

    if args.whiten:
        net.pca = net.pca[args.whiten]
        args.whiten = {'whitenp': args.whitenp, 'whitenv': args.whitenv, 'whitenm': args.whitenm}
    else:
        net.pca = None
        args.whiten = None

    # Evaluate
    res = eval_model(dataset, net, args.trfs, pooling=args.pooling, gemp=args.gemp, detailed=args.detailed,
        threads=args.threads, dbg=args.dbg, whiten=args.whiten,  aqe=args.aqe, adba=args.adba,
        save_feats=args.save_feats, load_feats=args.load_feats, load_distractors=args.load_distractors)
    print(' * '+'\n * '.join(['%s = %g'%p for p in res.items()]))

    if args.out_json:
        # write to file
        try:
            data = json.load(open(args.out_json))
        except IOError:
            data = {}
        data[args.dataset] = res
        mkdir(args.out_json)
        open(args.out_json,'w').write(json.dumps(data, indent=1))
        print("saved to "+args.out_json)



