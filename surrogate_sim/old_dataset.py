import numpy as np

import matplotlib.pyplot as plt

import os
import time
import json
from multiprocessing import Pool

DATASET_ROOT = '/n/fs/mm-iga/Varmint/slurm_experiments/datasetlarge'


class Ckpt:
    def __init__(self, ckpt_path):
        self.path = ckpt_path
        with open(os.path.join(ckpt_path, 'metadata.json')) as f:
            self.metadata = json.loads(f.read())

        self.fixed_locs = np.load(os.path.join(ckpt_path, 'fixed_locs.npy'))
        self.new_p = np.load(os.path.join(ckpt_path, 'new_p.npy'))
        self.new_q = np.load(os.path.join(ckpt_path, 'new_q.npy'))
        self.old_p = np.load(os.path.join(ckpt_path, 'old_p.npy'))
        self.old_q = np.load(os.path.join(ckpt_path, 'old_q.npy'))
        self.ref_ctrl = np.load(os.path.join(ckpt_path, 'ref_ctrl.npy'))


def parse_instance(instance):
    ckpts_path = os.path.join(DATASET_ROOT, instance, 'optckpts')
    ckpts = os.listdir(ckpts_path)

    return [Ckpt(os.path.join(ckpts_path, ckpt)) for ckpt in ckpts]


def read_dataset(permute=True, use_cache=True):
    start_t = time.time()

    if use_cache:
        if os.path.exists(os.path.join(DATASET_ROOT, 'train_new_q.npy')):
            print('reading from cache')
            train_old_q = np.load(os.path.join(
                DATASET_ROOT, 'train_old_q.npy'))
            train_old_p = np.load(os.path.join(
                DATASET_ROOT, 'train_old_p.npy'))
            train_new_q = np.load(os.path.join(
                DATASET_ROOT, 'train_new_q.npy'))

            test_old_q = np.load(os.path.join(DATASET_ROOT, 'test_old_q.npy'))
            test_old_p = np.load(os.path.join(DATASET_ROOT, 'test_old_p.npy'))
            test_new_q = np.load(os.path.join(DATASET_ROOT, 'test_new_q.npy'))

            print(f'read dataset in {time.time() - start_t} seconds')
            return (train_old_q, train_old_p, train_new_q), (test_old_q, test_old_p, test_new_q)

    instances = [f for f in os.listdir(DATASET_ROOT) if os.path.isdir(
        os.path.join(DATASET_ROOT, f)) and 'randomradii' in f]
    ntrain = int(len(instances) * 0.8)

    train_instances = instances[:ntrain]
    test_instances = instances[ntrain:]

    print('reading in train')
    train_ckpts = []
    total = 0
    with Pool(processes=80) as pool:
        for ckpts in pool.imap_unordered(parse_instance, train_instances):
            total += 1
            print(f'read {total}')
            train_ckpts.extend(ckpts)

    print('creating arrays')
    train_old_q = np.stack([ckpt.old_q for ckpt in train_ckpts])
    train_old_p = np.stack([ckpt.old_p for ckpt in train_ckpts])
    train_new_q = np.stack([ckpt.new_q for ckpt in train_ckpts])

    if permute:
        perm = np.random.RandomState(0).permutation(train_old_q.shape[0])
        train_old_q = train_old_q[perm]
        train_old_p = train_old_p[perm]
        train_new_q = train_new_q[perm]

    print('reading in test')
    test_ckpts = []
    with Pool(processes=20) as pool:
        for ckpts in pool.imap_unordered(parse_instance, test_instances):
            test_ckpts.extend(ckpts)

    print('creating arrays')
    test_old_q = np.stack([ckpt.old_q for ckpt in test_ckpts])
    test_old_p = np.stack([ckpt.old_p for ckpt in test_ckpts])
    test_new_q = np.stack([ckpt.new_q for ckpt in test_ckpts])

    if use_cache:
        print('caching')
        np.save(os.path.join(DATASET_ROOT, 'train_old_q.npy'), train_old_q)
        np.save(os.path.join(DATASET_ROOT, 'train_old_p.npy'), train_old_p)
        np.save(os.path.join(DATASET_ROOT, 'train_new_q.npy'), train_new_q)

        np.save(os.path.join(DATASET_ROOT, 'test_old_q.npy'), test_old_q)
        np.save(os.path.join(DATASET_ROOT, 'test_old_p.npy'), test_old_p)
        np.save(os.path.join(DATASET_ROOT, 'test_new_q.npy'), test_new_q)

    print(f'read dataset in {time.time() - start_t} seconds')
    return (train_old_q, train_old_p, train_new_q), (test_old_q, test_old_p, test_new_q)
