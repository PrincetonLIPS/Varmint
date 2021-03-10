import numpy as np

import os
import time
import json
from multiprocessing import Pool

from varmint.cell2d import Cell2D
from analysis_utils import load_dynamics_simulation


class Ckpt:
  def __init__(self, ckpt_path):
    self.path = ckpt_path
    with open(os.path.join(ckpt_path, 'args.txt')) as f:
      self.args = json.load(f)

    sim_path = os.path.join(ckpt_path, 'sim_ckpt')
    self.QQ, self.PP, self.TT, self.radii, self.cell = load_dynamics_simulation(sim_path)


def parse_instance(ds_root, instance):
  ckpt_path = os.path.join(ds_root, instance)
  return Ckpt(ckpt_path)


def read_dataset(ds_root):
  start_t = time.time()
  instances = [f for f in os.listdir(ds_root) \
      if os.path.isdir(os.path.join(ds_root, f)) and 'cantileverdataset' in f]
  ntrain = int(len(instances) * 0.8)

  train_instances = instances[:ntrain]
  test_instances  = instances[ntrain:]

  train_ckpts = [parse_instance(ds_root, i) for i in train_instances]
  test_ckpts  = [parse_instance(ds_root, i) for i in test_instances]

  print(f'dataset read in {time.time() - start_t} seconds')
  return train_ckpts, test_ckpts


def get_any_ckpt(ds_root):
  instances = [f for f in os.listdir(ds_root) \
      if os.path.isdir(os.path.join(ds_root, f)) and 'cantileverdataset' in f]
  return parse_instance(ds_root, instances[0])


def combine_instances(instances, verbose=False):
  ninstances = len(instances)

  QQs = np.stack([np.stack(i.QQ) for i in instances])
  PPs = np.stack([np.stack(i.PP) for i in instances])
  TTs = np.expand_dims(np.stack([np.stack(i.TT) for i in instances]), -1)
  TTs = np.diff(TTs, axis=1)
  radii = np.expand_dims(np.stack([i.radii for i in instances]), 1)
  radii = np.broadcast_to(radii, QQs.shape[:2] + radii.shape[2:])

  if verbose:
    print(f'QQ array shape: {QQs.shape}')
    print(f'PP array shape: {PPs.shape}')
    print(f'TT array shape: {TTs.shape}')
    print(f'Radii shape: {radii.shape}')
    print(f'Are times the same: {np.allclose(TTs, 0.005)}')
  
  assert QQs.shape == PPs.shape
  return QQs, PPs, TTs, radii


def assemble_into_inputs(QQs, PPs, radii):
  old_q = QQs[:, :(QQs.shape[1]-1), :]
  old_p = PPs[:, :(PPs.shape[1]-1), :]

  radii = radii.reshape((radii.shape[0], radii.shape[1], -1))
  radii = radii[:, :(radii.shape[1]-1), :]
  new_q = QQs[:, 1:, :]

  old_q = old_q.reshape(-1, old_q.shape[-1])
  old_p = old_p.reshape(-1, old_p.shape[-1])
  radii = radii.reshape(-1, radii.shape[-1])
  new_q = new_q.reshape(-1, new_q.shape[-1])

  return old_q, old_p, radii, new_q


def parse_tensors_with_cache(ds_root):
  print('Loading dataset.')
  cache_dir = os.path.join(ds_root, '_cache')
  if os.path.exists(cache_dir):
    print('Loading from cache.')
    train_oq = np.load(os.path.join(cache_dir, 'train_oq.npy'))
    train_op = np.load(os.path.join(cache_dir, 'train_op.npy'))
    train_rd = np.load(os.path.join(cache_dir, 'train_rd.npy'))
    train_nq = np.load(os.path.join(cache_dir, 'train_nq.npy'))

    test_oq = np.load(os.path.join(cache_dir, 'test_oq.npy'))
    test_op = np.load(os.path.join(cache_dir, 'test_op.npy'))
    test_rd = np.load(os.path.join(cache_dir, 'test_rd.npy'))
    test_nq = np.load(os.path.join(cache_dir, 'test_nq.npy'))
  else:
    print('Could not find in cache.')
    train, test = dataset.read_dataset(ds_root)

    print('Combining instances.')
    QQs, PPs, TTs, radii = combine_instances(train, verbose=True)
    train_oq, train_op, train_rd, train_nq = assemble_into_inputs(QQs, PPs, radii)

    QQs, PPs, TTs, radii = combine_instances(test, verbose=True)
    test_oq, test_op, test_rd, test_nq = assemble_into_inputs(QQs, PPs, radii)

    os.mkdir(cache_dir)
    np.save(os.path.join(cache_dir, 'train_oq.npy'), train_oq)
    np.save(os.path.join(cache_dir, 'train_op.npy'), train_op)
    np.save(os.path.join(cache_dir, 'train_rd.npy'), train_rd)
    np.save(os.path.join(cache_dir, 'train_nq.npy'), train_nq)

    np.save(os.path.join(cache_dir, 'test_oq.npy'), test_oq)
    np.save(os.path.join(cache_dir, 'test_op.npy'), test_op)
    np.save(os.path.join(cache_dir, 'test_rd.npy'), test_rd)
    np.save(os.path.join(cache_dir, 'test_nq.npy'), test_nq)

  return train_oq, train_op, train_rd, train_nq, test_oq, test_op, test_rd, test_nq
