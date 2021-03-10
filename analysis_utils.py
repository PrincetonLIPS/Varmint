import jax
import jax.numpy as np

import os
import pickle

def inspect_residuals(res_fun, old_q, new_q, p, dt, ref_ctrl, fixed_locs, diagD=1.0):
  resids_before = res_fun(old_q * diagD, (old_q, p, dt, (ref_ctrl, fixed_locs)))
  resids_after  = res_fun(new_q * diagD, (old_q, p, dt, (ref_ctrl, fixed_locs)))

  resid_grad = jax.jacfwd(res_fun)(old_q * diagD, (old_q, p, dt, (ref_ctrl, fixed_locs)))

  print(f'Shape of resids: {resids_before.shape}')
  print(f'Shape of resid grads: {resid_grad.shape}')
  print(f'Rank of Jacobian: {np.linalg.matrix_rank(resid_grad)}')
  print(f'Rank of GN Matrix: {np.linalg.matrix_rank(resid_grad.T @ resid_grad)}')
  print(f'Condition number of GN Matrix: {np.linalg.cond(resid_grad.T @ resid_grad)}')
  print(f'Before residual norm: {np.linalg.norm(resids_before)}')
  print(f'After norm: {np.linalg.norm(resids_after)}')


def save_dynamics_simulation(path, QQ, PP, TT, radii, cell):
  np.savez(os.path.join(path, 'positions.npz'), *QQ)
  np.savez(os.path.join(path, 'velocities.npz'), *PP)
  np.savez(os.path.join(path, 'times.npz'), *TT)
  np.savez(os.path.join(path, 'radii.npz'), radii)

  with open(os.path.join(path, 'cell.pkl'), 'wb') as f:
    pickle.dump(cell, f)


def load_dynamics_simulation(path):
  with open(os.path.join(path, 'positions.npz'), 'rb') as f:
    npz = np.load(f)
    QQ = [npz[i] for i in npz.files]

  with open(os.path.join(path, 'velocities.npz'), 'rb') as f:
    npz = np.load(f)
    PP = [npz[i] for i in npz.files]

  with open(os.path.join(path, 'times.npz'), 'rb') as f:
    npz = np.load(f)
    TT = [npz[i] for i in npz.files]

  with open(os.path.join(path, 'radii.npz'), 'rb') as f:
    radii = np.load(f)['arr_0']

  with open(os.path.join(path, 'cell.pkl'), 'rb') as f:
    cell = pickle.load(f)

  return QQ, PP, TT, radii, cell


def save_optimization(args, old_q, old_p, t, dt, ref_ctrl, fixed_locs, new_q, new_p):
  save_dir = os.path.join(args.exp_dir, 'optckpts', str(float(t)))
  print(f'saving to dir {save_dir}')

  assert not os.path.exists(save_dir)
  os.makedirs(save_dir)

  metadata = {}
  metadata['time'] = float(t)
  metadata['dt'] = float(dt)
  metadata['global_params'] = vars(args)

  with open(os.path.join(save_dir, 'metadata.json'), 'w') as f:
    json.dump(metadata, f)

  onp.save(os.path.join(save_dir, 'old_q.npy'), onp.asarray(old_q))
  onp.save(os.path.join(save_dir, 'old_p.npy'), onp.asarray(old_p))
  onp.save(os.path.join(save_dir, 'ref_ctrl.npy'), onp.asarray(ref_ctrl))
  onp.save(os.path.join(save_dir, 'fixed_locs.npy'), onp.asarray(fixed_locs))
  onp.save(os.path.join(save_dir, 'new_q.npy'), onp.asarray(new_q))
  onp.save(os.path.join(save_dir, 'new_p.npy'), onp.asarray(new_p))
