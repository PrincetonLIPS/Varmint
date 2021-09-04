import os
import numpy as np
import numpy.linalg as npla

import scipy.sparse.linalg

from jax.config import config
config.update("jax_enable_x64", True)

from steihaug import *
import preconditioners as prec


def load_varmint_optim_data_in_sequence():
  jac_dir = '/n/fs/mm-iga/Varmint/savedoptcheckpoints'
  numjacs = len([f for f in os.listdir(jac_dir) if 'jacs_' in f])
  all_jacs = []
  all_eps = []
  all_grads = []
  all_niter = []
  all_xs = []
  all_deltas = []

  for i in range(numjacs):
    with open(jac_dir + f'/jacs_{i}.npy', 'rb') as f:
        curr_jacs = np.load(f)
    with open(jac_dir + f'/grads_{i}.npy', 'rb') as f:
        curr_grads = np.load(f)
    with open(jac_dir + f'/epsilons_{i}.npy', 'rb') as f:
        curr_eps = np.load(f)
    with open(jac_dir + f'/nHvcalls_periter_{i}.npy', 'rb') as f:
        curr_niter = np.load(f)
    with open(jac_dir + f'/xs_{i}.npy', 'rb') as f:
        curr_xs = np.load(f)
    with open(jac_dir + f'/deltas_{i}.npy', 'rb') as f:
        curr_deltas = np.load(f)

    for j, jac in enumerate(curr_jacs):
        if np.all(jac == 0.0):
            break
        else:
            all_jacs.append(jac)
            all_eps.append(curr_eps[j])
            all_grads.append(curr_grads[j])
            all_niter.append(curr_niter[j])
            all_xs.append(curr_xs[j])
            all_deltas.append(curr_deltas[j])
    if i % 10 == 0:
        print(f"{j} jacs added from iteration {i}")
    
  return {
      'jacs': np.stack(all_jacs),
      'grads': np.stack(all_grads),
      'eps': np.stack(all_eps),
      'niters': np.stack(all_niter),
      'xs': np.stack(all_xs),
      'deltas': np.stack(all_deltas),
  }


def get_preconditioner(name, A, args=()):
    if name == 'diagonal':
        return prec.DiagonalPreconditioner(A)
    elif name == 'fixeddiagonal':
        return prec.FixedDiagonalPreconditioner(A, *args)
    elif name == 'exact':
        return prec.ExactPreconditioner(A)
    elif name == 'fixedmatrix':
        return prec.FixedMatrixPreconditioner(A, *args)
    elif name == 'identity':
        return prec.IdentityPreconditioner(A)
    else:
        raise ValueError(f'Invalid preconditioner name {name}.')

def main():
    optim_data = load_varmint_optim_data_in_sequence()
    optim_data['JTJs'] = optim_data['jacs'].transpose(0, 2, 1) @ optim_data['jacs']

    def JTJfun(x, v, args=()):
        (i,) = args
        return optim_data['JTJs'][i] @ v
    cg_solver = get_cg_steihaug_solver(JTJfun, precond=True, jittable=False)


    def cgsim(i, prec_name, prec_args=()):
        preconditioner = get_preconditioner(prec_name, optim_data['JTJs'][i], prec_args)

        return cg_solver(optim_data['grads'][i], optim_data['xs'][i],
                        optim_data['deltas'][i], optim_data['eps'][i], hv_args=(i,),
                        precond=(preconditioner.get_apply_fn(), preconditioner.get_apply_T_fn()))

    N = optim_data['JTJs'].shape[0]
    def sim_on_all(prec_name, start=0, end=N, prec_args=()):
        print(f'Testing {prec_name} preconditioner.')
        total_calls = 0
        for i in range(start, end):
            if i % 5 == 0:
                print(f'Finished {i} systems.')
            x_new, hit_bounds, ncalls = cgsim(i, prec_name, prec_args)
            norm = npla.norm(optim_data['JTJs'][i] @ (x_new - optim_data['xs'][i]) + optim_data['grads'][i])
            eps  = optim_data['eps'][i]
            if norm > 2 * eps:
                print('convergence did not happen!!')
                print(f'hit bounds: {hit_bounds}')
                print(f'achieved: {norm}')
                print(f'wanted: {eps}')
            total_calls += ncalls
        print(f'Total number of calls: {total_calls}')


    # Test identity
    sim_on_all('exact', 0, 10)
    sim_on_all('fixedmatrix', 0, 10, prec_args=(optim_data['JTJs'][0],))
    sim_on_all('diagonal', 0, 10)
    sim_on_all('fixeddiagonal', 0, 10, prec_args=(optim_data['JTJs'][0],))
    sim_on_all('identity', 0, 10)

if __name__ == '__main__':
    main()