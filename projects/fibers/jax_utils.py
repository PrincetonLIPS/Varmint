import functools 

import numpy as static_np
import jax 
import jax.numpy as np 
import jax.tree_util as tree_util
import optax

from varmint.utils.typing import *

def tree_stack(trees):
        """Takes a list of trees and stacks every corresponding leaf.
        For example, given two trees ((a, b), c) and ((a', b'), c'), returns
        ((stack(a, a'), stack(b, b')), stack(c, c')).
        Useful for turning a list of objects into something you can feed to a
        vmapped function.
        """
        leaves_list = []
        treedef_list = []
        for tree in trees:
            leaves, treedef = tree_util.tree_flatten(tree)
            leaves_list.append(leaves)
            treedef_list.append(treedef)

        grouped_leaves = zip(*leaves_list)
        result_leaves = [np.stack(l) for l in grouped_leaves]
        return treedef_list[0].unflatten(result_leaves)

def divides(a: int, b: int) -> bool: 
    return a % b == 0 

def array_size(arr: ndarray) -> int:
    # TODO unnecessary
    return static_np.array(arr).nbytes

def _vectorize(signature: str, excluded: tuple=()) -> callable: 
    def decorator(f: callable) -> callable:
        vectorized: callable = np.vectorize(f, excluded=excluded, signature=signature)
        return vectorized
    return decorator

def _jit_vectorize(signature: str, excluded: tuple =()) -> callable:
    """Applies the jax.jit and jax.numpy.vectorize transformations 
    to the tagged function.
    """
    def decorator(f: callable) -> callable:
        vectorized: callable = np.vectorize(f, excluded=excluded, signature=signature)
        jitted_and_vectorized: callable = jax.jit(vectorized, static_argnums=excluded)
        return jitted_and_vectorized
    return decorator

@functools.partial(jax.jit, static_argnums=(1,))
def custom_norm(x: ndarray, numeric_type: type=FP64) -> float: 
    """Utility function for computing the 2-norm of an array `x` 
    in a method that is safe under differentiation/tracing. 
    Parameters
    ----------
    x : ndarray
        array for which to compute the 2-norm. 
    numeric_type : type
        numeric type of the returned zero element (in the case where 
        the given array has norm zero). 
    Returns
    -------
    norm : float 
        2-norm of x. 
    """
    squared_sum: float = x.dot(x)
    is_zero: callable = lambda _: numeric_type(0) 
    is_nonzero: callable = lambda x: np.sqrt(x) 
    norm: float = jax.lax.cond(squared_sum == 0, is_zero, is_nonzero, operand=squared_sum)
    return norm 

def divide00(numerator: ndarray, denominator: ndarray, numeric_type: type=FP32) -> ndarray:
    """Computes the quotient of the given numerator and denominator 
    such that zero divided by zero equals zero, and differentiation 
    works. 
    Parameters
    ----------
    numerator : ndarray 
        numerator of the quotient 
    denominator : ndarray 
        denominator of the quotient 
    numeric_type : type 
        numeric type of the result (default FP32)
    Returns 
    -------
    quotient : ndarray 
        autodiff safe quotient. 
    """
    force_zero: ndarray = np.logical_and(numerator == 0, denominator == 0)
    quotient: ndarray = np.where(force_zero, numeric_type(0.0), numerator) / np.where(force_zero, numeric_type(1.0), denominator)
    return quotient 

def zero_one_sign(arr: ndarray) -> ndarray:
    """Returns an array of the same shape as the input with 
    the value 1. where the input array is greater than or equal 
    to zero and 0. where the input array is less than zero. 
    Parameters
    ----------
    arr : ndarray 
        input array 
    Returns 
    -------
    binary_arr : ndarray 
        result with shape of `arr` and value 1. where arr >= 0. 
        and value 0. otherwise. 
    """
    binary_arr: ndarray = 0.5 * (1.0 + np.sign(arr))
    return binary_arr

def setup_optimizer(args, params: tuple) -> tuple: 
    # TODO update for RenderingConfig
    if args.optimizer == "adam": 
        optimizer = optax.adam(args.step_size)
    elif args.optimizer == "rmsprop": 
        optimizer = optax.rmsprop(args.step_size)
    else: 
        raise(NotImplementedError) 

    optimizer_state = optimizer.init(params)
    return optimizer, optimizer_state, jax.jit(optimizer.update)

def vectorized_cond(predicate, true_function, false_function, operand) -> ndarray:
    # true_fun and false_fun must act elementwise (i.e. be vectorized)
    true_op = np.where(predicate, operand, 0)
    false_op = np.where(predicate, 0, operand)
    return np.where(predicate, true_function(true_op), false_function(false_op))

