import jax
import jax.numpy as np

from functools import partial
import matplotlib.patches as patches


@partial(jax.vmap, in_axes=(0, None, None))
def is_r1(p, cs, bs):
    return (cs + bs / 3. < p[0] < cs + bs / 3. + 1.0) & \
           (cs + bs < p[1] < cs + bs + 2 * cs)

def r1_points(N, cs, bs):
    xx = np.linspace(cs + bs / 3., cs + bs / 3. + 1.0, N)
    yy = np.linspace(cs + bs, cs + bs + 2 * cs, N)
    return np.stack(np.meshgrid(xx, yy), axis=-1).reshape(-1, 2)

@partial(jax.vmap, in_axes=(0, None, None))
def is_r2(p, cs, bs):
    return (cs + bs / 3. < p[0] < cs + bs / 3. + 1.0) & \
           (3 * cs + 2 * bs < p[1] < 3 * cs + 2 * bs + 2 * cs)

def r2_points(N, cs, bs):
    xx = np.linspace(cs + bs / 3., cs + bs / 3. + 1.0, N)
    yy = np.linspace(3 * cs + 2 * bs, 3 * cs + 2 * bs + 2 * cs, N)
    return np.stack(np.meshgrid(xx, yy), axis=-1).reshape(-1, 2)

@partial(jax.vmap, in_axes=(0, None, None))
def is_r3(p, cs, bs):
    return (3 * cs + bs + bs / 3. < p[0] < 3 * cs + bs + bs / 3. + 1.0) & \
           (cs + bs < p[1] < cs + bs + 2 * cs)

def r3_points(N, cs, bs):
    xx = np.linspace(3 * cs + bs + bs / 3., 3 * cs + bs + bs / 3. + 1.0, N)
    yy = np.linspace(cs + bs, cs + bs + 2 * cs, N)
    return np.stack(np.meshgrid(xx, yy), axis=-1).reshape(-1, 2)

@partial(jax.vmap, in_axes=(0, None, None))
def is_r4(p, cs, bs):
    return (3 * cs + bs + bs / 3. < p[0] < 3 * cs + bs + bs / 3. + 1.0) & \
           (3 * cs + 2 * bs < p[1] < 3 * cs + 2 * bs + 2 * cs)

def r4_points(N, cs, bs):
    xx = np.linspace(3 * cs + bs + bs / 3., 3 * cs + bs + bs / 3. + 1.0, N)
    yy = np.linspace(3 * cs + 2 * bs, 3 * cs + 2 * bs + 2 * cs, N)
    return np.stack(np.meshgrid(xx, yy), axis=-1).reshape(-1, 2)

@partial(jax.vmap, in_axes=(0, None, None))
def is_r5(p, cs, bs):
    return (1 * cs + bs < p[0] < 1 * cs + bs + 2 * cs) & \
           (1 * cs + bs / 3. < p[1] < 1 * cs + bs / 3. + 1.0)

def r5_points(N, cs, bs):
    xx = np.linspace(1 * cs + bs, 1 * cs + bs + 2 * cs, N)
    yy = np.linspace(1 * cs + bs / 3., 1 * cs + bs / 3. + 1.0, N)
    return np.stack(np.meshgrid(xx, yy), axis=-1).reshape(-1, 2)

@partial(jax.vmap, in_axes=(0, None, None))
def is_r6(p, cs, bs):
    return (1 * cs + bs < p[0] < 1 * cs + bs + 2 * cs) & \
           (3 * cs + 1 * bs + bs / 3. < p[1] < 3 * cs + 1 * bs + bs / 3. + 1.0)

def r6_points(N, cs, bs):
    xx = np.linspace(1 * cs + bs, 1 * cs + bs + 2 * cs, N)
    yy = np.linspace(3 * cs + 1 * bs + bs / 3., 3 * cs + 1 * bs + bs / 3. + 1.0, N)
    return np.stack(np.meshgrid(xx, yy), axis=-1).reshape(-1, 2)

@partial(jax.vmap, in_axes=(0, None, None))
def is_r7(p, cs, bs):
    return (1 * cs + bs < p[0] < 1 * cs + bs + 2 * cs) & \
           (5 * cs + 2 * bs + bs / 3. < p[1] < 5 * cs + 2 * bs + bs / 3. + 1.0)

def r7_points(N, cs, bs):
    xx = np.linspace(1 * cs + bs, 1 * cs + bs + 2 * cs, N)
    yy = np.linspace(5 * cs + 2 * bs + bs / 3., 5 * cs + 2 * bs + bs / 3. + 1.0, N)
    return np.stack(np.meshgrid(xx, yy), axis=-1).reshape(-1, 2)

def get_all_points_dict(N, cs, bs):
    return {
        '1': r1_points(N, cs, bs),
        '2': r2_points(N, cs, bs),
        '3': r3_points(N, cs, bs),
        '4': r4_points(N, cs, bs),
        '5': r5_points(N, cs, bs),
        '6': r6_points(N, cs, bs),
        '7': r7_points(N, cs, bs),
    }

# This was a bad idea...
def draw_mpl_patches(ax, cs, bs, alpha=0.7, fcolor='white'):
    # Patches to label where the segments should be.
    r1 = patches.Rectangle(
            (cs + bs / 3., cs + bs), 
            1.0, 2 * cs, linewidth=3, edgecolor='b', facecolor='none')
    ax.add_patch(r1)

    r2 = patches.Rectangle(
            (cs + bs / 3., 3 * cs + 2 * bs), 
            1.0, 2 * cs, linewidth=3, edgecolor='b', facecolor='none')
    ax.add_patch(r2)

    r3 = patches.Rectangle(
            (3 * cs + bs + bs / 3., cs + bs), 
            1.0, 2 * cs, linewidth=3, edgecolor='b', facecolor='none')
    ax.add_patch(r3)

    r4 = patches.Rectangle(
            (3 * cs + bs + bs / 3., 3 * cs + 2 * bs), 
            1.0, 2 * cs, linewidth=3, edgecolor='b', facecolor='none')
    ax.add_patch(r4)

    r5 = patches.Rectangle(
            (1 * cs + bs, 1 * cs + bs / 3.), 
            2 * cs, 1.0, linewidth=3, edgecolor='b', facecolor='none')
    ax.add_patch(r5)

    r6 = patches.Rectangle(
            (1 * cs + bs, 3 * cs + 1 * bs + bs / 3.), 
            2 * cs, 1.0, linewidth=3, edgecolor='b', facecolor='none')
    ax.add_patch(r6)

    r7 = patches.Rectangle(
            (1 * cs + bs, 5 * cs + 2 * bs + bs / 3.), 
            2 * cs, 1.0, linewidth=3, edgecolor='b', facecolor='none')
    ax.add_patch(r7)

    # Patches that cover the rest of the material

    # Bottom left
    c1 = patches.Rectangle(
            (0.0, 0.0), 
            1 * cs + 1 * bs, 1 * cs + 1 * bs * 2./3., linewidth=3, edgecolor='none', facecolor=fcolor, alpha=alpha)
    ax.add_patch(c1)

    # Top left
    c8 = patches.Rectangle(
            (0.0, cs * 5 + bs * 2 + bs * 1/3), 
            1 * cs + 1 * bs, 1 * cs + 1 * bs * 2./3., linewidth=3, edgecolor='none', facecolor=fcolor, alpha=alpha)
    ax.add_patch(c8)

    # Bottom middle
    c2 = patches.Rectangle(
            (1 * cs + 1 * bs, 0.0), 
            2 * cs, 1 * cs + 1 * bs / 3, linewidth=3, edgecolor='none', facecolor=fcolor, alpha=alpha)
    ax.add_patch(c2)

    # Top middle
    c2 = patches.Rectangle(
            (1 * cs + 1 * bs, cs * 5 + bs * 2 + bs * 2/3), 
            2 * cs, 1 * cs + 1 * bs / 3, linewidth=3, edgecolor='none', facecolor=fcolor, alpha=alpha)
    ax.add_patch(c2)

    # Bottom right
    c3 = patches.Rectangle(
            (3 * cs + 1 * bs, 0.0), 
            1 * cs + 1 * bs, 1 * cs + 1 * bs * 2./3., linewidth=3, edgecolor='none', facecolor=fcolor, alpha=alpha)
    ax.add_patch(c3)

    # Top right
    c3 = patches.Rectangle(
            (3 * cs + 1 * bs, cs * 5 + bs * 2 + bs * 1/3), 
            1 * cs + 1 * bs, 1 * cs + 1 * bs * 2./3., linewidth=3, edgecolor='none', facecolor=fcolor, alpha=alpha)
    ax.add_patch(c3)

    c4 = patches.Rectangle(
            (0.0, cs + bs * 2./3), 
            1 * cs + 1 * bs * 2/3, bs * 1./3., linewidth=3, edgecolor='none', facecolor=fcolor, alpha=alpha)
    ax.add_patch(c4)

    c4 = patches.Rectangle(
            (0.0, 3* cs + bs + bs * 2./3), 
            1 * cs + 1 * bs * 2/3, bs * 1./3., linewidth=3, edgecolor='none', facecolor=fcolor, alpha=alpha)
    ax.add_patch(c4)

    c4 = patches.Rectangle(
            (0.0, 3* cs + bs + bs * 1./3), 
            1 * cs + 1 * bs * 3/3, bs * 1./3., linewidth=3, edgecolor='none', facecolor=fcolor, alpha=alpha)
    ax.add_patch(c4)

    c4 = patches.Rectangle(
            (0.0, 3* cs + bs), 
            1 * cs + 1 * bs * 2/3, bs * 1./3., linewidth=3, edgecolor='none', facecolor=fcolor, alpha=alpha)
    ax.add_patch(c4)

    c4 = patches.Rectangle(
            (0.0, 5* cs + 2 * bs), 
            1 * cs + 1 * bs * 2/3, bs * 1./3., linewidth=3, edgecolor='none', facecolor=fcolor, alpha=alpha)
    ax.add_patch(c4)

    c5 = patches.Rectangle(
            (3 * cs + 1 * bs + bs * 1/3, cs + bs * 2./3), 
            1 * cs + 1 * bs * 2/3, bs * 1./3., linewidth=3, edgecolor='none', facecolor=fcolor, alpha=alpha)
    ax.add_patch(c5)

    c5 = patches.Rectangle(
            (3 * cs + 1 * bs + bs * 1/3, 3 * cs + bs + bs * 2./3), 
            1 * cs + 1 * bs * 2/3, bs * 1./3., linewidth=3, edgecolor='none', facecolor=fcolor, alpha=alpha)
    ax.add_patch(c5)

    c5 = patches.Rectangle(
            (3 * cs + 1 * bs, 3 * cs + bs + bs * 1./3), 
            1 * cs + 1 * bs * 3/3, bs * 1./3., linewidth=3, edgecolor='none', facecolor=fcolor, alpha=alpha)
    ax.add_patch(c5)

    c5 = patches.Rectangle(
            (3 * cs + 1 * bs + bs * 1/3, 3 * cs + bs), 
            1 * cs + 1 * bs * 2/3, bs * 1./3., linewidth=3, edgecolor='none', facecolor=fcolor, alpha=alpha)
    ax.add_patch(c5)

    c5 = patches.Rectangle(
            (3 * cs + 1 * bs + bs * 1/3, 5 * cs + 2 * bs), 
            1 * cs + 1 * bs * 2/3, bs * 1./3., linewidth=3, edgecolor='none', facecolor=fcolor, alpha=alpha)
    ax.add_patch(c5)

    # Middle bottom square batch
    c6 = patches.Rectangle(
            (1 * cs + 1 * bs * 2/3, cs + bs * 2./3), 
            2 * cs + 1 * bs * 2/3, 2 * cs + 1 * bs * 2/3, linewidth=3, edgecolor='none', facecolor=fcolor, alpha=alpha)
    ax.add_patch(c6)

    # Middle top square batch
    c7 = patches.Rectangle(
            (1 * cs + 1 * bs * 2/3, 3 * cs + bs + bs * 2./3), 
            2 * cs + 1 * bs * 2/3, 2 * cs + 1 * bs * 2/3, linewidth=3, edgecolor='none', facecolor=fcolor, alpha=alpha)
    ax.add_patch(c7)

    # Left side bottom
    c7 = patches.Rectangle(
            (0.0, 1 * cs + 1 * bs), 
            1 * cs + 1 * bs * 1/3, 2 * cs, linewidth=3, edgecolor='none', facecolor=fcolor, alpha=alpha)
    ax.add_patch(c7)

    c7 = patches.Rectangle(
            (0.0, 3 * cs + 2 * bs), 
            1 * cs + 1 * bs * 1/3, 2 * cs, linewidth=3, edgecolor='none', facecolor=fcolor, alpha=alpha)
    ax.add_patch(c7)

    c7 = patches.Rectangle(
            (3 * cs + bs + bs * 2/3, 1 * cs + 1 * bs), 
            1 * cs + 1 * bs * 1/3, 2 * cs, linewidth=3, edgecolor='none', facecolor=fcolor, alpha=alpha)
    ax.add_patch(c7)

    c7 = patches.Rectangle(
            (3 * cs + bs + bs * 2/3, 3 * cs + 2 * bs), 
            1 * cs + 1 * bs * 1/3, 2 * cs, linewidth=3, edgecolor='none', facecolor=fcolor, alpha=alpha)
    ax.add_patch(c7)

