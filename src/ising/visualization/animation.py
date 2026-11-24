"""Generate a GIF using the browser demonstration's light-theme palette."""
import numpy as np
from ising import IsingModel, Simulator

MINUS_COLOR = '#2c4a6b'
PLUS_COLOR = '#e3952f'
WALL_COLOR = '#14263a'
BACKGROUND_COLOR = '#eef1f4'
TEXT_COLOR = '#16202b'


def save_animation(model: IsingModel, output, *, dynamics: str = "heatbath", backend: str = "numba", init: str = "minus", seed: int = 0, frames: int = 150, stride: int = 5):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.colors import ListedColormap, BoundaryNorm

    sim = Simulator(model, init, seed, dynamics=dynamics, backend=backend)
    # Display codes are separate from physical spins: 0 = wall, 1 = minus, 2 = plus.
    pixels = np.zeros(sim.state.shape, dtype=np.uint8)
    cmap = ListedColormap([WALL_COLOR, MINUS_COLOR, PLUS_COLOR])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)

    def display_state():
        pixels[1:-1, 1:-1] = 1 + (sim.state[1:-1, 1:-1] == 1)
        return pixels

    fig, ax = plt.subplots(figsize=(5, 5), facecolor=BACKGROUND_COLOR)
    im = ax.imshow(display_state(), cmap=cmap, norm=norm, interpolation='nearest')
    ax.set_axis_off()
    title = ax.set_title('', color=TEXT_COLOR, fontsize=10)

    def update_display():
        im.set_data(display_state())
        title.set_text(f'{dynamics}  beta={model.beta}  lam={model.lam}  '
                       f't={sim.t:g}  m={sim.magnetization:+.2f}')
        return im, title

    def step(_frame):
        sim.advance(stride)
        return update_display()

    FuncAnimation(fig, step, init_func=update_display, frames=frames, blit=False).save(
        output, writer=PillowWriter(fps=15))
    plt.close(fig)
    return output
