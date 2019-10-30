import numpy as np
import matplotlib

matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
from matplotlib import animation

fig, ax = plt.subplots(1, 2)

x = np.arange(0, 2 * np.pi, 0.01)
line, = ax[0].plot(x, np.sin(x))
scarr, = ax[1].plot(x, np.sin(x), 'ro', markersize=0.5)
ax[0].plot([5, ], [0.5, ], 'ro')


def init():  # only required for blitting to give a clean slate.
    line.set_ydata([np.nan] * len(x))
    scarr.set_ydata([np.nan] * len(x))
    # return line,


def animate(i):
    if i % 2 == 0:
        line.set_ydata(np.sin(x + i / 100))
        scarr.set_ydata(np.sin(x + i / 100))  # update the data.
    else:
        line.set_ydata(np.sin(x * 0))
        scarr.set_ydata(np.sin(x * 0))

    # return line,


ani = animation.FuncAnimation(
    fig, animate, init_func=init, interval=1000, blit=False, save_count=50)

# To save the animation, use e.g.
#
# ani.save("movie.mp4")
#
# or
#
# from matplotlib.animation import FFMpegWriter
# writer = FFMpegWriter(fps=15, metadata=dict(artist='Me'), bitrate=1800)
# ani.save("movie.mp4", writer=writer)

plt.show()
