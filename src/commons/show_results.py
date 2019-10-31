import numpy as np
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
from matplotlib import animation


def show_result():
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    fig.tight_layout(pad=3)
    ax[0].set_xlim(15 - 15 * 0.2, 65 + 65 * 0.2)
    ax[0].set_ylim(40 - 40 * 0.2, 90 + 90 * 0.2)
    ax[0].set_yticks(np.linspace(40, 90, 11))
    ax[0].set_xticks(np.linspace(15, 65, 11))
    ax[0].set_title("Classify")
    ax[0].set_xlabel("Exam 1 Score")
    ax[0].set_ylabel("Exam 2 Score")

    # x = np.arange(0, 2 * np.pi, 0.01)
    # line, = ax[0].plot(x, np.sin(x))
    # scarr, = ax[1].plot(x, np.sin(x), 'ro', markersize=0.5)
    # ax[0].plot([5, ], [0.5, ], 'ro')


    def init():  # only required for blitting to give a clean slate.
        # line.set_ydata([np.nan] * len(x))
        # scarr.set_ydata([np.nan] * len(x))
        # return line,
        pass

    def animate(i):
        # if i % 2 == 0:
        #     line.set_ydata(np.sin(x + i / 100))
        #     scarr.set_ydata(np.sin(x * 0))
        # else:
        #     line.set_ydata(np.sin(x * 0))
        #
        #     scarr.set_ydata(np.sin(x + i / 100))

        # return line,
        pass

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

show_result()