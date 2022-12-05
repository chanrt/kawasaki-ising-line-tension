from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from pickle import load


def animate(i):
    plt.title(f"Frame: {i}")
    im.set_array(data[i])
    return [im]


if __name__ == '__main__':
    data = load(open('outputs/lattice_data.pkl', 'rb'))
    num_frames = len(data)

    proper_frame = 0
    for frame in range(num_frames):
        frame_shape = data[frame].shape
        if frame_shape[0] == frame_shape[1]:
            proper_frame = frame
            break

    fig = plt.figure(figsize=(5, 5))
    im = plt.imshow(data[proper_frame], origin='lower')
    anim = FuncAnimation(fig, animate, frames=num_frames, interval=1, repeat=False)
    plt.show()