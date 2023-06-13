import imageio.v3 as iio
import matplotlib.pyplot as plt

# im = iio.imread('imageio:chelsea.png')
# print(im.shape)

# index=None means: read all images in the file and stack along first axis
frames = iio.imread("imageio:newtonscradle.gif", index=None)
# ndarray with (num_frames, height, width, channel)
print(frames.shape)  # (36, 150, 200, 3)
print(frames[0].shape)
#
# plt.imshow(frames[35])
# plt.show()
