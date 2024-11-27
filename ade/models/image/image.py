
import numpy as np
import scipy
import cv2

import scipy
from scipy import ndimage
from scipy.stats import poisson
from scipy.stats import multivariate_normal
from scipy import signal

from skimage.color import rgb2gray
from skimage.transform import warp
from skimage.registration import optical_flow_tvl1, optical_flow_ilk


x_size = 32
y_size = 40
rate = 10
duration = 2.0

def align_images(images):

    # TODO: alignment should shrink the final image, instead of padding with noise

    alignments = []

    for i in range(images.shape[0] - 1):
    
        x_vel, y_vel = dense_optical_flow(images[i], images[i + 1])

        alignments.append([np.rint(x_vel), np.rint(y_vel)])

    x_shift = 0
    y_shift = 0

    new_images = [images[0]]

    out_shape = (images[0].shape[1], images[0].shape[0])

    for i in range(images.shape[0] - 1):

        align = alignments[i]

        x_shift = int(x_shift + align[0])
        y_shift = int(y_shift + align[1])

        translation_matrix = np.float32([[1, 0, -x_shift], [0, 1, -y_shift]])

        new_img = cv2.warpAffine(images[i + 1], translation_matrix, out_shape)

        # TODO: this should be sensitive to sign, and also handle y
        synth_background = make_background()
        img = cv2.resize(synth_background, out_shape)
        new_img[:, -x_shift:] = img[:, -x_shift:]

        new_images.append(new_img)

    return np.array(new_images)


def dense_optical_flow(img0, img1):
    radius = 50
    v, u = optical_flow_ilk(img0, img1, radius=radius)

    return np.mean(u), np.mean(v)


def fft_filter(data):
    # (32, 40)
    x_size = data.shape[1]
    y_size = data.shape[2]

    output_img = np.zeros((x_size, y_size), dtype=int)

    for y in range(y_size):
        for x in range(x_size):

            sig = data[-10:, x, y]
            
            freqs = scipy.fftpack.rfftfreq(sig.size, d=0.1)
            rfft = scipy.fftpack.rfft(sig)
            amps = 2/20 * rfft

            rfft[0] = 0
            res = scipy.fftpack.irfft(rfft)
    
            # filter
            k = 0.5
            shift = sig_mean - res_mean
            amp = amps[1:].max()
            thresh = np.abs((shift - amp) * k)
            res[res <= thresh] = 0

            res[res <= 0] = 0
            
            output_img[x, y] = res.max()

    return output_img


def fft_filter_id(data, stage):
    # (32, 40)
    x_size = data.shape[1]
    y_size = data.shape[2]


    output_img = np.zeros((x_size, y_size), dtype=int)
    freq_img = np.zeros((x_size, y_size), dtype=int)

    for y in range(y_size):
        for x in range(x_size):

            sig = data[-10:, x, y]
            
            freqs = scipy.fftpack.rfftfreq(sig.size, d=0.1)
            rfft = scipy.fftpack.rfft(sig)
            amps = 2/20 * rfft
            
            rfft[0] = 0
            # rfft[-1] = 0
            res = scipy.fftpack.irfft(rfft)
    
            ### Filter
            k = 0.5
            shift = sig_mean - res_mean
            amp = amps[1:].max()
            thresh = np.abs((shift - amp) * k)
            res[res <= thresh] = 0
            res[res <= 0] = 0

            max_freq = np.argmax(np.abs(rfft))
            max_f = freqs[max_freq]
            
            output_img[x, y] = res.max()
            freq_img[x, y] = int(max_f)

    
    return output_img, freq_img


def create_synth_image_moving():

    times = np.arange(0, 5, 0.1)
    length = times.shape[0]
        
    poses = []
    for t in times:
        # create pose
        pose = {"x": t, "y": 0.0, "z": 0.0}
        poses.append(pose)

    # [x, y, z, period*10]
    beacon_arr = [[1.0, 1.0, 0.0, 0.2, 1.0], [1.0, -1.0, 0.0, 0.4, 1.0], [-1.0, -1.0, 0.0, 1.0, 1.0], [-1.0, 1.0, 0.0, 0.2, 1.0],
                  [1.0, 0.0, 0.0, 1.0, 0.0], [0.0, -1.0, 0.0, 1.0, 0.0], [-1.0, 0.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0, 0.0]]

    x_size = 32
    y_size = 40
    
    scale = 100
    x_off = 256
    y_off = 120
    side = 30


    for i, pose in enumerate(poses):
        synth_img = make_background()
        for x in beacon_arr:

            period = x[3]
            if i % ((period / 2.0) * rate) == 0:
                x[4] = x[4] * -1.0

            if x[4] >= 0:
                x_ind = int(x[0] * scale + x_off)
                y_ind = int(x[1] * scale + y_off + i * 16)
                x_0 = int(x_ind-side/2)
                x_1 = int(x_ind+side/2)
                y_0 = int(y_ind-side/2)
                y_1 = int(y_ind+side/2)
    
                synth_img[x_0:x_1, y_0:y_1] = make_beacon()
    
        ### reduce resolution from (512, 640) to (64, 80)
        images.append(synth_img)
        img = cv2.resize(synth_img, (y_size, x_size))

        yield {"data": img,
               "timestamp": 0,
               "topic": "images",
               "name": "synth_ir"}




def binarization(img, min=0, max=255):
    
    ret, im = cv2.threshold(img.astype(np.uint8), min, max, cv2.THRESH_BINARY)
    
    contours = cv2.findContours(im, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    centroids = []
    
    for contour in contours[0]:
        x_list = []
        y_list = []
        for point in contour:
            y = point[0][0]
            x = point[0][1]
            x_list.append(x)
            y_list.append(y)

        a_max = np.argmax(img[x_list, y_list])

        x_mean = x_list[a_max]
        y_mean = y_list[a_max]
    
        centroids.append([x_mean, y_mean])

    return centroids, im



def create_synth_image(t, pose):
    # [x, y, z, period*10]
    beacon_arr = [[1.0, 1.0, 0.0, 5.0], [1.0, -1.0, 0.0, 4.0], [-1.0, -1.0, 0.0, 5.0], [-1.0, 1.0, 0.0, 4.0], 
                  [1.0, 0.0, 0.0, 1.0], [0.0, -1.0, 0.0, 1.0], [-1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0]]

    scale = 100
    x_off = 256
    y_off = 320
    side = 30
    
    synth_img = make_background()
    for x in beacon_arr:
        period = x[3]
        ts = t * 10
        remainder = np.rint(ts % period)
        if period == 2.5:
            print("ts: ", ts, " period: ", period, " remainder... ", remainder)
        if remainder == 0:
            x_ind = int(x[0] * scale + x_off)
            y_ind = int(x[1] * scale + y_off)
            x_0 = int(x_ind-side/2)
            x_1 = int(x_ind+side/2)
            y_0 = int(y_ind-side/2)
            y_1 = int(y_ind+side/2)

            # every 0.1 second shifts the beacons one pixel
            shift = pose["x"] * 10

            synth_img[x_0:x_1, y_0:y_1] = make_beacon()

    img = cv2.resize(synth_img, (x_size, y_size))

    return img


def make_background():
    mean_dark = 15.5375
    std_dark = 2.143087613521848
    window_shape = (512, 640)
    synth_img = np.random.normal(mean_dark, std_dark, window_shape)
    return synth_img


def make_beacon():
    std_beacon = 41.72827445376682
    mean_dark = 15.5375
    std_dark = 2.143087613521848
    window_shape = (512, 640)
    beacon_max = 200
    mean_dark = 15.5375
    bound_x = 15
    bound_y = 15
    step = 1
    x, y = np.mgrid[-bound_x:bound_x:step, -bound_y:bound_y:step]
    pos = np.dstack((x, y))
    mean_x = 0
    mean_y = 0
    std_beacon = np.std(beacon) # 41.72827445376682
    std_x = std_beacon
    std_y = std_beacon
    rv = multivariate_normal([mean_x, mean_y], [[std_x, 0], [0, std_y]])
    z = rv.pdf(pos)
    diff_beacon_background = beacon.max() - np.mean(dark)
    fake_beacon = ((z - z.min()) / z.max()) * diff_beacon_background + np.mean(dark)
    return fake_beacon


