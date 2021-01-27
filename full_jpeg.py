"""
Generate the JPEG images with full (64) DCT coefficients, use these images
instead of the original images from datasets to make sure that the image 
sizes are only dependent on the number of DCT coefficients, not dependent on
the compression parameters when JPEG are generated.
"""

# import the necessary packages
import numpy as np
import cv2
import sys
import glob
import os

# Input image folder
IM_DIR = 'test_images'

# Get path to current working directory
CWD_PATH = os.getcwd()
PATH_TO_IMAGES = os.path.join(CWD_PATH, IM_DIR)
images = glob.glob(PATH_TO_IMAGES + '/*')

# Output image folder
OUT_DIR = 'test_images'
B = 8

for (x, image_path) in enumerate(images):

    image_name = image_path.split("/")[-1]
    image = cv2.imread(image_path)
    h, w = np.array(image.shape[:2]) // B * B
    image = image[:h, :w]

    transcol = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    SSV = 1
    SSH = 1
    crf = cv2.boxFilter(transcol[:, :, 1], ddepth=-1, ksize=(2, 2))
    cbf = cv2.boxFilter(transcol[:, :, 2], ddepth=-1, ksize=(2, 2))
    crsub = crf[::SSV, ::SSH]
    cbsub = cbf[::SSV, ::SSH]
    imSub = [transcol[:, :, 0], crsub, cbsub]

    TransAll = []

    for idx, channel in enumerate(imSub):        
        channelrows = channel.shape[0]
        channelcols = channel.shape[1]
        Trans = np.zeros((channelrows, channelcols), np.float32)
        TransQuant = np.zeros((channelrows, channelcols), np.float32)
        blocksV = channelrows // B
        blocksH = channelcols // B
        vis0 = np.zeros((channelrows, channelcols), np.float32)
        vis0[:channelrows, :channelcols] = channel
        vis0 = vis0 - 128
        for row in range(blocksV):
            for col in range(blocksH):
                currentblock = cv2.dct(vis0[row * B:(row + 1) * B, col * B:(col + 1) * B])
                Trans[row * B:(row + 1) * B, col * B:(col + 1) * B] = currentblock

        TransAll.append(Trans)

    """
    Decoding
    """
    DecAll = np.zeros((h, w, 3), np.uint8)
    for idx, channel in enumerate(TransAll):
        channelrows = channel.shape[0]
        channelcols = channel.shape[1]
        blocksV = channelrows // B
        blocksH = channelcols // B
        back0 = np.zeros((channelrows, channelcols), np.uint8)
        for row in range(blocksV):
            for col in range(blocksH):
                dequantblock = channel[row * B:(row + 1) * B, col * B:(col + 1) * B]
                currentblock = np.round(cv2.idct(dequantblock)) + 128
                currentblock[currentblock > 255] = 255
                currentblock[currentblock < 0] = 0
                back0[row * B:(row + 1) * B, col * B:(col + 1) * B] = currentblock
        back1 = cv2.resize(back0, (w, h))
        DecAll[:, :, idx] = np.round(back1)

    reImg = cv2.cvtColor(DecAll, cv2.COLOR_YCrCb2BGR)
    cv2.imwrite('%s/%s' %(OUT_DIR, image_name), reImg)
