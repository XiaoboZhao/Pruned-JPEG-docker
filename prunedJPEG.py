# -*- coding: utf-8 -*-
"""
input: image
output: 64 pruned JPEG images, and their sizes

for a given image, return 64 images by 1~64 DCT coefficients in each 8x8 block 
and their sizes

@author: Xiaobo Zhao
"""

import cv2
import numpy as np
import zigzag as zz

def pjpeg(image):

    B = 8
    h, w = np.array(image.shape[:2]) // B * B
    image = image[:h, :w]

    # Transform BGR to YCrCb and Subsample Chrominance Channels
    transcol = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    SSV = 1
    SSH = 1
    crf = cv2.boxFilter(transcol[:, :, 1], ddepth=-1, ksize=(2, 2))
    cbf = cv2.boxFilter(transcol[:, :, 2], ddepth=-1, ksize=(2, 2))
    crsub = crf[::SSV, ::SSH]
    cbsub = cbf[::SSV, ::SSH]
    imSub = [transcol[:, :, 0], crsub, cbsub]

    
    TransAll = []
    TransAllQuant = []
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

    output = []
    cof_n_pos = []
    image_size = []

    for n in range(64):
        temp = np.zeros((h, w, 3), np.uint8)
        pos_n_channel = []
        for idx, channel in enumerate(TransAll):
            channelrows = channel.shape[0]
            channelcols = channel.shape[1]
            blocksV = channelrows // B
            blocksH = channelcols // B
            back0 = np.zeros((channelrows, channelcols), np.uint8)
            pos_n = []
            for row in range(blocksV):
                for col in range(blocksH):
                    n_cof = np.zeros(B * B, np.float32)
                    all_cof = zz.zigzag(channel[row * B:(row + 1) * B, col * B:(col + 1) * B])
                    pos_n.append(all_cof[n])
                    n_cof[:n + 1] = all_cof[:n + 1]
                    subblock = zz.inverse_zigzag(n_cof, B, B)
                    currentblock = np.round(cv2.idct(subblock)) + 128
                    currentblock[currentblock > 255] = 255
                    currentblock[currentblock < 0] = 0
                    back0[row * B:(row + 1) * B, col * B:(col + 1) * B] = currentblock
            pos_n_channel += pos_n
            back1 = cv2.resize(back0, (w, h))
            temp[:, :, idx] = np.round(back1)

        cof_n_pos += pos_n_channel
        reImg = cv2.cvtColor(temp, cv2.COLOR_YCrCb2BGR)
        cv2.imwrite('Image.jpg', reImg)
        image_size.append(len(open('Image.jpg', 'rb').read()))
        output.append(reImg)

    return output, np.array(image_size)
