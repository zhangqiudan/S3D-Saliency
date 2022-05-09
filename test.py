import numpy as np
from os import listdir, makedirs
from os.path import isfile, join
import sys, getopt
import cv2
from S3DModel import S3DModel
import os
import glob

SCALE_LIST = [1, 3, 5]
input_shape = (224, 224, 3)


def main(argv):
    img_dir = ''

    if argv:
        opts, args = getopt.getopt(argv, "i:")
    else:
        print('Usage: python2 test.py -i <input_dir>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-i':
            img_dir = arg

    model_path = './models_checkpoint/'
    model_path_dir = glob.glob(model_path+'*.h5')
    model_path_dir.sort()
    out_dir = './save/'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    for i in range(1,3):

        model_weights = model_path_dir[i]
        model_name = os.path.basename(model_weights).split('.')[0]
        print('model_weights:',model_weights)
        print('model_name:',model_name)
        save_dir = out_dir  + model_name
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        images = [f for f in listdir(img_dir) if isfile(join(img_dir, f))]

        s = S3DModel(model_weights)

        for img_name in images:
            smap = s.compute_saliency(img_path=join(img_dir, img_name))
            img_name = img_name.split('.')[0] + '.jpg'
            smap = smap * 255
            cv2.imwrite(join(save_dir, img_name), smap)


if __name__ == "__main__":
    main(sys.argv[1:])
