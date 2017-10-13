import sys
import os
import random

if len(sys.argv) == 2:
    gpu_id = sys.argv[1]
else:
    gpu_id = "gpu0"
    cnmem = "0.80"
print("Argument: gpu={}, mem={}".format(gpu_id, cnmem))
os.environ["THEANO_FLAGS"] = "device=" + gpu_id + ", lib.cnmem=" + cnmem

from models import net
from batch_generators import load_names, load_images, load_names_val

import warnings
warnings.filterwarnings("ignore")

random.seed(1769)
from matplotlib import pyplot as plt
from keras.callbacks import EarlyStopping, ModelCheckpoint,Callback


class PlotLosses(Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []

        self.fig = plt.figure()

        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.i += 1
        plt.plot(self.x, self.losses, label="loss")
        plt.plot(self.x, self.val_losses, label="val_loss")
        plt.legend()
        #plt.show();
        self.fig.savefig(WEIGHT+'/plot_losses.png')
        self.fig = plt.figure()


if __name__ == '__main__':

    # image dimensions
    rows = 424
    cols = 512

    # deep parameters
    patience = 100
    batch_size = 10
    n_epoch = 100

    # training parameters
    data_augmentation = False
    b_crop = False
    b_rescale = True
    b_scale = True
    b_normcv2 = False
    b_tanh = False
    limit_train = 20
    #limit_train = -1
    limit_test = 10

    #limit_test = -1
    b_debug = False
    fulldepth = False
    removeBackground = False
    equalize = False

    WEIGHT = 'weights'

    model = net(input_shape=(1, rows, cols))

    model.summary()

    # load train name
    train_data_names = load_names(dataset=0)
    train_data_names = train_data_names + load_names(dataset=1)
    # train_data_names = load_names(dataset=2)
    random.shuffle(train_data_names)
    # load validation name
    val_data_names = load_names_val()
    # data augmentation
    if data_augmentation:
        for i in range(1,9):
            tmp = load_names(augm=i)
            train_data_names = train_data_names + tmp

    # train data cut
    random.shuffle(train_data_names)
    if limit_train == -1:
        limit_train = len(train_data_names)
    train_data_names = train_data_names[:limit_train]
    random.shuffle(val_data_names)
    # validation data cut
    if limit_test == -1:
        limit_test = len(val_data_names)
    val_data_names = val_data_names[:limit_test]

    def generator():
        random.shuffle(train_data_names)
        while True:
            for it in range(0, len(train_data_names), batch_size):
                X, Y = load_images(train_data_names[it:it + batch_size],division=8,b_debug=b_debug,crop=b_crop, rescale=b_rescale, scale=b_scale, normcv2=b_normcv2, fulldepth=fulldepth, rows=rows, cols=cols,removeBackground=removeBackground,equalize=equalize)
                yield X, Y

    def generator_val():
        while True:
            for it in range(0, len(val_data_names), batch_size):
                val_data_X, val_data_Y = load_images(val_data_names[it:it + batch_size],division=8, crop=b_crop, rescale=b_rescale, scale=b_scale, b_debug=b_debug, normcv2=b_normcv2, rows=rows,fulldepth=fulldepth, cols=cols,removeBackground=removeBackground,equalize=equalize)
                yield val_data_X,val_data_Y

    plot_losses = PlotLosses()
    print('start train')
    model.fit_generator(generator(),
                        nb_epoch=n_epoch,
                        validation_data=generator_val(),
                        validation_steps=len(val_data_names)/batch_size,
                        verbose=1,
                        steps_per_epoch=len(train_data_names)/batch_size,
                        callbacks=[plot_losses,EarlyStopping(patience=patience),
                        ModelCheckpoint(WEIGHT+"/weights.{epoch:03d}-{val_loss:.5f}.hdf5", save_best_only=True)]
                            )
