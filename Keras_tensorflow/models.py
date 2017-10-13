from keras.models import *
from keras.layers import *
from keras.optimizers import SGD

def vgg19(input):

    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', name='conv1_1', padding='same')(input)
    x = Conv2D(64, (3, 3), activation='relu', name='conv1_2', padding='same')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool1_stage1')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', name='conv2_1', padding='same')(x)
    x = Conv2D(128, (3, 3), activation='relu', name='conv2_2', padding='same')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool2_stage1')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', name='conv3_1', padding='same')(x)
    x = Conv2D(256, (3, 3), activation='relu', name='conv3_2', padding='same')(x)
    x = Conv2D(256, (3, 3), activation='relu', name='conv3_3', padding='same')(x)
    x = Conv2D(256, (3, 3), activation='relu', name='conv3_4', padding='same')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool3_stage1')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', name='conv4_1', padding='same')(x)
    x = Conv2D(512, (3, 3), activation='relu', name='conv4_2', padding='same')(x)
    x = Conv2D(256, (3, 3), activation='relu', name='conv4_3_CPM', padding='same')(x)
    x = Conv2D(128, (3, 3), activation='relu', name='conv4_4_CPM', padding='same')(x)

    return x

def stage_1(vgg_out):

    x = Conv2D(128, (3, 3), name='conv5_1_CPM_L2', padding='same', activation='relu')(vgg_out)
    x = Conv2D(128, (3, 3), name='conv5_2_CPM_L2', padding='same', activation='relu')(x)
    x = Conv2D(128, (3, 3), name='conv5_3_CPM_L2', padding='same', activation='relu')(x)
    x = Conv2D(512, (1, 1), name='conv5_4_CPM_L2', padding='same', activation='relu')(x)
    x2 = Conv2D(1, (1, 1), name='conv5_5_CPM_L2', padding='same', activation='linear')(x)
    return x2

def stage_n(prevl2_out, vgg_out, stage_number):

    if stage_number <= 1:
        raise ValueError('[\'stage_number\' = {}]: \'stage_number\' must be > 1'.format(stage_number))

    stage='_stage{}_'.format(stage_number)

    concat_axis = 1
    input = Concatenate(axis=concat_axis, name='contact_stage' + str(stage_number))([ prevl2_out, vgg_out])

    x = Conv2D(128, (7, 7), name='Mconv1' + stage + 'L2', padding='same', activation='relu')(input)
    x = Conv2D(128, (7, 7), name='Mconv2' + stage + 'L2', padding='same', activation='relu')(x)
    x = Conv2D(128, (7, 7), name='Mconv3' + stage + 'L2', padding='same', activation='relu')(x)
    x = Conv2D(128, (7, 7), name='Mconv4' + stage + 'L2', padding='same', activation='relu')(x)
    x = Conv2D(128, (7, 7), name='Mconv5' + stage + 'L2', padding='same', activation='relu')(x)
    x = Conv2D(128, (1, 1), name='Mconv6' + stage + 'L2', padding='same', activation='relu')(x)
    x2 = Conv2D(1, (1, 1), name='Mconv7' + stage + 'L2', padding='same', activation='linear')(x)
    return (x2)

def net(input_shape=None, weights_path=None):

    if input_shape == None:
        input_shape = (1, None, None)
    input_image = Input(shape=input_shape, name='input')
    # VGG19
    vgg_out = vgg19(input_image)
    # Stage 1
    x2 = stage_1(vgg_out)
    # Stage 2, 3
    for i in range(2,4):
        x2 = stage_n(x2, vgg_out, stage_number=i)
    # final layer
    x2 = Conv2D(2, (1, 1), name='FINAL', padding='same', activation=None)(x2)
    x2 = Permute((2, 3, 1))(x2)
    x2 = Activation("softmax")(x2)
    # creating model
    model = Model(inputs=input_image, outputs=[x2])
    if weights_path != None:
        print('loading weights')
        model.load_weights(filepath=weights_path)
        print('done')
    opt = SGD(lr=0.01, momentum=0.8, decay=1e-6, nesterov=False)
    model.compile(optimizer=opt, loss='binary_crossentropy')
    return model

if __name__ == '__main__':
    rows = 424
    cols = 512

    model = net(input_shape=(1, rows, cols))
    model.summary()

