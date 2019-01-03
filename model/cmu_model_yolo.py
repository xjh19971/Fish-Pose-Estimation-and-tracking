from keras.models import Model
from keras.layers.merge import Concatenate
from keras.layers import Activation, Input, Lambda, ZeroPadding2D
from keras.layers.convolutional import Conv2D, UpSampling2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import Multiply
from keras.regularizers import l2
from keras.initializers import random_normal, constant
from keras.layers import BatchNormalization, Add, ReLU
import keras.backend as K

KEY_POINT_NUM = 3 + 1
KEY_POINT_LINK = 2 * 2


def STEM_block(input_tensor, filters, stage, weight_decay):
    """conv_block is the block that has a conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    Note that from stage 3, the first conv layer at main path is with strides=(2,2)
    And the shortcut should have strides=(2,2) as well
    """
    filters4 = filters
    bn_axis = 1 if K.image_data_format() == 'channels_first' else -1
    conv_name_base = 'inception' + str(stage) + '_branch'
    bn_name_base = 'bn' + str(stage) + '_branch'

    '''x1 = conv(input_tensor, filters1[0], 1, conv_name_base + 'a1', weight_decay)
    x1 = BatchNormalization(axis=bn_axis, name=bn_name_base + 'a1', epsilon=1e-5, momentum=0.9)(x1)
    x1 = relu(x1)
    x1 = pooling(x1, 2, 2)'''

    x4 = conv(input_tensor, filters4[0], 3, conv_name_base + 'c1', weight_decay, strides=(2, 2))
    x4 = BatchNormalization(axis=bn_axis, name=bn_name_base + 'c1', epsilon=1e-5, momentum=0.9)(x4)
    x4 = relu(x4)
    x4 = conv(x4, filters4[1], 3, conv_name_base + 'c2', weight_decay, strides=(2, 2))
    x4 = BatchNormalization(axis=bn_axis, name=bn_name_base + 'c2', epsilon=1e-5, momentum=0.9)(x4)
    x = relu(x4)
    return x


def relu(x):
    return ReLU(6.)(x)


def conv(x, nf, ks, name, weight_decay, strides=None):
    kernel_reg = l2(weight_decay[0]) if weight_decay else None
    if strides == None:
        x = Conv2D(nf, (ks, ks), padding='same', name=name,
                   kernel_regularizer=kernel_reg,
                   kernel_initializer=random_normal(stddev=0.01),
                   use_bias=False
                   )(x)
    else:
        x = Conv2D(nf, (ks, ks), padding='same', name=name, strides=strides,
                   kernel_regularizer=kernel_reg,
                   kernel_initializer=random_normal(stddev=0.01),
                   use_bias=False)(x)
    return x


def pooling(x, ks, st):
    x = MaxPooling2D((ks, ks), strides=(st, st))(x)
    return x


def vgg_block(x, weight_decay):
    ''' # Block 1
    x = conv(x, 64, 3, "conv1_1", (weight_decay, 0))
    x = relu(x)
    x = conv(x, 64, 3, "conv1_2", (weight_decay, 0))
    x = relu(x)
    x = pooling(x, 2, 2, "pool1_1")

    # Block 2
    x = conv(x, 128, 3, "conv2_1", (weight_decay, 0))
    x = relu(x)
    x = conv(x, 128, 3, "conv2_2", (weight_decay, 0))
    x = relu(x)
    x = pooling(x, 2, 2, "pool2_1")

    # Block 3
    x = conv(x, 256, 3, "conv3_1", (weight_decay, 0))
    x = relu(x)
    x = conv(x, 256, 3, "conv3_2", (weight_decay, 0))
    x = relu(x)
    x = conv(x, 256, 3, "conv3_3", (weight_decay, 0))
    x = relu(x)
    x = conv(x, 256, 3, "conv3_4", (weight_decay, 0))
    x = relu(x)
    x = pooling(x, 2, 2, "pool3_1")

    # Block 4
    x = conv(x, 512, 3, "conv4_1", (weight_decay, 0))
    x = relu(x)
    x = conv(x, 512, 3, "conv4_2", (weight_decay, 0))
    x = relu(x)

    # Additional non vgg layers
    x = conv(x, 256, 3, "conv4_3_CPM", (weight_decay, 0))
    x = relu(x)
    x = conv(x, 128, 3, "conv4_4_CPM", (weight_decay, 0))
    x = relu(x)'''

    '''
    x = conv(x, 64, 3, "conv1_1", (weight_decay, 0),strides=(2,2))                 #64
    x = BatchNormalization(axis=bn_axis, epsilon=1e-5, momentum=0.9)(x)
    x = relu(x)
    x = conv(x, 64, 3, "conv1_2", (weight_decay, 0))                               #64
    x = BatchNormalization(axis=bn_axis, epsilon=1e-5, momentum=0.9)(x)
    x = relu(x)
    # Block 2
    x = conv(x, 128, 3, "conv2_1", (weight_decay, 0),strides=(2,2))                 #128
    x = BatchNormalization(axis=bn_axis, epsilon=1e-5, momentum=0.9)(x)
    x = relu(x)
    x = conv(x, 128, 3, "conv2_2", (weight_decay, 0),strides=(2,2))                 #256
    x = BatchNormalization(axis=bn_axis, epsilon=1e-5, momentum=0.9)(x)
    x = relu(x)
    '''
    x = STEM_block(x, [64, 64], 1, (weight_decay, 0))
    return x


def stage1_block(x, num_p, branch, weight_decay):
    bn_axis = 1 if K.image_data_format() == 'channels_first' else -1
    # Block 1
    x1 = x
    x = conv(x, 64, 3, "Mconv1_stage1_L%d" % branch, (weight_decay, 0))
    x = BatchNormalization(axis=bn_axis, epsilon=1e-5, momentum=0.9)(x)
    x = relu(x)
    x = conv(x, 64, 3, "Mconv2_stage1_L%d" % branch, (weight_decay, 0))
    x = BatchNormalization(axis=bn_axis, epsilon=1e-5, momentum=0.9)(x)
    x = Add()([x, x1])
    x = relu(x)
    x = conv(x, 128, 1, "Mconv3_stage1_L%d" % branch, (weight_decay, 0))
    x = BatchNormalization(axis=bn_axis, epsilon=1e-5, momentum=0.9)(x)
    x = relu(x)
    x2 = x
    x = conv(x, 128, 3, "Mconv4_stage1_L%d" % branch, (weight_decay, 0))
    x = BatchNormalization(axis=bn_axis, epsilon=1e-5, momentum=0.9)(x)
    x = relu(x)
    x = conv(x, 128, 3, "Mconv5_stage1_L%d" % branch, (weight_decay, 0))
    x = BatchNormalization(axis=bn_axis, epsilon=1e-5, momentum=0.9)(x)
    x = Add()([x, x2])
    x= relu(x)
    '''x = conv(x, 256, 1, "Mconv4_stage1_L%d" % branch, (weight_decay, 0))
    x = BatchNormalization(axis=bn_axis, epsilon=1e-5, momentum=0.9)(x)
    x = relu(x)'''
    return x


def stageT_block(x, num_p, stage, branch, weight_decay):
    bn_axis = 1 if K.image_data_format() == 'channels_first' else -1
    # Block 1
    x = conv(x, 128, 1, "Mconv1_stage%d_L%d" % (stage, branch), (weight_decay, 0))
    x = BatchNormalization(axis=bn_axis, epsilon=1e-5, momentum=0.9)(x)
    x1 = relu(x)
    x = conv(x, 128, 3, "Mconv2_stage%d_L%d" % (stage, branch), (weight_decay, 0), strides=(2, 2))
    x = BatchNormalization(axis=bn_axis, epsilon=1e-5, momentum=0.9)(x)
    x = relu(x)
    x = conv(x, 128, 3, "Mconv3_stage%d_L%d" % (stage, branch), (weight_decay, 0))
    x = BatchNormalization(axis=bn_axis, epsilon=1e-5, momentum=0.9)(x)
    x1 = conv(x1, 128, 1, "Mconv6_stage%d_L%d" % (stage, branch), (weight_decay, 0), strides=(2, 2))
    x = BatchNormalization(axis=bn_axis, epsilon=1e-5, momentum=0.9)(x)
    x = Add()([x, x1])
    x = relu(x)
    x2 = x
    x = conv(x, 128, 3, "Mconv4_stage%d_L%d" % (stage, branch), (weight_decay, 0))
    x = BatchNormalization(axis=bn_axis, epsilon=1e-5, momentum=0.9)(x)
    x = relu(x)
    x = conv(x, 128, 3, "Mconv5_stage%d_L%d" % (stage, branch), (weight_decay, 0))
    x = BatchNormalization(axis=bn_axis, epsilon=1e-5, momentum=0.9)(x)
    x = Add()([x, x2])
    x= relu(x)
    '''x = conv(x, 256, 1, "Mconv6_stage%d_L%d" % (stage, branch), (weight_decay, 0))
    x = BatchNormalization(axis=bn_axis, epsilon=1e-5, momentum=0.9)(x)
    x = relu(x)'''
    return x


def get_training_model(weight_decay):
    bn_axis = 1 if K.image_data_format() == 'channels_first' else -1
    stages = 3
    np_branch1 = KEY_POINT_LINK
    np_branch2 = KEY_POINT_NUM
    img_size = 368
    img_input_shape = (img_size, img_size, 3)
    vec_input_shape = (46, 46, KEY_POINT_LINK)
    heat_input_shape = (46, 46, KEY_POINT_NUM)

    inputs = []
    outputs = []
    outputstemp = []

    img_input = Input(shape=img_input_shape)
    vec_weight_input = Input(shape=vec_input_shape)
    heat_weight_input = Input(shape=heat_input_shape)

    inputs.append(img_input)
    inputs.append(vec_weight_input)
    inputs.append(heat_weight_input)

    img_normalized = Lambda(lambda x: x / 256 - 0.5)(img_input)  # [-0.5, 0.5]

    # VGG
    stage0_out = vgg_block(img_normalized, weight_decay)

    # stage 1 - branch 1 (PAF)
    stage1_branch1_out = stage1_block(stage0_out, np_branch1, 1, weight_decay)

    # stage 1 - branch 2 (confidence maps)
    stage1_branch2_out = stage1_block(stage0_out, np_branch2, 2, weight_decay)

    x = Concatenate()([stage1_branch1_out, stage1_branch2_out, stage0_out])

    outputstemp.append(stage1_branch1_out)
    outputstemp.append(stage1_branch2_out)
    # stage sn >= 2
    for sn in range(2, stages + 1):
        # stage SN - branch 1 (PAF)
        stageT_branch1_out = stageT_block(x, np_branch1, sn, 1, weight_decay)
        # stage SN - branch 2 (confidence maps)
        stageT_branch2_out = stageT_block(x, np_branch2, sn, 2, weight_decay)
        outputstemp.append(stageT_branch1_out)
        outputstemp.append(stageT_branch2_out)
        if (sn < stages):
            x = Concatenate()([stageT_branch1_out, stageT_branch2_out])
    w1up = None
    w2up = None
    for sn in range(stages, 0, -1):
        w1 = outputstemp[2 * sn - 2]
        w2 = outputstemp[2 * sn - 1]
        w1_name = "weight_stage%d_L%d" % (sn, 1)
        w2_name = "weight_stage%d_L%d" % (sn, 2)
        if sn != stages:
            w1 = Concatenate()([w1, w1up])
            w2 = Concatenate()([w2, w2up])
            outputstemp[2 * sn - 2] = w1
            outputstemp[2 * sn - 1] = w2
        w1 = conv(w1, np_branch1, 1, "up_stage%d_L%d" % (sn, 1), (weight_decay, 0))
        w1 = BatchNormalization(name=w1_name, axis=bn_axis, epsilon=1e-5, momentum=0.9)(w1)
        w2 = conv(w2, np_branch2, 1, "up_stage%d_L%d" % (sn, 2), (weight_decay, 0))
        w2 = BatchNormalization(name=w2_name, axis=bn_axis, epsilon=1e-5, momentum=0.9)(w2)
        if sn != 1:
            w1up = UpSampling2D(size=(2, 2), data_format=None)(outputstemp[2 * sn - 2])
            w2up = UpSampling2D(size=(2, 2), data_format=None)(outputstemp[2 * sn - 1])
        outputs.append(w1)
        outputs.append(w2)
    model = Model(inputs=inputs, outputs=outputs)
    model.summary()
    return model


def get_testing_model():
    bn_axis = 1 if K.image_data_format() == 'channels_first' else -1
    stages = 3
    np_branch1 = KEY_POINT_LINK
    np_branch2 = KEY_POINT_NUM

    img_input_shape = (None, None, 3)

    img_input = Input(shape=img_input_shape)

    img_normalized = Lambda(lambda x: x / 256 - 0.5)(img_input)  # [-0.5, 0.5]

    # VGG
    stage0_out = vgg_block(img_normalized, None)

    # stage 1 - branch 1 (PAF)
    stage1_branch1_out = stage1_block(stage0_out, np_branch1, 1, None)

    # stage 1 - branch 2 (confidence maps)
    stage1_branch2_out = stage1_block(stage0_out, np_branch2, 2, None)

    x = Concatenate()([stage1_branch1_out, stage1_branch2_out, stage0_out])
    outputstemp = []
    outputs = []
    # stage t >= 2
    outputstemp.append(stage1_branch1_out)
    outputstemp.append(stage1_branch2_out)
    # stage sn >= 2
    for sn in range(2, stages + 1):
        # stage SN - branch 1 (PAF)
        stageT_branch1_out = stageT_block(x, np_branch1, sn, 1, None)
        # stage SN - branch 2 (confidence maps)
        stageT_branch2_out = stageT_block(x, np_branch2, sn, 2, None)
        outputstemp.append(stageT_branch1_out)
        outputstemp.append(stageT_branch2_out)
        if (sn < stages):
            x = Concatenate()([stageT_branch1_out, stageT_branch2_out])
    w1up = None
    w2up = None
    for sn in range(stages, 0, -1):
        w1 = outputstemp[2 * sn - 2]
        w2 = outputstemp[2 * sn - 1]
        w1_name = "weight_stage%d_L%d" % (sn, 1)
        w2_name = "weight_stage%d_L%d" % (sn, 2)
        if sn != stages:
            w1 = Concatenate()([w1, w1up])
            w2 = Concatenate()([w2, w2up])
            outputstemp[2 * sn - 2] = w1
            outputstemp[2 * sn - 1] = w2
        w1 = conv(w1, np_branch1, 1, "up_stage%d_L%d" % (sn, 1), (None, 0))
        w1 = BatchNormalization(name=w1_name, axis=bn_axis, epsilon=1e-5, momentum=0.9)(w1)
        w2 = conv(w2, np_branch2, 1, "up_stage%d_L%d" % (sn, 2), (None, 0))
        w2 = BatchNormalization(name=w2_name, axis=bn_axis, epsilon=1e-5, momentum=0.9)(w2)
        if sn != 1:
            w1up = UpSampling2D(size=(2, 2), data_format=None)(outputstemp[2 * sn - 2])
            w2up = UpSampling2D(size=(2, 2), data_format=None)(outputstemp[2 * sn - 1])
    outputs.append(w1)
    outputs.append(w2)
    model = Model(inputs=[img_input], outputs=outputs)
    model.summary()
    return model
