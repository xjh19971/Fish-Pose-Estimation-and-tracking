from keras.models import Model
from keras.layers.merge import Concatenate
from keras.layers import Activation, Input, Lambda, ZeroPadding2D, UpSampling2D, Conv2DTranspose
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.regularizers import l2
from keras.initializers import he_normal
from keras.layers import BatchNormalization, add, ReLU, DepthwiseConv2D
import keras.backend as K

KEY_POINT_NUM = 3 + 1
KEY_POINT_LINK = 2 * 2


def STEM_block_trans(input_tensor, filters, stage, weight_decay, change=False, branch=None, strides=(1, 1)):
    filters4 = filters
    bn_axis = 1 if K.image_data_format() == 'channels_first' else -1
    if branch != None:
        conv_name_base = 'inception' + str(stage) + '_branch' + str(branch)
        bn_name_base = 'bn' + str(stage) + '_branch' + str(branch)
    else:
        conv_name_base = 'inception' + str(stage) + '_branch'
        bn_name_base = 'bn' + str(stage) + '_branch'

    x4 = convtrans(input_tensor, filters4[0], 3, conv_name_base + 'd1', weight_decay)
    x4 = BatchNormalization(axis=bn_axis, name=bn_name_base + 'd1')(x4)
    x4 = relu(x4)
    x4 = convtrans(x4, filters4[1], 3, conv_name_base + 'd2', weight_decay, strides=strides)
    x4 = BatchNormalization(axis=bn_axis, name=bn_name_base + 'd2')(x4)
    if change == True:
        shortcut = convtrans(input_tensor, filters4[1], 1, conv_name_base + '1', weight_decay, strides=strides)
        shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)
    else:
        shortcut = input_tensor
    x = add([x4, shortcut])
    x = relu(x)
    return x


def STEM_block(input_tensor, filters, stage, weight_decay, change=False, branch=None):
    filters4 = filters
    bn_axis = 1 if K.image_data_format() == 'channels_first' else -1
    if branch != None:
        conv_name_base = 'inception' + str(stage) + '_branch' + str(branch)
        bn_name_base = 'bn' + str(stage) + '_branch' + str(branch)
    else:
        conv_name_base = 'inception' + str(stage) + '_branch'
        bn_name_base = 'bn' + str(stage) + '_branch'

    x4 = conv(input_tensor, filters4[0], 3, conv_name_base + 'd1', weight_decay)
    x4 = BatchNormalization(axis=bn_axis, name=bn_name_base + 'd1')(x4)
    x4 = relu(x4)
    x4 = conv(x4, filters4[1], 3, conv_name_base + 'd2', weight_decay)
    x4 = BatchNormalization(axis=bn_axis, name=bn_name_base + 'd2')(x4)
    if change == True:
        shortcut = conv(input_tensor, filters4[1], 1, conv_name_base + '1', weight_decay)
        shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + 'd3')(shortcut)
    else:
        shortcut = input_tensor
    x = add([x4, shortcut])
    x = relu(x)
    return x


def tiny_inception_block(input_tensor, filters, stage, branch, weight_decay):
    filters2 = filters
    bn_axis = 1 if K.image_data_format() == 'channels_first' else -1
    conv_name_base = 'tinyinception' + str(stage) + '_branch' + str(branch)
    bn_name_base = 'bn' + str(stage) + '_branch' + str(branch)

    x2 = conv(input_tensor, filters2[0], 3, conv_name_base + 'b1', weight_decay)
    x2 = BatchNormalization(axis=bn_axis, name=bn_name_base + 'b1')(x2)
    x2 = relu(x2)
    x2 = conv(x2, filters2[1], 3, conv_name_base + 'b2', weight_decay)
    x2 = BatchNormalization(axis=bn_axis, name=bn_name_base + 'b2')(x2)
    x = x2
    x = add([x, input_tensor])
    x = relu(x)
    return x


def relu(x):
    return Activation('relu')(x)


def convtrans(x, nf, ks, name, weight_decay, strides=(1, 1), use_bias=False):
    kernel_reg = l2(weight_decay[0]) if weight_decay else None
    bias_reg = l2(weight_decay[0]) if weight_decay else None
    x = Conv2DTranspose(nf, (ks, ks), padding='same', name=name, strides=strides,
                        kernel_regularizer=kernel_reg,
                        bias_regularizer=bias_reg,
                        kernel_initializer=he_normal(),
                        use_bias=use_bias)(x)
    return x


def conv(x, nf, ks, name, weight_decay, strides=(1, 1), use_bias=False):
    kernel_reg = l2(weight_decay[0]) if weight_decay else None
    x = Conv2D(nf, (ks, ks), padding='same', name=name, strides=strides,
               kernel_regularizer=kernel_reg,
               kernel_initializer=he_normal(),
               use_bias=use_bias)(x)
    return x


def pooling(x, ks, st):
    x = MaxPooling2D((ks, ks), strides=(st, st))(x)
    return x


def vgg_block(x, weight_decay):
    bn_axis = 1 if K.image_data_format() == 'channels_first' else -1
    x = ZeroPadding2D((2, 2))(x)  # 对图片界面填充0，保证特征图的大小#
    x = conv(x, 64, 7, strides=(2, 2), name='conv1', weight_decay=(weight_decay, 0))  # 定义卷积层#
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)  # 批标准化#
    x = Activation('relu')(x)  # 激活函数#
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)  # 最大池化层#

    x = STEM_block(x, [64, 64], 1, (weight_decay, 0))
    x = STEM_block(x, [64, 64], 2, (weight_decay, 0))
    x = pooling(x, 2, 2)
    x = STEM_block(x, [128, 128], 3, (weight_decay, 0), change=True)
    x = STEM_block(x, [128, 128], 4, (weight_decay, 0))
    x = STEM_block(x, [128, 128], 5, (weight_decay, 0))
    x = STEM_block(x, [128, 128], 6, (weight_decay, 0))
    x1 = x
    x = STEM_block(x, [256, 256], 7, (weight_decay, 0), change=True)
    x = STEM_block(x, [256, 256], 8, (weight_decay, 0))
    x = STEM_block(x, [256, 256], 9, (weight_decay, 0))
    x = STEM_block(x, [256, 256], 10, (weight_decay, 0))
    return x, x1


def stage1_block(x, num_p, branch, weight_decay):
    bn_axis = 1 if K.image_data_format() == 'channels_first' else -1
    # Block 1
    x = conv(x, 128, 1, "Mconv1_stage1_L%d" % branch, (weight_decay, 0))
    x = BatchNormalization(axis=bn_axis)(x)
    x = relu(x)
    x = tiny_inception_block(x, [128, 128], 2 * 1 - 1, branch, (weight_decay, 0))
    x = tiny_inception_block(x, [128, 128], 2 * 1, branch, (weight_decay, 0))
    x1 = convtrans(x, num_p, 4, "Output_1_%d" % branch, (weight_decay, 0), strides=(2, 2), use_bias=True)
    x2 = conv(x1, int(x.shape[3]), 4, "Output2_1_%d" % branch, (weight_decay, 0), strides=(2, 2))
    x2 = BatchNormalization(name="Outputbn2_1_%d" % branch, axis=bn_axis)(x2)
    x2 = relu(x2)
    x = Concatenate()([x, x2])
    x = conv(x, int(int(x.shape[3]) / 2), 1, "Output3_1_%d" % branch, (weight_decay, 0))
    x = BatchNormalization(name="Outputbn3_1_%d" % branch, axis=bn_axis)(x)
    x = relu(x)
    return x, x1


def stageT_block(x, num_p, stage, branch, weight_decay):
    bn_axis = 1 if K.image_data_format() == 'channels_first' else -1
    # Block 1
    x = conv(x, 128, 1, "Mconv1_stage%d_L%d" % (stage, branch), (weight_decay, 0))
    x = BatchNormalization(axis=bn_axis)(x)
    x = relu(x)
    x = tiny_inception_block(x, [128, 128], 2 * stage - 1, branch, (weight_decay, 0))
    x = tiny_inception_block(x, [128, 128], 2 * stage, branch, (weight_decay, 0))
    x1 = convtrans(x, num_p, 4, "Output_%d_%d" % (stage, branch), (weight_decay, 0), strides=(2, 2), use_bias=True)
    x2 = conv(x1, int(x.shape[3]), 4, "Output2_%d_%d" % (stage, branch), (weight_decay, 0), strides=(2, 2))
    x2 = BatchNormalization(name="Outputbn2_%d_%d" % (stage, branch), axis=bn_axis)(x2)
    x2 = relu(x2)
    x = Concatenate()([x, x2])
    x = conv(x, int(int(x.shape[3]) / 2), 1, "Output3_%d_%d" % (stage, branch), (weight_decay, 0))
    x = BatchNormalization(name="Outputbn3_%d_%d" % (stage, branch), axis=bn_axis)(x)
    x = relu(x)
    return x, x1


def get_training_model(weight_decay):
    stages = 5
    np_branch1 = KEY_POINT_LINK
    np_branch2 = KEY_POINT_NUM
    img_size = 368
    img_input_shape = (img_size, img_size, 3)
    vec_input_shape = (None, None, KEY_POINT_LINK)
    heat_input_shape = (None, None, KEY_POINT_NUM)

    inputs = []
    outputs = []

    img_input = Input(shape=img_input_shape)
    vec_weight_input = Input(shape=vec_input_shape)
    heat_weight_input = Input(shape=heat_input_shape)

    inputs.append(img_input)
    inputs.append(vec_weight_input)
    inputs.append(heat_weight_input)

    img_normalized = Lambda(lambda x: x / 256 - 0.5)(img_input)  # [-0.5, 0.5]

    # VGG
    stage0_out, x1 = vgg_block(img_normalized, weight_decay)
    x = Concatenate()([x1, stage0_out])
    # stage 1 - branch 1 (PAF)
    stage1_branch1_out, realout1 = stage1_block(x, np_branch1, 1, weight_decay)

    # stage 1 - branch 2 (confidence maps)
    stage1_branch2_out, realout2 = stage1_block(x, np_branch2, 2, weight_decay)

    x = Concatenate()([stage1_branch1_out, stage1_branch2_out, x])

    outputs.append(realout1)
    outputs.append(realout2)

    # stage sn >= 2
    for sn in range(2, stages + 1):
        # stage SN - branch 1 (PAF)
        stageT_branch1_out, realout1 = stageT_block(x, np_branch1, sn, 1, weight_decay)
        # stage SN - branch 2 (confidence maps)
        stageT_branch2_out, realout2 = stageT_block(x, np_branch2, sn, 2, weight_decay)
        outputs.append(realout1)
        outputs.append(realout2)
        if (sn < stages):
            x = Concatenate()([stageT_branch1_out, stageT_branch2_out, x])
    model = Model(inputs=inputs, outputs=outputs)
    model.summary()
    return model


def get_testing_model(weight_decay=None):
    stages = 5
    np_branch1 = KEY_POINT_LINK
    np_branch2 = KEY_POINT_NUM
    img_size = None
    img_input_shape = (img_size, img_size, 3)
    vec_input_shape = (None, None, KEY_POINT_LINK)
    heat_input_shape = (None, None, KEY_POINT_NUM)

    inputs = []
    outputs = []

    img_input = Input(shape=img_input_shape)
    vec_weight_input = Input(shape=vec_input_shape)
    heat_weight_input = Input(shape=heat_input_shape)

    inputs.append(img_input)
    #inputs.append(vec_weight_input)
    #inputs.append(heat_weight_input)

    img_normalized = Lambda(lambda x: x / 256 - 0.5)(img_input)  # [-0.5, 0.5]

    # VGG
    stage0_out, x1 = vgg_block(img_normalized, weight_decay)
    x = Concatenate()([x1, stage0_out])
    # stage 1 - branch 1 (PAF)
    stage1_branch1_out, realout1 = stage1_block(x, np_branch1, 1, weight_decay)

    # stage 1 - branch 2 (confidence maps)
    stage1_branch2_out, realout2 = stage1_block(x, np_branch2, 2, weight_decay)

    x = Concatenate()([stage1_branch1_out, stage1_branch2_out, x])

    outputs.append(realout1)
    outputs.append(realout2)

    # stage sn >= 2
    for sn in range(2, stages + 1):
        # stage SN - branch 1 (PAF)
        stageT_branch1_out, realout1 = stageT_block(x, np_branch1, sn, 1, weight_decay)
        # stage SN - branch 2 (confidence maps)
        stageT_branch2_out, realout2 = stageT_block(x, np_branch2, sn, 2, weight_decay)
        outputs.append(realout1)
        outputs.append(realout2)
        if (sn < stages):
            x = Concatenate()([stageT_branch1_out, stageT_branch2_out, x])
    outputs=[realout1,realout2]
    model = Model(inputs=inputs, outputs=outputs)
    model.summary()
    return model