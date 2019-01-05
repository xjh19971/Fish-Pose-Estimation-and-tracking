from keras.models import Model
from keras.layers.merge import Concatenate
from keras.layers import Activation, Input, Lambda, ZeroPadding2D
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import Multiply
from keras.regularizers import l2
from keras.initializers import random_normal, constant
from keras.layers import BatchNormalization, add, ReLU,DepthwiseConv2D
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

    x4 = conv(input_tensor, filters4[0], 3, conv_name_base + 'd1', weight_decay)
    x4 = BatchNormalization(axis=bn_axis, name=bn_name_base + 'd1', epsilon=1e-5, momentum=0.9)(x4)
    x4 = relu(x4)
    x4 = conv(x4, filters4[1], 3, conv_name_base + 'd2', weight_decay)
    x4 = BatchNormalization(axis=bn_axis, name=bn_name_base + 'd2', epsilon=1e-5, momentum=0.9)(x4)
    x4 = relu(x4)
    shortcut = conv(input_tensor, filters4[1], 1, conv_name_base + '1', weight_decay)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)
    x = add([x4, shortcut])
    x = relu(x)
    return x


def tiny_inception_block(input_tensor, filters, stage, branch, weight_decay):
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
    filters1, filters2, filters3 = filters
    bn_axis = 1 if K.image_data_format() == 'channels_first' else -1
    conv_name_base = 'tinyinception' + str(stage) + '_branch' + str(branch)
    bn_name_base = 'bn' + str(stage) + '_branch' + str(branch)

    x1 = conv(input_tensor, filters1[0], 3, conv_name_base + 'a1', weight_decay)
    x1 = BatchNormalization(axis=bn_axis, name=bn_name_base + 'a1', epsilon=1e-5, momentum=0.9)(x1)
    x1 = relu(x1)

    x2 = conv(input_tensor, filters2[0], 3, conv_name_base + 'b1', weight_decay)
    x2 = BatchNormalization(axis=bn_axis, name=bn_name_base + 'b1', epsilon=1e-5, momentum=0.9)(x2)
    x2 = relu(x2)
    x2 = conv(x2, filters2[1], 3, conv_name_base + 'b2', weight_decay)
    x2 = BatchNormalization(axis=bn_axis, name=bn_name_base + 'b2', epsilon=1e-5, momentum=0.9)(x2)
    x2 = relu(x2)

    x3 = conv(input_tensor, filters3[0], 3, conv_name_base + 'c1', weight_decay)
    x3 = BatchNormalization(axis=bn_axis, name=bn_name_base + 'c1', epsilon=1e-5, momentum=0.9)(x3)
    x3 = relu(x3)
    x3 = conv(x3, filters3[1], 3, conv_name_base + 'c2', weight_decay)
    x3 = BatchNormalization(axis=bn_axis, name=bn_name_base + 'c2', epsilon=1e-5, momentum=0.9)(x3)
    x3 = relu(x3)
    x3 = conv(x3, filters3[2], 3, conv_name_base + 'c3', weight_decay)
    x3 = BatchNormalization(axis=bn_axis, name=bn_name_base + 'c3', epsilon=1e-5, momentum=0.9)(x3)
    x3 = relu(x3)
    x = Concatenate()([x1, x2, x3])
    x = conv(x, 128, 1, conv_name_base + 'd', weight_decay)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + 'd', epsilon=1e-5, momentum=0.9)(x)
    x = add([x, input_tensor])
    x = relu(x)
    return x


def relu(x):
    return ReLU(6.)(x)


def conv(x, nf, ks, name, weight_decay, strides=(1,1)):
    kernel_reg = l2(weight_decay[0]) if weight_decay else None
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
    bn_axis = 1 if K.image_data_format() == 'channels_first' else -1
    x = ZeroPadding2D((2, 2))(x)  # 对图片界面填充0，保证特征图的大小#
    x = conv(x, 64, 7, strides=(2, 2), name='conv1', weight_decay=(weight_decay, 0))  # 定义卷积层#
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)  # 批标准化#
    x = Activation('relu')(x)  # 激活函数#
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)  # 最大池化层#

    x = STEM_block(x, [128, 128], 1, (weight_decay, 0))
    x = STEM_block(x, [128, 128], 2, (weight_decay, 0))
    x = pooling(x, 2, 2)
    return x


def stage1_block(x, num_p, branch, weight_decay):
    bn_axis = 1 if K.image_data_format() == 'channels_first' else -1
    # Block 1
    x = conv(x, 128, 1, "Mconv1_stage1_L%d" % branch, (weight_decay, 0))
    x = BatchNormalization(axis=bn_axis, epsilon=1e-5, momentum=0.9)(x)
    x = relu(x)
    x1 = x
    x = conv(x, 128, 3, "Mconv2_stage1_L%d" % branch, (weight_decay, 0))
    x = BatchNormalization(axis=bn_axis, epsilon=1e-5, momentum=0.9)(x)
    x = relu(x)
    x = conv(x, 128, 3, "Mconv3_stage1_L%d" % branch, (weight_decay, 0))
    x = BatchNormalization(axis=bn_axis, epsilon=1e-5, momentum=0.9)(x)
    x = add([x1, x])
    x = relu(x)
    x2 = x
    x = conv(x, 128, 3, "Mconv4_stage1_L%d" % branch, (weight_decay, 0))
    x = BatchNormalization(axis=bn_axis, epsilon=1e-5, momentum=0.9)(x)
    x = relu(x)
    x = conv(x, 128, 3, "Mconv5_stage1_L%d" % branch, (weight_decay, 0))
    x = BatchNormalization(axis=bn_axis, epsilon=1e-5, momentum=0.9)(x)
    x = add([x2, x])
    x = relu(x)
    x3 = x
    x = conv(x, 128, 3, "Mconv6_stage1_L%d" % branch, (weight_decay, 0))
    x = BatchNormalization(axis=bn_axis, epsilon=1e-5, momentum=0.9)(x)
    x = relu(x)
    x = conv(x, 128, 3, "Mconv7_stage1_L%d" % branch, (weight_decay, 0))
    x = BatchNormalization(axis=bn_axis, epsilon=1e-5, momentum=0.9)(x)
    x = add([x3, x])
    x = relu(x)
    x4 = x
    x = conv(x, num_p, 1, "Mconv8_stage1_L%d" % branch, (weight_decay, 0))
    x = BatchNormalization(axis=bn_axis, epsilon=1e-5, momentum=0.9)(x)
    return x,x3


def stageT_block(x, num_p, stage, branch, weight_decay):
    bn_axis = 1 if K.image_data_format() == 'channels_first' else -1
    # Block 1
    x = conv(x, 128, 1, "Mconv1_stage%d_L%d" % (stage, branch), (weight_decay, 0))
    x = BatchNormalization(axis=bn_axis, epsilon=1e-5, momentum=0.9)(x)
    x = relu(x)
    x1 = x
    x = conv(x, 128, 3, "Mconv2_stage%d_L%d" % (stage, branch), (weight_decay, 0))
    x = BatchNormalization(axis=bn_axis, epsilon=1e-5, momentum=0.9)(x)
    x = relu(x)
    x = conv(x, 128, 3, "Mconv3_stage%d_L%d" % (stage, branch), (weight_decay, 0))
    x = BatchNormalization(axis=bn_axis, epsilon=1e-5, momentum=0.9)(x)
    x = add([x1, x])
    x = relu(x)
    x2 = x
    x = conv(x, 128, 3, "Mconv4_stage%d_L%d" % (stage, branch), (weight_decay, 0))
    x = BatchNormalization(axis=bn_axis, epsilon=1e-5, momentum=0.9)(x)
    x = relu(x)
    x = conv(x, 128, 3, "Mconv5_stage%d_L%d" % (stage, branch), (weight_decay, 0))
    x = BatchNormalization(axis=bn_axis, epsilon=1e-5, momentum=0.9)(x)
    x = add([x2, x])
    x = relu(x)
    x3 = x
    x = conv(x, 128, 3, "Mconv6_stage%d_L%d" % (stage, branch), (weight_decay, 0))
    x = BatchNormalization(axis=bn_axis, epsilon=1e-5, momentum=0.9)(x)
    x = relu(x)
    x = conv(x, 128, 3, "Mconv7_stage%d_L%d" % (stage, branch), (weight_decay, 0))
    x = BatchNormalization(axis=bn_axis, epsilon=1e-5, momentum=0.9)(x)
    x = add([x3, x])
    x = relu(x)
    x4 = x
    x = conv(x, num_p, 1,"Mconv8_stage%d_L%d" % (stage, branch), (weight_decay, 0))
    x = BatchNormalization(axis=bn_axis, epsilon=1e-5, momentum=0.9)(x)
    return x,x4


def apply_mask(x, mask1, mask2, num_p, stage, branch, is_weight):
    w_name = "weight_stage%d_L%d" % (stage, branch)
    if is_weight:
        w = Multiply(name=w_name)([x, mask1])  # vec_weight

    else:
        w = Multiply(name=w_name)([x, mask2])  # vec_heat
    return w


def get_training_model(weight_decay):
    stages = 6
    np_branch1 = KEY_POINT_LINK
    np_branch2 = KEY_POINT_NUM
    img_size = 320
    img_input_shape = (img_size, img_size, 3)
    vec_input_shape = (None, None, KEY_POINT_LINK)
    heat_input_shape = (None, None, KEY_POINT_NUM)

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
    stage1_branch1_out,x2 = stage1_block(stage0_out, np_branch1, 1, weight_decay)
    outputstemp.append(x2)
    w1 = apply_mask(stage1_branch1_out, vec_weight_input, heat_weight_input, np_branch1, 1, 1, True)

    # stage 1 - branch 2 (confidence maps)
    stage1_branch2_out,x3 = stage1_block(stage0_out, np_branch2, 2, weight_decay)
    outputstemp.append(x3)
    w2 = apply_mask(stage1_branch2_out, vec_weight_input, heat_weight_input, np_branch2, 1, 2, False)

    x = Concatenate()([x2, x3, stage0_out])

    outputs.append(w1)
    outputs.append(w2)

    # stage sn >= 2
    for sn in range(2, stages + 1):
        # stage SN - branch 1 (PAF)
        stageT_branch1_out,x3 = stageT_block(x, np_branch1, sn, 1, weight_decay)
        outputstemp.append(x3)
        w1 = apply_mask(stageT_branch1_out, vec_weight_input, heat_weight_input, np_branch1, sn, 1, is_weight=True)

        # stage SN - branch 2 (confidence maps)
        stageT_branch2_out,x3 = stageT_block(x, np_branch2, sn, 2, weight_decay)
        outputstemp.append(x3)
        w2 = apply_mask(stageT_branch2_out, vec_weight_input, heat_weight_input, np_branch2, sn, 2, is_weight=False)

        outputs.append(w1)
        outputs.append(w2)

        if (sn < stages):
            x = stage0_out
            for snc in range(1,sn+1):
                x = Concatenate()([outputstemp[2*snc-2], outputstemp[2*snc-1], x])

    model = Model(inputs=inputs, outputs=outputs)
    model.summary()
    return model


def get_testing_model():
    stages = 6
    np_branch1 = KEY_POINT_LINK
    np_branch2 = KEY_POINT_NUM

    img_input_shape = (None, None, 3)

    img_input = Input(shape=img_input_shape)

    img_normalized = Lambda(lambda x: x / 256 - 0.5)(img_input)  # [-0.5, 0.5]

    # VGG
    stage0_out = vgg_block(img_normalized, None)
    outputstemp=[]
    # stage 1 - branch 1 (PAF)
    stage1_branch1_out,x2 = stage1_block(stage0_out, np_branch1, 1, None)
    outputstemp.append(x2)
    # stage 1 - branch 2 (confidence maps)
    stage1_branch2_out,x3 = stage1_block(stage0_out, np_branch2, 2, None)
    outputstemp.append(x3)
    x = Concatenate()([x2, x3, stage0_out])

    # stage t >= 2
    for sn in range(2, stages + 1):
        stageT_branch1_out,x3 = stageT_block(x, np_branch1, sn, 1, None)
        outputstemp.append(x3)
        stageT_branch2_out,x3 = stageT_block(x, np_branch2, sn, 2, None)
        outputstemp.append(x3)

        if (sn < stages):
            x = stage0_out
            for snc in range(1,sn+1):
                x = Concatenate()([outputstemp[2*snc-2], outputstemp[2*snc-1], x])

    
    model = Model(inputs=[img_input], outputs=[stageT_branch1_out, stageT_branch2_out])
    model.summary()
    return model
