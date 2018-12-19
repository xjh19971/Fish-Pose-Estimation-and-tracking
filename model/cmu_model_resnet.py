from keras.models import Model
from keras.layers.merge import Concatenate
from keras.layers import Activation, Input, Lambda,ZeroPadding2D
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import Multiply
from keras.regularizers import l2
from keras.initializers import random_normal,constant
from keras.layers import  BatchNormalization,add
import keras.backend as K
KEY_POINT_NUM=3+1
KEY_POINT_LINK=2*2


def identity_block(input_tensor, kernel_size, filters, stage, block, weight_decay):
    """The identity block is the block that has no conv layer at shortcut.
    # Arguments
        input_tensor: input tensor  #输入变量#
        kernel_size: defualt 3, the kernel size of middle conv layer at main path #卷积核的大小#
        filters: list of integers, the filterss of 3 conv layer at main path  #卷积核的数目#
        stage: integer, current stage label, used for generating layer names #当前阶段的标签#
        block: 'a','b'..., current block label, used for generating layer names #当前块的标签#
    # Returns
        Output tensor for the block.  #返回块的输出变量#
    """
    filters1, filters2, filters3 = filters  # 滤波器的名称#
    bn_axis = 1 if K.image_data_format() == 'channels_first' else -1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    #conv(x, 64, 3, "conv1_1", (weight_decay, 0))
    #x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
    '''x = conv(input_tensor, filters1, 1, conv_name_base + '2a', weight_decay)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a', epsilon=1e-5, momentum=0.9)(x)
    x = Activation('relu')(x)  # 卷积层，BN层，激活函数#

    #x = Conv2D(filters2, kernel_size,
    #           padding='same', name=conv_name_base + '2b')(x)
    x = conv(x, filters2, kernel_size, conv_name_base + '2b', weight_decay)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b', epsilon=1e-5, momentum=0.9)(x)
    x = Activation('relu')(x)

    #x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = conv(x, filters3, 1, conv_name_base + '2c', weight_decay)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c', epsilon=1e-5, momentum=0.9)(x)'''
    x = conv(input_tensor, filters3, kernel_size, conv_name_base + '2a', weight_decay)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a', epsilon=1e-5, momentum=0.9)(x)
    x = Activation('relu')(x)
    x = conv(x, filters3, kernel_size, conv_name_base + '2b', weight_decay)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b', epsilon=1e-5, momentum=0.9)(x)
    x = Activation('relu')(x)
    x = add([x, input_tensor])
    x = Activation('relu')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, weight_decay, strides=(2, 2)):
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
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    ''''#x = Conv2D(filters1, (1, 1), strides=strides,
    #           name=conv_name_base + '2a')(input_tensor)
    x = conv(input_tensor, filters1, 1, conv_name_base + '2a', weight_decay, strides=strides)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a', epsilon=1e-5, momentum=0.9)(x)
    x = Activation('relu')(x)

    #x = Conv2D(filters2, kernel_size, padding='same',
    #           name=conv_name_base + '2b')(x)
    x = conv(x, filters2, kernel_size, conv_name_base + '2b', weight_decay)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b', epsilon=1e-5, momentum=0.9)(x)
    x = Activation('relu')(x)

    #x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = conv(x, filters3, 1, conv_name_base + '2c', weight_decay)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c', epsilon=1e-5, momentum=0.9)(x)'''
    x = conv(input_tensor, filters3, kernel_size, conv_name_base + '2a', weight_decay,strides=strides)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a', epsilon=1e-5, momentum=0.9)(x)
    x = Activation('relu')(x)
    x = conv(x, filters3, kernel_size, conv_name_base + '2b', weight_decay)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b', epsilon=1e-5, momentum=0.9)(x)
    x = Activation('relu')(x)
    #shortcut = Conv2D(filters3, (1, 1), strides=strides,
    #                  name=conv_name_base + '1')(input_tensor)
    shortcut = conv(input_tensor, filters3, 1, conv_name_base + '1', weight_decay, strides=strides)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1', epsilon=1e-5, momentum=0.9)(shortcut)

    x = add([x, shortcut])
    x = Activation('relu')(x)
    return x



def relu(x): return Activation('relu')(x)


def conv(x, nf, ks, name,  weight_decay, strides = None):
    kernel_reg = l2(weight_decay[0]) if weight_decay else None
    if strides == None:
        x = Conv2D(nf, (ks, ks), padding='same', name=name,
               kernel_regularizer=kernel_reg,
               kernel_initializer=random_normal(stddev=0.01),
	           use_bias=False
               )(x)
    else:
        x = Conv2D(nf, (ks, ks), padding='same', name=name,strides=strides,
               kernel_regularizer=kernel_reg,
               kernel_initializer=random_normal(stddev=0.01),
               use_bias=False)(x)
    return x


def pooling(x, ks, st, name):
    x = MaxPooling2D((ks, ks), strides=(st, st), name=name)(x)
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
    bn_axis = 1 if K.image_data_format() == 'channels_first' else -1

    x = ZeroPadding2D((2, 2))(x)  # 对图片界面填充0，保证特征图的大小#
    x = conv(x, 64, 7, strides=(2, 2), name='conv1', weight_decay=(weight_decay, 0))  # 定义卷积层#
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)  # 批标准化#
    x = Activation('relu')(x)  # 激活函数#
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)  # 最大池化层#

    # Block 2
    x = conv(x, 64, 3, "conv2_1", (weight_decay, 0))
    x = BatchNormalization(axis=bn_axis, epsilon=1e-5, momentum=0.9)(x)
    x = relu(x)
    x = conv(x, 64, 3, "conv2_2", (weight_decay, 0))
    x = BatchNormalization(axis=bn_axis, epsilon=1e-5, momentum=0.9)(x)
    x = relu(x)
    # Block 3
    x = conv(x, 128, 3, "conv3_1", (weight_decay, 0),strides=(2,2))
    x = BatchNormalization(axis=bn_axis, epsilon=1e-5, momentum=0.9)(x)
    x = relu(x)
    x = conv(x, 128, 3, "conv3_2", (weight_decay, 0))
    x = BatchNormalization(axis=bn_axis, epsilon=1e-5, momentum=0.9)(x)
    x = relu(x)
    x = conv(x, 128, 3, "conv3_3", (weight_decay, 0))
    x = BatchNormalization(axis=bn_axis, epsilon=1e-5, momentum=0.9)(x)
    x = relu(x)
    return x


def stage1_block(x, num_p, branch, weight_decay):
    bn_axis = 1 if K.image_data_format() == 'channels_first' else -1
    # Block 1
    x = conv(x, 64, 3, "Mconv1_stage1_L%d" % branch, (weight_decay, 0))
    x = BatchNormalization(axis=bn_axis, epsilon=1e-5, momentum=0.9)(x)
    x = relu(x)
    x = conv(x, 64, 3, "Mconv2_stage1_L%d" % branch, (weight_decay, 0))
    x = BatchNormalization(axis=bn_axis, epsilon=1e-5, momentum=0.9)(x)
    x = relu(x)
    x = conv(x, 64, 3, "Mconv3_stage1_L%d" % branch, (weight_decay, 0))
    x = BatchNormalization(axis=bn_axis, epsilon=1e-5, momentum=0.9)(x)
    x = relu(x)
    x = conv(x, 256, 1, "Mconv4_stage1_L%d" % branch, (weight_decay, 0))
    x = BatchNormalization(axis=bn_axis, epsilon=1e-5, momentum=0.9)(x)
    x = relu(x)
    x = conv(x, num_p, 1, "Mconv5_stage1_L%d" % branch, (weight_decay, 0))
    x = BatchNormalization(axis=bn_axis, epsilon=1e-5, momentum=0.9)(x)

    return x


def stageT_block(x, num_p, stage, branch, weight_decay):
    bn_axis = 1 if K.image_data_format() == 'channels_first' else -1
    # Block 1
    x = conv(x, 64, 3, "Mconv1_stage%d_L%d" % (stage, branch), (weight_decay, 0))
    x = BatchNormalization(axis=bn_axis, epsilon=1e-5, momentum=0.9)(x)
    x = relu(x)
    x = conv(x, 64, 3, "Mconv2_stage%d_L%d" % (stage, branch), (weight_decay, 0))
    x = BatchNormalization(axis=bn_axis, epsilon=1e-5, momentum=0.9)(x)
    x = relu(x)
    x = conv(x, 64, 3, "Mconv3_stage%d_L%d" % (stage, branch), (weight_decay, 0))
    x = BatchNormalization(axis=bn_axis, epsilon=1e-5, momentum=0.9)(x)
    x = relu(x)
    x = conv(x, 64, 3, "Mconv4_stage%d_L%d" % (stage, branch), (weight_decay, 0))
    x = BatchNormalization(axis=bn_axis, epsilon=1e-5, momentum=0.9)(x)
    x = relu(x)
    x = conv(x, 64, 3, "Mconv5_stage%d_L%d" % (stage, branch), (weight_decay, 0))
    x = BatchNormalization(axis=bn_axis, epsilon=1e-5, momentum=0.9)(x)
    x = relu(x)
    x = conv(x, 64, 1, "Mconv6_stage%d_L%d" % (stage, branch), (weight_decay, 0))
    x = BatchNormalization(axis=bn_axis, epsilon=1e-5, momentum=0.9)(x)
    x = relu(x)
    x = conv(x, num_p, 1, "Mconv7_stage%d_L%d" % (stage, branch), (weight_decay, 0))
    x = BatchNormalization(axis=bn_axis, epsilon=1e-5, momentum=0.9)(x)

    return x


def apply_mask(x, mask1, mask2, num_p, stage, branch, is_weight):
    w_name = "weight_stage%d_L%d" % (stage, branch)
    if is_weight:
        w = Multiply(name=w_name)([x, mask1]) # vec_weight

    else:
        w = Multiply(name=w_name)([x, mask2])  # vec_heat
    return w


def get_training_model(weight_decay):

    stages = 2
    np_branch1 = KEY_POINT_LINK
    np_branch2 = KEY_POINT_NUM

    img_size=368
    img_input_shape = (img_size, img_size, 3)
    vec_input_shape = (46, 46, KEY_POINT_LINK)
    heat_input_shape = (46, 46, KEY_POINT_NUM)

    inputs = []
    outputs = []

    img_input = Input(shape=img_input_shape)
    vec_weight_input = Input(shape=vec_input_shape)
    heat_weight_input = Input(shape=heat_input_shape)

    inputs.append(img_input)
    inputs.append(vec_weight_input)
    inputs.append(heat_weight_input)

    img_normalized = Lambda(lambda x: x / 256 - 0.5)(img_input) # [-0.5, 0.5]

    # VGG
    stage0_out = vgg_block(img_normalized, weight_decay)

    # stage 1 - branch 1 (PAF)
    stage1_branch1_out = stage1_block(stage0_out, np_branch1, 1, weight_decay)
    w1 = apply_mask(stage1_branch1_out, vec_weight_input, heat_weight_input, np_branch1, 1, 1,True)

    # stage 1 - branch 2 (confidence maps)
    stage1_branch2_out = stage1_block(stage0_out, np_branch2, 2, weight_decay)
    w2 = apply_mask(stage1_branch2_out, vec_weight_input, heat_weight_input, np_branch2, 1, 2,False)

    x = Concatenate()([stage1_branch1_out, stage1_branch2_out, stage0_out])

    outputs.append(w1)
    outputs.append(w2)

    # stage sn >= 2
    for sn in range(2, stages + 1):
        # stage SN - branch 1 (PAF)
        stageT_branch1_out = stageT_block(x, np_branch1, sn, 1, weight_decay)
        w1 = apply_mask(stageT_branch1_out, vec_weight_input, heat_weight_input, np_branch1, sn, 1,is_weight=True)

        # stage SN - branch 2 (confidence maps)
        stageT_branch2_out = stageT_block(x, np_branch2, sn, 2, weight_decay)
        w2 = apply_mask(stageT_branch2_out, vec_weight_input, heat_weight_input, np_branch2, sn, 2,is_weight=False)

        outputs.append(w1)
        outputs.append(w2)

        if (sn < stages):
            x = Concatenate()([stageT_branch1_out, stageT_branch2_out, stage0_out])

    model = Model(inputs=inputs, outputs=outputs)
    model.summary()
    return model


def get_testing_model():
    stages = 2
    np_branch1 = KEY_POINT_LINK
    np_branch2 = KEY_POINT_NUM

    img_input_shape = (None, None, 3)

    img_input = Input(shape=img_input_shape)

    img_normalized = Lambda(lambda x: x / 256 - 0.5)(img_input) # [-0.5, 0.5]

    # VGG
    stage0_out = vgg_block(img_normalized, None)

    # stage 1 - branch 1 (PAF)
    stage1_branch1_out = stage1_block(stage0_out, np_branch1, 1, None)

    # stage 1 - branch 2 (confidence maps)
    stage1_branch2_out = stage1_block(stage0_out, np_branch2, 2, None)

    x = Concatenate()([stage1_branch1_out, stage1_branch2_out, stage0_out])

    # stage t >= 2
    stageT_branch1_out = None
    stageT_branch2_out = None
    for sn in range(2, stages + 1):
        stageT_branch1_out = stageT_block(x, np_branch1, sn, 1, None)
        stageT_branch2_out = stageT_block(x, np_branch2, sn, 2, None)

        if (sn < stages):
            x = Concatenate()([stageT_branch1_out, stageT_branch2_out, stage0_out])

    model = Model(inputs=[img_input], outputs=[stageT_branch1_out, stageT_branch2_out])

    return model
