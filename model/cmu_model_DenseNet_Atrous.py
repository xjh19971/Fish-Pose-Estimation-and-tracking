from keras.models import Model
from keras.layers.merge import Concatenate
from keras.layers import Activation, Input, Lambda, ZeroPadding2D,Conv2DTranspose
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import Multiply
from keras.regularizers import l2
from keras.initializers import random_normal, constant
from keras.layers import BatchNormalization, add, ReLU, DepthwiseConv2D
import keras.backend as K

KEY_POINT_NUM = 3 + 1
KEY_POINT_LINK = 2 * 2

bn_axis = 1 if K.image_data_format() == 'channels_first' else -1
def Dense_block(input_tensor, filters, stage, densenum, weight_decay,rate=1):

    conv_name_base = 'dense' + str(stage)+'_'+ str(densenum)
    bn_name_base = 'densebn' + str(stage)+'_'+ str(densenum)
    x = conv(input_tensor, filters[0]*4, 1, conv_name_base + '1', weight_decay)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '1', epsilon=1e-5, momentum=0.9)(x)
    x = relu(x)

    x = conv(x, filters[0], 3, conv_name_base + '2', weight_decay,rate=(rate,rate))
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2', epsilon=1e-5, momentum=0.9)(x)
    x = relu(x)

    return x

def Trans_block(input_tensor, filters, stage, weight_decay,strides=(1,1)):
    conv_name_base = 'trans' + str(stage)
    bn_name_base = 'transbn' + str(stage)

    x = conv(input_tensor, filters[0]*4, 1, conv_name_base + '1', weight_decay)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '1', epsilon=1e-5, momentum=0.9)(x)
    x = relu(x)

    x = conv(x, filters[0], 3, conv_name_base + '2', weight_decay,strides=strides)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2', epsilon=1e-5, momentum=0.9)(x)
    x = relu(x)

    return x

def Deconv_block(input_tensor, filters, stage, weight_decay,strides=(2,2)):
    conv_name_base = 'deconv' + str(stage)
    bn_name_base = 'deconvbn' + str(stage)

    x = conv(input_tensor, filters[0], 1, conv_name_base + '1', weight_decay)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '1', epsilon=1e-5, momentum=0.9)(x)
    x = relu(x)

    x = convtrans(x, filters[0], 3, conv_name_base + '2', weight_decay,strides=strides)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2', epsilon=1e-5, momentum=0.9)(x)
    x = relu(x)

    return x
def relu(x):
    return Activation('relu')(x)


def conv(x, nf, ks, name, weight_decay, strides=(1, 1),rate=(1, 1)):
    kernel_reg = l2(weight_decay[0]) if weight_decay else None
    nf=int(nf)
    x = Conv2D(nf, (ks, ks), padding='same', name=name, strides=strides,
               kernel_regularizer=kernel_reg,
               kernel_initializer=random_normal(stddev=0.01),
               use_bias=False,
               dilation_rate=rate)(x)
    return x

def convtrans(x, nf, ks, name, weight_decay, strides=(1, 1)):
    kernel_reg = l2(weight_decay[0]) if weight_decay else None
    x = Conv2DTranspose(nf, (ks, ks), padding='same', name=name, strides=strides,
               kernel_regularizer=kernel_reg,
               kernel_initializer=random_normal(stddev=0.01),
               use_bias=False)(x)
    return x


def vgg_block(x, weight_decay,stagenums,k=24,sigma=0.5):

    bn_axis = 1 if K.image_data_format() == 'channels_first' else -1
    x = ZeroPadding2D((2, 2))(x)  # 对图片界面填充0，保证特征图的大小#
    x = conv(x, k*2, 7, strides=(2, 2), name='conv1', weight_decay=(weight_decay, 0))  # 定义卷积层#
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)  # 批标准化#
    x = Activation('relu')(x)  # 激活函数#
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)  # 最大池化层#
    xout = [x]
    for i in range(len(stagenums)):
        n = 0
        while n < stagenums[i]:
            x1 = Dense_block(x, [k], i, n, (weight_decay, 0))
            x = Concatenate()([x, x1])
            n = n+1
        if i <= 1:
            x = Trans_block(x, [stagenums[i] * k * sigma], i, (weight_decay, 0), strides=(2, 2))
            if i == 0:
                xout.append(x)
        else:
            x = Trans_block(x, [stagenums[i] * k * sigma], i, (weight_decay, 0), strides=(1, 1))
    xout.append(x)
    return xout

def stage1_block(x, weight_decay,rates,k=24):
    input = x
    for i in range(len(rates)):
        x1 = Dense_block(x, [k], i+5, 1, (weight_decay, 0), rate=rates[i])
        x = Concatenate()([x, x1])
    x = Concatenate()([x, input])
    return x

def stage2_block(x, stage0, weight_decay, num_p1,num_p2):
    x = Deconv_block(x, [128], 1, (weight_decay,0))
    x = Concatenate()([x, stage0[1]])
    x = Deconv_block(x, [128], 2, (weight_decay,0))
    x = Concatenate()([x, stage0[0]])
    x1 = conv(x, num_p1, 1, 'final1', (weight_decay,0))
    x1 = BatchNormalization(axis=bn_axis, name='final1bn', epsilon=1e-5, momentum=0.9)(x1)
    x2 = conv(x, num_p2, 1, 'final2', (weight_decay, 0))
    x2 = BatchNormalization(axis=bn_axis, name='final2bn', epsilon=1e-5, momentum=0.9)(x2)
    return x1,x2

def apply_mask(x, mask1, mask2):
    w_name = "weight_stage_out"
    w1 = Multiply(name=w_name + '_1')([x[0], mask1])  # vec_heat
    w2 = Multiply(name=w_name + '_2')([x[1], mask2])  # vec_heat
    return w1,w2


def get_training_model(weight_decay):
    np_branch1 = KEY_POINT_LINK
    np_branch2 = KEY_POINT_NUM
    img_size = 400
    img_input_shape = (img_size, img_size, 3)
    vec_input_shape = (None, None, KEY_POINT_LINK)
    heat_input_shape = (None, None, KEY_POINT_NUM)

    inputs = []

    img_input = Input(shape=img_input_shape)
    vec_weight_input = Input(shape=vec_input_shape)
    heat_weight_input = Input(shape=heat_input_shape)

    inputs.append(img_input)
    inputs.append(vec_weight_input)
    inputs.append(heat_weight_input)

    img_normalized = Lambda(lambda x: x / 256 - 0.5)(img_input)  # [-0.5, 0.5]

    # VGG
    stage0_out = vgg_block(img_normalized, weight_decay, [6, 12, 18, 24])

    stage1_out = stage1_block(stage0_out[2], weight_decay, rates=[1, 2, 3])

    stage2_out = stage2_block(stage1_out, stage0_out, weight_decay, np_branch1, np_branch2)

    outputs = apply_mask(stage2_out, vec_weight_input, heat_weight_input)

    model = Model(inputs=inputs, outputs=outputs)
    model.summary()
    return model


def get_testing_model():
    stages = 5
    np_branch1 = KEY_POINT_LINK
    np_branch2 = KEY_POINT_NUM
    img_size = 368
    img_input_shape = (None, None, 3)

    inputs = []
    outputs = []
    img_input = Input(shape=img_input_shape)
    inputs.append(img_input)

    img_normalized = Lambda(lambda x: x / 256 - 0.5)(img_input)  # [-0.5, 0.5]

    stage0_out,x1 = vgg_block(img_normalized,None)
    x = Concatenate()([x1,stage0_out])
    # stage 1 - branch 1 (PAF)
    stage1_branch1_out = stage1_block(x, np_branch1, 1,None)

    # stage 1 - branch 2 (confidence maps)
    stage1_branch2_out = stage1_block(x, np_branch2, 2, None)

    x = Concatenate()([stage1_branch1_out, stage1_branch2_out, x])

    # stage sn >= 2
    for sn in range(2, stages + 1):
        # stage SN - branch 1 (PAF)
        stageT_branch1_out = stageT_block(x, np_branch1, sn, 1, None)
        # stage SN - branch 2 (confidence maps)
        stageT_branch2_out = stageT_block(x, np_branch2, sn, 2, None)
        if (sn < stages):
            x = Concatenate()([stageT_branch1_out, stageT_branch2_out, x])
    outputs.append(stageT_branch1_out)
    outputs.append(stageT_branch2_out)
    model = Model(inputs=inputs, outputs=outputs)
    model.summary()
    return model
