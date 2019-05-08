from keras.models import Model
from keras.layers.merge import Concatenate
from keras.layers import Activation, Input, Lambda, ZeroPadding2D,Conv2DTranspose,UpSampling2D,AveragePooling2D
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
def Dense_block(x, filters, stage, densenum, weight_decay, rate=1, alpha=4):

    conv_name_base = 'dense' + str(stage)+'_'+ str(densenum)
    bn_name_base = 'densebn' + str(stage)+'_'+ str(densenum)

    x = conv(x, filters[0]*alpha, 1, conv_name_base + '1', weight_decay)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '1', epsilon=1e-5, momentum=0.9)(x)
    x = relu(x)

    x = conv(x, filters[0], 3, conv_name_base + '2', weight_decay,rate=(rate,rate))
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2', epsilon=1e-5, momentum=0.9)(x)
    x = relu(x)

    return x

def Trans_block(x, filters, stage, weight_decay,strides=(1,1)):
    conv_name_base = 'trans' + str(stage)
    bn_name_base = 'transbn' + str(stage)


    x = conv(x, filters[0], 1, conv_name_base + '1', weight_decay)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '1', epsilon=1e-5, momentum=0.9)(x)
    #x = relu(x)
    x1=x
    if strides!=(1,1):
        x = AveragePooling2D()(x)


    return x,x1
'''
def Deconv_block(x, filters, stage, weight_decay):
    conv_name_base = 'deconv' + str(stage)
    bn_name_base = 'deconvbn' + str(stage)

    x = conv(x, filters[0], 1, conv_name_base + '1', weight_decay)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '1', epsilon=1e-5, momentum=0.9)(x)
    x = relu(x)

    x = convtrans(x, filters[0], 3, conv_name_base + '2', weight_decay)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2', epsilon=1e-5, momentum=0.9)(x)
    x = relu(x)

    return x
    '''
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

def convtrans(x, nf, ks, name, weight_decay, strides=(2, 2)):
    kernel_reg = l2(weight_decay[0]) if weight_decay else None
    x = Conv2DTranspose(nf, (ks, ks), padding='same', name=name, strides=strides,
               kernel_regularizer=kernel_reg,
               kernel_initializer=random_normal(stddev=0.01),
               use_bias=False)(x)
    return x


def vgg_block(x, weight_decay,stagenums,k=32,sigma=0.5):
    xout = []
    bn_axis = 1 if K.image_data_format() == 'channels_first' else -1
    x = ZeroPadding2D((2, 2))(x)  # 对图片界面填充0，保证特征图的大小#
    x = conv(x, k*2, 7, strides=(2, 2), name='conv1', weight_decay=(weight_decay, 0))  # 定义卷积层#
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)  # 批标准化#
    x = Activation('relu')(x)  # 激活函数#
    #x = ZeroPadding2D((1, 1))(x)  # 对图片界面填充0，保证特征图的大小#
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)  # 最大池化层#
    for i in range(len(stagenums)):
        n = 0
        while n < stagenums[i]:
            x1 = Dense_block(x, [k], i, n, (weight_decay, 0))
            x = Concatenate()([x, x1])
            n = n+1
        if i <= 2:
            x,x1 = Trans_block(x, [int(x.shape[3]) * sigma], i, (weight_decay, 0), strides=(2, 2))
        else:
            x,x1 = Trans_block(x, [int(x.shape[3]) * sigma], i, (weight_decay, 0))
        xout.append(x1)
    return xout

def stage1_block(x, weight_decay, rates, k=64,sigma=0.5):
    input = x
    for i in range(len(rates)):
        x1 = Dense_block(x, [k], i+4, 1, (weight_decay, 0), rate=rates[i])
        x = Concatenate()([x, x1])
    x = Concatenate()([x, input])
    x,x4 = Trans_block(x,[int(x.shape[3])*sigma],i+4,(weight_decay,0))
    return x

def stage2_block(x, stage0, weight_decay, num_p1,num_p2,stagenums,k=32,sigma=0.5):
    for i in range(len(stagenums)):
        x = UpSampling2D()(x)
        x3 = stage0[-i - 2]
        x = Concatenate()([x, x3])
        n = 0
        while n < stagenums[i]:
            x1 = Dense_block(x, [k], i+4, n, (weight_decay, 0))
            x = Concatenate()([x, x1])
            n = n+1
        if i <= 1:
            x, x4 = Trans_block(x, [int(x.shape[3]) * sigma], i+4, (weight_decay, 0), strides=(1, 1))
    x1 = conv(x, num_p1, 1, 'final1', (weight_decay, 0))
    x1 = BatchNormalization(axis=bn_axis, name='final1bn', epsilon=1e-5, momentum=0.9)(x1)
    x2 = conv(x, num_p2, 1, 'final2', (weight_decay, 0))
    x2 = BatchNormalization(axis=bn_axis, name='final2bn', epsilon=1e-5, momentum=0.9)(x2)
    return x1,x2



def get_training_model(weight_decay):
    np_branch1 = KEY_POINT_LINK
    np_branch2 = KEY_POINT_NUM
    img_size = 320
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

    #stage1_out = stage1_block(stage0_out[-1], weight_decay, rates=[1, 2, 3, 5])

    stage2_out = stage2_block(stage0_out[-1], stage0_out, weight_decay, np_branch1, np_branch2, [18, 12, 6])

    model = Model(inputs=inputs, outputs=stage2_out)
    model.summary()
    return model


def get_testing_model():
    np_branch1 = KEY_POINT_LINK
    np_branch2 = KEY_POINT_NUM
    img_size = None
    img_input_shape = (img_size, img_size, 3)


    inputs = []

    img_input = Input(shape=img_input_shape)

    weight_decay=None

    inputs.append(img_input)


    img_normalized = Lambda(lambda x: x / 256 - 0.5)(img_input)  # [-0.5, 0.5]

    stage0_out = vgg_block(img_normalized, weight_decay, [6, 12, 18, 24])

    # stage1_out = stage1_block(stage0_out[-1], weight_decay, rates=[1, 2, 3])

    stage2_out = stage2_block(stage0_out[-1], stage0_out, weight_decay, np_branch1, np_branch2, [18, 12, 6])

    model = Model(inputs=inputs, outputs=stage2_out)
    model.summary()
    return model