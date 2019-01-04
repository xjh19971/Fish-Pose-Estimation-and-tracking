from keras.models import Model
from keras.layers.merge import Concatenate
from keras.layers import Activation, Input, Lambda
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import Multiply
from keras.regularizers import l2
from keras.layers import  BatchNormalization,add,ReLU,DepthwiseConv2D
import keras.backend as K

KEY_POINT_NUM=3+1
KEY_POINT_LINK=2*2


def Relu6(x, **kwargs):
    return ReLU(6.)(x)

def InvertedResidualBlock(x, expand, out_channels, repeats, stride, weight_decay, block_id):
    '''
    This function defines a sequence of 1 or more identical layers, referring to Table 2 in the original paper.
    :param x: Input Keras tensor in (B, H, W, C_in)
    :param expand: expansion factor in bottlenect residual block
    :param out_channels: number of channels in the output tensor
    :param repeats: number of times to repeat the inverted residual blocks including the one that changes the dimensions.
    :param stride: stride for the 1x1 convolution
    :param weight_decay: hyperparameter for the l2 penalty
    :param block_id: as its name tells
    :return: Output tensor (B, H_new, W_new, out_channels)
    '''
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    in_channels = K.int_shape(x)[channel_axis]
    x = Conv2D(expand * in_channels, 1, padding='same', strides=stride, use_bias=False,
                kernel_regularizer=l2(weight_decay), name='conv_%d_0' % block_id)(x)
    x = BatchNormalization(epsilon=1e-5, momentum=0.9, name='conv_%d_0_bn' % block_id)(x)
    x = Relu6(x, name='conv_%d_0_act_1' % block_id)
    x = DepthwiseConv2D((3, 3),
                        padding='same',
                        depth_multiplier=1,
                        strides=1,
                        use_bias=False,
                        kernel_regularizer=l2(weight_decay),
                        name='conv_dw_%d_0' % block_id )(x)
    x = BatchNormalization(axis=channel_axis, epsilon=1e-5, momentum=0.9, name='conv_dw_%d_0_bn' % block_id)(x)
    x = Relu6(x, name='conv_%d_0_act_2' % block_id)
    x = Conv2D(out_channels, 1, padding='same', strides=1, use_bias=False,
               kernel_regularizer=l2(weight_decay), name='conv_bottleneck_%d_0' % block_id)(x)
    x = BatchNormalization(axis=channel_axis, epsilon=1e-5, momentum=0.9, name='conv_bottlenet_%d_0_bn' % block_id)(x)

    for i in range(1, repeats):
        x1 = Conv2D(expand*out_channels, 1, padding='same', strides=1, use_bias=False,
                    kernel_regularizer=l2(weight_decay), name='conv_%d_%d' % (block_id, i))(x)
        x1 = BatchNormalization(axis=channel_axis, epsilon=1e-5,momentum=0.9,name='conv_%d_%d_bn' % (block_id, i))(x1)
        x1 = Relu6(x1,name='conv_%d_%d_act_1' % (block_id, i))
        x1 = DepthwiseConv2D((3, 3),
                            padding='same',
                            depth_multiplier=1,
                            strides=1,
                            use_bias=False,
                            kernel_regularizer=l2(weight_decay),
                            name='conv_dw_%d_%d' % (block_id, i))(x1)
        x1 = BatchNormalization(axis=channel_axis, epsilon=1e-5,momentum=0.9, name='conv_dw_%d_%d_bn' % (block_id, i))(x1)
        x1 = Relu6(x1, name='conv_dw_%d_%d_act_2' % (block_id, i))
        x1 = Conv2D(out_channels, 1, padding='same', strides=1, use_bias=False,
                    kernel_regularizer=l2(weight_decay),name='conv_bottleneck_%d_%d' % (block_id, i))(x1)
        x1 = BatchNormalization(axis=channel_axis, epsilon=1e-5, momentum=0.9, name='conv_bottlenet_%d_%d_bn' % (block_id, i))(x1)
        x = add([x, x1], name='block_%d_%d_output' % (block_id, i))
    return x

def conv_block(inputs, filters, weight_decay, name, kernel=(3, 3), strides=(1, 1)):
    '''
    Normal convolution block performs conv+bn+relu6 operations.
    :param inputs: Input Keras tensor in (B, H, W, C_in)
    :param filters: number of filters in the convolution layer
    :param name: name for the convolutional layer
    :param kernel: kernel size
    :param strides: strides for convolution
    :return: Output tensor in (B, H_new, W_new, filters)
    '''
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    x = Conv2D(filters, kernel,
               padding='same',
               use_bias=False,
               kernel_regularizer=l2(weight_decay),
               strides=strides,
               name=name)(inputs)
    x = BatchNormalization(axis=channel_axis, epsilon=1e-5,momentum=0.9,name=name+'_bn')(x)
    return Relu6(x, name=name+'_relu')


def relu(x): return Activation('relu')(x)


def conv(x, nf, ks, name,  weight_decay, strides = None,expand= 6,change=False,last=False):
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    in_channels = K.int_shape(x)[channel_axis]
    if change is False:
        input = x
    else:
        input = Conv2D(nf, 1, padding='same', strides=1, use_bias=False,
                kernel_regularizer=l2(weight_decay))(x)
        input = BatchNormalization(axis=channel_axis, epsilon=1e-5, momentum=0.9)(
            input)
    x = DepthwiseConv2D((ks, ks),
                         padding='same',
                         depth_multiplier=1,
                         strides=1,
                         use_bias=False,
                         kernel_regularizer=l2(weight_decay),
                         )(x)
    x = BatchNormalization(axis=channel_axis, epsilon=1e-5, momentum=0.9)(x)
    x = Relu6(x)
    x = Conv2D(nf, 1, padding='same', strides=1, use_bias=False,name=name,
                kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization(axis=channel_axis, epsilon=1e-5, momentum=0.9,
                            )(x)
    if last is False:
        x= add([input,x])
    return x


def pooling(x, ks, st, name):
    x = MaxPooling2D((ks, ks), strides=(st, st), name=name)(x)
    return x


def vgg_block(x, weight_decay):

    x = conv_block(x, 32, weight_decay=weight_decay, name='conv1', strides=(2, 2))
    x = InvertedResidualBlock(x, expand=1, out_channels=16, repeats=1, stride=1, weight_decay=weight_decay, block_id=1)
    x = InvertedResidualBlock(x, expand=6, out_channels=24, repeats=2, stride=2, weight_decay=weight_decay, block_id=2)
    x = InvertedResidualBlock(x, expand=6, out_channels=32, repeats=3, stride=2, weight_decay=weight_decay, block_id=3)
    x = InvertedResidualBlock(x, expand=6, out_channels=64, repeats=4, stride=1, weight_decay=weight_decay, block_id=4)
    x = InvertedResidualBlock(x, expand=6, out_channels=96, repeats=3, stride=1, weight_decay=weight_decay, block_id=5)
    return x


def stage1_block(x, num_p, branch, weight_decay):
    # Block 1
    x = conv(x, 64, 3, "Mconv1_stage1_L%d" % branch, weight_decay,change=True)
    x = conv(x, 64, 3, "Mconv2_stage1_L%d" % branch, weight_decay)
    x = conv(x, 64, 3, "Mconv3_stage1_L%d" % branch, weight_decay)
    x = conv(x, 256, 1, "Mconv4_stage1_L%d" % branch, weight_decay,change=True)
    x = conv(x, num_p, 1, "Mconv5_stage1_L%d" % branch, weight_decay,change=True,last=True)

    return x


def stageT_block(x, num_p, stage, branch, weight_decay):
    # Block 1
    x = conv(x, 64, 7, "Mconv1_stage%d_L%d" % (stage, branch), weight_decay,change=True)
    x = conv(x, 64, 7, "Mconv2_stage%d_L%d" % (stage, branch), weight_decay)
    x = conv(x, 64, 7, "Mconv3_stage%d_L%d" % (stage, branch), weight_decay)
    x = conv(x, 64, 7, "Mconv4_stage%d_L%d" % (stage, branch), weight_decay)
    x = conv(x, 64, 7, "Mconv5_stage%d_L%d" % (stage, branch), weight_decay)
    x = conv(x, 64, 1, "Mconv6_stage%d_L%d" % (stage, branch), weight_decay)
    x = conv(x, num_p, 1, "Mconv7_stage%d_L%d" % (stage, branch), weight_decay,change=True,last=True)

    return x


def apply_mask(x, mask1, mask2, num_p, stage, branch, is_weight):
    w_name = "weight_stage%d_L%d" % (stage, branch)
    if is_weight:
        w = Multiply(name=w_name)([x, mask1]) # vec_weight

    else:
        w = Multiply(name=w_name)([x, mask2])  # vec_heat
    return w


def get_training_model(weight_decay):

    stages = 4
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

    img_normalized = Lambda(lambda x: x / 256 - 0.5)(img_input)  # [-0.5, 0.5]

    # VGG
    stage0_out = vgg_block(img_normalized, weight_decay)

    # stage 1 - branch 1 (PAF)
    stage1_branch1_out = stage1_block(stage0_out, np_branch1, 1, weight_decay)
    w1 = apply_mask(stage1_branch1_out, vec_weight_input, heat_weight_input, np_branch1, 1, 1, True)

    # stage 1 - branch 2 (confidence maps)
    stage1_branch2_out = stage1_block(stage0_out, np_branch2, 2, weight_decay)
    w2 = apply_mask(stage1_branch2_out, vec_weight_input, heat_weight_input, np_branch2, 1, 2, False)

    x = Concatenate()([stage1_branch1_out, stage1_branch2_out, stage0_out])

    outputs.append(w1)
    outputs.append(w2)

    # stage sn >= 2
    for sn in range(2, stages + 1):
        # stage SN - branch 1 (PAF)
        stageT_branch1_out = stageT_block(x, np_branch1, sn, 1, weight_decay)
        w1 = apply_mask(stageT_branch1_out, vec_weight_input, heat_weight_input, np_branch1, sn, 1, is_weight=True)

        # stage SN - branch 2 (confidence maps)
        stageT_branch2_out = stageT_block(x, np_branch2, sn, 2, weight_decay)
        w2 = apply_mask(stageT_branch2_out, vec_weight_input, heat_weight_input, np_branch2, sn, 2, is_weight=False)

        outputs.append(w1)
        outputs.append(w2)

        if (sn < stages):
            x = Concatenate()([stageT_branch1_out, stageT_branch2_out, stage0_out])

    model = Model(inputs=inputs, outputs=outputs)
    model.summary()
    return model


def get_testing_model():
    stages = 3
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
