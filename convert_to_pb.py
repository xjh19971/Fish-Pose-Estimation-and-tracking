import argparse

import tensorflow as tf
from keras import backend as K
import tensorflow.contrib.tensorrt as trt
from config_reader import config_reader
from model.cmu_model_inception_improve import get_testing_model


def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True,input_names=None):
    """
    Freezes the state of a session into a pruned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    from tensorflow.python.framework.graph_util import convert_variables_to_constants
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        # Graph -> GraphDef ProtoBuf
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = convert_variables_to_constants(session, input_graph_def,
                                                      output_names, freeze_var_names)
        return frozen_graph,output_names,input_names

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='model/keras/model.h5', help='path to the weights file')

    args = parser.parse_args()
    keras_weights_file = args.model


    print('start processing...')

    # load model

    # authors of original model don't use
    # vgg normalization (subtracting mean) on input images
    #model = get_testing_model()
    #model.load_weights(keras_weights_file)

    # load config
    params, model_params = config_reader()
    K.set_learning_phase(0)
    # load model

    # authors of original model don't use
    # vgg normalization (subtracting mean) on input images
    model = get_testing_model()
    model.load_weights(keras_weights_file)
    frozen_graph, output_names, input_names = freeze_session(K.get_session(),
                                                             output_names=[out.op.name for out in model.outputs],
                                                             input_names=[out.op.name for out in model.inputs])
    tf.train.write_graph(frozen_graph, ".", "tf_model_real.pb", as_text=False)
    trt_graph = trt.create_inference_graph(
        input_graph_def=frozen_graph,
        outputs=['batch_normalization_17/FusedBatchNorm_1','batch_normalization_22/FusedBatchNorm_1'],
        max_batch_size=1,
        max_workspace_size_bytes=4000000000,
        precision_mode='FP16',
        minimum_segment_size=50
    )
    #trt_graph=trt.calib_graph_to_infer_graph(trt_graph)
    tf.train.write_graph(trt_graph, ".", "tf_model.pb", as_text=False)

