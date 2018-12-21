import tensorflow as tf
from tensorflow.python.platform import gfile

# 这是从二进制格式的pb文件加载模型
graph = tf.get_default_graph()
graphdef = graph.as_graph_def()
graphdef.ParseFromString(gfile.FastGFile("/data/TensorFlowAndroidMNIST/app/src/main/expert-graph.pb", "rb").read())
_ = tf.import_graph_def(graphdef, name="")