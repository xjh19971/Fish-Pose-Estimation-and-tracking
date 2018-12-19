from keras.optimizers import Optimizer
from keras import backend as K
from keras.legacy import interfaces
class Adam(Optimizer):

    """

    # 参数

        lr: float >= 0. 学习速率、学习步长，值越大则表示权值调整动作越大，对应上图算法中的参数 alpha；

        beta_1:  接近 1 的常数，（有偏）一阶矩估计的指数衰减因子；

        beta_2:  接近 1 的常数，（有偏）二阶矩估计的指数衰减因子；

        epsilon: 大于但接近 0 的数，放在分母，避免除以 0 ；

        decay:  学习速率衰减因子，【2】算法中无这个参数；

    """



    def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999,

                 epsilon=1e-8, decay=0., **kwargs): # 【2】中的默认参数设置

        super(Adam, self).__init__(**kwargs)

        self.iterations = K.variable(0)    # 迭代次数变量

        self.lr = K.variable(lr)        # 学习速率

        self.beta_1 = K.variable(beta_1) 

        self.beta_2 = K.variable(beta_2)

        self.epsilon = epsilon

        self.decay = K.variable(decay)

        self.initial_decay = decay



    def get_updates(self, params, constraints, loss):

        grads = self.get_gradients(loss, params)    # 计算梯度 g_t

        self.updates = [K.update_add(self.iterations, 1)] # 迭代次数加 1



        lr = self.lr

        if self.initial_decay > 0: # 如果初始学习速率衰减因子不为0，

                                                # 则随着迭代次数增加，学习速率将不断减小

            lr *= (1. / (1. + self.decay * self.iterations))



        t = self.iterations + 1 # 就是公式中的 t

        # 有偏估计到无偏估计的校正值

        # 这里将循环内的公共计算提到循环外面，提高速度

        lr_t = lr * (K.sqrt(1. - K.pow(self.beta_2, t)) /

                     (1. - K.pow(self.beta_1, t))) 



        shapes = [K.get_variable_shape(p) for p in params] # 获得权值形状

        ms = [K.zeros(shape) for shape in shapes] # 一阶矩估计初始值

        vs = [K.zeros(shape) for shape in shapes] # 二阶矩估计初始值

        self.weights = [self.iterations] + ms + vs 



        for p, g, m, v in zip(params, grads, ms, vs):

            m_t = (self.beta_1 * m) + (1. - self.beta_1) * g # 一阶矩估计

            v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square(g) # 二阶矩估计

            p_t = p - lr_t * m_t / (K.sqrt(v_t) + self.epsilon) # 权值更新



            self.updates.append(K.update(m, m_t))

            self.updates.append(K.update(v, v_t))



            new_p = p_t

            # 对权值加约束

            if p in constraints:

                c = constraints[p]

                new_p = c(new_p)

            self.updates.append(K.update(p, new_p))

        return self.updates



    # 获取当前超参数

    def get_config(self):

        config = {'lr': float(K.get_value(self.lr)),

                  'beta_1': float(K.get_value(self.beta_1)),

                  'beta_2': float(K.get_value(self.beta_2)),

                  'decay': float(K.get_value(self.decay)),

                  'epsilon': self.epsilon}

        base_config = super(Adam, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))