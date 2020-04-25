from keras.layers.core import Layer
from streetnumber.model import *

# class PoolHelper(Layer):
#     def __init__(self, **kwargs):
#         super(PoolHelper, self).__init__(**kwargs)
# 
#     def call(self, x, mask=None):
#         return x[:,:,1:,1:]
# 
#     def get_config(self):
#         config = {}
#         base_config = super(PoolHelper, self).get_config()
#         return dict(list(base_config.items()) + list(config.items()))

class LRN(Layer):

    def __init__(self, alpha=0.0001, k=1, beta=0.75, n=5, **kwargs):
        self.alpha = alpha
        self.k = k
        self.beta = beta
        self.n = n
        super(LRN, self).__init__(**kwargs)

    def call(self, x, mask=None):
        b, ch, r, c = x.shape
        half_n = self.n // 2 # half the local region
        input_sqr = K.square(x) # square the input
        if K.backend() == 'theano':
            # make an empty tensor with zero pads along channel dimension
            zeros = T.alloc(0., b, ch + 2*half_n, r, c)
            # set the center to be the squared input
            input_sqr = T.set_subtensor(zeros[:, half_n:half_n+ch, :, :], input_sqr)
        else:
            input_sqr = tf.pad(input_sqr, [[0, 0], [half_n, half_n], [0, 0], [0, 0]])
        scale = self.k # offset for the scale
        norm_alpha = self.alpha / self.n # normalized alpha
        for i in range(self.n):
            scale += norm_alpha * input_sqr[:, i:i+ch, :, :]
        scale = scale ** self.beta
        x = x / scale
        return x

    def get_config(self):
        config = {"alpha": self.alpha,
                  "k": self.k,
                  "beta": self.beta,
                  "n": self.n}
        base_config = super(LRN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class FeatureExtractor(BaseModel):
    """ Scaled-Down GoogLeNet """
    def __init__(self, **kwargs):
        super().__init__(input_shape = (32, 32, 3), output_shape = (2048, ), **kwargs)

    def prepare_model(self):
        inputs = Input(shape = self.input_shape)
        print (inputs)
        input_pad = ZeroPadding2D(padding=(3, 3))(inputs)
        conv_1 = Conv2D(64, (7,7), strides=(2,2), padding='valid', name='conv_1_7x7_s2', kernel_regularizer=l2(0.0002))(input_pad)
        print (conv_1.shape)
        conv_1_zero_pad = ZeroPadding2D(padding=(1, 1))(conv_1)
        print (conv_1_zero_pad.shape)
        # pool_1_helper = PoolHelper()(conv_1_zero_pad)
        pool_1_helper = conv_1_zero_pad
        pool_1 = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid', name='pool_1_3x3_s2')(pool_1_helper)
        pool_1_norm = LRN(name="pool_1_norm")(pool_1)

        # 1x1 Convolution
        conv_2_reduce = Conv2D(64, (1,1), padding='valid', activation='relu', name='conv_2_reduce', kernel_regularizer=l2(0.0002))(pool_1_norm)
        print (conv_2_reduce.shape)
        conv_2 = Conv2D(192, (3,3), padding='same', activation='relu', name='conv_2', kernel_regularizer=l2(0.0002))(conv_2_reduce)
        print (conv_2)
        conv_2_norm = LRN(name='conv_2_norm')(conv_2)
        conv_2_zero_pad = ZeroPadding2D(padding=(1, 1))(conv_2_norm)
        # pool_2_helper = PoolHelper()(conv_2_zero_pad)
        pool_2_helper = conv_2_zero_pad
        pool_2 = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid', name='pool_2')(pool_2_helper)

        inception_3a_1x1 = Conv2D(64, (1,1), padding='same', activation='relu', name='inception_3a_1x1', kernel_regularizer=l2(0.0002))(pool_2)
        inception_3a_3x3_reduce = Conv2D(96, (1,1), padding='same', activation='relu', name='inception_3a_3x3_reduce', kernel_regularizer=l2(0.0002))(pool_2)
        inception_3a_3x3_pad = ZeroPadding2D(padding=(1, 1))(inception_3a_3x3_reduce)
        inception_3a_3x3 = Conv2D(128, (3,3), padding='valid', activation='relu', name='inception_3a_3x3', kernel_regularizer=l2(0.0002))(inception_3a_3x3_pad)
        inception_3a_5x5_reduce = Conv2D(16, (1,1), padding='same', activation='relu', name='inception_3a_5x5_reduce', kernel_regularizer=l2(0.0002))(pool_2)
        inception_3a_5x5_pad = ZeroPadding2D(padding=(2, 2))(inception_3a_5x5_reduce)
        inception_3a_5x5 = Conv2D(32, (5,5), padding='valid', activation='relu', name='inception_3a_5x5', kernel_regularizer=l2(0.0002))(inception_3a_5x5_pad)
        inception_3a_pool = MaxPooling2D(pool_size=(3,3), strides=(1,1), padding='same', name='inception_3a_pool')(pool_2)
        inception_3a_pool_proj = Conv2D(32, (1,1), padding='same', activation='relu', name='inception_3a_pool_proj', kernel_regularizer=l2(0.0002))(inception_3a_pool)
        inception_3a_output = Concatenate(axis=3, name='inception_3a_output')([inception_3a_1x1,inception_3a_3x3,inception_3a_5x5,inception_3a_pool_proj])

        # 3B
        inception_3b_1x1 = Conv2D(128, (1,1), padding='same', activation='relu', name='inception_3b_1x1', kernel_regularizer=l2(0.0002))(inception_3a_output)
        inception_3b_3x3_reduce = Conv2D(128, (1,1), padding='same', activation='relu', name='inception_3b_3x3_reduce', kernel_regularizer=l2(0.0002))(inception_3a_output)
        inception_3b_3x3_pad = ZeroPadding2D(padding=(1, 1))(inception_3b_3x3_reduce)
        inception_3b_3x3 = Conv2D(192, (3,3), padding='valid', activation='relu', name='inception_3b_3x3', kernel_regularizer=l2(0.0002))(inception_3b_3x3_pad)
        inception_3b_5x5_reduce = Conv2D(32, (1,1), padding='same', activation='relu', name='inception_3b_5x5_reduce', kernel_regularizer=l2(0.0002))(inception_3a_output)
        inception_3b_5x5_pad = ZeroPadding2D(padding=(2, 2))(inception_3b_5x5_reduce)
        inception_3b_5x5 = Conv2D(96, (5,5), padding='valid', activation='relu', name='inception_3b_5x5', kernel_regularizer=l2(0.0002))(inception_3b_5x5_pad)
        inception_3b_pool = MaxPooling2D(pool_size=(3,3), strides=(1,1), padding='same', name='inception_3b_pool')(inception_3a_output)
        inception_3b_pool_proj = Conv2D(64, (1,1), padding='same', activation='relu', name='inception_3b_pool_proj', kernel_regularizer=l2(0.0002))(inception_3b_pool)
        inception_3b_output = Concatenate(axis=3, name='inception_3b_output')([inception_3b_1x1,inception_3b_3x3,inception_3b_5x5,inception_3b_pool_proj])

        inception_3b_output_zero_pad = ZeroPadding2D(padding=(1, 1))(inception_3b_output)
        # pool3_helper = PoolHelper()(inception_3b_output_zero_pad)
        pool3_helper = inception_3b_output_zero_pad
        pool3_3x3 = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid', name='pool3_3x3')(pool3_helper)

        inception_4a_1x1 = Conv2D(192, (1,1), padding='same', activation='relu', name='inception_4a_1x1', kernel_regularizer=l2(0.0002))(pool3_3x3)
        inception_4a_3x3_reduce = Conv2D(96, (1,1), padding='same', activation='relu', name='inception_4a_3x3_reduce', kernel_regularizer=l2(0.0002))(pool3_3x3)
        inception_4a_3x3_pad = ZeroPadding2D(padding=(1, 1))(inception_4a_3x3_reduce)
        inception_4a_3x3 = Conv2D(208, (3,3), padding='valid', activation='relu', name='inception_4a_3x3' ,kernel_regularizer=l2(0.0002))(inception_4a_3x3_pad)
        inception_4a_5x5_reduce = Conv2D(16, (1,1), padding='same', activation='relu', name='inception_4a_5x5_reduce', kernel_regularizer=l2(0.0002))(pool3_3x3)
        inception_4a_5x5_pad = ZeroPadding2D(padding=(2, 2))(inception_4a_5x5_reduce)
        inception_4a_5x5 = Conv2D(48, (5,5), padding='valid', activation='relu', name='inception_4a_5x5', kernel_regularizer=l2(0.0002))(inception_4a_5x5_pad)
        inception_4a_pool = MaxPooling2D(pool_size=(3,3), strides=(1,1), padding='same', name='inception_4a_pool')(pool3_3x3)
        inception_4a_pool_proj = Conv2D(64, (1,1), padding='same', activation='relu', name='inception_4a_pool_proj', kernel_regularizer=l2(0.0002))(inception_4a_pool)
        inception_4a_output = Concatenate(axis=3, name='inception_4a_output')([inception_4a_1x1,inception_4a_3x3,inception_4a_5x5,inception_4a_pool_proj])

        # loss1_ave_pool = AveragePooling2D(pool_size=(5,5), strides=(3,3), name='loss1_ave_pool')(inception_4a_output)
        loss1_ave_pool = AveragePooling2D(pool_size=(2,2), strides=(1,1), name='loss1_ave_pool')(inception_4a_output)
        loss1_conv = Conv2D(128, (1,1), padding='same', activation='relu', name='loss1_conv', kernel_regularizer=l2(0.0002))(loss1_ave_pool)
        loss1_flat = Flatten()(loss1_conv)
        loss1_fc = Dense(1024, activation='relu', name='loss1_fc', kernel_regularizer=l2(0.0002))(loss1_flat)
        # loss1_drop_fc = Dropout(rate=0.7)(loss1_fc)
        loss1_drop_fc = Dropout(rate=0.4)(loss1_fc)
        loss1_classifier = Dense(10, name='loss1_classifier', kernel_regularizer=l2(0.0002))(loss1_drop_fc)
        loss1_classifier_act = Activation('softmax')(loss1_classifier)

        # pool5_7x7 = AveragePooling2D(pool_size=(7,7), strides=(1,1), name='pool5_7x7_s2')(inception_5b_output)
        # loss3_flat = Flatten()(pool5_7x7)
        # pool5_drop_7x7 = Dropout(rate=0.4)(loss3_flat)
        # loss3_classifier = Dense(1000, name='loss3_classifier', kernel_regularizer=l2(0.0002))(pool5_drop_7x7)
        # loss3_classifier_act = Activation('softmax', name='prob')(loss3_classifier)

        # inception_4b_1x1 = Conv2D(160, (1,1), padding='same', activation='relu', name='inception_4b_1x1', kernel_regularizer=l2(0.0002))(inception_4a_output)
        # inception_4b_3x3_reduce = Conv2D(112, (1,1), padding='same', activation='relu', name='inception_4b_3x3_reduce', kernel_regularizer=l2(0.0002))(inception_4a_output)
        # inception_4b_3x3_pad = ZeroPadding2D(padding=(1, 1))(inception_4b_3x3_reduce)
        # inception_4b_3x3 = Conv2D(224, (3,3), padding='valid', activation='relu', name='inception_4b_3x3', kernel_regularizer=l2(0.0002))(inception_4b_3x3_pad)
        # inception_4b_5x5_reduce = Conv2D(24, (1,1), padding='same', activation='relu', name='inception_4b_5x5_reduce', kernel_regularizer=l2(0.0002))(inception_4a_output)
        # inception_4b_5x5_pad = ZeroPadding2D(padding=(2, 2))(inception_4b_5x5_reduce)
        # inception_4b_5x5 = Conv2D(64, (5,5), padding='valid', activation='relu', name='inception_4b_5x5', kernel_regularizer=l2(0.0002))(inception_4b_5x5_pad)
        # inception_4b_pool = MaxPooling2D(pool_size=(3,3), strides=(1,1), padding='same', name='inception_4b_pool')(inception_4a_output)
        # inception_4b_pool_proj = Conv2D(64, (1,1), padding='same', activation='relu', name='inception_4b_pool_proj', kernel_regularizer=l2(0.0002))(inception_4b_pool)
        # inception_4b_output = Concatenate(axis=3, name='inception_4b_output')([inception_4b_1x1,inception_4b_3x3,inception_4b_5x5,inception_4b_pool_proj])

        # inception_4c_1x1 = Conv2D(128, (1,1), padding='same', activation='relu', name='inception_4c_1x1', kernel_regularizer=l2(0.0002))(inception_4b_output)
        # inception_4c_3x3_reduce = Conv2D(128, (1,1), padding='same', activation='relu', name='inception_4c_3x3_reduce', kernel_regularizer=l2(0.0002))(inception_4b_output)
        # inception_4c_3x3_pad = ZeroPadding2D(padding=(1, 1))(inception_4c_3x3_reduce)
        # inception_4c_3x3 = Conv2D(256, (3,3), padding='valid', activation='relu', name='inception_4c_3x3', kernel_regularizer=l2(0.0002))(inception_4c_3x3_pad)
        # inception_4c_5x5_reduce = Conv2D(24, (1,1), padding='same', activation='relu', name='inception_4c_5x5_reduce', kernel_regularizer=l2(0.0002))(inception_4b_output)
        # inception_4c_5x5_pad = ZeroPadding2D(padding=(2, 2))(inception_4c_5x5_reduce)
        # inception_4c_5x5 = Conv2D(64, (5,5), padding='valid', activation='relu', name='inception_4c_5x5', kernel_regularizer=l2(0.0002))(inception_4c_5x5_pad)
        # inception_4c_pool = MaxPooling2D(pool_size=(3,3), strides=(1,1), padding='same', name='inception_4c_pool')(inception_4b_output)
        # inception_4c_pool_proj = Conv2D(64, (1,1), padding='same', activation='relu', name='inception_4c_pool_proj', kernel_regularizer=l2(0.0002))(inception_4c_pool)
        # inception_4c_output = Concatenate(axis=3, name='inception_4c_output')([inception_4c_1x1,inception_4c_3x3,inception_4c_5x5,inception_4c_pool_proj])

        # inception_4d_1x1 = Conv2D(112, (1,1), padding='same', activation='relu', name='inception_4d_1x1', kernel_regularizer=l2(0.0002))(inception_4c_output)
        # inception_4d_3x3_reduce = Conv2D(144, (1,1), padding='same', activation='relu', name='inception_4d_3x3_reduce', kernel_regularizer=l2(0.0002))(inception_4c_output)
        # inception_4d_3x3_pad = ZeroPadding2D(padding=(1, 1))(inception_4d_3x3_reduce)
        # inception_4d_3x3 = Conv2D(288, (3,3), padding='valid', activation='relu', name='inception_4d_3x3', kernel_regularizer=l2(0.0002))(inception_4d_3x3_pad)
        # inception_4d_5x5_reduce = Conv2D(32, (1,1), padding='same', activation='relu', name='inception_4d_5x5_reduce', kernel_regularizer=l2(0.0002))(inception_4c_output)
        # inception_4d_5x5_pad = ZeroPadding2D(padding=(2, 2))(inception_4d_5x5_reduce)
        # inception_4d_5x5 = Conv2D(64, (5,5), padding='valid', activation='relu', name='inception_4d_5x5', kernel_regularizer=l2(0.0002))(inception_4d_5x5_pad)
        # inception_4d_pool = MaxPooling2D(pool_size=(3,3), strides=(1,1), padding='same', name='inception_4d_pool')(inception_4c_output)
        # inception_4d_pool_proj = Conv2D(64, (1,1), padding='same', activation='relu', name='inception_4d_pool_proj', kernel_regularizer=l2(0.0002))(inception_4d_pool)
        # inception_4d_output = Concatenate(axis=3, name='inception_4d_output')([inception_4d_1x1,inception_4d_3x3,inception_4d_5x5,inception_4d_pool_proj])

        # loss2_ave_pool = AveragePooling2D(pool_size=(5,5), strides=(3,3), name='loss2_ave_pool')(inception_4d_output)
        # loss2_conv = Conv2D(128, (1,1), padding='same', activation='relu', name='loss2_conv', kernel_regularizer=l2(0.0002))(loss2_ave_pool)
        # loss2_flat = Flatten()(loss2_conv)
        # loss2_fc = Dense(1024, activation='relu', name='loss2_fc', kernel_regularizer=l2(0.0002))(loss2_flat)
        # loss2_drop_fc = Dropout(rate=0.7)(loss2_fc)
        # loss2_classifier = Dense(1000, name='loss2_classifier', kernel_regularizer=l2(0.0002))(loss2_drop_fc)
        # loss2_classifier_act = Activation('softmax')(loss2_classifier)

        # inception_4e_1x1 = Conv2D(256, (1,1), padding='same', activation='relu', name='inception_4e_1x1', kernel_regularizer=l2(0.0002))(inception_4d_output)
        # inception_4e_3x3_reduce = Conv2D(160, (1,1), padding='same', activation='relu', name='inception_4e_3x3_reduce', kernel_regularizer=l2(0.0002))(inception_4d_output)
        # inception_4e_3x3_pad = ZeroPadding2D(padding=(1, 1))(inception_4e_3x3_reduce)
        # inception_4e_3x3 = Conv2D(320, (3,3), padding='valid', activation='relu', name='inception_4e_3x3', kernel_regularizer=l2(0.0002))(inception_4e_3x3_pad)
        # inception_4e_5x5_reduce = Conv2D(32, (1,1), padding='same', activation='relu', name='inception_4e_5x5_reduce', kernel_regularizer=l2(0.0002))(inception_4d_output)
        # inception_4e_5x5_pad = ZeroPadding2D(padding=(2, 2))(inception_4e_5x5_reduce)
        # inception_4e_5x5 = Conv2D(128, (5,5), padding='valid', activation='relu', name='inception_4e_5x5', kernel_regularizer=l2(0.0002))(inception_4e_5x5_pad)
        # inception_4e_pool = MaxPooling2D(pool_size=(3,3), strides=(1,1), padding='same', name='inception_4e_pool')(inception_4d_output)
        # inception_4e_pool_proj = Conv2D(128, (1,1), padding='same', activation='relu', name='inception_4e_pool_proj', kernel_regularizer=l2(0.0002))(inception_4e_pool)
        # inception_4e_output = Concatenate(axis=3, name='inception_4e_output')([inception_4e_1x1,inception_4e_3x3,inception_4e_5x5,inception_4e_pool_proj])

        # inception_4e_output_zero_pad = ZeroPadding2D(padding=(1, 1))(inception_4e_output)
        # # pool4_helper = PoolHelper()(inception_4e_output_zero_pad)
        # pool4_helper = inception_4e_output_zero_pad
        # pool4_3x3 = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid', name='pool4_3x3_s2')(pool4_helper)

        # inception_5a_1x1 = Conv2D(256, (1,1), padding='same', activation='relu', name='inception_5a_1x1', kernel_regularizer=l2(0.0002))(pool4_3x3)
        # inception_5a_3x3_reduce = Conv2D(160, (1,1), padding='same', activation='relu', name='inception_5a_3x3_reduce', kernel_regularizer=l2(0.0002))(pool4_3x3)
        # inception_5a_3x3_pad = ZeroPadding2D(padding=(1, 1))(inception_5a_3x3_reduce)
        # inception_5a_3x3 = Conv2D(320, (3,3), padding='valid', activation='relu', name='inception_5a_3x3', kernel_regularizer=l2(0.0002))(inception_5a_3x3_pad)
        # inception_5a_5x5_reduce = Conv2D(32, (1,1), padding='same', activation='relu', name='inception_5a_5x5_reduce', kernel_regularizer=l2(0.0002))(pool4_3x3)
        # inception_5a_5x5_pad = ZeroPadding2D(padding=(2, 2))(inception_5a_5x5_reduce)
        # inception_5a_5x5 = Conv2D(128, (5,5), padding='valid', activation='relu', name='inception_5a_5x5', kernel_regularizer=l2(0.0002))(inception_5a_5x5_pad)
        # inception_5a_pool = MaxPooling2D(pool_size=(3,3), strides=(1,1), padding='same', name='inception_5a_pool')(pool4_3x3)
        # inception_5a_pool_proj = Conv2D(128, (1,1), padding='same', activation='relu', name='inception_5a_pool_proj', kernel_regularizer=l2(0.0002))(inception_5a_pool)
        # inception_5a_output = Concatenate(axis=3, name='inception_5a_output')([inception_5a_1x1,inception_5a_3x3,inception_5a_5x5,inception_5a_pool_proj])

        # inception_5b_1x1 = Conv2D(384, (1,1), padding='same', activation='relu', name='inception_5b_1x1', kernel_regularizer=l2(0.0002))(inception_5a_output)
        # inception_5b_3x3_reduce = Conv2D(192, (1,1), padding='same', activation='relu', name='inception_5b_3x3_reduce', kernel_regularizer=l2(0.0002))(inception_5a_output)
        # inception_5b_3x3_pad = ZeroPadding2D(padding=(1, 1))(inception_5b_3x3_reduce)
        # inception_5b_3x3 = Conv2D(384, (3,3), padding='valid', activation='relu', name='inception_5b_3x3', kernel_regularizer=l2(0.0002))(inception_5b_3x3_pad)
        # inception_5b_5x5_reduce = Conv2D(48, (1,1), padding='same', activation='relu', name='inception_5b_5x5_reduce', kernel_regularizer=l2(0.0002))(inception_5a_output)
        # inception_5b_5x5_pad = ZeroPadding2D(padding=(2, 2))(inception_5b_5x5_reduce)
        # inception_5b_5x5 = Conv2D(128, (5,5), padding='valid', activation='relu', name='inception_5b_5x5', kernel_regularizer=l2(0.0002))(inception_5b_5x5_pad)
        # inception_5b_pool = MaxPooling2D(pool_size=(3,3), strides=(1,1), padding='same', name='inception_5b_pool')(inception_5a_output)
        # inception_5b_pool_proj = Conv2D(128, (1,1), padding='same', activation='relu', name='inception_5b_pool_proj', kernel_regularizer=l2(0.0002))(inception_5b_pool)
        # inception_5b_output = Concatenate(axis=3, name='inception_5b_output')([inception_5b_1x1,inception_5b_3x3,inception_5b_5x5,inception_5b_pool_proj])

        # pool5_7x7 = AveragePooling2D(pool_size=(7,7), strides=(1,1), name='pool5_7x7_s2')(inception_5b_output)
        # loss3_flat = Flatten()(pool5_7x7)
        # pool5_drop_7x7 = Dropout(rate=0.4)(loss3_flat)
        # loss3_classifier = Dense(1000, name='loss3_classifier', kernel_regularizer=l2(0.0002))(pool5_drop_7x7)
        # loss3_classifier_act = Activation('softmax', name='prob')(loss3_classifier)

        return Model(inputs=inputs, outputs=loss1_classifier_act, name = self.name) # , loss2_classifier_act, loss3_classifier_act])
        
    @property
    def loss(self):
        return "categorical_crossentropy"
    
    @property
    def optimizer(self):
        # return SGD(lr=0.08, momentum=0.9, nesterov=True, clipvalue=1.0) # decay=1e-6, 
        return Adadelta()

