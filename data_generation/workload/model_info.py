from __future__ import division

import six
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Activation,
    Dense,
    Flatten
)
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    AveragePooling2D
)
from tensorflow.keras.layers import add
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D , Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import tensorflow as tf

def select_model(model_name, input_shape, num_classes):
    
    def LeNet5(input_shape, num_classes):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Conv2D(6,kernel_size=(5, 5),
                                         activation='relu', input_shape=input_shape))
        model.add(tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2)))
        model.add(tf.keras.layers.Conv2D(16, kernel_size=(5, 5), activation='relu'))
        model.add(tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2)))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(120, activation='relu'))
        model.add(tf.keras.layers.Dense(84, activation='relu'))
        model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
        return model

    def ResNetSmall(input_shape, num_classes):
        # Build ResNet-small model
        def res_block(x,filter,stride,name):
            input = x
            if stride != 1:
                input = tf.keras.layers.Conv2D(
                    filters=filter,kernel_size=1,strides=stride,name=name+'_pooling_conv')(input)
                input = tf.keras.layers.BatchNormalization(
                    name=name+'_pooling_bn')(input)

            x = tf.keras.layers.Conv2D(
                filters=filter,kernel_size=1,strides=stride,padding='same',name=name+'_conv1')(x)
            x = tf.keras.layers.BatchNormalization(name=name+'_bn1')(x)
            x = tf.nn.relu(x,name=name+'_relu1')

            x = tf.keras.layers.Conv2D(filters=filter,
                                       kernel_size=1,strides=1,padding='same',name=name+'_conv2')(x)
            x = tf.keras.layers.BatchNormalization(name=name+'_bn2')(x)
            x = tf.keras.layers.add([input,x],name=name+'_add')

            x = tf.nn.relu(x,name=name+'_relu2')
            return x

        def model_builder(x,attention):
            x = tf.keras.layers.Conv2D(filters=64,kernel_size=7,strides=2,
                                       activation='relu',padding='same',name='conv1')(x)
            x = tf.keras.layers.BatchNormalization(name='conv1_bn')(x)
            x = tf.keras.layers.MaxPool2D(
                pool_size=3,strides=2,padding='same',name='conv1_max_pool')(x)

            x = res_block(x,64,2,'ResBlock21')
            x = res_block(x,64,1,'ResBlock22')
            x = res_block(x,128,2,'ResBlock31')
            x = res_block(x,128,1,'ResBlock32')

            x =tf.keras.layers.GlobalAveragePooling2D(name='GAP')(x) 
            pred = tf.keras.layers.Dense(num_classes,activation='softmax')(x)

            return pred

        inputs = tf.keras.Input(shape=input_shape)
        pred_normal = model_builder(inputs,None)
        model = tf.keras.Model(inputs=inputs,outputs=pred_normal)
        return model

    def VGG11(input_shape, num_classes):
        model = tf.keras.models.Sequential()
        model.add(Conv2D(input_shape=input_shape,
                         filters=64,kernel_size=(3,3),padding="same", activation="relu"))
        model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
        model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
        model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
        model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
        model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
        model.add(Flatten())
        model.add(Dense(units=4096,activation="relu"))
        model.add(Dense(units=4096,activation="relu"))
        model.add(Dense(units=num_classes, activation="softmax"))
        return model
    
    def VGGSmall(input_shape, num_classes):
        model = tf.keras.models.Sequential()
        model.add(Conv2D(input_shape=input_shape,
                         filters=32,kernel_size=(3,3),padding="same", activation="relu"))
        model.add(Conv2D(filters=32,kernel_size=(3,3),padding="same", activation="relu"))
        model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
        model.add(Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
        model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
        model.add(Flatten())
        model.add(Dense(units=1024,activation="relu"))
        model.add(Dense(units=1024,activation="relu"))
        model.add(Dense(units=num_classes, activation="softmax"))
        return model
    
    def VGG13(input_shape, num_classes):
        model = tf.keras.models.Sequential()
        model.add(Conv2D(input_shape=input_shape,
                         filters=64,kernel_size=(3,3),padding="same", activation="relu"))
        model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
        model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
        model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
        model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
        model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
        model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
        model.add(Flatten())
        model.add(Dense(units=4096,activation="relu"))
        model.add(Dense(units=4096,activation="relu"))
        model.add(Dense(units=num_classes, activation="softmax"))
        return model

    def VGG16(input_shape, num_classes):
        model = tf.keras.models.Sequential()
        model.add(Conv2D(input_shape=input_shape,
                         filters=64,kernel_size=(3,3),padding="same", activation="relu"))
        model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
        model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
        model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
        model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
        model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
        model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
        model.add(Flatten())
        model.add(Dense(units=4096,activation="relu"))
        model.add(Dense(units=4096,activation="relu"))
        model.add(Dense(units=num_classes, activation="softmax"))
        return model

    def VGG19(input_shape, num_classes):
        model = tf.keras.models.Sequential()
        model.add(Conv2D(input_shape=input_shape,
                         filters=64,kernel_size=(3,3),padding="same", activation="relu"))
        model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
        model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
        model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
        model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
        model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
        model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
        model.add(Flatten())
        model.add(Dense(units=4096,activation="relu"))
        model.add(Dense(units=4096,activation="relu"))
        model.add(Dense(units=num_classes, activation="softmax"))
        return model

    def ResNet(input_shape, num_classes, resnet_number):

        def _bn_relu(input):
            """Helper to build a BN -> relu block
            """
            norm = BatchNormalization(axis=CHANNEL_AXIS)(input)
            return Activation("relu")(norm)


        def _conv_bn_relu(**conv_params):
            """Helper to build a conv -> BN -> relu block
            """
            filters = conv_params["filters"]
            kernel_size = conv_params["kernel_size"]
            strides = conv_params.setdefault("strides", (1, 1))
            kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
            padding = conv_params.setdefault("padding", "same")
            kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1.e-4))

            def f(input):
                conv = Conv2D(filters=filters, kernel_size=kernel_size,
                              strides=strides, padding=padding,
                              kernel_initializer=kernel_initializer,
                              kernel_regularizer=kernel_regularizer)(input)
                return _bn_relu(conv)

            return f


        def _bn_relu_conv(**conv_params):
            """Helper to build a BN -> relu -> conv block.
            This is an improved scheme proposed in http://arxiv.org/pdf/1603.05027v2.pdf
            """
            filters = conv_params["filters"]
            kernel_size = conv_params["kernel_size"]
            strides = conv_params.setdefault("strides", (1, 1))
            kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
            padding = conv_params.setdefault("padding", "same")
            kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1.e-4))

            def f(input):
                activation = _bn_relu(input)
                return Conv2D(filters=filters, kernel_size=kernel_size,
                              strides=strides, padding=padding,
                              kernel_initializer=kernel_initializer,
                              kernel_regularizer=kernel_regularizer)(activation)

            return f


        def _shortcut(input, residual):
            """Adds a shortcut between input and residual block and merges them with "sum"
            """
            # Expand channels of shortcut to match residual.
            # Stride appropriately to match residual (width, height)
            # Should be int if network architecture is correctly configured.
            input_shape = K.int_shape(input)
            residual_shape = K.int_shape(residual)
            stride_width = int(round(input_shape[ROW_AXIS] / residual_shape[ROW_AXIS]))
            stride_height = int(round(input_shape[COL_AXIS] / residual_shape[COL_AXIS]))
            equal_channels = input_shape[CHANNEL_AXIS] == residual_shape[CHANNEL_AXIS]

            shortcut = input
            # 1 X 1 conv if shape is different. Else identity.
            if stride_width > 1 or stride_height > 1 or not equal_channels:
                shortcut = Conv2D(filters=residual_shape[CHANNEL_AXIS],
                                  kernel_size=(1, 1),
                                  strides=(stride_width, stride_height),
                                  padding="valid",
                                  kernel_initializer="he_normal",
                                  kernel_regularizer=l2(0.0001))(input)

            return add([shortcut, residual])


        def _residual_block(block_function, filters, repetitions, is_first_layer=False):
            """Builds a residual block with repeating bottleneck blocks.
            """
            def f(input):
                for i in range(repetitions):
                    init_strides = (1, 1)
                    if i == 0 and not is_first_layer:
                        init_strides = (2, 2)
                    input = block_function(filters=filters, init_strides=init_strides,
                                           is_first_block_of_first_layer=(is_first_layer and i == 0))(input)
                return input

            return f


        def basic_block(filters, init_strides=(1, 1), is_first_block_of_first_layer=False):
            """Basic 3 X 3 convolution blocks for use on resnets with layers <= 34.
            Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf
            """
            def f(input):

                if is_first_block_of_first_layer:
                    # don't repeat bn->relu since we just did bn->relu->maxpool
                    conv1 = Conv2D(filters=filters, kernel_size=(3, 3),
                                   strides=init_strides,
                                   padding="same",
                                   kernel_initializer="he_normal",
                                   kernel_regularizer=l2(1e-4))(input)
                else:
                    conv1 = _bn_relu_conv(filters=filters, kernel_size=(3, 3),
                                          strides=init_strides)(input)

                residual = _bn_relu_conv(filters=filters, kernel_size=(3, 3))(conv1)
                return _shortcut(input, residual)

            return f


        def bottleneck(filters, init_strides=(1, 1), is_first_block_of_first_layer=False):
            """Bottleneck architecture for > 34 layer resnet.
            Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf
            Returns:
                A final conv layer of filters * 4
            """
            def f(input):

                if is_first_block_of_first_layer:
                    # don't repeat bn->relu since we just did bn->relu->maxpool
                    conv_1_1 = Conv2D(filters=filters, kernel_size=(1, 1),
                                      strides=init_strides,
                                      padding="same",
                                      kernel_initializer="he_normal",
                                      kernel_regularizer=l2(1e-4))(input)
                else:
                    conv_1_1 = _bn_relu_conv(filters=filters, kernel_size=(1, 1),
                                             strides=init_strides)(input)

                conv_3_3 = _bn_relu_conv(filters=filters, kernel_size=(3, 3))(conv_1_1)
                residual = _bn_relu_conv(filters=filters * 4, kernel_size=(1, 1))(conv_3_3)
                return _shortcut(input, residual)

            return f


        def _handle_dim_ordering():
            global ROW_AXIS
            global COL_AXIS
            global CHANNEL_AXIS
            ROW_AXIS = 1
            COL_AXIS = 2
            CHANNEL_AXIS = 3
        #     if K.image_dim_ordering() == 'tf':
        #         ROW_AXIS = 1
        #         COL_AXIS = 2
        #         CHANNEL_AXIS = 3
        #     else:
        #         CHANNEL_AXIS = 1
        #         ROW_AXIS = 2
        #         COL_AXIS = 3

        def _get_block(identifier):
            if isinstance(identifier, six.string_types):
                res = globals().get(identifier)
                if not res:
                    raise ValueError('Invalid {}'.format(identifier))
                return res
            return identifier


    #     class ResnetBuilder(object):
    #         @staticmethod
        def build(input_shape, num_outputs, block_fn, repetitions):
            _handle_dim_ordering()
            if len(input_shape) != 3:
                raise Exception("Input shape should be a tuple (nb_channels, nb_rows, nb_cols)")

            # Permute dimension order if necessary
            input_shape = (input_shape[1], input_shape[2], input_shape[0])
    #         if K.image_dim_ordering() == 'tf':
    #             input_shape = (input_shape[1], input_shape[2], input_shape[0])

            # Load function from str if needed.
            block_fn = _get_block(block_fn)

            input = Input(shape=input_shape)
            conv1 = _conv_bn_relu(filters=64, kernel_size=(7, 7), strides=(2, 2))(input)
            pool1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")(conv1)

            block = pool1
            filters = 64
            for i, r in enumerate(repetitions):
                block = _residual_block(block_fn, filters=filters,
                                        repetitions=r, is_first_layer=(i == 0))(block)
                filters *= 2

            # Last activation
            block = _bn_relu(block)

            # Classifier block
            block_shape = K.int_shape(block)
            pool2 = AveragePooling2D(pool_size=(block_shape[ROW_AXIS], block_shape[COL_AXIS]),
                                     strides=(1, 1))(block)
            flatten1 = Flatten()(pool2)
            dense = Dense(units=num_outputs, kernel_initializer="he_normal",
                          activation="softmax")(flatten1)

            model = Model(inputs=input, outputs=dense)
            return model

    #     @staticmethod
        def build_resnet_18(input_shape, num_outputs):
            return build(input_shape, num_outputs, basic_block, [2, 2, 2, 2])

    #     @staticmethod
        def build_resnet_34(input_shape, num_outputs):
            return build(input_shape, num_outputs, basic_block, [3, 4, 6, 3])

    #     @staticmethod
        def build_resnet_50(input_shape, num_outputs):
            return build(input_shape, num_outputs, bottleneck, [3, 4, 6, 3])

    #     @staticmethod
        def build_resnet_101(input_shape, num_outputs):
            return build(input_shape, num_outputs, bottleneck, [3, 4, 23, 3])

    #     @staticmethod
        def build_resnet_152(input_shape, num_outputs):
            return build(input_shape, num_outputs, bottleneck, [3, 8, 36, 3])

        row, col, channel = input_shape
        input_shape = (channel, row, col)

        if resnet_number == 18:
            return build_resnet_18(input_shape, num_classes)
        elif resnet_number == 34:
            return build_resnet_34(input_shape, num_classes)
        elif resnet_number == 50:
            return build_resnet_50(input_shape, num_classes)
        else:
            print('model select error')
            return 0

    def MNIST_CNN(input_shape, num_classes):
        model = tf.keras.Sequential([
            tf.keras.Input(shape=input_shape),
            tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(num_classes, activation='softmax'),
        ])
        return model

    def CIFAR10_CNN(input_shape, num_classes):
        model = tf.keras.Sequential()
        model.add(
            tf.keras.layers.Conv2D(
                32, (3, 3), padding='same', input_shape=input_shape))
        model.add(tf.keras.layers.Activation('relu'))
        model.add(tf.keras.layers.Conv2D(32, (3, 3)))
        model.add(tf.keras.layers.Activation('relu'))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(tf.keras.layers.Dropout(0.25))

        model.add(tf.keras.layers.Conv2D(64, (3, 3), padding='same'))
        model.add(tf.keras.layers.Activation('relu'))
        model.add(tf.keras.layers.Conv2D(64, (3, 3)))
        model.add(tf.keras.layers.Activation('relu'))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(tf.keras.layers.Dropout(0.25))

        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(512))
        model.add(tf.keras.layers.Activation('relu'))
        model.add(tf.keras.layers.Dropout(0.5))
        model.add(tf.keras.layers.Dense(num_classes))
        model.add(tf.keras.layers.Activation('softmax'))
        return model

    def FLOWER_CNN(input_shape, num_classes):
        model = tf.keras.Sequential()
        model.add(
            tf.keras.layers.Conv2D(
                16, 3, padding='same', activation='relu', input_shape=input_shape))
        model.add(tf.keras.layers.MaxPooling2D())
        model.add(tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'))
        model.add(tf.keras.layers.MaxPooling2D())
        model.add(tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'))
        model.add(tf.keras.layers.MaxPooling2D())
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(128, activation='relu'))
        model.add(tf.keras.layers.Dense(num_classes))
        return model

    def AlexNet(input_shape, num_classes):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Conv2D(96,
                                        (11, 11),
                                        input_shape=input_shape,
                                        padding='same',
                                        kernel_regularizer=tf.keras.regularizers.l2(0.)))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Activation('relu'))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

        # Layer 2
        model.add(tf.keras.layers.Conv2D(256, (5, 5), padding='same'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Activation('relu'))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

        # Layer 3
        model.add(tf.keras.layers.ZeroPadding2D((1, 1)))
        model.add(tf.keras.layers.Conv2D(512, (3, 3), padding='same'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Activation('relu'))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

        # Layer 4
        model.add(tf.keras.layers.ZeroPadding2D((1, 1)))
        model.add(tf.keras.layers.Conv2D(1024, (3, 3), padding='same'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Activation('relu'))

        # Layer 5
        model.add(tf.keras.layers.ZeroPadding2D((1, 1)))
        model.add(tf.keras.layers.Conv2D(1024, (3, 3), padding='same'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Activation('relu'))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

        # Layer 6
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(3072))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Activation('relu'))
        model.add(tf.keras.layers.Dropout(0.5))

        # Layer 7
        model.add(tf.keras.layers.Dense(4096))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Activation('relu'))
        model.add(tf.keras.layers.Dropout(0.5))

        # Layer 8
        model.add(tf.keras.layers.Dense(num_classes))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Activation('softmax'))
        return model

    def InceptionV3(input_shape, num_classes):
        model = tf.keras.applications.InceptionV3(
            weights=None,
            input_shape=input_shape,
            classes=num_classes)
        return model

    def InceptionResNetV2(input_shape, num_classes):
        model = tf.keras.applications.InceptionResNetV2(
            weights=None,
            input_shape=input_shape,
            classes=num_classes)
        return model

    def Xception(input_shape, num_classes):
        model = tf.keras.applications.Xception(
            weights=None,
            input_shape=input_shape,
            classes=num_classes)
        return model

    def EfficientNetB0(input_shape, num_classes):
        model = tf.keras.applications.EfficientNetB0(
            weights=None,
            input_shape=input_shape,
            classes=num_classes)
        return model

    def MobileNetV2(input_shape, num_classes):
        model = tf.keras.applications.MobileNetV2(
            weights=None,
            input_shape=input_shape,
            classes=num_classes)
        return model

    if model_name == 'LeNet5':
        return LeNet5(input_shape, num_classes)
    elif model_name == 'VGGSmall':
        return VGGSmall(input_shape, num_classes)
    elif model_name == 'VGG11':
        return VGG11(input_shape, num_classes)
    elif model_name == 'VGG13':
        return VGG13(input_shape, num_classes)
    elif model_name == 'VGG16':
        return VGG16(input_shape, num_classes)
    elif model_name == 'VGG19':
        return VGG19(input_shape, num_classes)
    elif model_name == 'ResNetSmall':
        return ResNetSmall(input_shape, num_classes)
    elif model_name == 'ResNet18':
        return ResNet(input_shape, num_classes, 18)
    elif model_name == 'ResNet34':
        return ResNet(input_shape, num_classes, 34)
    elif model_name == 'ResNet50':
        return ResNet(input_shape, num_classes, 50)
    elif model_name == 'MNIST_CNN':
        return MNIST_CNN(input_shape, num_classes)
    elif model_name == 'CIFAR10_CNN':
        return CIFAR10_CNN(input_shape, num_classes)
    elif model_name == 'FLOWER_CNN':
        return FLOWER_CNN(input_shape, num_classes)
    elif model_name == 'AlexNet':
        return AlexNet(input_shape, num_classes)
    elif model_name == 'InceptionV3':
        return InceptionV3(input_shape, num_classes)
    elif model_name == 'InceptionResNetV2':
        return InceptionResNetV2(input_shape, num_classes)
    elif model_name == 'Xception':
        return Xception(input_shape, num_classes)
    elif model_name == 'EfficientNetB0':
        return EfficientNetB0(input_shape, num_classes)
    elif model_name == 'MobileNetV2':
        return MobileNetV2(input_shape, num_classes)
    else:
        return 0
