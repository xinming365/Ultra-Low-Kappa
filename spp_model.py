import tensorflow as tf
import keras.backend as K
class SpatialPyramidPooling():
    """Spatial pyramid pooling layer for 2D inputs.
        See Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition,
        K. He, X. Zhang, S. Ren, J. Sun
        # Arguments
            pool_list: list of int
                List of pooling regions to use. The length of the list is the number of pooling regions,
                each int in the list is the number of regions in that pool. For example [1,2,4] would be 3
                regions with 1, 2x2 and 4x4 max pools, so 21 outputs per feature map
        # Input shape
            4D tensor with shape:
            `(samples, channels, rows, cols)` if dim_ordering='th'
        # Output shape
            2D tensor with shape:
            `(samples, channels * sum([i * i for i in pool_list])`
    """
    def __init__(self,pool_list):
        self.pool_list = pool_list
        self.num_outputs_per_channel = sum([i*i for i in pool_list])

    def output_shape(self,input_shape):
        return input_shape[0], input_shape[1] * self.num_outputs_per_channel

    def call(self, x):
        # x:numpy array
        input_shape = x.shape
        num_rows = input_shape[2]
        num_cols = input_shape[3]

        row_length = [num_rows/i for i in self.pool_list]
        col_length = [num_cols/i for i in self.pool_list]

        outputs = []

        for pool_num, num_pool_regions in enumerate(self.pool_list):
            for x in range(num_pool_regions):
                for y in range(num_pool_regions):
                    x1 = y * col_length[pool_num]
                    x2 = y * col_length[pool_num] + col_length[pool_num]
                    y1 = x * row_length[pool_num]
                    y2 = x * row_length[pool_num] + row_length[pool_num]

                    new_shape = [input_shape[0],input_shape[1],y2-y1,x2-x1]
                    x_crop = x[:,:,y1:y2,x1:x2]
                    xm = tf.reshape(x_crop, new_shape)
                    pooled_val = tf.reduce_mean(xm, axis=(2,3))
                    outputs.append(pooled_val)
        outputs = K.concatenate(outputs)

        return outputs



