class k_max_pool(mx.operator.CustomOp):

    """
    https://github.com/CNevd/DeepLearning-Mxnet/blob/master/DCNN/dcnn_train.py#L15
    """

    def __init__(self, k):
        super(k_max_pool, self).__init__()
        self.k = int(k)

    def forward(self, is_train, req, in_data, out_data, aux):
        x = in_data[0].asnumpy()
        # assert(4 == len(x.shape))
        ind = np.argsort(x, axis=2)
        sorted_ind = np.sort(ind[:, :, -(self.k):, :], axis=2)
        dim0, dim1, dim2, dim3 = sorted_ind.shape
        self.indices_dim0 = np.arange(dim0).repeat(dim1 * dim2 * dim3)
        self.indices_dim1 = np.transpose(
            np.arange(dim1).repeat(dim2 * dim3).reshape((dim1 * dim2 * dim3, 1)).repeat(dim0, axis=1)).flatten()
        self.indices_dim2 = sorted_ind.flatten()
        self.indices_dim3 = np.transpose(
            np.arange(dim3).repeat(dim2).reshape((dim2 * dim3, 1)).repeat(dim0 * dim1, axis=1)).flatten()
        y = x[self.indices_dim0, self.indices_dim1, self.indices_dim2, self.indices_dim3].reshape(sorted_ind.shape)
        self.assign(out_data[0], req[0], mx.nd.array(y))

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        x = out_grad[0].asnumpy()
        y = in_data[0].asnumpy()
        # assert(4 == len(x.shape))
        # assert(4 == len(y.shape))
        y[:, :, :, :] = 0
        y[self.indices_dim0, self.indices_dim1, self.indices_dim2, self.indices_dim3] \
            = x.reshape([x.shape[0] * x.shape[1] * x.shape[2] * x.shape[3], ])
        self.assign(in_grad[0], req[0], mx.nd.array(y))


@mx.operator.register("k_max_pool")
class k_max_poolProp(mx.operator.CustomOpProp):
    def __init__(self, k):
        self.k = int(k)
        super(k_max_poolProp, self).__init__(True)

    def list_argument(self):
        return ['data']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        data_shape = in_shape[0]
        assert (len(data_shape) == 4)
        out_shape = (data_shape[0], data_shape[1], self.k, data_shape[3])
        return [data_shape], [out_shape]

    def create_operator(self, ctx, shapes, dtypes):
        return k_max_pool(self.k)