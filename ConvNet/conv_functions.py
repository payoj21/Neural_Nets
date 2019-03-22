from functions import Function
from numba import jit
import np_utils


class Convolution2D(Function):

    @staticmethod
    @jit(nopython=True)
    def example_helper_func():
        """
        Example of an accelerated function, Notice the Numba jit decorator on top.
        """
        pass

    def forward(self, stride, padding, *args):
        """
        Forward pass of the convolution operation between two four dimensional tensors.
        :param stride: Convolution stride, defaults to 1.
        :param padding: Convolution padding, defaults to 0.
        :param args: Operands of convolution operation (input(batch_size, in_channels, H, W), kernel(out_channels, in_channels, Hk, Wk)).
        :return: Output of the convolution operation.
        """
        #TODO
        return None

    def backward(self, gradient):
        """
        Sets the gradients for operands of convolution operation.
        :param gradient: Upstream gradient.
        """
        #TODO
        pass


class Reshape(Function):
    def forward(self, shape, *args):
        """
        Forward pass of the reshape operation on a tensor
        :param shape: tuple of required dimension.
        :param args: Input tensor to be reshaped.
        :return: reshaped tensor.
        """
        #TODO
        return None

    def backward(self, gradient):
        """
        Sets the gradient for input of reshape operation.
        :param gradient: Upstream gradient.
        """
        #TODO
        pass



