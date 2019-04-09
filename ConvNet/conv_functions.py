from functions import Function
from numba import jit
import np_utils
import numpy

class Convolution2D(Function):
    
    parents = []
    @staticmethod
    @jit(nopython=False)
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
        self.parents = list(args)
        

        (batch_size, in_channels, H, W) , (out_kchannels, in_kchannels, Hk, Wk) = args[0].value.shape, args[1].value.shape
        
        output_images = batch_size
        channels_out = 1+(in_channels - in_kchannels + 2*padding)/stride
        h_out = (H - Hk + 2*padding)/stride + 1
        w_out = (W - Wk + 2*padding)/stride + 1
        channels_out, h_out, w_out = int(channels_out), int(h_out), int(w_out)

        image, kernel = args[0].value, args[1].value

        output = numpy.zeros((batch_size,channels_out*out_kchannels,h_out,w_out))
        
        for each_image in range(batch_size):
            for start_z in range(channels_out):
                for start_y in range(h_out):
                    for start_x in range(w_out):
                        for each_kernel in range(out_kchannels):
                            end_x, end_y, end_z = (start_x+Wk-1), (start_y+Hk-1), (start_z+in_kchannels-1)
                            image_temp = image[each_image, start_z:end_z+1, start_y:end_y+1, start_x:end_x+1]
                            kernel_temp = kernel[each_kernel, :, :, :]
                            if(image_temp.shape == kernel_temp.shape):
                                temp = numpy.multiply(image_temp, kernel_temp)
                                output[each_image][each_kernel+start_z][start_y][start_x] = numpy.sum(temp)
        return output
                            

    def backward(self, gradient):
        """
        Sets the gradients for operands of convolution operation.
        :param gradient: Upstream gradient.
        """
        #TODO
#         print('Conv2D backward')
        input_image = self.parents[0]
        kernel = self.parents[1]
        
        (batch_size, in_channels, H, W)      = input_image.value.shape
        (out_channels, in_channels, Hk, Wk)  = kernel.value.shape
        
        (no_images, channels, height, width) = gradient.shape
        
        for each_kernel in range(out_channels):
            for image in range(no_images):
                for h in range(height):
                    for w in range(width):
                        temp = gradient[image, each_kernel, h, w]
                        delete1 = input_image.grad[image, :, h:h+Hk, w:w+Wk]
                        delete2 = input_image.value[image, :, h:h+Hk, w:w+Wk]

                        if(delete1.shape == kernel.value[each_kernel].shape):
                            input_image.grad[image, :, h:h+Hk, w:w+Wk] += kernel.value[each_kernel] * temp
                        if(kernel.grad[each_kernel].shape == delete2.shape):
                            kernel.grad[each_kernel] += input_image.value[image, :, h:h+Hk, w:w+Wk] * temp



class Reshape(Function):
    parents = []
    def forward(self, shape, *args):
        """
        Forward pass of the reshape operation on a tensor
        :param shape: tuple of required dimension.
        :param args: Input tensor to be reshaped.
        :return: reshaped tensor.
        """
        self.parents = list(args)
        return numpy.reshape(self.parents[0].value, shape)
    
    def backward(self, gradient):
        """
        Sets the gradient for input of reshape operation.
        :param gradient: Upstream gradient.
        """
#         print('Reshape backward')
        input_image = self.parents[0]
        reshaped_gradient = numpy.reshape(gradient, input_image.value.shape)
  
        input_image.grad += reshaped_gradient
        



