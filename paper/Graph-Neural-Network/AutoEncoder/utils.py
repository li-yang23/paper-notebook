""" This file contains different utility functions that are not connected
in anyway to the networks presented in the tutorials, but rather help in
processing the outputs into a more understandable way.

For example ``tile_raster_images`` helps in generating a easy to grasp
image from a set of samples or weights.
"""
import numpy

def scale_tp_unit_interval(ndar,eps=1e-8):
    """Scales all values in the ndarray ndar to be between 0 and 1"""
    ndat = ndar.copy()
    ndar -= ndar.min()
    ndat *= 1.0 / (ndar.max() + eps)
    return ndar

def tile_raster_images(X,img_shape,tile_shape,tile_spacing=(0,0),
                        scale_rows_to_unit_interval=True,
                        output_pixel_vals=True):
    """Transform array with one flattened image per row, 
    into an array in which images are reshaped and 
    layed out like tile on a floor.

    This function is useful for visualizing datasets whose rows are images, 
    and also columns of matrices for transforming those rows
    
    Arguments:
        X {numpy.array or tuple[numpy.arrays*4]} -- 二维数组，每行是一个展平的图像
        img_shape {tuple} -- 图像的原始形状
        tile_shape {[type]} -- 要平铺的图像数
    
    Keyword Arguments:
        tile_spacing {tuple} --  (default: {(0,0)})
        scale_rows_to_unit_interval {bool} -- 绘制到0-1之间需要做缩放 (default: {True})
        output_pixel_vals {bool} -- 判断输出是不是应该是像素值(int8)(True)或者float (default: {True})
    """
    assert len(img_shape) == 2
    assert len(tile_shape) == 2
    assert len(tile_spacing) == 2
    
    out_shape = [
        (ishp+tsp)*tshp-tsp
        for ishp,tshp,tsp in zip(img_shape,tile_shape,tile_spacing)
    ]
    if isinstance(X,tuple):
        assert len(X) == 4
        # 创建一个存储图像的nupy.array
        if output_pixel_vals:
            out_array = numpy.zeros((out_shape[0],out_shape[1],4),dtype='uint8')
        else:
            out_array = numpy.zeros((out_shape[0],out_shape[1],4),dtype=X.dtype)
        if output_pixel_vals:
            channel_defaults = [0,0,0,255]
        else:
            channel_defaults = [0.,0.,0.,1.]
        for i in range(4):
            if X[i] is None:
                # 如果通道是None，用0填充，注意0的类型
                dt = out_array.dtype
                if output_pixel_vals:
                    dt = 'uint8'
                out_array[:,:,i] = numpy.zeros(out_shape,dtype=dt) + channel_defaults[i]
            else:
                out_array[:,:,i] = tile_raster_images(X[i], img_shape, tile_shape,tile_shape,
                                                        scale_rows_to_unit_interval,output_pixel_vals)
        return out_array
    else:
        # if we are dealing with only one channel
        # 如果只有一个通道
        H,W = img_shape
        Hs,Ws = tile_spacing

        dt = X.dtype
        if output_pixel_vals:
            dt = 'unit8'
        out_array = numpy.zeros(out_shape,dtype=dt)

        for tile_row in range(tile_shape[0]):
            for tile_col in range(tile_shape[1]):
                if tile_row * tile_shape[1] + tile_col < X.shape[0]:
                    this_x = X[tile_row * tile_shape[1] + tile_col]
                    if scale_rows_to_unit_interval:
                        this_img = scale_tp_unit_interval(this_x.reshape(img_shape))
                    else:
                        this_img = this_x.reshape(img_shape)
                    c = 1
                    if output_pixel_vals:
                        c = 255
                    out_array[
                        tile_row*(H+Hs):tile_row*(H+Hs)+H,
                        tile_col*(W+Ws):tile_col*(W+Ws)+W
                    ] = this_img * c
        return out_array