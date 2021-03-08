import numpy as np

def deconv_math(input, output, stride, d, pad, out_pad, kernel_size):
    if stride == None:
        stride = [1, 1, 1]

    if pad == None:
        pad = [0, 0, 0]

    if out_pad == None:
        out_pad = [0, 0, 0]

    if d == None:
        d = [1, 1, 1]

    D_in = input[0]
    H_in = input[1]
    W_in = input[2]

    D_out = output[0]
    H_out = output[1]
    W_out = output[2]

    if kernel_size == None:
        ker_sz0 = 1 + ((D_out + 2 * pad[0] - out_pad[0] - 1 - (D_in - 1) * stride[0]) / d[0])
        ker_sz1 = 1 + ((H_out + 2 * pad[1] - out_pad[1] - 1 - (H_in - 1) * stride[1]) / d[1])
        ker_sz2 = 1 + ((W_out + 2 * pad[2] - out_pad[2] - 1 - (W_in - 1) * stride[2]) / d[2])

        ker_sz = [ker_sz0, ker_sz1, ker_sz2]
        return ker_sz
    else:
        # for i in ker_sz:
        #     # print(i)
        #     if i.is_integer() is False:
        #         break
        #     else:
        #         continue
        s0 = (D_out + 2 * pad[0] - out_pad[0] - 1 - (d[0] * (kernel_size[0] - 1))) / (D_in - 1)
        s1 = (H_out + 2 * pad[1] - out_pad[1] - 1 - (d[1] * (kernel_size[1] - 1))) / (H_in - 1)
        s2 = (W_out + 2 * pad[2] - out_pad[2] - 1 - (d[2] * (kernel_size[2] - 1))) / (W_in - 1)

        str = [s0, s1, s2]
        # print(str)
        return str

print(deconv_math(input=[23, 31, 31], output=[24, 32, 32], stride=[1,1,1], pad=None,
                  d=None, out_pad=None, kernel_size=None))