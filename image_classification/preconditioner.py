import torch
import math
import time
import pytorch_minimax


def householder(src, tar):
    N = src.shape[0]
    v = tar - src
    v = v / v.norm()
    return torch.eye(N) - 2 * v.view(N, 1) @ v.view(1, N)


Qs = [[], [torch.ones(1), 1.0]]
Qqs = [torch.tensor(1.0), torch.ones(1)]
Qmax = [1.0, 1.0]


def init(max_bs):
    for i in range(2, max_bs+1):
        e1 = torch.zeros(i)
        e1[0] = 1
        ones = torch.ones(i) / math.sqrt(i)
        H = householder(e1, ones)
        Hmax = H.abs().max()
        Qs.append([H, Hmax])
        Qqs.append(H)
        Qmax.append(Hmax)


class Preconditioner:
    def __init__(self, x, num_bits, left=True):
        self.left = left
        self.x_shape = x.shape
        self.num_bins = 2 ** num_bits - 1

        self.x = self.flatten(x)
        self.Tx = self.transform(self.x)

    def flatten(self, x):
        self.x_shape2 = x.shape
        self.x_shape_double = torch.cat([x, x], dim=0).shape
        return x.view(x.shape[0], -1)

    def deflatten(self, Tx):
        try:
            x = Tx.view(*self.x_shape2)
        except:
            x = Tx.view(*self.x_shape_double)
        return x

    def forward(self):
        return self.Tx

    def inverse(self, Tx):
        x = self.inverse_transform(Tx)
        return self.deflatten(x)


class ScalarPreconditioner(Preconditioner):
    # y = (x - z) * scale
    # x = y / scale + z
    def __init__(self, x, num_bits, left=True):
        super(ScalarPreconditioner, self).__init__(x, num_bits, left)

    def transform(self, x):
        with torch.no_grad():
            mn = min(x.min() - 1e-8, 0)
            mx = max(x.max() + 1e-8, 0)

        self.zero_point = mn
        self.scale = self.num_bins / (mx - mn)

        qzero = -self.zero_point * self.scale
        iqzero = torch.floor(qzero)
        mx = (iqzero - self.num_bins) * mn / iqzero
        self.scale = self.num_bins / (mx - mn)

        return (x - self.zero_point) * self.scale

    def inverse_transform(self, x):
        return x / self.scale + self.zero_point


class ScalarPreconditionerAct(Preconditioner):
    # y = (x - z) * scale
    # x = y / scale + z
    def __init__(self, x, num_bits, left=True):
        super(ScalarPreconditionerAct, self).__init__(x, num_bits, left)

    def transform(self, x):
        with torch.no_grad():
            mn = x.min() - 1e-8
            mx = x.max() + 1e-8

        self.zero_point = mn
        self.scale = self.num_bins / (mx - mn)

        return (x - self.zero_point) * self.scale

    def inverse_transform(self, x):
        return x / self.scale + self.zero_point

class TwoLayerPreconditioner(Preconditioner):
    # Y = D (Y - z 1^\top)
    # X = D^-1 Y + z 1^\top
    def __init__(self, x, num_bits, left=True):
        super(TwoLayerPreconditioner, self).__init__(x, num_bits, left)

    def transform(self, x, debug=False):
        with torch.no_grad():
            mn = min(x.min() - 1e-8, 0)
            mx = max(x.max() + 1e-8, 0)

        if debug:
            print("mn is {}, mx is {}".format(mn, mx))

        self.zero_point1 = mn
        self.scale1 = self.num_bins / (mx - mn)

        qzero = -self.zero_point1 * self.scale1
        iqzero = torch.floor(qzero)

        if debug:
            print(qzero, iqzero, "qzero, iqzero")

        if iqzero <= 0:
            with open("debug.txt", "a") as f:
                f.write("part 1 break, x is {}, iqzero is {} \n".format(x, iqzero))

        mx = (iqzero - self.num_bins) * mn / iqzero
        self.scale1 = self.num_bins / (mx - mn)

        if debug:
            print(mx, self.scale1, "mx, scale1")

        if torch.isnan(self.scale1):
            print(mx, mn, iqzero)
            with torch.no_grad():
                mn = min(x.min() - 1e-8, 0)
                mx = max(x.max() + 1e-8, 0)
            print("mn is {}, mx is {}".format(mn, mx))
            self.zero_point1 = mn
            self.scale1 = self.num_bins / (mx - mn)
            print("scale1 is {}".format(self.scale1))

            qzero = -self.zero_point1 * self.scale1
            iqzero = torch.floor(qzero)

            if iqzero <= 0:
                with open("debug.txt", "a") as f:
                    f.write("part 1 break, x is {}, iqzero is {} \n".format(x, iqzero))

            mx = (iqzero - self.num_bins) * mn / iqzero
            self.scale1 = self.num_bins / (mx - mn)


        first_transform = (x - self.zero_point1) * self.scale1
        # noise = first_transform.new(first_transform.shape).uniform_(-0.5, 0.5)
        # first_transform.add_(noise)
        first_transform.clamp_(0.0, self.num_bins).round_()
        first_quantize = first_transform / self.scale1 + self.zero_point1

        residual = x - first_quantize

        with torch.no_grad():
            mn = min(residual.min() - 1e-8, 0)
            mx = max(residual.max() + 1e-8, 0)

        self.zero_point2 = mn
        self.scale2 = self.num_bins / (mx - mn)

        qzero = -self.zero_point2 * self.scale2
        iqzero = torch.floor(qzero)

        if iqzero <= 0:
            with open("debug.txt", "a") as f:
                f.write("part 2 break, x is {}, iqzero is {} \n".format(x, iqzero))

        mx = (iqzero - self.num_bins) * mn / iqzero
        self.scale2 = self.num_bins / (mx - mn)

        if torch.isnan(self.scale2):
            # torch.save(x, 'image_classification/ckpt/precon.pt')
            with torch.no_grad():
                mn = min(residual.min() - 1e-8, 0)
                mx = max(residual.max() + 1e-8, 0)

            print(mn, mx, "mn, mx")
            self.zero_point2 = mn
            self.scale2 = self.num_bins / (mx - mn)

            qzero = -self.zero_point2 * self.scale2
            iqzero = torch.floor(qzero)

            print(qzero, iqzero, "qzero, iqzero")
            if iqzero <= 0:
                with open("debug.txt", "a") as f:
                    f.write("part 2 break, x is {}, iqzero is {} \n".format(x, iqzero))

            mx = (iqzero - self.num_bins) * mn / iqzero
            self.scale2 = self.num_bins / (mx - mn)

        second_transform = (residual - self.zero_point2) * self.scale2
        output = torch.cat([first_transform, second_transform], dim=0)
        # print("the integer is {}".format(output))
        # print("quantize shape is {}".format(output.shape))

        return output

    def inverse_transform(self, x):
        half_shape = int(x.shape[0] / 2)
        first, second = torch.split(x, [half_shape, half_shape], dim=0)
        # print("first shape is {}, second shape is {}".format(first.shape, second.shape))
        first = first / self.scale1 + self.zero_point1
        second = second / self.scale2 + self.zero_point2
        dequantize = torch.cat([first, second], dim=0)
        # print("scale1 is {}, scale2 is {}, zero 1 {}, zero 2 {}".format(self.scale1, self.scale2, self.zero_point1, self.zero_point2))
        # print("dequantize shape:{}".format(dequantize.shape))
        # print("input is {}, dequantize is {}, difference is ".format(x, dequantize, x-dequantize))
        if torch.isnan(dequantize[0, 0]):
            print("scale1 is {}, scale2 is {}, zero 1 {}, zero 2 {}".format(self.scale1, self.scale2, self.zero_point1, self.zero_point2))

        return dequantize

