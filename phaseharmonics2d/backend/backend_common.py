import torch


# substract spatial mean (complex valued input)
class SubInitSpatialMeanC(object):
    def __init__(self):
        self.minput = None

    def __call__(self, input):
        if self.minput is None:
            minput = input.clone().detach()
            minput = torch.mean(minput, dim=(-2, -1), keepdim=True)
            if minput.dim() == 3:
                minput = minput.mean(0)
            self.minput = minput
            print('sum of minput', self.minput.sum())

        output = input - self.minput
        return output


class DivInitStd(object):
    def __init__(self, stdcut=0):
        self.stdinput = None
        self.eps = stdcut
        print('DivInitStd:stdcut', stdcut)

    def __call__(self, input):
        if self.stdinput is None:
            stdinput = input.clone().detach()  # input size:(M,N)
            stdinput = (torch.abs(stdinput) ** 2).mean(dim=(-2, -1), keepdim=True) ** 0.5
            if stdinput.dim() == 3:
                stdinput = stdinput.mean(0)
            self.stdinput = stdinput + self.eps
            print('stdinput max,min:', self.stdinput.max(), self.stdinput.min())

        output = input / self.stdinput
        return output


# substract spatial mean (complex valued input), average over ell
class SubInitSpatialMeanCL(object):
    def __init__(self):
        self.minput = None

    def __call__(self, input):  # input: (J,L2,K,M,N)
        if self.minput is None:
            minput = input.clone().detach()
            minput = torch.mean(minput, dim=(-4, -2, -1), keepdim=True)
            if minput.dim() == 6:
                minput = minput.mean(0)
            self.minput = minput
            print('minput size', self.minput.shape)
            print('sum of minput', self.minput.sum())

        output = input - self.minput
        return output


# divide by std, average over ell
class DivInitStdL(object):
    def __init__(self):
        self.stdinput = None

    def __call__(self, input):  # input: (J,L2,K,M,N)
        if self.stdinput is None:
            stdinput = input.clone().detach()
            stdinput = torch.abs(stdinput) ** 2  # (J,L2,K,M,N)
            stdinput = torch.mean(stdinput, dim=(-4, -2, -1), keepdim=True)
            if stdinput.dim() == 6:
                stdinput = stdinput.mean(0)  # batch average
            self.stdinput = torch.sqrt(stdinput)
            print('stdinput size', self.stdinput.shape)
            print('stdinput max,min:', self.stdinput.max(), self.stdinput.min())

        output = input / self.stdinput
        return output


def phaseHarmonicsIsoFn(z, k):
    r = torch.abs(z)  # (J, L2, M, N)
    theta = torch.angle(z)  # (J, L2, M, N) # Can replace with angle (assuming z is torch.complex64)
    ktheta = k.unsqueeze(-1).unsqueeze(-1) * theta.unsqueeze(-3)  # (J, L2, K, M, N)
    eiktheta = torch.cos(ktheta) + 1.j * torch.sin(ktheta)
    # eiktheta.size(): (J, L2, K, M, N)
    return r.unsqueeze(-3) * eiktheta
