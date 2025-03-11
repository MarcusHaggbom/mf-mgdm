# isotropic case, implement basic phase harmonics
# assume both isotropic and stationary
# but no line reflection / plain symmetry

__all__ = ['PhaseHarmonics2d']

import torch
import numpy as np
import scipy.io as sio
from .backend import SubInitSpatialMeanC, SubInitSpatialMeanCL, DivInitStd, DivInitStdL, phaseHarmonicsIsoFn
from pathlib import Path
import os


class PhaseHarmonics2d(object):
    def __init__(self, M, N, J, L, delta_j, delta_l, delta_k,
                 nb_chunks, chunk_id, stdnorm=0, kmax=None):
        self.M, self.N, self.J, self.L = M, N, J, L  # size of image, max scale, number of angles [0,pi]
        self.dj = delta_j  # max scale interactions
        self.dl = delta_l  # max angular interactions
        self.dk = delta_k  #
        if kmax is None:
            self.K = 2 ** self.dj + self.dk + 1
        else:
            assert (kmax >= 0)
            self.K = min(kmax + 1, 2 ** self.dj + self.dk + 1)
        self.k = torch.arange(0, self.K).to(torch.float32)  # vector between [0,..,K-1]
        self.nb_chunks = nb_chunks  # number of chunks to cut whp cov
        self.chunk_id = chunk_id
        assert (self.chunk_id < self.nb_chunks)  # chunk_id = 0..nb_chunks-1, are the wph cov
        if self.dl != self.L:
            raise (ValueError('delta_l must be = L'))
        self.stdnorm = stdnorm
        self.pre_pad = False  # no padding
        self.cache = False  # cache filter bank
        self.nbcov = 0
        self.build()

    def build(self):
        check_for_nan = False  # True
        # self.phase_harmonics = PhaseHarmonicsIso.apply
        self.phase_harmonics = phaseHarmonicsIsoFn
        self.M_padded, self.N_padded = self.M, self.N
        self.filters_tensor()
        if self.chunk_id < self.nb_chunks:
            self.idx_wph = self.compute_idx()
            self.this_wph = self.get_this_chunk(self.nb_chunks, self.chunk_id)
            self.subinitmean = SubInitSpatialMeanCL()
            self.subinitmeanJ = SubInitSpatialMeanC()
            if self.stdnorm == 1:
                self.divinitstd = DivInitStdL()
                self.divinitstdJ = DivInitStd()
        else:
            assert 0

    def filters_tensor(self):
        # TODO load bump steerable wavelets
        assert (self.M == self.N)
        print('path parent', Path(__file__).parent)
        matfilters = sio.loadmat(os.path.join(Path(__file__).parent, 'matlab', 'filters', 'bumpsteerableg1_fft2d_N' + str(self.N) + '_J' + str(self.J) + '_L' + str(self.L) + '.mat'))
        self.hatpsi = torch.from_numpy(matfilters['filt_fftpsi'].astype(np.complex64))  # (J,L2,M,N)
        self.hatphi = torch.from_numpy(matfilters['filt_fftphi'].astype(np.complex64))  # (M,N)

    def get_this_chunk(self, nb_chunks, chunk_id):
        # cut self.idx_wph into smaller pieces
        nb_cov = len(self.idx_wph['la1'])
        max_chunk = nb_cov // nb_chunks
        nb_cov_chunk = np.zeros(nb_chunks, dtype=np.int32)
        for idxc in range(nb_chunks):
            if idxc < nb_chunks - 1:
                nb_cov_chunk[idxc] = int(max_chunk)
            else:
                nb_cov_chunk[idxc] = int(nb_cov - max_chunk * (nb_chunks - 1))
                assert (nb_cov_chunk[idxc] > 0)

        this_wph = dict()
        offset = int(0)
        for idxc in range(nb_chunks):
            if idxc == chunk_id:
                this_wph['la1'] = self.idx_wph['la1'][offset:offset + nb_cov_chunk[idxc]]
                this_wph['la2'] = self.idx_wph['la2'][offset:offset + nb_cov_chunk[idxc]]
            offset = offset + nb_cov_chunk[idxc]

        print('this chunk', chunk_id, ' size is ', len(this_wph['la1']), ' among ', nb_cov)

        return this_wph

    def compute_idx(self):
        L = self.L
        L2 = L * 2
        Q = L2
        J = self.J
        dj = self.dj
        dl = self.dl
        dk = self.dk
        K = self.K
        assert (K >= 2)

        idx_la1 = []
        idx_la2 = []

        # j1=j2, k1=1, k2=0 or 1
        for j1 in range(J):
            for q1 in range(L2):  # from q=0..2L-1
                k1 = 1
                j2 = j1
                q2 = q1
                k2 = 0
                idx_la1.append(K * Q * j1 + K * q1 + k1)
                idx_la2.append(K * Q * j2 + K * q2 + k2)
                # print('add j1='+str(j1)+'q1='+str(q1)+'k1='+\
                #      str(k1)+'j2='+str(j2)+'q2='+str(q2)+'k2='+str(k2))
                self.nbcov += 2
                k2 = 1
                idx_la1.append(K * Q * j1 + K * q1 + k1)
                idx_la2.append(K * Q * j2 + K * q2 + k2)
                # print('add j1='+str(j1)+'q1='+str(q1)+'k1='+\
                #      str(k1)+'j2='+str(j2)+'q2='+str(q2)+'k2='+str(k2))
                self.nbcov += 1

        # k1 = 0
        # k2 = 0
        # j1 = j2
        for j1 in range(J):
            for q1 in range(L2):
                k1 = 0
                j2 = j1
                q2 = q1
                k2 = 0
                idx_la1.append(K * Q * j1 + K * q1 + k1)
                idx_la2.append(K * Q * j2 + K * q2 + k2)
                # print('add j1='+str(j1)+'q1='+str(q1)+'k1='+\
                #      str(k1)+'j2='+str(j2)+'q2='+str(q2)+'k2='+str(k2))
                self.nbcov += 1

        # k1 = 0
        # k2 = 0,1,2
        # j1+1 <= j2 <= min(j1+dj,J-1)
        for j1 in range(J):
            for q1 in range(L2):
                k1 = 0
                for j2 in range(j1 + 1, min(j1 + dj + 1, J)):
                    q2 = q1
                    for k2 in range(min(K, 3)):
                        idx_la1.append(K * Q * j1 + K * q1 + k1)
                        idx_la2.append(K * Q * j2 + K * q2 + k2)
                        # print('add j1='+str(j1)+'q1='+str(q1)+'k1='+\
                        #      str(k1)+'j2='+str(j2)+'q2='+str(q2)+'k2='+str(k2))
                        if k2 == 0:
                            self.nbcov += 1
                        else:
                            self.nbcov += 2

        # k1 = 1
        # k2 = 2^(j2-j1)±dk
        # j1+1 <= j2 <= min(j1+dj,J-1)
        for j1 in range(J):
            for q1 in range(L2):
                k1 = 1
                q2 = q1
                for j2 in range(j1 + 1, min(j1 + dj + 1, J)):
                    for k2 in range(max(0, 2 ** (j2 - j1) - dk), min(K, 2 ** (j2 - j1) + dk + 1)):
                        idx_la1.append(K * Q * j1 + K * q1 + k1)
                        idx_la2.append(K * Q * j2 + K * q2 + k2)
                        # print('add j1='+str(j1)+'q1='+str(q1)+'k1='+\
                        #      str(k1)+'j2='+str(j2)+'q2='+str(q2)+'k2='+str(k2))
                        self.nbcov += 2

        self.nbcov += 1

        idx_wph = dict()
        idx_wph['la1'] = torch.tensor(idx_la1)
        idx_wph['la2'] = torch.tensor(idx_la2)

        return idx_wph

    def cuda(self, devid=0):
        return self.to(f'cuda:{devid}')

    def cpu(self):
        return self.to('cpu')

    def to(self, device):
        if self.chunk_id < self.nb_chunks:
            self.this_wph['la1'] = self.this_wph['la1'].to(device)
            self.this_wph['la2'] = self.this_wph['la2'].to(device)
        self.hatpsi = self.hatpsi.to(device)
        self.hatphi = self.hatphi.to(device)
        self.k = self.k.to(device)
        return self

    def forward(self, input):
        J = self.J
        M = self.M
        N = self.N
        L2 = self.L * 2
        Q = L2

        dj = self.dj
        dl = self.dl
        dk = self.dk
        K = self.K
        k = self.k

        # input: (..., M, N)
        x_c = input  # add zeros to imag part -> (M,N)
        hatx_c = torch.fft.fft2(x_c)  # fft2 -> (M,N)
        if self.chunk_id < self.nb_chunks:
            hatpsi_la = self.hatpsi  # (J,L2,M,N)
            hatx_bc = hatx_c.unsqueeze(-3).unsqueeze(-3)  # (1,1,M,N)
            hatxpsi_bc = hatpsi_la * hatx_bc  # (J,L2,M,N)
            xpsi_bc = torch.fft.ifft2(hatxpsi_bc)
            xpsi_ph_bc = self.phase_harmonics(xpsi_bc, k)  # (J,L2,K,M,N)
            xpsi_ph_bc0 = self.subinitmean(xpsi_ph_bc)
            if self.stdnorm == 1:
                xpsi_ph_bc0 = self.divinitstd(xpsi_ph_bc0)
            # fft in angles
            xpsi_iso_bc = torch.fft.fft(xpsi_ph_bc0, norm='ortho', dim=-4)  # (J,Q,K,M,N)
            # reshape to (1,J*L*K,M,N)
            xpsi_iso_bc0_ = torch.flatten(xpsi_iso_bc, -5, -3)
            # select la1, et la2, P_c = number of |la1| in this chunk
            xpsi_bc_la1 = torch.index_select(xpsi_iso_bc0_, -3, self.this_wph['la1'])  # (P_c,M,N)
            xpsi_bc_la2 = torch.index_select(xpsi_iso_bc0_, -3, self.this_wph['la2'])  # (P_c,M,N)

            # compute mean spatial
            corr_xpsi_bc = xpsi_bc_la1 * torch.conj(xpsi_bc_la2)  # (P_c,M,N)
            corr_bc = torch.mean(corr_xpsi_bc, dim=(-2, -1))  # (P_c)
            Sout = torch.cat([corr_bc.real, corr_bc.imag], -1)

            if self.chunk_id == self.nb_chunks - 1:
                # ADD 1 channel for spatial phiJ
                # add l2 phiJ to last channel
                hatxphi_c = hatx_c * self.hatphi  # (M,N)
                xphi_c = torch.fft.ifft2(hatxphi_c)
                # submean from spatial M N
                xphi0_c = self.subinitmeanJ(xphi_c)
                if self.stdnorm == 1:
                    xphi0_c = self.divinitstdJ(xphi0_c)
                xphi0_mod = torch.abs(xphi0_c)  # (M,N)
                xphi0_mod2 = xphi0_mod ** 2  # (M,N)
                xphi0_mean = torch.mean(xphi0_mod2, dim=(-2, -1), keepdim=True).squeeze(-1)
                Sout = torch.cat((Sout, xphi0_mean), dim=-1)

        return Sout

    def __call__(self, input):
        return self.forward(input)
