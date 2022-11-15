from complex_generator import ComplexGenerator
from holo_encoder import HoloEncoder
from utils import *
from propagation_ASM import *

class selfholo(nn.Module):

    def __init__(self):
        super().__init__()
        self.network1 = ComplexGenerator()
        self.network2 = HoloEncoder()

        self.wavelength = 532e-9
        self.feature_size = [8e-6, 8e-6]
        self.z = 0.3
        self.precomputed_H = None
        self.pren = None
        self.prem = None
        self.pref = None
        self.return_H = None

    def forward(self, source, ikk):

        if self.precomputed_H == None:
            self.precomputed_H = propagation_ASM(torch.empty(1, 1, 1072, 1072), feature_size=[8e-6, 8e-6],
                                                 wavelength=self.wavelength, z=0.30, return_H=True)
            self.precomputed_H = self.precomputed_H.to('cuda').detach()
            self.precomputed_H.requires_grad = False

        if self.pren == None:
            self.pren = propagation_ASM(torch.empty(1, 1, 1072, 1072), feature_size=[8e-6, 8e-6],
                                        wavelength=self.wavelength, z=-0.30, return_H=True)
            self.pren = self.pren.to('cuda').detach()
            self.pren.requires_grad = False

        if self.prem == None:
            self.prem = propagation_ASM(torch.empty(1, 1, 1072, 1072), feature_size=[8e-6, 8e-6],
                                        wavelength=self.wavelength, z=-0.31, return_H=True)
            self.prem = self.prem.to('cuda').detach()
            self.prem.requires_grad = False

        if self.pref == None:
            self.pref = propagation_ASM(torch.empty(1, 1, 1072, 1072), feature_size=[8e-6, 8e-6],
                                        wavelength=self.wavelength, z=-0.32, return_H=True)
            self.pref = self.pref.to('cuda').detach()
            self.pref.requires_grad = False

        target_amp, target_phase = self.network1(source)
        obj_r, obj_i = polar_to_rect(target_amp, target_phase)
        target_field = torch.complex(obj_r, obj_i)

        slm_field = propagation_ASM(target_field, self.feature_size, self.wavelength, self.z, precomped_H=self.precomputed_H)
        slm_amp, slm_phase = rect_to_polar(slm_field.real, slm_field.imag)
        slm_field = torch.cat([slm_amp, slm_phase], dim=-3)

        holo = self.network2(slm_field)
        H_real, H_imag = polar_to_rect(torch.ones(holo.shape).cuda(), holo)
        holo_field = torch.complex(H_real, H_imag)

        distance = [-0.30, -0.31, -0.32]
        dis = distance[ikk]
        pre_kernel = [self.pren, self.prem, self.pref]
        pre_kernel = pre_kernel[ikk]
        recon_field = propagation_ASM(holo_field, self.feature_size, self.wavelength, dis, precomped_H=pre_kernel)

        return holo, slm_amp, recon_field