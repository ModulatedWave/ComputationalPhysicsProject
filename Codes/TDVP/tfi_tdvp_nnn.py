from tenpy.models.model import CouplingMPOModel, NearestNeighborModel
from tenpy.models.lattice import Chain
from tenpy.networks.site import SpinHalfSite

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import scienceplots
plt.style.use(['science'])

from tenpy.networks.mps import MPS
from tenpy.algorithms import tdvp

class nnnTFIModel(CouplingMPOModel):

    def init_sites(self, model_params):
        conserve = model_params.get('conserve', 'parity')
        assert conserve != 'Sz'
        if conserve == 'best':
            conserve = 'parity'
            self.logger.info("%s: set conserve to %s", self.name, conserve)
        sort_charge = model_params.get('sort_charge', True)
        site = SpinHalfSite(conserve=conserve, sort_charge=sort_charge)
        return site

    def init_terms(self, model_params):
        J = np.asarray(model_params.get('J', 1.))
        g = np.asarray(model_params.get('g', 1.))
        for u in range(len(self.lat.unit_cell)):
            self.add_onsite(-g, u, 'Sigmaz')
        for u1, u2, dx in self.lat.pairs['nearest_neighbors']:
            self.add_coupling(-J, u1, 'Sigmax', u2, 'Sigmax', dx)
        for u1, u2, dx in self.lat.pairs['next_nearest_neighbors']:
            self.add_coupling(-J, u1, 'Sigmax', u2, 'Sigmax', dx)
    
J = 1
g = 1
L = 10

model_params = {'J': J, 
                'g': g,
                'L': L,
                'bc_MPS': 'finite',
                'sort_charge': False,
                'lattice': Chain}

H = nnnTFIModel(model_params)

steps = np.arange(1, 5, 1)

otoc = np.zeros((len(steps), L)) # Store the OTOC for each site and each step

for n in steps:
    
    print('---')
    print(f'Step {n}')
    print('---')
    
    for i in tqdm(range(L)):
    
        # Prepare two copies of matrix product states
    
        psi = MPS.from_lat_product_state(H.lat, [['up']])
        phi = psi.copy()
    
        # e^iHt W_i(t) e^-iHt W_0(L//2) |psi>
    
        psi.apply_local_op(L//2, 'Sigmaz', unitary=True)
        
        tdvp_params = {
            'N_steps': n,
            'dt': -0.1,
            'preserve_norm': True,
            'trunc_params': {'chi_max': 100, 'svd_min': 1.e-12}
            }
        
        eng = tdvp.TwoSiteTDVPEngine(psi, H, tdvp_params)
        
        eng.run()
        
        psi.apply_local_op(i, 'Sigmaz', unitary=True)
        
        tdvp_params = {
            'N_steps': n,
            'dt': 0.1,
            'preserve_norm': True,
            'trunc_params': {'chi_max': 100, 'svd_min': 1.e-12}
            }
        
        eng = tdvp.TwoSiteTDVPEngine(psi, H, tdvp_params)
        
        eng.run()
        
        # W_0(L//2) e^iHt W_i(t) e^-iHt |phi>
        
        tdvp_params = {
            'N_steps': n,
            'dt': -0.1,
            'preserve_norm': True,
            'trunc_params': {'chi_max': 100, 'svd_min': 1.e-12}
            }
        
        eng = tdvp.TwoSiteTDVPEngine(phi, H, tdvp_params)
        
        eng.run()
        
        phi.apply_local_op(i, 'Sigmaz', unitary=True)
        
        tdvp_params = {
            'N_steps': n,
            'dt': 0.1,
            'preserve_norm': True,
            'trunc_params': {'chi_max': 100, 'svd_min': 1.e-12}
            }
        
        eng = tdvp.TwoSiteTDVPEngine(phi, H, tdvp_params)
        
        eng.run()
        
        phi.apply_local_op(L//2, 'Sigmaz', unitary=True)
        
        # Append the overlap to the OTOC array
        
        otoc[n-1, i] = psi.overlap(phi).real
        

np.savetxt('otoc_lightcone_nnn_tfi.txt', otoc)
plt.figure(figsize=(8, 6))
plt.imshow(otoc, aspect='auto', origin='lower', extent=[0, L, 0, steps[-1]])
plt.colorbar()
plt.savefig('figures/nnn_tfi_tdvp.pdf')