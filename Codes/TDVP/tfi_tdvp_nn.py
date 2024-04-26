from tenpy.models.tf_ising import TFIChain

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import scienceplots
plt.style.use(['science'])

from tenpy.networks.mps import MPS
from tenpy.algorithms import tdvp
    
J = 1
g = 1
L = 33

model_params = {'J': J, 
                'g': g,
                'L': L,
                'bc_MPS': 'finite',
                'sort_charge': False}

H = TFIChain(model_params)

steps = np.arange(1, 100, 1)

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
        

np.savetxt('otoc_lightcone_tfi_custom_nn.txt', otoc)
plt.figure(figsize=(8, 6))
plt.imshow(otoc, aspect='auto', origin='lower', extent=[0, L, 0, steps[-1]])
plt.colorbar()