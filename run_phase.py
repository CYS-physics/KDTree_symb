from KDTree_symbreak import AGran
import numpy as np
import os
import sys
import time
from tqdm import trange
from scipy.spatial import cKDTree
from scipy import sparse


start = time.time()


AR = float(sys.argv[1])
seed = int(sys.argv[2])
rho = float(sys.argv[3])
name = str(sys.argv[4])

S1 = AGran(Lx=200,Ly=200,r0=0.8,AR = AR,T = 5, rho = rho,r_tr=5,factor = 0.01, mode='free',v0 = 1,eta = 6,mu =0.05, mur =0.15 , k=200,trN=10,v_drag = 0, tracer=False)
S1.N = 10000#20000
S1.Lx = np.sqrt(S1.N/rho)
S1.Ly = np.sqrt(S1.N/rho)
S1.initialize()


np.random.seed(seed)
direc = 'data/phase/'+name+'/'+str(rho)
os.makedirs(direc,exist_ok=True)
state = os.getcwd()+'/'+direc+'/'+str(AR)+'_'+str(seed)+'.npz'

S1.update()

N_init = 100000
N_measure = 200#10000
N_skip = 150
X_tr = 0
Y_tr = 0

# dx = 5/np.sqrt(rho)
dx = 2/np.sqrt(rho)
Nmark = int(np.ceil(S1.Lx/S1.r0))
marker = np.zeros((Nmark**2,2))
for i in range(Nmark-1):
    for j in range(Nmark-1):
        marker[Nmark*i+j,:] = np.array([(S1.r0)*i,(S1.r0)*j])

tree1 = cKDTree(marker,boxsize=[S1.Lx,S1.Ly])
Nmax = np.ceil((dx/S1.r0)**2)

cbins = np.arange(Nmax)
sbins = np.linspace(0,Nmax,50)
ebins = np.linspace(0,1,50)


dxp_traj = np.zeros((N_measure,100))
dyp_traj = np.zeros((N_measure,100))
dop_traj = np.zeros((N_measure,100))

p_traj = np.zeros(N_measure)
n_traj = np.zeros(N_measure)

for _ in range(N_init):
    S1.update()
N_hist = 10
# r_list = np.linspace(0,dx,N_hist+1)
# r_list = (r_list[1:]+r_list[:-1])/2
dens_hist = np.zeros((len(cbins)-1))
pol_hist = np.zeros((len(sbins)-1))
nem_hist = np.zeros((len(sbins)-1))
poln_hist = np.zeros((len(ebins)-1))
nemn_hist = np.zeros((len(ebins)-1))
# corr_dens = np.zeros(Nmark)
# corr_pol = np.zeros(Nmark)
# corr_nem = np.zeros(Nmark)
# corr_poln = np.zeros(Nmark)
# corr_nemn = np.zeros(Nmark)

for i in range(N_measure):
    dxp = 0
    dyp = 0
    dop = 0


    for _ in range(N_skip):
        S1.update()
        # dxp +=S1.dxp[:100]
        # dyp +=S1.dyp[:100]
        # dop +=S1.dop[:100]
    
    # for k in range(N_hist):
    tree2 = cKDTree(S1.pos,boxsize=[S1.Lx,S1.Ly])
    dist = tree1.sparse_distance_matrix(tree2, max_distance=dx,output_type='coo_matrix')
    _, indices1 = tree2.query(S1.pos,k=7,distance_upper_bound=S1.r0*3)
    _, indices2 = tree2.query(marker,k=1)


    count = np.ones(S1.N)[dist.col]

    count_mat = sparse.coo_matrix((count,(dist.row,dist.col)), shape=dist.get_shape())
    
    density = np.squeeze(np.asarray(count_mat.sum(axis=1)))
  

    density = density.reshape(Nmark,Nmark)
    
    dens_hist_temp,_=np.histogram(density,bins=cbins)
    dens_hist+=dens_hist_temp

    neighbor_orient = S1.orient[indices1]
    pmag1 = np.sqrt((np.mean(np.cos(neighbor_orient),axis=1)**2+np.mean(np.sin(neighbor_orient),axis=1)**2))
    nmag1 = np.sqrt((np.mean(np.cos(2*neighbor_orient),axis=1)**2+np.mean(np.sin(2*neighbor_orient),axis=1)**2))

    pol_hist_temp,_ = np.histogram(pmag1,bins=ebins)
    nem_hist_temp,_ = np.histogram(nmag1,bins=ebins)
    pol_hist += pol_hist_temp
    nem_hist += nem_hist_temp

    poln_hist_temp,_ = np.histogram(pmag1[indices2],bins=ebins)
    nemn_hist_temp,_ = np.histogram(nmag1[indices2],bins=ebins)
    poln_hist += poln_hist_temp
    nemn_hist += nemn_hist_temp


    dxp_traj[i] = dxp
    dyp_traj[i] = dyp
    dop_traj[i] = dop
    p_traj[i] = np.sum(np.cos(S1.orient))**2+np.sum(np.sin(S1.orient))**2
    n_traj[i] = np.sum(np.cos(2*S1.orient))**2+np.sum(np.sin(2*S1.orient))**2
    

# corr_dens /= N_measure
# corr_pol /= N_measure
# corr_nem /= N_measure
# corr_poln /= N_measure
# corr_nemn /= N_measure

# rho /= N_measure
# px /= N_measure
# py /= N_measure
# v_loc /= N_measure
# F_loc /= N_measure
# D_loc /= N_measure


end = time.time()

save_dict={}
# save_dict['marker'] = S1.marker
save_dict['dt'] = S1.dt
save_dict['N_measure'] = N_measure
save_dict['N_skip'] = N_skip
save_dict['dx'] = dx

save_dict['AR'] = S1.AR
save_dict['N'] = S1.N

save_dict['eta'] = S1.eta
# save_dict['density'] = density
save_dict['pos'] = S1.pos
save_dict['orient'] = S1.orient
save_dict['seed'] = seed

# save_dict['rho'] = rho
# save_dict['px'] = px
# save_dict['py'] = py



save_dict['dxp_traj'] = dxp_traj
save_dict['dyp_traj'] = dyp_traj
save_dict['dop_traj'] = dop_traj
save_dict['p_traj'] = p_traj
save_dict['n_traj'] = n_traj
 
save_dict['dens_hist'] = dens_hist
save_dict['pol_hist'] = pol_hist
save_dict['nem_hist'] = nem_hist
save_dict['poln_hist'] = poln_hist
save_dict['nemn_hist'] = nemn_hist

save_dict['cbins'] = cbins
save_dict['sbins'] = sbins
save_dict['ebins'] = ebins

# save_dict['corr_dens'] = corr_dens
# save_dict['corr_pol'] = corr_pol
# save_dict['corr_nem'] = corr_nem
# save_dict['corr_poln'] = corr_poln 
# save_dict['corr_nemn'] = corr_nemn

save_dict['time'] =str(f"{end - start:.5f} sec")


np.savez(state, **save_dict)