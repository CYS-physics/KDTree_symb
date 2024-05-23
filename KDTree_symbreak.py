import numpy as np
import scipy as sp
from scipy import sparse
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
import matplotlib.patches as patches
from tqdm import trange

class AGran:
    def __init__(self,Lx = 100.0,Ly=100.0,AR = 1.5,r0 = 1.,rho=0.25, r_tr = 8, T = 10,factor = 0.1, v0 = 1.2, eta = 1500,mu = 0.0007, mur = 0.0001,mu_tr = 0.0001, k = 200,trN=1,v_drag=1, mode='drag',tracer=True):
        self.Lx = Lx   # system size
        self.Ly = Ly
        self.AR = AR   # aspect ratio
        self.r0 = r0/np.sqrt(AR)   # particle size
        self.r_tr = r_tr  # tracer size
        
    
        rho = rho          # system packing fraction

        self.T=T  # number of steps for damping

        factor =factor       # one step jump length
        self.v0 = v0   # propulsion speed
        self.dt = r0*factor/(self.v0)        # time discretization
        self.eta = eta          # noise strength
        self.mu = mu         # mobility
        self.mu_tr = mu_tr
        self.mur = mur             # angular mobility
        self.f0 = self.v0/(self.T*self.mu*self.dt)
        # w0 = 0.2
        # tau0 = w0*10/mur
        self.k = k              # wca strength
        self.v_drag = v_drag
        self.mode =mode
        self.tracer = tracer




        self.pos_tr = np.zeros(2)
        self.pos_tr[0] = self.Lx/2
        self.pos_tr[1] = self.Ly/2

        self.N_tr = int(trN*6*np.pi*self.r_tr/self.r0)
        self.body_tr = np.zeros((self.N_tr,2))
        self.theta_tr = np.linspace(0,2*np.pi,self.N_tr)

        N_layer = 12
        self.marker = np.zeros((20*N_layer,2))
        self.theta_marker = np.linspace(0,2*np.pi,20)

        for i in range(N_layer):
            self.marker[i*20:(i+1)*20,0] = self.pos_tr[0]+(self.r_tr+self.r0*1.3*(i+2))*np.cos(self.theta_marker)
            self.marker[i*20:(i+1)*20,1] = self.pos_tr[1]+(self.r_tr+self.r0*1.3*(i+2))*np.sin(self.theta_marker)


        if tracer:
            self.N = int(rho*(self.Lx*self.Ly-np.pi*self.r_tr**2)/(self.AR*np.pi*self.r0**2))
        else:
            self.N = int(rho*(self.Lx*self.Ly)/(self.AR*np.pi*self.r0**2))
            # self.k = 0
        # N=10
        self.initialize()

    def initialize(self):
        self.pos = np.zeros((self.N,2))
        self.stress = np.zeros(self.N)

        self.pos[:,0] = np.random.uniform(-self.Lx/2+self.r_tr+self.r0,self.Lx/2-self.r_tr-self.r0,size=self.N)
        self.pos[:,1] = np.random.uniform(-self.Ly/2+self.r_tr+self.r0,self.Ly/2-self.r_tr-self.r0,size=self.N)
        self.orient = np.random.uniform(-np.pi, np.pi,size=self.N)
        self.len_or_traj = 10000
        self.or_traj = np.zeros((self.len_or_traj,self.N))
        self.mom_trans = np.zeros((self.N,2))
        self.iter = 0
        self.mom_ang = np.zeros(self.N)

        self.set_coord()
        self.relax()


        self.VX_avg = 0
        self.VY_avg = 0

    
    def set_coord(self):
        self.pos[:,0] = self.pos[:,0]%self.Lx
        self.pos[:,1] = self.pos[:,1]%self.Ly
        self.pos_tr[0] = self.pos_tr[0]%self.Lx
        self.pos_tr[1] = self.pos_tr[1]%self.Ly

        self.armb1 = (self.AR-(1/self.AR))*self.r0*np.array([np.cos(self.orient),np.sin(self.orient)]).T
        self.armb2 = - (self.AR-(1/self.AR))*self.r0*np.array([np.cos(self.orient),np.sin(self.orient)]).T
    #     armb1 = (AR-(1/AR))*r0*np.array([np.cos(orient+np.pi/2),np.sin(orient+np.pi/2)]).T
    #     armb2 = - (AR-(1/AR))*r0*np.array([np.cos(orient+np.pi/2),np.sin(orient+np.pi/2)]).T


        self.pos1 = self.pos + self.armb1
        self.pos2 = self.pos + self.armb2
        self.pos1[:,0] = self.pos1[:,0]%self.Lx
        self.pos1[:,1] = self.pos1[:,1]%self.Ly
        self.pos2[:,0] = self.pos2[:,0]%self.Lx
        self.pos2[:,1] = self.pos2[:,1]%self.Ly
        self.postot = np.concatenate([self.pos1,self.pos2])

        if self.mode=='free':

            self.body_tr[:,0] = (self.pos_tr[0]+self.r_tr*np.cos(self.theta_tr))%self.Lx
            self.body_tr[:,1] = (self.pos_tr[1]+self.r_tr*np.sin(self.theta_tr))%self.Ly



    def WCA(self,rsq,r_unit,k):
        return self.k*(6*np.divide((r_unit)**6,(rsq)**(7/2),out=np.zeros_like(rsq),where=rsq!=0) - 12*np.divide((r_unit)**12,(rsq)**(13/2),out=np.zeros_like(rsq),where=rsq!=0))
    def FWCA(self,p1,p2,k,r_ref,typ):


        tree1 = cKDTree(p1,boxsize=[self.Lx,self.Ly])
        tree2 = cKDTree(p2,boxsize=[self.Lx,self.Ly])
        dist = tree1.sparse_distance_matrix(tree2, max_distance=(r_ref)*2**(1/6),output_type='coo_matrix')

        dx = -p1[dist.row,0]+p2[dist.col,0]
        dy = -p1[dist.row,1]+p2[dist.col,1]

        dx[dx>self.Lx/2] -=self.Lx
        dx[dx<-self.Lx/2]+=self.Lx
        dy[dy>self.Ly/2] -=self.Ly
        dy[dy<-self.Ly/2]+=self.Ly

        force = self.WCA(dx**2+dy**2,r_ref,self.k)
        if typ==0:
            filt = (~np.isnan(force))*(np.abs(force)<np.abs(self.WCA((r_ref*0.8)**2,r_ref,self.k)))
        else:
            filt = (~np.isnan(force))
        angle = np.angle(dx+1j*dy)
        fx = sparse.coo_matrix((force[filt]*np.cos(angle[filt]),(dist.row[filt],dist.col[filt])), shape=dist.get_shape())
        fy = sparse.coo_matrix((force[filt]*np.sin(angle[filt]),(dist.row[filt],dist.col[filt])), shape=dist.get_shape())
        Fx = np.squeeze(np.asarray(fx.sum(axis=1)))
        Fy = np.squeeze(np.asarray(fy.sum(axis=1)))

        return (Fx,Fy)






    def Torque(self,Fx,Fy, rx,ry):
        return (rx*Fy-ry*Fx)

    def check_neighbor(self):
        tree0 = cKDTree(self.pos,boxsize=[self.Lx,self.Ly])
        tree1 = cKDTree(self.pos1,boxsize=[self.Lx,self.Ly])
        tree2 = cKDTree(self.pos2,boxsize=[self.Lx,self.Ly])

        dist00 = tree0.sparse_distance_matrix(tree0, max_distance=self.r0*2*1.1,output_type='coo_matrix')
        dist01 = tree0.sparse_distance_matrix(tree1, max_distance=self.r0*(1+1/self.AR)*1.1,output_type='coo_matrix')
        dist02 = tree0.sparse_distance_matrix(tree2, max_distance=self.r0*(1+1/self.AR)*1.1,output_type='coo_matrix')
        dist10 = tree1.sparse_distance_matrix(tree0, max_distance=self.r0*(1+1/self.AR)*1.1,output_type='coo_matrix')
        dist11 = tree1.sparse_distance_matrix(tree1, max_distance=self.r0*(2/self.AR)*1.1,output_type='coo_matrix')
        dist12 = tree1.sparse_distance_matrix(tree2, max_distance=self.r0*(2/self.AR)*1.1,output_type='coo_matrix')
        dist20 = tree2.sparse_distance_matrix(tree0, max_distance=self.r0*(1+1/self.AR)*1.1,output_type='coo_matrix')
        dist21 = tree2.sparse_distance_matrix(tree1, max_distance=self.r0*(2/self.AR)*1.1,output_type='coo_matrix')
        dist22 = tree2.sparse_distance_matrix(tree2, max_distance=self.r0*(2/self.AR)*1.1,output_type='coo_matrix')

        dist_list = [dist00,dist01,dist02,dist10,dist11,dist12,dist20,dist21,dist22]
        count_sum = sparse.coo_matrix((self.N, self.N ), dtype=bool)
        for dist in dist_list:
            count = np.full(self.N,True)[dist.col]
            count[dist.row==dist.col] = False
            count_mat = sparse.coo_matrix((count,(dist.row,dist.col)), shape=dist.get_shape())
            count_sum+=count_mat
        neighbor_count = np.squeeze(np.asarray(count_sum.sum(axis=1)))

        return neighbor_count


    




    def update(self):



        # interaction
        FX = np.zeros(self.N)
        FY = np.zeros(self.N)
        TAU = np.zeros(self.N)
        stress = np.zeros(self.N)
        self.set_coord()


        # volume exclusion (2body)

        (Fxvol1,Fyvol1) = self.FWCA(self.pos1,self.postot,self.k,self.r0*2/self.AR,0)
        FX += Fxvol1
        FY += Fyvol1
        TAU += self.Torque(Fxvol1,Fyvol1,self.armb1[:,0],self.armb1[:,1])
        (Fxvol2,Fyvol2) = self.FWCA(self.pos2,self.postot,self.k,self.r0*2/self.AR,0)
        FX += Fxvol2
        FY += Fyvol2
        TAU += self.Torque(Fxvol2,Fyvol2,self.armb2[:,0],self.armb2[:,1])
        
        (Fxvol10,Fyvol10) = self.FWCA(self.pos1,self.pos,self.k,self.r0*(1+1/self.AR),0)
        FX += Fxvol10
        FY += Fyvol10
        TAU += self.Torque(Fxvol10,Fyvol10,self.armb1[:,0],self.armb1[:,1])
        (Fxvol20,Fyvol20) = self.FWCA(self.pos2,self.pos,self.k,self.r0*(1+1/self.AR),0)
        FX += Fxvol20
        FY += Fyvol20
        TAU += self.Torque(Fxvol20,Fyvol20,self.armb2[:,0],self.armb2[:,1])
        (Fxvol01,Fyvol01) = self.FWCA(self.pos,self.pos1,self.k,self.r0*(1+1/self.AR),0)
        FX += Fxvol01
        FY += Fyvol01
        (Fxvol02,Fyvol02) = self.FWCA(self.pos,self.pos2,self.k,self.r0*(1+1/self.AR),0)
        FX += Fxvol02
        FY += Fyvol02
        (Fxvol00,Fyvol00) = self.FWCA(self.pos,self.pos,self.k,self.r0*2,0)
        FX += Fxvol00
        FY += Fyvol00

        stress += np.abs(Fxvol1+1j*Fyvol1) + np.abs(Fxvol2+1j*Fyvol2)
        stress += np.abs(Fxvol10+1j*Fyvol10) + np.abs(Fxvol01+1j*Fyvol01)   + np.abs(Fxvol20+1j*Fyvol20) + np.abs(Fxvol02+1j*Fyvol02)  + np.abs(Fxvol00+1j*Fyvol00)  
        

        # tracer dynamics
        if self.tracer:
            (Fxtr1,Fytr1) = self.FWCA(self.pos1,self.body_tr,self.k,self.r0*(1+1/self.AR),1)
            (Fxtr2,Fytr2) = self.FWCA(self.pos2,self.body_tr,self.k,self.r0*(1+1/self.AR),1)
            (Fxtr0,Fytr0) = self.FWCA(self.pos,self.body_tr,self.k,self.r0*2,1)
            FX += Fxtr0+Fxtr1+Fxtr2
            FY += Fytr0+Fytr1+Fytr2
            stress += np.abs((Fxtr0+Fxtr1+Fxtr2) + 1j*(Fytr0+Fytr1+Fytr2))


            TAU += self.Torque(Fxtr1,Fytr1,self.armb1[:,0],self.armb1[:,1])+ self.Torque(Fxtr2,Fytr2,self.armb2[:,0],self.armb2[:,1])
            VX = -self.mu_tr*np.sum(Fxtr0+Fxtr1+Fxtr2)
            VY = -self.mu_tr*np.sum(Fytr0+Fytr1+Fytr2)
        else:
            VX = 0
            VY = 0

        self.stress = stress
    #     pos_tr[0]-=mu_tr*np.sum(Fxtr0+Fxtr1+Fxtr2)
    #     pos_tr[1]-=mu_tr*np.sum(Fytr0+Fytr1+Fytr2)

        # in object frame
        # self.pos[:,0] +=-VX
        # self.pos[:,1] +=-VY
        if self.mode=='drag':
            self.pos[:,0] -= self.v_drag*self.dt
        else:
            self.pos_tr[0] +=VX*self.dt
            self.pos_tr[1] +=VY*self.dt
            self.pos_tr[0] = self.pos_tr[0]%self.Lx
            self.pos_tr[1] = self.pos_tr[1]%self.Ly

            

        self.VX_avg = VX*0.1+self.VX_avg*0.9
        self.VY_avg = VY*0.1+self.VY_avg*0.9
        # propulsion force
        FX += self.f0*np.cos(self.orient)
        FY += self.f0*np.sin(self.orient)
    #     TAU += tau0
    #     TAU[:int(N/2)] +=tau0
    #     TAU[int(N/2):] -=tau0




        # drag force
        FX -= self.mom_trans[:,0]/(self.T*self.dt)  
        FY -= self.mom_trans[:,1]/(self.T*self.dt)
        TAU -= self.mom_ang/(self.T*self.dt)

        # noise
        FX += self.eta*(1/np.sqrt(self.dt))*np.random.uniform(-1, 1, size=self.N)
        FY += self.eta*(1/np.sqrt(self.dt))*np.random.uniform(-1, 1, size=self.N)
        if (self.AR-1)**2>0.0001:
            TAU+=3*self.eta*(1/np.sqrt(self.dt))*np.random.uniform(-np.pi, np.pi, size=self.N)

        else:
            TAU = 3*self.eta*(1/np.sqrt(self.dt))*np.random.uniform(-np.pi, np.pi, size=self.N)

        




        # momentum update
        self.mom_trans[:,0]+=FX*self.dt
        self.mom_trans[:,1]+=FY*self.dt
        self.mom_ang[:]+=TAU*self.dt


        # position update
        self.pos[:,0] += self.mu*self.mom_trans[:,0]*self.dt
        self.pos[:,1] += self.mu*self.mom_trans[:,1]*self.dt

        self.dxp = self.mu*self.mom_trans[:,0]*self.dt
        self.dyp = self.mu*self.mom_trans[:,1]*self.dt
    #     orient   += mur*TAU*dt
        self.orient += self.mur*self.mom_ang[:]*self.dt
        self.dop = self.mur*self.mom_ang[:]*self.dt
        self.or_traj[self.iter,:] = self.dop
        self.iter = (self.iter+1)%self.len_or_traj
        # self.pos[:,0] +=self.mu*FX*self.dt
        # self.pos[:,1] +=self.mu*FY*self.dt
        # self.orient +=self.mur*TAU*self.dt


    # periodic boundary
        self.pos[:,0] = self.pos[:,0]%self.Lx
        self.pos[:,1] = self.pos[:,1]%self.Ly
        self.orient = self.orient%(2*np.pi)

        self.set_coord()
    def measure(self):
        if self.tracer:
            self.pointing = np.angle(self.VX_avg+1j*self.VY_avg)

            for i in range(8):
                self.marker[i*20:(i+1)*20,0] = self.pos_tr[0]+(self.r_tr+self.r0*(i+2))*np.cos(self.theta_marker+self.pointing)
                self.marker[i*20:(i+1)*20,1] = self.pos_tr[1]+(self.r_tr+self.r0*(i+2))*np.sin(self.theta_marker+self.pointing)
        else:
            self.pointing=0
            for i in range(8):
                self.marker[i*20:(i+1)*20,0] = self.pos_tr[0]+(self.r_tr+self.r0*(i+2))*np.cos(self.theta_marker)
                self.marker[i*20:(i+1)*20,1] = self.pos_tr[1]+(self.r_tr+self.r0*(i+2))*np.sin(self.theta_marker)

        self.marker[:,0] = self.marker[:,0]%self.Lx
        self.marker[:,1] = self.marker[:,1]%self.Ly

        tree1 = cKDTree(self.marker,boxsize=[self.Lx,self.Ly])
        tree2 = cKDTree(self.pos,boxsize=[self.Lx,self.Ly])
        dist = tree1.sparse_distance_matrix(tree2, max_distance=self.r0*4,output_type='coo_matrix')

        rho = np.ones(self.N)[dist.col]
        rho_mat = sparse.coo_matrix((rho,(dist.row,dist.col)), shape=dist.get_shape())
        self.rho = np.squeeze(np.asarray(rho_mat.sum(axis=1)))

        px = np.cos(self.orient[dist.col]-self.pointing)
        py = np.sin(self.orient[dist.col]-self.pointing)
        px_mat = sparse.coo_matrix((px,(dist.row,dist.col)), shape=dist.get_shape())
        py_mat = sparse.coo_matrix((py,(dist.row,dist.col)), shape=dist.get_shape())
        self.px = np.squeeze(np.asarray(px_mat.sum(axis=1)))
        self.py = np.squeeze(np.asarray(py_mat.sum(axis=1)))

        v_loc = np.sqrt(self.dxp**2+self.dyp**2)[dist.col]
        F_loc = self.stress[dist.col]
        dop = np.mean(self.or_traj,axis=0)
        D_loc = (dop**2)[dist.col]
        v_mat = sparse.coo_matrix((v_loc,(dist.row,dist.col)), shape=dist.get_shape())
        F_mat = sparse.coo_matrix((F_loc,(dist.row,dist.col)), shape=dist.get_shape())
        D_mat = sparse.coo_matrix((D_loc,(dist.row,dist.col)), shape=dist.get_shape())

        v_loc = np.squeeze(np.asarray(v_mat.sum(axis=1)))
        self.F_loc = np.squeeze(np.asarray(F_mat.sum(axis=1)))
        D_loc = np.squeeze(np.asarray(D_mat.sum(axis=1)))
        self.v_loc = np.divide(v_loc,self.rho,out=np.zeros_like(v_loc), where=self.rho!=0)
        self.D_loc = np.divide(D_loc,self.rho,out=np.zeros_like(D_loc), where=self.rho!=0)




    # relaxation of initial position to avoid overlapping
    def relax(self):
        self.set_coord()
        tree = cKDTree(self.pos,boxsize=[self.Lx,self.Ly])
        # treeall = cKDTree(np.concatenate([pos,wall]),boxsize=[Lx,Ly])
        dist = tree.sparse_distance_matrix(tree, max_distance=self.r0*2*2**(1/6),output_type='coo_matrix')
        while (len(dist.col)>self.N):
            filt = (dist.col!=dist.row)
            self.pos[dist.col[filt][0]][0] = np.random.uniform(-self.Lx/2+self.r_tr+self.r0,self.Lx/2-self.r_tr-self.r0,size=1)
            self.pos[dist.col[filt][0]][1] = np.random.uniform(-self.Ly/2+self.r_tr+self.r0,self.Ly/2-self.r_tr-self.r0,size=1)
            self.set_coord()

            tree = cKDTree(self.pos,boxsize=[self.Lx,self.Ly])
        #     treeall = cKDTree(np.concatenate([pos,wall]),boxsize=[Lx,Ly])
            dist = tree.sparse_distance_matrix(tree, max_distance=self.r0*1.9*2**(1/6),output_type='coo_matrix')





