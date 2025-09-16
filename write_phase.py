import numpy as np
import os

# config_list = [['420',0.3,7,200],['421',0.2,7,200],['422',0.1,7,200],['423',0.03,7,200],['424',0.01,7,200],['425',0.003,7,200],['426',0.001,7,200],['427',0.4,7,200]]
# config_list = [['420',0.02,7,200],['428',0.05,7,200],['429',0.15,7,200],['430',0.2,7,200],['431',0.35,7,200],['432',0.07,7,200]]

# config_list = [['1',0.2],['2',]]

AR_list = [0.9,0.96,1.0,1.1,1.3,1.5,1.7]#[0.9,0.96,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7]
rho_list = [0.3,0.2,0.1,0.05,0.02,0.01]#,0.005]#,0.002,0.001,0.0005]

# AR_list = [1.08,1.25,1.12,1.14,1.16,1.33,1.36,1.43,1.46,1.55,1.65,0.9,0.93,0.96,0.98,1.0,1.02,1.04,1.06,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.31,1.32,1.34,1.35,1.37,1.38,1.39,1.22,1.27]
file = open('jobs.txt','a')
name='8'#'7'#'6'#'5'#'4'#'3'#'2'#'1'
for rho in rho_list:

    

    direc = 'data/phase/'+name+'/'+str(rho)
    # os.makedirs(direc,exist_ok=True)
    # file1=open(direc+'/settings.txt','a')
    # file1.write('name, rho, R, L = '+str(config)+'\n')
    # file1.close()
    
    for AR in AR_list:
        for i in range(3):
            state = os.getcwd()+'/'+direc+'/'+str(AR)+'_'+str(i)+'.npz'
            if os.path.exists(state):
                pass
            else:
                file.write('/pds/pds21/yunsik/miniconda3/bin/python run_phase.py %f %i %f  %s   \n' % (AR,i,rho,name))

file.close()

