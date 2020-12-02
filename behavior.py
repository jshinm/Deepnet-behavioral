import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class behavior:

    def generate_gaussian_parity(self, n, mean=np.array([-1, -1]), cov_scale=1, angle_params=None, k=1, acorn=None):
        if acorn is not None:
            np.random.seed(acorn)
            
        d = len(mean)
        lim = abs(mean[0])
        
        if mean[0] == -1 and mean[1] == -1:
            mean = mean + 1 / 2**k
        elif mean[0] == -2 and mean[1] == -2:
            mean = mean + 1
        
        mnt = np.random.multinomial(n, 1/(4**k) * np.ones(4**k))
        cumsum = np.cumsum(mnt)
        cumsum = np.concatenate(([0], cumsum))
        
        Y = np.zeros(n)
        X = np.zeros((n, d))
        
        for i in range(2**k):
            for j in range(2**k):
                temp = np.random.multivariate_normal(mean, cov_scale * np.eye(d), 
                                                    size=mnt[i*(2**k) + j])
                if abs(mean[0]) == 0.5:
                    temp[:, 0] += i*(1/2**(k-1))
                    temp[:, 1] += j*(1/2**(k-1))
                    
                elif abs(mean[0]) == 1:
                    temp[:, 0] += i*2
                    temp[:, 1] += j*2

                # screen out values outside the boundary
                idx_oob = np.where(abs(temp) > lim)
                
                for l in idx_oob:
                    
                    while True:
                        temp2 = np.random.multivariate_normal(mean, cov_scale * np.eye(d), 
                                                    size=1)

                        if (abs(temp2) < lim).all():
                            temp[l] = temp2
                            break
                
                X[cumsum[i*(2**k) + j]:cumsum[i*(2**k) + j + 1]] = temp
                
                if i % 2 == j % 2:
                    Y[cumsum[i*(2**k) + j]:cumsum[i*(2**k) + j + 1]] = 0
                else:
                    Y[cumsum[i*(2**k) + j]:cumsum[i*(2**k) + j + 1]] = 1
                    
        if d == 2:
            if angle_params is None:
                angle_params = np.random.uniform(0, 2*np.pi)
                
    #         R = generate_2d_rotation(angle_params)
    #         X = X @ R
            
        else:
            raise ValueError('d=%i not implemented!'%(d))
        
        return X, Y.astype(int)