import numpy as np
from src.dataprocessing import CartesianSnapshot


class Particles:
    def __init__(self, X, Y, mask=None ):

        if not(mask is None):
            X = X[~mask]
            Y = Y[~mask]
            self._mask = mask

        self.posx = X
        self.posy = Y
        self.n_particles = X.shape[0]
        

    def step(self, field:CartesianSnapshot, dt=None, method = "Euler" ):

        if method == "Euler":
            u, v, p = field.sample(self.posx,self.posy)
            X, Y = self._euler_step(u,v,dt)
            valid = ~(np.isnan(X) | np.isnan(Y))
            X= X[valid]
            Y = Y[valid]

        return Particles(X,Y)

    def _euler_step(self, u,v,dt):
        X = self.posx+u*dt
        Y = self.posy+v*dt

        return X, Y
    
    # def _rk4_step(self, u,v, dt):
    #     k1 = 2*dt*u
    #     pass