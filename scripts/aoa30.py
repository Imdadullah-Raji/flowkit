from src.io import foam_to_netcdf
from src.dataprocessing import Dataset
import matplotlib.pyplot as plt


casepath = '/home/raji/Research/Thesis/static_airfoil/re500_aoa30'
outpath =  '/home/raji/Research/analyze_cfd/data/re500_aoa30.nc'
figsavepath = '/home/raji/Research/analyze_cfd/figures/'

#out = foam_to_netcdf(casepath=casepath, outpath= outpath, patch='airfoil')

ds = Dataset.from_netcdf(outpath)
aoa = 30
ds = ds.rotate(aoa).crop(-1, 10, -5, 5)
snapshot = ds.snapshot(200.076175)

plt.scatter(snapshot.x, snapshot.y, c= snapshot.u, s=1)
plt.show()