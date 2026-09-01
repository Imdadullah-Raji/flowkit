from src.io import foam_to_netcdf
from src.dataprocessing import Dataset
import matplotlib.pyplot as plt

casepath = '/home/raji/OpenFOAM/raji-dev/run/sq_new'
outpath =  '/home/raji/Research/analyze_cfd/data/squar.nc'
figsavepath = '/home/raji/Research/analyze_cfd/figures/'

#out = foam_to_netcdf(casepath=casepath, outpath= outpath, patch='square')

dataset = Dataset.from_netcdf(outpath)
snapshot = dataset.snapshot(20)

# print("Length Scale:", dataset.length_scale)
# cropped_ds = dataset.crop_relative(-2, 8, -2, 2, length_scale=0.5)
# cropped_snap = cropped_ds.snapshot(20)
# plt.scatter(cropped_snap.x, cropped_snap.y, c= cropped_snap.u)
# plt.show()
plt.scatter(snapshot.x, snapshot.y, c= snapshot.u)
plt.show()
next_snap = snapshot.next()

plt.scatter(next_snap.x, next_snap.y, c= next_snap.u)
plt.show()




