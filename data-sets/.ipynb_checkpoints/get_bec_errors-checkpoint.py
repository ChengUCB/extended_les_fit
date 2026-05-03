from ase.io import read
import numpy as np
import sys
xyzs = read(sys.argv[1], index=':')
epsilon_r = 1.78 # float(sys.argv[2])
# collect all the bec tensors

bec_refs = []
bec_preds = []

for xyz in xyzs:
    # get the bec tensor
    bec_ref = xyz.get_array('BEC')
    bec_pred = xyz.get_array('MACE_BEC') * epsilon_r**0.5

    # append to the list
    bec_refs.append(bec_ref)
    bec_preds.append(bec_pred)

# convert to numpy arrays
bec_refs = np.array(bec_refs)
bec_preds = np.array(bec_preds)

# calculate the errors
bec_errors = bec_refs - bec_preds
# calculate the mean and std
mae = np.mean(np.abs(bec_errors))
rmse = np.sqrt(np.mean(bec_errors**2))
r2 = 1 - np.sum(bec_errors**2) / np.sum((bec_refs - np.mean(bec_refs))**2)
# print the results
print('MAE: ', mae)
print('RMSE: ', rmse)
print('R2: ', r2)
