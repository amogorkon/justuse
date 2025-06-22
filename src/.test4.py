import numpy as np
import scipy

U = np.array([[-0.-1.j, 0.+0.j, 0.+0.j, 0.+0.j],
 [ 0.+0.j, 0.+1.j, 0.+0.j, 0.+0.j],
 [ 0.+0.j, 0.+0.j, -0.-1.j, 0.+0.j],
 [ 0.+0.j, 0.+0.j, 0.+0.j, 0.+1.j]])
eigenvalues, eigenvectors = scipy.linalg.eig(U)
print(eigenvalues)