import numpy as np
from scipy.spatial.transform import Rotation as R

def _make_transform(rot, pos):
    tf = np.eye(4, dtype=np.float64)
    tf[:3, :3] = np.asarray(rot, dtype=np.float64)
    tf[:3, 3] = np.asarray(pos, dtype=np.float64).reshape(3)
    return tf


camera_to_apple = [[ 0.83931,  0.06824,  0.53935, -0.09377],[-0.16931,  0.97556,  0.14004,  0.30348],[-0.51662, -0.20886,  0.83035, -0.00217],[ 0,       0, 0,       1,     ]]

rot = R.from_quat([-0.6446, 0.328, -0.326, 0.6086]).as_matrix()
base_to_camera = _make_transform(rot, [-0.469, 0.529, 0.5896])

base_to_apple =  base_to_camera @ camera_to_apple

print(base_to_apple)