
import numpy as np

def get_dist_from_center_x(map_height: int, map_width: int) -> np.ndarray:
    pos = np.linspace(0, 2, map_width, dtype=np.float32)[None, :].repeat(map_height, axis=0)
    return np.abs(1 - pos)[None, None, :, :]



p = get_dist_from_center_x(32, 32);
print(p)
