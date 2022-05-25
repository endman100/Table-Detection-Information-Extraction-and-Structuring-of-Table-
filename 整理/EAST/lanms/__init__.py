import numpy as np
from .adaptor import merge_quadrangle_n9 as nms_impl


def merge_quadrangle_n9(polys, thres=0.3, precision=10000):
    if len(polys) == 0:
        return np.array([], dtype='float32')
    p = polys.copy()
    p[:, :8] *= precision
    ret = np.array(nms_impl(p, thres), dtype='float32')
    ret[:, :8] /= precision
    return ret
    
# cl adaptor.cpp ./include/clipper/clipper.cpp /I ./include /I "C:\Users\endman100\AppData\Local\Programs\Python\Python36\include" /LD /Fe:adaptor.pyd /link/LIBPATH:"C:\Users\endman100\AppData\Local\Programs\Python\Python36\libs"
