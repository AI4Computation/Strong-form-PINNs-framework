"""Geometry-only points used to audit each compact neural branch."""
import numpy as np


def support_probes(domain, centres, halfwidths, minimum_raw=1e-8):
    points=[];qualities=[];grid_sizes=[]
    for c,h in zip(centres,halfwidths):
        chosen=None
        for size in [33,65,129]:
            v=np.linspace(-.99,.99,size)
            z=np.stack(np.meshgrid(v,v),axis=-1).reshape(-1,2)
            xy=c+z*h
            valid=domain.contains(xy)
            score=np.prod(np.maximum(1-z*z,0)**3,axis=1)
            score[~valid]=-1
            idx=int(np.argmax(score))
            if score[idx]>=minimum_raw:
                chosen=xy[idx];qualities.append(float(score[idx]));grid_sizes.append(size)
                break
        if chosen is None:
            raise ValueError('No valid support probe; geometry support cannot be exercised.')
        points.append(chosen)
    return np.array(points),dict(minimum_local_window=float(min(qualities)),grid_sizes=grid_sizes)
