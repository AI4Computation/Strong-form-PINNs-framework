"""Positive quadrature on exact vertical rock intervals of polygon/ellipse domains.

Break x at polygon vertices and ellipse extrema. Quadratic endpoint maps remove
ellipse square-root endpoints. No cut-cell point rejection or FEM data is used.
"""
import numpy as np
from numpy.polynomial.legendre import leggauss


def polygon_intervals(vertices,x):
    vertices=np.asarray(vertices);ys=[]
    for a,b in zip(vertices,np.roll(vertices,-1,axis=0)):
        if min(a[0],b[0])<x<max(a[0],b[0]):
            ys.append(float(a[1]+(x-a[0])*(b[1]-a[1])/(b[0]-a[0])))
    ys.sort();assert len(ys)%2==0,(x,ys)
    return list(zip(ys[::2],ys[1::2]))


def ellipse_data(hole):
    angle=hole.get('angle',0.);c,s=np.cos(angle),np.sin(angle)
    R=np.array([[c,-s],[s,c]]);a=np.asarray(hole['axes'])
    Q=R@np.diag(1/a**2)@R.T
    extent=np.sqrt((a[0]*c)**2+(a[1]*s)**2)
    return Q,float(hole['center'][0]-extent),float(hole['center'][0]+extent)


def solid_intervals(domain,x):
    outer=polygon_intervals(domain.outer,x);void=[]
    for hole in domain.holes:
        if hole['kind']=='polygon':void+=polygon_intervals(hole['vertices'],x)
        else:
            Q,lo,hi=ellipse_data(hole)
            if lo<x<hi:
                dx=x-hole['center'][0]
                centre=hole['center'][1]-Q[0,1]/Q[1,1]*dx
                radius=np.sqrt(max(0.,(1-(Q[0,0]-Q[0,1]**2/Q[1,1])*dx**2)/Q[1,1]))
                void.append((centre-radius,centre+radius))
    void.sort();rock=[]
    for low,high in outer:
        current=low
        for a,b in void:
            if b<=current or a>=high:continue
            if a>current:rock.append((current,min(a,high)))
            current=max(current,b)
            if current>=high:break
        if current<high:rock.append((current,high))
    return [(a,b) for a,b in rock if b-a>1e-14]


class SliceQuadrature:
    def __init__(self,domain,base_depth=5,order=5,**unused):
        self.domain,self.order=domain,order
        self.spacing=float(np.max(domain.hi-domain.lo))/2**base_depth
        values=list(domain.outer[:,0]);special=[]
        for hole in domain.holes:
            if hole['kind']=='polygon':values.extend(np.asarray(hole['vertices'])[:,0])
            else:
                _,a,b=ellipse_data(hole);values.extend([a,b]);special.extend([a,b])
        values=sorted(float(v) for v in values if domain.lo[0]-1e-13<=v<=domain.hi[0]+1e-13)
        critical=[]
        for x in values:
            if not critical or x-critical[-1]>1e-13:critical.append(x)
        breaks=[]
        for a,b in zip(critical,critical[1:]):
            count=max(1,int(np.ceil((b-a)/self.spacing)))
            if count==1 and any(abs(a-v)<1e-13 for v in special) and any(abs(b-v)<1e-13 for v in special):count=2
            nodes=np.linspace(a,b,count+1);breaks.extend(zip(nodes[:-1],nodes[1:]))
        self.panels=breaks;self.special=special
        self.vertical_panels=max(1,int(np.ceil((domain.hi[1]-domain.lo[1])/self.spacing)))

    def points(self,order=None):
        nodes,weights=leggauss(order or self.order);t=(nodes+1)/2;w=weights/2
        points=[];mass=[];ny=self.vertical_panels
        yt=((np.arange(ny)[:,None]+t)/ny).ravel();yw=np.tile(w/ny,ny)
        for a,b in self.panels:
            left=any(abs(a-v)<1e-13 for v in self.special)
            right=any(abs(b-v)<1e-13 for v in self.special)
            assert not(left and right)
            if left:x=a+(b-a)*t**2;wx=w*2*(b-a)*t
            elif right:x=b-(b-a)*(1-t)**2;wx=w*2*(b-a)*(1-t)
            else:x=a+(b-a)*t;wx=w*(b-a)
            for xi,wi in zip(x,wx):
                for low,high in solid_intervals(self.domain,float(xi)):
                    points.append(np.column_stack([np.full(len(yt),xi),low+(high-low)*yt]))
                    mass.append(wi*(high-low)*yw)
        xy,weights=np.concatenate(points),np.concatenate(mass)
        assert np.all(weights>0) and self.domain.contains(xy).all()
        return dict(xy=xy,w=weights,area=float(weights.sum()),cells=len(self.panels)*ny,
                    geometry_exact_slices=True)
