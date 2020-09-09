import os
import pickle
import numpy as np
from scipy.io import loadmat
from tqdm import tqdm
import common.gui.pyzview as zview
from common.gui.pyzviewutils.addColor import add_color
from projects.DensePose.scripts.densepose_methods import DensePoseMethods

DP = DensePoseMethods()
zv = zview.interface()

def get_mesh():
    ALP_UV = loadmat(os.path.join(os.path.dirname(__file__), '../DensePoseData/UV_data/UV_Processed.mat'))
    with open('projects/DensePose/DensePoseData/basicModel_m_lbs_10_207_0_v1.1.0.pkl', 'rb') as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        data = u.load()
    v = data['v_template']
    f = data['f']

    return v, f


def iuv2p(i, u, v, verts):
    fbc = DP.IUV2FBC(i, u, v)
    p = DP.FBC2PointOnSurface(fbc, verts)
    return p


if __name__ == "__main__":
    zv.removeShape(-1)
    verts, f = get_mesh()
    v_iuv = np.ones((verts.shape[0], 3))*255
    v_err = np.ones(verts.shape[0])*np.inf

    h_mesh=zv.addColoredMesh("SMPL",addColor(verts,'r'),f)


    ug,vg = np.meshgrid(np.linspace(0,1,512),np.linspace(0,1,512))
    ug=ug.flatten()
    vg=vg.flatten()


    for i in tqdm(DP.Index_Symmetry_List):
        p = iuv2p(i, ug, vg, verts)
        ok = ~np.isnan(p[:,0])
        iuv = np.c_[ug*0+i,ug,vg][ok]
        p = p[ok]

        d=np.linalg.norm(verts.reshape(-1, 3, 1).transpose([0, 2, 1])-p.reshape(-1,3,1).transpose([2,0,1]),axis=2)
        indx=np.argmin(d,axis=1)
        minerr = np.min(d,axis=1)
        do_update = (minerr<v_err) & (minerr < 0.1)
        v_err[do_update]=minerr[do_update]
        v_iuv[do_update]=iuv[indx[do_update]]
        zv.updateColoredPoints(h_mesh, addColor(verts, v_iuv / [25.5, 1, 1]))




    np.savez('v_iuv', v=np.c_[verts,v_iuv],f=f)
