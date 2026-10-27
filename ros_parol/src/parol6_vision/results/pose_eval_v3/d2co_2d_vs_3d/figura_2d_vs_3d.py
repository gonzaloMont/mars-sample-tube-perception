#!/usr/bin/env python3
"""
figura_2d_vs_3d.py — Registro de D2CO: imagen (proyección) frente a vista
cenital (profundidad). Toma las poses de d2co_malla_test60.csv (columna T_est)
y dibuja los 3 casos con menor error 2D bidireccional y 3 en torno a la
mediana. Escribe figura_d2co_2d_vs_3d.png.
"""
import csv, json, os, numpy as np, cv2, open3d as o3d
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
HERE=os.path.dirname(os.path.abspath(__file__))
D=os.path.expanduser('~/parol6_pose_eval_dataset_v3')
STL=os.path.join(HERE,'..','..','..','..','parol6_d2co','models','mars2020tube_scaled.stl')
gt=json.load(open(D+'/bop/scene_gt.json')); cam=json.load(open(D+'/bop/scene_camera.json'))
meta={json.loads(l)['im_id']:json.loads(l) for l in open(D+'/meta.jsonl')}
m=o3d.io.read_triangle_mesh(STL); o3d.utility.random.seed(0)
pts=np.asarray(m.sample_points_uniformly(3000).points); pts-=(pts.min(0)+pts.max(0))/2
rows=[r for r in csv.DictReader(open(os.path.join(HERE,'d2co_malla_test60.csv'))) if r['pose_ok']=='1']
def proj(K,R,t): p=pts@R.T+t; q=p@K.T; return q[:,:2]/q[:,2:], p
sel=[]
for r in rows:
    g=gt[r['im_id']][int(r['inst'])]; R=np.array(g['cam_R_m2c']).reshape(3,3); t=np.array(g['cam_t_m2c'])
    K=np.array(cam[r['im_id']]['cam_K']).reshape(3,3); T=np.array([float(v) for v in r['T_est'].split()]).reshape(3,4)
    a,_=proj(K,R,t); b,_=proj(K,T[:,:3],T[:,3])
    d=np.sqrt(((b[:,None]-a[None])**2).sum(-1)); e2=max(d.min(1).mean(), d.min(0).mean())   # bidireccional
    sel.append((e2,r,R,t,K,T))
sel.sort(key=lambda s:s[0]); med=len(sel)//2
good=sel[:3]+sel[med-1:med+2]          # 3 con mejor registro 2D + 3 en torno a la mediana
fig,axs=plt.subplots(len(good),2,figsize=(11,3.6*len(good)),gridspec_kw={'width_ratios':[1.5,1]})
for i,(e2,r,R,t,K,T) in enumerate(good):
    img=cv2.imread(D+'/'+meta[int(r['im_id'])]['file'],0)
    a,pa=proj(K,R,t); b,pb=proj(K,T[:,:3],T[:,3])
    lo=np.minimum(a.min(0),b.min(0))-25; hi=np.maximum(a.max(0),b.max(0))+25
    x0,y0=np.maximum(lo,0).astype(int); x1,y1=np.minimum(hi,1023).astype(int)
    ax=axs[i,0]; ax.imshow(img[y0:y1,x0:x1],cmap='gray',vmin=0,vmax=255)
    ax.scatter(a[:,0]-x0,a[:,1]-y0,s=1,c='lime',label='pose real'); ax.scatter(b[:,0]-x0,b[:,1]-y0,s=1,c='red',alpha=.6,label='D2CO')
    ax.set_title(f"im{r['im_id']}#{r['inst']}: error 2D bidireccional {e2:.1f} px",fontsize=9); ax.axis('off')
    if i==0: ax.legend(loc='lower left',markerscale=8,fontsize=8)
    ax=axs[i,1]
    ax.scatter(pa[:,0],pa[:,2],s=1,c='lime'); ax.scatter(pb[:,0],pb[:,2],s=1,c='red',alpha=.6)
    zs=np.r_[pa[:,2],pb[:,2]]; xs=np.r_[pa[:,0],pb[:,0]]
    for p in (pa,pb):
        c=p.mean(0); ax.plot([0,c[0]*1.2],[0,c[2]*1.2],color='0.75',lw=.6,zorder=0)   # línea de visión
    ax.set_xlim(xs.min()-.05,xs.max()+.05); ax.set_ylim(zs.min()-.05,zs.max()+.05); ax.set_aspect('equal')
    ax.set_xlabel('X cámara (m)',fontsize=8); ax.set_ylabel('profundidad Z (m)',fontsize=8); ax.tick_params(labelsize=7)
    ax.set_title(f"vista cenital: ADD-S {float(r['add_s_mm']):.0f} mm, radial {float(r['err_radial_mm']):+.0f} mm",fontsize=9)
plt.tight_layout(); plt.savefig(os.path.join(HERE,'figura_d2co_2d_vs_3d.png'),dpi=110); print('ok', len(good))
