import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image
from ase import Atoms

# ==============================
# CONSTANTS
# ==============================
HARTREE_TO_KCAL = 627.509
R_KCAL = 0.0019872041
TEMP = 298.15
NBINS = 72
BOOT_N = 2000
BOOT_CI = 95

CHIRAL_XYZ  = "crest_conformers_chiral.xyz"
ACHIRAL_XYZ = "crest_conformers_achiral.xyz"
BG_IMAGE    = "Picture1.png"

TORSIONS = {
    "left_COAr":  {"arm":"left","chiral":(35,23,21,4),"achiral":(27,23,21,4)},
    "right_COAr": {"arm":"right","chiral":(37,25,22,6),"achiral":(29,25,22,6)},
}

# ==============================
# IO
# ==============================
def to_zero(t):
    return tuple(i-1 for i in t)

def read_crest_xyz(filename):
    atoms_list, energies = [], []
    with open(filename) as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        n = int(lines[i])
        e = float(lines[i+1])
        sym, xyz = [], []
        for j in range(n):
            p = lines[i+2+j].split()
            sym.append(p[0])
            xyz.append([float(p[1]),float(p[2]),float(p[3])])
        atoms_list.append(Atoms(symbols=sym, positions=xyz))
        energies.append(e)
        i += n+2
    return atoms_list, np.array(energies)

def boltz_weights(E):
    dE = (E-E.min())*HARTREE_TO_KCAL
    w = np.exp(-dE/(R_KCAL*TEMP))
    return w/w.sum()

# ==============================
# GEOMETRY
# ==============================
def dihedral(p0,p1,p2,p3):
    b0=p1-p0; b1=p2-p1; b2=p3-p2
    b1/=np.linalg.norm(b1)
    v=b0-np.dot(b0,b1)*b1
    w=b2-np.dot(b2,b1)*b1
    return np.degrees(np.arctan2(
        np.dot(np.cross(b1,v),w),
        np.dot(v,w)))%360

def get_angles(atoms_list, t):
    a,b,c,d=t
    return np.array([
        dihedral(at.get_positions()[a],
                 at.get_positions()[b],
                 at.get_positions()[c],
                 at.get_positions()[d])
        for at in atoms_list])

# ==============================
# METRICS
# ==============================
def circ_mean_rad(a,w):
    ang=np.radians(a)
    return np.arctan2(np.sum(w*np.sin(ang)),
                      np.sum(w*np.cos(ang)))

def circ_rms_deg(a,w):
    ang=np.radians(a)
    mu=circ_mean_rad(a,w)
    d=np.angle(np.exp(1j*(ang-mu)))
    return np.degrees(np.sqrt(np.sum(w*d*d)))

def central_width(a,w,frac=0.68):
    a=np.mod(a,360)
    o=np.argsort(a)
    a=a[o]; w=w[o]/w.sum()
    ae=np.concatenate([a,a+360])
    we=np.concatenate([w,w])
    c=np.cumsum(we)
    best=360
    j=0
    for i in range(len(a)):
        tgt=c[i]+frac
        while j<len(c) and c[j]<tgt:
            j+=1
        if j<len(c):
            best=min(best,ae[j]-ae[i])
    return best

def circ_entropy(a,w):
    h,_=np.histogram(a,bins=np.linspace(0,360,NBINS+1),weights=w)
    p=h/h.sum()
    p=p[p>0]
    return -np.sum(p*np.log(p))

# ==============================
# HISTOGRAM CSV
# ==============================
def save_hist_csv(name, ang_c, w_c, ang_a, w_a):
    bins = np.linspace(0,360,NBINS+1)
    centers = 0.5*(bins[:-1]+bins[1:])
    hc,_ = np.histogram(ang_c,bins=bins,weights=w_c)
    ha,_ = np.histogram(ang_a,bins=bins,weights=w_a)
    pd.DataFrame({
        "bin_center_deg":centers,
        "chiral_weight":hc,
        "achiral_weight":ha
    }).to_csv(f"{name}_hist.csv",index=False)

# ==============================
# BAYESIAN BOOTSTRAP
# ==============================
def bayesian_bootstrap_arm(angle_sets, base_weights, metric):
    N=len(base_weights)
    draws=np.empty(BOOT_N)
    for b in range(BOOT_N):
        d = np.random.dirichlet(np.ones(N))
        w = base_weights*d
        w/=w.sum()
        vals=[metric(a,w) for a in angle_sets]
        draws[b]=np.mean(vals)
    return draws

def ci(x):
    lo=(100-BOOT_CI)/2
    hi=100-lo
    return np.percentile(x,[lo,hi])

def log_ratio_ci(num,den,eps=1e-12):
    num=np.clip(num,eps,None)
    den=np.clip(den,eps,None)
    lr=np.log(num)-np.log(den)
    lo,hi=ci(lr)
    return np.exp(lo),np.exp(hi)

# ==============================
# POLAR PLOTTING (ORIGINAL STYLE)
# ==============================
def add_background_image(ax,path,xy=(0.5,0.5),zoom=0.5):
    img=Image.open(path)
    oi=OffsetImage(img,zoom=zoom)
    ab=AnnotationBbox(oi,xy,xycoords="axes fraction",
                      frameon=False,zorder=0)
    ax.add_artist(ab)

def load_arm_csv(csv):
    df=pd.read_csv(csv)
    ang=(360-df["bin_center_deg"].to_numpy())%360
    v1=df["chiral_weight"].to_numpy()
    v2=df["achiral_weight"].to_numpy()
    m=(ang>=0)&(ang<=180)
    ang,v1,v2=ang[m],v1[m],v2[m]
    o=np.argsort(ang)
    return ang[o],v1[o],v2[o]

def plot_semicircle(csv,out,mirror=False,bg_xy=(0.5,0.5)):
    deg,v1,v2=load_arm_csv(csv)
    if mirror: deg=-deg
    th=np.deg2rad(deg)
    bw=np.diff(np.sort(np.abs(deg))).mean() if len(deg)>1 else 5
    w=np.deg2rad(bw)
    off=w*0.25
    fig,ax=plt.subplots(figsize=(6,6),
                        subplot_kw={"projection":"polar"})
    add_background_image(ax,BG_IMAGE,bg_xy)
    ax.bar(th-off,v1,width=w*0.9,alpha=0.7,label="Chiral")
    ax.bar(th+off,v2,width=w*0.9,alpha=0.5,label="Achiral")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    if mirror:
        ax.set_thetamin(-180); ax.set_thetamax(0)
    else:
        ax.set_thetamin(0); ax.set_thetamax(180)
    ax.legend()
    plt.savefig(out,dpi=300,bbox_inches="tight")
    plt.close()

def plot_two_arm(left,right,out):
    Ld,L1,L2=load_arm_csv(left)
    Rd,R1,R2=load_arm_csv(right)
    deg=np.concatenate([-Ld,Rd])
    v1=np.concatenate([L1,R1])
    v2=np.concatenate([L2,R2])
    th=np.deg2rad(deg)
    bw=np.diff(Ld).mean() if len(Ld)>1 else 5
    w=np.deg2rad(bw)
    off=w*0.25
    fig,ax=plt.subplots(figsize=(7,7),
                        subplot_kw={"projection":"polar"})
    add_background_image(ax,BG_IMAGE)
    ax.bar(th-off,v1,width=w*0.9,alpha=0.7,label="Chiral")
    ax.bar(th+off,v2,width=w*0.9,alpha=0.5,label="Achiral")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_thetamin(0); ax.set_thetamax(360)
    ticks=np.arange(0,360,30)
    labels=[f"{t}°" if t<=180 else f"-{360-t}°" for t in ticks]
    ax.set_thetagrids(ticks,labels=labels)
    ax.legend()
    plt.savefig(out,dpi=300,bbox_inches="tight")
    plt.close()

# ==============================
# MAIN
# ==============================
c_atoms,E_c = read_crest_xyz(CHIRAL_XYZ)
a_atoms,E_a = read_crest_xyz(ACHIRAL_XYZ)
w_c = boltz_weights(E_c)
w_a = boltz_weights(E_a)

arms={"left":{"c":[],"a":[]},
      "right":{"c":[],"a":[]}}

for name,info in TORSIONS.items():
    ac=get_angles(c_atoms,to_zero(info["chiral"]))
    aa=get_angles(a_atoms,to_zero(info["achiral"]))
    arms[info["arm"]]["c"].append(ac)
    arms[info["arm"]]["a"].append(aa)
    save_hist_csv(name,ac,w_c,aa,w_a)

rows=[]
for arm in ["left","right"]:
    cs=arms[arm]["c"]
    aa=arms[arm]["a"]

    rms_c=np.mean([circ_rms_deg(a,w_c) for a in cs])
    rms_a=np.mean([circ_rms_deg(a,w_a) for a in aa])

    w68_c=np.mean([central_width(a,w_c) for a in cs])
    w68_a=np.mean([central_width(a,w_a) for a in aa])

    ent_c=np.mean([circ_entropy(a,w_c) for a in cs])
    ent_a=np.mean([circ_entropy(a,w_a) for a in aa])

    br_c=bayesian_bootstrap_arm(cs,w_c,circ_rms_deg)
    br_a=bayesian_bootstrap_arm(aa,w_a,circ_rms_deg)
    bw_c=bayesian_bootstrap_arm(cs,w_c,central_width)
    bw_a=bayesian_bootstrap_arm(aa,w_a,central_width)
    be_c=bayesian_bootstrap_arm(cs,w_c,circ_entropy)
    be_a=bayesian_bootstrap_arm(aa,w_a,circ_entropy)

    rows.append([
        arm,
        rms_c,*ci(br_c),
        rms_a,*ci(br_a),
        w68_c,*ci(bw_c),
        w68_a,*ci(bw_a),
        ent_c,*ci(be_c),
        ent_a,*ci(be_a),
        rms_a/rms_c,*log_ratio_ci(br_a,br_c),
        w68_a/w68_c,*log_ratio_ci(bw_a,bw_c),
        ent_a/ent_c,*log_ratio_ci(be_a,be_c)
    ])

df=pd.DataFrame(rows,columns=[
"group",
"rms_spread_chiral","rms_lo","rms_hi",
"rms_spread_achiral","rmsa_lo","rmsa_hi",
"width68_chiral","w68_lo","w68_hi",
"width68_achiral","w68a_lo","w68a_hi",
"entropy_chiral","ent_lo","ent_hi",
"entropy_achiral","enta_lo","enta_hi",
"rigidity_factor_rms","rig_rms_lo","rig_rms_hi",
"rigidity_factor_w68","rig_w68_lo","rig_w68_hi",
"rigidity_factor_entropy","rig_ent_lo","rig_ent_hi"
])

df.to_csv("rigidity_bootstrap_summary.csv",index=False)

# plots
plot_semicircle("right_COAr_hist.csv","right_arm.png",
                mirror=False,bg_xy=(0.25,0.5))
plot_semicircle("left_COAr_hist.csv","left_arm.png",
                mirror=True,bg_xy=(0.75,0.5))
plot_two_arm("left_COAr_hist.csv","right_COAr_hist.csv",
             "two_arm_circle_correct.png")

print("✓ Bootstrap + CSV + original polar plots complete.")
