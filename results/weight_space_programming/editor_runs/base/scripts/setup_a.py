# Setup part A (re-runnable after a kernel restart): data, helpers, grader, constants. No heavy computation.
import json, numpy as np, random, re, os, torch, time, warnings
import pandas as pd
from collections import Counter, defaultdict
warnings.filterwarnings('ignore')
torch.set_num_threads(64)
def ex(text):
    with open(os.path.join(WORK,'explanations.md'),'a') as f: f.write(text.strip()+'\n\n')
def note(text):
    with open(os.path.join(WORK,'notes.md'),'a') as f: f.write(text.strip()+'\n\n')
train=[json.loads(l) for l in open(os.path.join(PRACTICE,'mpts52_train.jsonl'))]
val=[json.loads(l) for l in open(os.path.join(PRACTICE,'mpts52_val.jsonl'))]
vby={d['id']:d for d in val}; tby={d['id']:d for d in train}
def natoms(d): return sum(1 for l in d['answer'].split('\n') if l.strip().endswith('1.0') and len(l.split())==7)
rng=np.random.RandomState(0); perm=rng.permutation(len(val)); VID=[val[i]['id'] for i in perm]
def sset(k,n=256): return VID[k*256:k*256+n]
rng=np.random.RandomState(1); tperm=rng.permutation(len(train)); TID=[train[i]['id'] for i in tperm]
PROMPT_PRE="<|im_start|>system\nYou are an expert in materials science and crystallography.<|im_end|>\n<|im_start|>user\n"
def prompt_of(d): return PROMPT_PRE+d['problem']+"<|im_end|>\n<|im_start|>assistant\n"
vnat={d['id']:natoms(d) for d in val}
DS=[i for i in VID[2560:] if vnat[i]<=10]
# pass-1 / pass-3 / pass-7 id lists (deterministic reconstruction)
def _tf1_ids():
    ids=[];tot=0;n_=0
    for i in TID:
        t=prompt_of(tby[i])+tby[i]['answer']+"<|im_end|>"; n=len(tok.encode(t))
        if n>4000: continue
        if tot+n>515000 or n_>=1024: break
        ids.append(i); tot+=n; n_+=1
    return ids
TF1_ids=_tf1_ids(); assert len(TF1_ids)==428
P3ids=[i for i in TID if i not in set(TF1_ids)][:200]
_used=set(TF1_ids)|set(P3ids)
P7ids=[i for i in TID if i not in _used][:880]
from pymatgen.core import Structure, Composition, Lattice
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
SM=StructureMatcher(stol=0.5,ltol=0.3,angle_tol=10)
REF={}
def ref(id_):
    if id_ not in REF: REF[id_]=Structure.from_str(vby[id_]['answer'] if id_ in vby else tby[id_]['answer'],fmt='cif')
    return REF[id_]
def grade(text,id_):
    r=dict(parseable=0,valid=0,comp_ok=0,match=0,rms=None,cat='not_parseable',err=None,nat=None)
    try: s=Structure.from_str(text,fmt='cif')
    except Exception as e:
        r['err']=type(e).__name__+':'+str(e)[:80]; return r
    r['parseable']=1; r['nat']=len(s)
    try:
        dm=s.distance_matrix; ok=True
        if len(s)>1:
            iu=np.triu_indices(len(s),1); ok=dm[iu].min()>=0.5
        ok=ok and s.volume>0.1
        if ok: SpacegroupAnalyzer(s,symprec=0.1).get_space_group_number()
        r['valid']=int(ok)
    except Exception as e: r['valid']=0
    if not r['valid']: r['cat']='invalid_structure'; return r
    R=ref(id_)
    r['comp_ok']=int(s.composition.reduced_formula==R.composition.reduced_formula)
    if not r['comp_ok']: r['cat']='valid_wrong_composition'; return r
    try:
        d=SM.get_rms_dist(s,R)
        if d is not None: r['match']=1; r['rms']=d[0]; r['cat']='match'
        else: r['cat']='valid_no_match'
    except Exception as e: r['cat']='valid_no_match'
    return r
HDR=["# generated using pymatgen","data_","_symmetry_space_group_name_H-M   'P 1'","_cell_length_a","_cell_length_b","_cell_length_c","_cell_angle_alpha","_cell_angle_beta","_cell_angle_gamma","_symmetry_Int_Tables_number   1","_chemical_formula_structural","_chemical_formula_sum","_cell_volume","_cell_formula_units_Z","loop_"," _symmetry_equiv_pos_site_id"," _symmetry_equiv_pos_as_xyz","  1  'x, y, z'","loop_"," _atom_site_type_symbol"," _atom_site_label"," _atom_site_symmetry_multiplicity"," _atom_site_fract_x"," _atom_site_fract_y"," _atom_site_fract_z"," _atom_site_occupancy"]
def hdr_ok(t):
    ls=t.split('\n')
    for i,h in enumerate(HDR):
        if i>=len(ls) or not ls[i].startswith(h): return i
    return len(HDR)
def atom_lines(t):
    ls=t.split('\n'); out=[]; started=False
    for l in ls:
        if l.strip()=='_atom_site_occupancy': started=True; continue
        if started:
            p=l.split()
            if len(p)>=6: out.append(p)
            else: break
    return out
def coords(t):
    out=[]
    for p in atom_lines(t):
        try: out.append((re.sub(r'[^A-Za-z]','',p[0]),tuple(round(float(x)%1,3) for x in p[-4:-1])))
        except Exception: pass
    return out
def fcount(i):
    d=vby[i] if i in vby else tby[i]
    f=d['input'].split('composition is ')[1].split('. It has')[0]
    try: return int(Composition(f).num_atoms)
    except Exception: return None
def cellp(d): return [float(l.split()[-1]) for l in d['answer'].split('\n')[3:9]]
def sgn(d): return int(re.search(r'number (\d+)',d['input']).group(1))
def bf16_round_stochastic(x,seed=0):
    g=torch.Generator().manual_seed(seed)
    b=x.contiguous().numpy().view(np.int32).copy()
    frac=(b & 0xFFFF).astype(np.float32)/65536.0
    lo=b & np.int32(-65536)
    up=torch.rand(x.shape,generator=g).numpy()<frac
    out=(lo+np.where(up,np.int32(65536),np.int32(0))).astype(np.int32)
    return torch.from_numpy(out.view(np.float32).reshape(x.shape))
IM_END=151645; NL=198; NLNL=271; SP=220; FENCE=[73594,13874]
DIG=[tok.encode(str(d))[0] for d in range(10)]
LN='model.layers.27.mlp.down_proj.weight'
U=load_original('lm_head.weight'); gam=load_original('model.norm.weight'); Ug=U*gam
W0=load_original(LN)
def lens_pre(h):
    return torch.log_softmax((h*torch.rsqrt((h*h).mean(1,keepdim=True)+1e-6)*gam)@U.T,dim=1)
def line_tokens(T,r,n):
    a=np.where(T[r]==77091)[0][-1]+2; lines=[];cur=[]
    for j in range(a,n):
        s=tok.decode([int(T[r,j])]); cur.append((j,s))
        if s.endswith('\n'): lines.append(cur); cur=[]
    if cur: lines.append(cur)
    return a,lines
print('setup A done', len(train), len(val), len(DS))
