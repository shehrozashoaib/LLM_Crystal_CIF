# Setup part B (re-runnable): pass-3 arrays, key matrix K (last-layer MLP activations at answer positions), second moment C (cached on disk), Cholesky, vocabulary whitening.
t0=time.time()
T3=np.load('fwd_3/tokens.npy'); LP3=np.load('fwd_3/token_logprobs.npy').astype(np.float32); TK3=np.load('fwd_3/topk_ids.npy'); TL3=np.load('fwd_3/topk_logprobs.npy').astype(np.float32)
m3=json.load(open('fwd_3/meta.json')); L3=np.array(m3['lengths'])
M3=np.load('fwd_3/mlp_L28.npy',mmap_mode='r'); H28=np.load('fwd_3/hidden_L28.npy',mmap_mode='r'); H27=np.load('fwd_3/hidden_L27.npy',mmap_mode='r')
LINES3=[]; ASTART3=[]
for r in range(200):
    a,lines=line_tokens(T3,r,L3[r]); LINES3.append(lines); ASTART3.append(a)
INFO=[]   # (text r, pos j, line idx of NEXT token, token idx in line of NEXT token, NEXT token str, NEXT token id)
for r in range(200):
    a=ASTART3[r]; nxt={}
    for li,l in enumerate(LINES3[r]):
        for ti,(j,s) in enumerate(l): nxt[j-1]=(li,ti,s,int(T3[r,j]))
    for j in range(a-1,L3[r]):
        INFO.append((r,j)+nxt[j] if j in nxt else (r,j,-1,-1,'<END>',-1))
pos2idx={(x[0],x[1]):i for i,x in enumerate(INFO)}
K=np.empty((len(INFO),18944),dtype=np.float32); i0=0
for r in range(200):
    a=ASTART3[r]; n=L3[r]-(a-1)
    K[i0:i0+n]=np.asarray(M3[r,a-1:L3[r]],dtype=np.float32); i0+=n
assert i0==len(INFO)
Kt=torch.from_numpy(K)
ir=np.array([x[0] for x in INFO]); fit=ir<160
ho=np.where(~fit)[0]; ho_pos={i:q for q,i in enumerate(ho)}
if os.path.exists('Cfit.pt'):
    Cfit=torch.load('Cfit.pt').double()
else:
    Cfit=torch.zeros(18944,18944,dtype=torch.float64); idx=np.where(fit)[0]
    for s in range(0,len(idx),20000):
        X=Kt[idx[s:s+20000]]; Cfit+=(X.T@X).double()
    Cfit/=len(idx); torch.save(Cfit.float(),'Cfit.pt')
tr=float(torch.trace(Cfit))/18944
Lc=torch.linalg.cholesky(Cfit+0.01*tr*torch.eye(18944,dtype=torch.float64))
mu_U=Ug.mean(0); _Xc=(Ug-mu_U).double(); SigU=(_Xc.T@_Xc)/Ug.shape[0]; del _Xc
SIGI=torch.linalg.inv(SigU+0.1*float(torch.trace(SigU))/3584*torch.eye(3584,dtype=torch.float64)).float()
def prenorm(idx):
    x=torch.from_numpy(np.stack([np.asarray(H28[INFO[i][0],INFO[i][1]],dtype=np.float32) for i in idx]))
    h27=torch.from_numpy(np.stack([np.asarray(H27[INFO[i][0],INFO[i][1]],dtype=np.float32) for i in idx]))
    hh=x/gam; t=h27+Kt[idx]@W0.T
    s=(hh*t).sum(1)/(hh*hh).sum(1)
    return hh,s
def prenorm_s(idx):
    idx=np.asarray(idx)
    if len(idx)>300: idx=np.random.RandomState(0).choice(idx,300,replace=False)
    hh,s=prenorm(np.sort(idx)); return s.numpy()
HHh,Sh=prenorm(ho)
nxt_ho=torch.tensor([INFO[i][5] for i in ho]); okn=(nxt_ho>=0).numpy()
print('setup B done',K.shape,time.time()-t0)
