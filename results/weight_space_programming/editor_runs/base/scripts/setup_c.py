# Setup part C (re-runnable): decision classes for the template edit (valued + protected), joint keys, value builders, assembly, offline check.
PREL={};SUFL={}
for li in range(26):
    lns=[L[li] for L in LINES3 if len(L)>li]
    p=0
    while True:
        c=Counter(l[p][1] if len(l)>p else None for l in lns).most_common(1)[0]
        if c[0] is None or c[1]<0.97*len(lns): break
        p+=1
        if p>=min(len(l) for l in lns): break
    q=0
    while True:
        c=Counter(l[-1-q][1] if len(l)>q else None for l in lns).most_common(1)[0]
        if c[0] is None or c[1]<0.97*len(lns): break
        q+=1
        if q>=min(len(l) for l in lns): break
    full=(p>=min(len(l) for l in lns))
    PREL[li]=p; SUFL[li]=0 if full else q
def build_classes():
    CL={}; PROT=defaultdict(list)
    def add(name,i,tid):
        c=CL.setdefault(name,dict(mem=[],kind='single',tid=tid,tids=Counter())); c['mem'].append(i); c['tids'][tid]+=1
    def addm(name,i):
        c=CL.setdefault(name,dict(mem=[],kind='multi')); c['mem'].append(i)
    for r in range(200):
        L=LINES3[r]
        for li in range(min(26,len(L))):
            l=L[li]
            for ti in range(min(PREL[li],len(l))):
                j,s=l[ti]; add(('h',li,'p',ti,s),pos2idx[(r,j-1)],int(T3[r,j]))
            for q in range(SUFL[li]):
                if len(l)>PREL[li]+q:
                    j,s=l[-1-q]
                    if li in (3,4,5,6,7,8,12): PROT[('prot','h',li,'s',q)].append(pos2idx[(r,j-1)])
                    else: add(('h',li,'s',q,s),pos2idx[(r,j-1)],int(T3[r,j]))
            var=l[PREL[li]:len(l)-SUFL[li]]
            if li in (3,4,5,6,7,8,12):
                strs=[s for _,s in var]
                if '.' in strs:
                    dp=strs.index('.')
                    for q,(j,s) in enumerate(var): PROT[('pv',li,max(q-dp,-2))].append(pos2idx[(r,j-1)])
            else:
                for q,(j,s) in enumerate(var): PROT[('pv',li,min(q,3))].append(pos2idx[(r,j-1)])
        for li in range(26,len(L)):
            l=L[li]
            if not (l[-1][1]=='\n' or l[-1][1]=='<|im_end|>') or len(l)<50: continue
            strs=[s for _,s in l]
            if strs[0]!=' ': continue
            m=strs.index(' ',1)-1
            if m<1 or m>3: continue
            if li==26: add(('a','start0'),pos2idx[(r,l[0][0]-1)],int(T3[r,l[0][0]]))
            else: addm(('a','startn'),pos2idx[(r,l[0][0]-1)])
            PROT[('pa','sym_end')].append(pos2idx[(r,l[m][0])])
            PROT[('pa','el')].append(pos2idx[(r,l[0][0])])
            PROT[('pa','lab')].append(pos2idx[(r,l[m+1][0])])
            for q in range(1,m): PROT[('pa','symmid')].append(pos2idx[(r,l[q][0]-1)])
            for e in (44,43,42,41,40,3,2,1):
                j,s=l[-1-e]; add(('a','e',e,s),pos2idx[(r,j-1)],int(T3[r,j]))
            for e in (39,38,29,28,27,26,17,16,15,14,5,4):
                j,s=l[-1-e]; PROT[('prot','a','e',e)].append(pos2idx[(r,j-1)])
            lab0=(li==26 and l[-46][1]=='0')
            if lab0: add(('a','label0'),pos2idx[(r,l[-46][0]-1)],int(T3[r,l[-46][0]]))
            for e in list(range(6,14))+list(range(18,26))+list(range(30,38))+[45,46,47,48]:
                if e<len(l)-1 and not (lab0 and e==45): PROT[('pa',e)].append(pos2idx[(r,l[-1-e][0]-1)])
            je=l[-1][0]
            if l[-1][1]=='<|im_end|>': addm(('a','stop'),pos2idx[(r,je-1)])
            else: addm(('a','e0n'),pos2idx[(r,je-1)])
    for k in [k for k,c in CL.items() if len(c['mem'])<60]: del CL[k]
    for k,c in CL.items():
        c['mem']=np.asarray(c['mem']); c['fit']=c['mem'][fit[c['mem']]]; c['ho']=c['mem'][~fit[c['mem']]]
        if c['kind']=='single': c['tid']=c['tids'].most_common(1)[0][0]
    PROT={k:np.asarray(v) for k,v in PROT.items() if fit[np.asarray(v)].sum()>=40}
    return CL,PROT
def solve_keys(CL,PROT,extra_means=None):
    NV=list(CL); NP=list(PROT)
    rows=[Kt[CL[k]['fit']].mean(0) for k in NV]+[Kt[PROT[k][fit[PROT[k]]]].mean(0) for k in NP]
    KBA=torch.stack(rows).double()
    CiK=torch.cholesky_solve(KBA.T,Lc); Mg=KBA@CiK
    WJ=torch.linalg.solve(Mg+1e-6*Mg.diag().mean()*torch.eye(Mg.shape[0],dtype=torch.float64),CiK.T)
    for ci,k in enumerate(NV): CL[k]['w']=WJ[ci]
    return NV,NP,WJ,float(torch.linalg.cond(Mg))
NNC={}
def nn_tokens(tid,n=6):
    if tid not in NNC:
        sim=torch.nn.functional.cosine_similarity(Ug,Ug[tid][None],dim=1); NNC[tid]=[int(t) for t in sim.topk(n+1).indices[1:]]
    return NNC[tid]
def multi_val2(c,targets,s=None):
    ids=list(targets); A=Ug[ids]
    if s is None: s=float(np.percentile(prenorm_s(c['fit']),85))
    SA=SIGI@A.T; G=(A@SA).double(); a=torch.linalg.solve(G+1e-3*G.diag().mean()*torch.eye(len(ids),dtype=torch.float64),torch.tensor([targets[i] for i in ids],dtype=torch.float64)*s)
    c.update(s=s,vec=(SA.double()@a),targets=targets)
def single_val3(c,margin):
    mem=c['fit']; tid=c['tid']
    if len(mem)>200: mem=np.random.RandomState(0).choice(mem,200,replace=False)
    comp=Counter(); gaps=[]
    for m in mem:
        r,j=INFO[m][0],INFO[m][1]; best=-99
        for t,l in zip(TK3[r,j],TL3[r,j]):
            if t!=tid: comp[int(t)]+=float(np.exp(l)); best=max(best,l)
        gaps.append(best-LP3[r,j+1])
    comps=[t for t,_ in comp.most_common(8)]; gap=float(np.percentile(gaps,98)); beta=max(gap,0)+margin
    tg={tid:beta}
    for t in comps+nn_tokens(tid): tg.setdefault(t,0.0)
    multi_val2(c,tg); c.update(gap=gap,beta=beta)
def assemble(CL,names,scale={}):
    D=torch.zeros(3584,18944,dtype=torch.float64)
    for k in names: D+=scale.get(k,1.0)*torch.outer(CL[k]['vec'],CL[k]['w'])
    return D
def offline_sub(Dq,sub):
    DH=(Kt[ho[sub]].double()@Dq.T).float(); out=[]
    for s0 in range(0,len(sub),3600):
        ii=sub[s0:s0+3600]; hp=HHh[ii]*Sh[ii,None]
        l0=lens_pre(hp); l1=lens_pre(hp+DH[s0:s0+3600]); n=nxt_ho[ii].clamp(min=0)
        out.append(torch.stack([l0.gather(1,n[:,None])[:,0],l1.gather(1,n[:,None])[:,0],(l0.argmax(1)==n).float(),(l1.argmax(1)==n).float(),(l0.argmax(1)==l1.argmax(1)).float(),(l1.exp()*(l1-l0)).sum(1)],1))
    return torch.cat(out).numpy()
def percls_check(CL,Dq,names,nper=12,nother=2500,seed=3):
    rs_=np.random.RandomState(seed); pick={}; allv=set()
    for k in names:
        hq=np.array([ho_pos[i] for i in CL[k]['ho']],dtype=int); allv.update(hq.tolist())
        pick[k]=rs_.choice(hq,min(nper,len(hq)),replace=False) if len(hq) else hq
    oq=np.array([q for q in range(len(ho)) if q not in allv and okn[q]]); oq=np.sort(rs_.choice(oq,nother,replace=False))
    sub=np.unique(np.concatenate([oq]+[pick[k] for k in names])).astype(int); O=offline_sub(Dq,sub); loc={q:a for a,q in enumerate(sub)}
    return pick,oq,O,loc
def report_check(CL,Dq,names):
    pick,oq,O,loc=percls_check(CL,Dq,names)
    bad=[]
    for k in names:
        if CL[k]['kind']!='single' or len(pick[k])==0: continue
        a=np.array([loc[q] for q in pick[k]])
        if O[a,1].min()<-0.3 or O[a,3].mean()<1: bad.append((k,round(float(O[a,1].mean()),3),round(float(O[a,1].min()),2),float(O[a,3].mean())))
    print('weak single classes:',bad)
    a=np.array([loc[q] for k in names if CL[k]['kind']=='single' for q in pick[k]]); print('all single: after mean %.4f min %.2f top1 %.4f'%(O[a,1].mean(),O[a,1].min(),O[a,3].mean()))
    a=np.array([loc[q] for q in oq]); print('other n=%d: ref lp before %.4f after %.4f top1 unchanged %.4f KL mean %.5f max %.3f n(KL>.1)=%d'%(len(a),O[a,0].mean(),O[a,1].mean(),O[a,4].mean(),O[a,5].mean(),O[a,5].max(),(O[a,5]>0.1).sum()))
    for nm in names:
        if CL[nm]['kind']=='multi' and len(pick[nm]):
            a=np.array([loc[q] for q in pick[nm]]); print(nm,'n=%d ref lp before %.2f after %.3f min %.2f top1 after %.3f'%(len(a),O[a,0].mean(),O[a,1].mean(),O[a,1].min(),O[a,3].mean()))
def e3_values(CL,margin=10.0):
    for k in CL:
        if CL[k]['kind']=='single': single_val3(CL[k],margin)
    multi_val2(CL[('a','e0n')],{**{d:-10.0 for d in DIG},NLNL:-10.0,NL:0.0,IM_END:0.0,SP:0.0})
    multi_val2(CL[('a','stop')],{IM_END:11.0,NL:0.0,SP:0.0,NLNL:-10.0,**{d:-10.0 for d in DIG}})
    multi_val2(CL[('a','startn')],{IM_END:8.0,SP:0.0,NL:-10.0,NLNL:-10.0,FENCE[0]:-10.0,FENCE[1]:-10.0})
print('setup C loaded')
