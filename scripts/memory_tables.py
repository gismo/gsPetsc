# Turns the CSV lines of mpi_memory_profile (--csv) into markdown tables of
# the maximum per-rank memory (MiB). Usage: python3 memory_tables.py sweep.log
import csv, sys, collections
rows=[]
for line in open(sys.argv[1]):
    if line.startswith("CSV,"):
        f=line.strip().split(",")
        k=1
        while k<len(f) and f[k].lstrip('-').isdigit(): k+=1
        hdr=f[1:k]; P,d,p,N,npch,geo=hdr[:6]; nr=hdr[6] if len(hdr)>6 else '0'
        var='block'
        if k<len(f) and f[k].startswith('v='): var=f[k][2:]; k+=1
        rest=[",".join(f[k:-4])]+f[-4:]
        rows.append(dict(var=var,P=int(P),d=int(d),p=int(p),N=int(N),patches=int(npch),geo=int(geo),nr=int(nr),
                         name=rest[0].strip(),kind=rest[1],mn=int(rest[2]),mx=int(rest[3]),sm=int(rest[4])))
MB=1024*1024
def table(filt, key, title):
    sel=[r for r in rows if filt(r)]
    keys=sorted(set(key(r) for r in sel))
    names=[]
    for r in sel:
        if r['name'] not in names: names.append(r['name'])
    print("\n## "+title)
    print("| object / stage | "+" | ".join(str(k) for k in keys)+" |")
    print("|---|"+"---:|"*len(keys))
    for n in names:
        vals=[]
        for k in keys:
            v=[r['mx'] for r in sel if r['name']==n and key(r)==k]
            vals.append("%.1f"%(v[0]/MB) if v else "")
        print("| "+n+" | "+" | ".join(vals)+" |")
mode=sys.argv[2] if len(sys.argv)>2 else "all"
if mode=="variants":
    for d,lo,hi,t in [(2,1e6,2e6,"2D, p=2, N~1.05M"),(3,2e5,4e5,"3D, p=2, N~275k")]:
        table(lambda r:r['d']==d and lo<r['N']<hi and r['P']==4, lambda r:r['var'], t+", P=4, max MiB per rank by variant")
    for v in sorted(set(r['var'] for r in rows if r['var']!='block')):
        sel=[r for r in rows if r['var']==v and r['d']==2 and 1e6<r['N']<2e6]
        if len(set(r['P'] for r in sel))>2:
            table(lambda r,v=v:r['var']==v and r['d']==2 and 1e6<r['N']<2e6, lambda r:r['P'], "2D N~1.05M, "+v+", max MiB per rank vs P")
    sys.exit(0)
table(lambda r:r['d']==2 and r['N']>1e6 and r['N']<2e6 and r['patches']==4 and not r['geo'] and r['p']==2 and not r['nr'], lambda r:r['P'], "2D, p=2, N~1.05M, max MiB per rank vs P")
table(lambda r:r['d']==3 and r['N']>2e5 and r['N']<4e5 and not r['geo'] and r['p']==2 and not r['nr'], lambda r:r['P'], "3D, p=2, N~275k, max MiB per rank vs P")
table(lambda r:r['d']==2 and r['P']==4 and r['patches']==4 and not r['geo'] and r['p']==2 and not r['nr'], lambda r:r['N'], "2D, p=2, P=4, max MiB per rank vs N")
table(lambda r:r['P']==4 and (r['geo'] or r['patches']>4 or r['p']>2 or r['nr']), lambda r:"%dD p%d N=%d np=%d%s%s"%(r['d'],r['p'],r['N'],r['patches']," geo" if r['geo'] else ""," noreserve" if r['nr'] else ""), "Variants, P=4, max MiB per rank")
