
import pandas as pd, numpy as np, re
from typing import Optional

# --- Load ---
p_g04 = "/mnt/data/2021Census_G04_NSW_SA2_full.csv"
p_g17 = "/mnt/data/2021Census_G17_NSW_SA2_full.csv"
p_g60 = "/mnt/data/2021Census_G60_NSW_SA2_full.csv"
p_g09 = "/mnt/data/2021Census_G09_NSW_SA2_full.csv"

g04 = pd.read_csv(p_g04); g17 = pd.read_csv(p_g17); g60 = pd.read_csv(p_g60); g09 = pd.read_csv(p_g09)

def norm_sa2_str(df, col="SA2_CODE_2021"):
    if col in df.columns:
        df[col] = df[col].astype(str).str.zfill(9)
    return df
g04 = norm_sa2_str(g04); g17 = norm_sa2_str(g17); g60 = norm_sa2_str(g60); g09 = norm_sa2_str(g09)
key = "SA2_CODE_2021"

# --- G04 tidy to 15+ Age×Sex ---
age_male_cols   = [c for c in g04.columns if re.match(r"^Age_yr_.*_M$", c)]
age_female_cols = [c for c in g04.columns if re.match(r"^Age_yr_.*_F$", c)]
def tidy_g04_age_sex(df):
    recs = []
    for c in age_male_cols + age_female_cols:
        sex = 'Male' if c.endswith('_M') else 'Female'
        age = c.replace('Age_yr_', '').replace('_M','').replace('_F','')
        recs.append(df[[key, c]].assign(age_band=age, sex=sex).rename(columns={c:'count'}))
    out = pd.concat(recs, ignore_index=True)
    return out[[key, 'age_band', 'sex', 'count']]
age_sex_all = tidy_g04_age_sex(g04)

AGE_BANDS_15P = [("15_19", 15, 19),("20_24", 20, 24),("25_34", 25, 34),("35_44", 35, 44),
                 ("45_54", 45, 54),("55_64", 55, 64),("65_74", 65, 74),("75_84", 75, 84),("85ov", 85, 200)]
def map_age_label_to_band(s):
    s=str(s)
    if s.isdigit():
        x=int(s)
        for lab,lo,hi in AGE_BANDS_15P:
            if lo<=x<=hi: return lab
        return None
    if "_" in s:
        parts=s.split("_"); 
        try: lo=int(parts[0])
        except: lo=None
        if lo is not None:
            for lab,lo2,hi2 in AGE_BANDS_15P:
                if lo2<=lo<=hi2: return lab
    if "over" in s or "ov" in s: return "85ov"
    return None

age_sex_15p = tidy_g04_age_sex(g04).copy()
age_sex_15p["age_band_15p"] = age_sex_15p["age_band"].map(map_age_label_to_band)
age_sex_15p = age_sex_15p.dropna(subset=["age_band_15p"]).drop(columns=["age_band"]).rename(columns={"age_band_15p":"age_band"})
age_sex_15p = (age_sex_15p.groupby([key, "age_band", "sex"], as_index=False)["count"].sum())

# --- G17 income collapse ---
inc_band_cols = [c for c in g17.columns if c.startswith('P_') and c.endswith('_Tot') and not c.startswith('P_Tot_')]
income_long = (g17[[key] + inc_band_cols].melt(id_vars=[key], var_name='income_band', value_name='count')[[key,'income_band','count']])
INCOME_COLLAPSE = [
    (r"^P_Neg_Nil_income_Tot$", "NEG_NIL"),
    (r"^P_1_149_Tot|P_150_299_Tot$", "1_299"),
    (r"^P_300_399_Tot|P_400_499_Tot|P_500_649_Tot$", "300_649"),
    (r"^P_650_799_Tot|P_800_999_Tot$", "650_999"),
    (r"^P_1000_1249_Tot|P_1250_1499_Tot|P_1500_1749_Tot|P_1750_1999_Tot$", "1000_1999"),
    (r"^P_2000_2999_Tot|P_3000_3499_Tot|P_3500_more_Tot$", "2000_plus"),
    (r"^P_PI_NS_ns_Tot$", "INC_NS"),
]
def collapse_income(income_long):
    def map_band(b):
        for pat,lab in INCOME_COLLAPSE:
            if re.match(pat,b): return lab
        return "OTHER"
    out=income_long.copy()
    out["income6"]=out["income_band"].map(map_band)
    out=(out.groupby([key,"income6"],as_index=False)["count"].sum()
           .rename(columns={"income6":"income_band"}))
    return out
income6 = collapse_income(income_long)

# --- G09 COB collapse Top-K ---
cob_tot_cols = [c for c in g09.columns if c.startswith('P_') and c.endswith('_Tot') and not c.startswith('P_Tot_')]
cob_long = (g09[[key] + cob_tot_cols].melt(id_vars=[key], var_name='cob_col', value_name='count')
            .assign(country=lambda d: d['cob_col'].str.replace(r'^P_|_Tot$', '', regex=True))[[key,'country','count']])
def collapse_cob_topk(cob_long, k=10):
    cob=cob_long.copy(); special={"Elsewhere","COB_NS"}
    cob["_rank"]=cob.groupby(key)["count"].rank(ascending=False, method="first")
    cob["country_collapsed"]=np.where((cob["_rank"]<=k) | (cob["country"].isin(special)), cob["country"], "Elsewhere")
    cob2=(cob.groupby([key,"country_collapsed"],as_index=False)["count"].sum()
            .rename(columns={"country_collapsed":"country"}))
    return cob2
cob_top = collapse_cob_topk(cob_long, k=10)

# --- Constraint builders ---
def build_age_sex_constraint(sa2, age_sex_long):
    order=["15_19","20_24","25_34","35_44","45_54","55_64","65_74","75_84","85ov"]
    df=age_sex_long[age_sex_long[key]==sa2].copy()
    A=df.pivot(index="age_band", columns="sex", values="count").fillna(0)
    for s in ["Male","Female"]:
        if s not in A.columns: A[s]=0
    return A.reindex(order).fillna(0)[["Male","Female"]]

def build_income_constraint(sa2, income_any):
    return (income_any[income_any[key]==sa2].set_index("income_band")["count"].astype(float).sort_index())

def build_cob_constraint_for_universe(sa2, cob_any, A_agesex_15p):
    C_all=(cob_any[cob_any[key]==sa2].set_index("country")["count"].astype(float).sort_index())
    if C_all.empty: return None
    total_all=float(C_all.sum()); total_15p=float(A_agesex_15p.values.sum())
    if total_all<=0 or total_15p<=0: return None
    return (C_all/total_all)*total_15p

# --- Seed, IPF, integerise, generate ---
def seed_joint(A,I,C):
    X=A.values.astype(float); ages=list(A.index); sexes=list(A.columns); incs=[]; cobs=[]
    if I is not None:
        incs=list(I.index); X = X[:,:,None] * I.values[None,None,:]
    if C is not None:
        cobs=list(C.index)
        if X.ndim==2: X=X[:,:,None]
        if X.ndim==3: X=X[:,:,:,None]
        X = X * (C.values[None,None,None,:])
    X = X * (A.values.sum() / (X.sum() + 1e-12))
    return X, ages, sexes, incs, cobs

def ipf_match_with_log(X, A=None, I=None, C=None, tol=1e-5, max_iter=200, ridge=0.05):
    eps=1e-12; logs=[]
    for it in range(max_iter):
        errs={}
        if A is not None:
            cur = X.sum(axis=tuple(range(2,X.ndim)))
            alpha=(A+ridge)/(cur+ridge+eps); X = X * alpha[(slice(None),slice(None)) + (None,)*(X.ndim-2)]
            errs["AS_max_relerr"]=float(np.max(np.abs(cur-A)/(np.maximum(A,eps))))
        if I is not None and X.ndim>=3:
            cur = X.sum(axis=(0,1) + tuple(range(3,X.ndim)))
            beta=(I+ridge)/(cur+ridge+eps); X = X * beta[(None,None,slice(None)) + (None,)*(X.ndim-3)]
            errs["I_max_relerr"]=float(np.max(np.abs(cur-I)/(np.maximum(I,eps))))
        if C is not None and X.ndim>=4:
            cur = X.sum(axis=(0,1,2))
            delta=(C+ridge)/(cur+ridge+eps); X = X * delta[(None,None,None,slice(None))]
            errs["C_max_relerr"]=float(np.max(np.abs(cur-C)/(np.maximum(C,eps))))
        if errs:
            errs["iter"]=it; errs["total"]=float(X.sum()); logs.append(errs)
            if max(v for k,v in errs.items() if k.endswith("relerr")) < tol: break
    return X, pd.DataFrame(logs)

def integerise_TRS(X, rng=None):
    if rng is None: rng=np.random.default_rng(11)
    base=np.floor(X).astype(int); need=int(round(X.sum()-base.sum()))
    if need<=0: return base
    frac=(X-base).ravel(); s=frac.sum()
    if s<=0: return base
    p=frac/s; idx=rng.choice(np.arange(frac.size), size=need, replace=False, p=p)
    out=base.ravel(); out[idx]+=1
    return out.reshape(X.shape)

def generate_people(X_int, ages, sexes, incs, cobs, sa2_name):
    recs=[]
    if X_int.ndim==4:
        for ia,a in enumerate(ages):
            for isx,s in enumerate(sexes):
                for ii,i in enumerate(incs):
                    for ic,c in enumerate(cobs):
                        recs += [[sa2_name,a,s,i,c]] * int(X_int[ia,isx,ii,ic])
    elif X_int.ndim==3:
        for ia,a in enumerate(ages):
            for isx,s in enumerate(sexes):
                for ii,i in enumerate(incs):
                    recs += [[sa2_name,a,s,i,None]] * int(X_int[ia,isx,ii])
    else:
        for ia,a in enumerate(ages):
            for isx,s in enumerate(sexes):
                recs += [[sa2_name,a,s,None,None]] * int(X_int[ia,isx])
    people = pd.DataFrame(recs, columns=["SA2_NAME_2021","age_band","sex","income_band","country"])
    people.insert(0,"person_id",[f"P{str(i).zfill(8)}" for i in range(len(people))])
    return people

def validate(people, A, I=None, C=None):
    reps=[]
    curA=(people.groupby(["age_band","sex"]).size().unstack(fill_value=0))
    curA=curA.reindex(index=A.index, columns=A.columns, fill_value=0)
    rA=(curA-A).stack().rename("abs_err").reset_index(); rA["rel_err"]=np.where(A.stack().values==0,0,np.abs(rA["abs_err"])/A.stack().values)
    rA["constraint"]="Age×Sex"; reps.append(rA)
    if I is not None and len(I)>0:
        curI=people.groupby("income_band").size().reindex(I.index, fill_value=0)
        rI=pd.DataFrame({"income_band":I.index,"abs_err":(curI-I).values,"rel_err":np.where(I.values==0,0,np.abs(curI-I)/I.values),"constraint":"Income"}); reps.append(rI)
    if C is not None and len(C)>0 and "country" in people.columns:
        curC=people.groupby("country").size().reindex(C.index, fill_value=0)
        rC=pd.DataFrame({"country":C.index,"abs_err":(curC-C).values,"rel_err":np.where(C.values==0,0,np.abs(curC-C)/C.values),"constraint":"Country"}); reps.append(rC)
    return pd.concat(reps, ignore_index=True)

# --- Choose SA2 & run ---
sa2_code="101021008"; sa2_name="Karabar"
A = build_age_sex_constraint(sa2_code, age_sex_15p)
I = build_income_constraint(sa2_code, income6)
C = build_cob_constraint_for_universe(sa2_code, cob_top, A)
# Rescale income to match A total (composition)
I_scaled = (I / float(I.sum())) * float(A.values.sum())

X0, ages, sexes, incs, cobs = seed_joint(A, I_scaled, C)
Xf, ipf_log = ipf_match_with_log(X0.copy(), A=A.values.astype(float), I=I_scaled.values.astype(float), C=C.values.astype(float))
X_int = integerise_TRS(Xf)
people = generate_people(X_int, ages, sexes, incs, cobs, sa2_name)
report = validate(people, A, I_scaled, C)

people.to_csv("/mnt/data/synth_people_101021008.py_out.csv", index=False)
report.to_csv("/mnt/data/validation_101021008.py_out.csv", index=False)
ipf_log.to_csv("/mnt/data/ipf_log_101021008.py_out.csv", index=False)

print("Done. Rows:", len(people))
