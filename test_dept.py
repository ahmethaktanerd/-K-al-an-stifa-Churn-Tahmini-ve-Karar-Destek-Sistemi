import pandas as pd

df = pd.read_csv("data/raw/veri_seti.csv")
dept_jobs = df.groupby('department_name')['job_title'].unique().apply(list).to_dict()

for k, v in dept_jobs.items():
    print(f"{k}: {len(v)} jobs -> {v[:3]}")

