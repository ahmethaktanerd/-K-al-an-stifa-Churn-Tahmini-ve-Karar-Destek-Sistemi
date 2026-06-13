import pandas as pd

df = pd.read_csv('final-project/data/raw/veri_seti.csv')
print("Years available:", sorted(df['STATUS_YEAR'].unique()))
print("\nNumber of records per year:")
print(df['STATUS_YEAR'].value_counts().sort_index())
print("\nUnique employees:", df['EmployeeID'].nunique())
print("\nExample of terminated employee (e.g. EmployeeID 1338):")
print(df[df['EmployeeID'] == 1338][['EmployeeID', 'STATUS_YEAR', 'age', 'length_of_service', 'STATUS', 'termreason_desc']])
