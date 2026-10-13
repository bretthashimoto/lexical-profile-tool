import pandas as pd

CSV_PATH = r"C:\Users\bretthas\PycharmProjects\Lexical profile tool\COCA data.csv"

df = pd.read_csv(CSV_PATH)
for row in df.itertuples():
    # if row.word.endswith("_N"):
    #     print(row)
    if "their" in row.word:
        print(row)
