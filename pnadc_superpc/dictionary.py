import pandas as pd

from .config import preferred_raw_path


def load_dictionary():
    path = preferred_raw_path("dictionary")
    return pd.read_csv(path, dtype=str)


def build_label_map(tables):
    dic = load_dictionary()
    dic = dic[dic["id_tabela"].isin(tables)].copy()
    dic = dic[dic["chave"].notna() & (dic["chave"].str.strip() != "")].copy()
    dic["col_lower"] = dic["nome_coluna"].astype(str).str.strip().str.lower()
    dic["chave_num"] = pd.to_numeric(dic["chave"].astype(str).str.strip(), errors="coerce")

    label_map = {}
    for col_lower, grp in dic.groupby("col_lower"):
        mapping = {}
        for _, row in grp.iterrows():
            if pd.isna(row["chave_num"]):
                continue
            mapping[int(row["chave_num"])] = str(row["valor"]).strip()
        if mapping:
            label_map[col_lower] = mapping
    return label_map


def apply_label_columns(df, label_map):
    col_lookup = {column.lower(): column for column in df.columns}
    for col_lower, mapping in label_map.items():
        actual = col_lookup.get(col_lower)
        if actual is None:
            continue
        df[actual + "_label"] = (
            pd.to_numeric(df[actual], errors="coerce").map(mapping).astype("category")
        )
    return df

