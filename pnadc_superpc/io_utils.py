import pandas as pd


def actual_rename_map(df_columns, rename_map):
    lookup = {column.lower(): column for column in df_columns}
    actual = {}
    for original, readable in rename_map.items():
        raw_name = lookup.get(original.lower())
        if raw_name:
            actual[raw_name] = readable
    return actual


def rename_with_labels(df, rename_map):
    actual = actual_rename_map(df.columns, rename_map)
    label_rename = {}
    for raw_name, readable in actual.items():
        label_name = raw_name + "_label"
        if label_name in df.columns:
            label_rename[label_name] = readable + "_label"
    return df.rename(columns={**actual, **label_rename}, errors="ignore")


def downsample_individuals(df, sample_frac, seed, panel_cols=None):
    if not sample_frac or sample_frac >= 1:
        return df.copy()

    panel_cols = panel_cols or ["id_domicilio", "num_ordem"]
    unique_ids = df[panel_cols].drop_duplicates()
    n_sample = max(1, int(round(len(unique_ids) * sample_frac)))
    sampled = unique_ids.sample(n=n_sample, random_state=seed)
    return df.merge(sampled, on=panel_cols, how="inner")


def build_label_maps_from_frame(df):
    label_maps = {}
    for column in df.columns:
        if not column.endswith("_label"):
            continue
        base = column[:-6]
        if base not in df.columns:
            continue
        pairs = df[[base, column]].dropna().drop_duplicates()
        mapping = {}
        for _, row in pairs.iterrows():
            try:
                mapping[float(row[base])] = str(row[column])
            except (TypeError, ValueError):
                continue
        if mapping:
            label_maps[base] = mapping
    return label_maps


def decode_series(series, column_name, label_maps):
    base = column_name
    for suffix in ("_t1", "_t2", "_f1"):
        if column_name.endswith(suffix):
            base = column_name[: -len(suffix)]
            break

    mapping = label_maps.get(base, {})
    if not mapping:
        return series

    numeric = pd.to_numeric(series, errors="coerce")
    mapped = numeric.map(mapping)
    return mapped.where(mapped.notna(), series.astype(str).replace("nan", pd.NA))


def best_column(df, column):
    label_column = column + "_label"
    return label_column if label_column in df.columns else column
