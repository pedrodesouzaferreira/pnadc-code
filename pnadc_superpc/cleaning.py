import numpy as np
import pandas as pd
from pathlib import Path

from .config import CLEANED_DIR, RAW_DIR, cleaned_path, ensure_dir, preferred_raw_path
from .dictionary import apply_label_columns, build_label_map
from .io_utils import rename_with_labels
from .mappings import EDUCACAO_RENAME_MAP, HOURS_COLUMNS, INCOME_COLUMNS, MICRO_RENAME_MAP


DEFAULT_SUPERIOR_VALUES = (7,)
DTA_RENAME_MAP = {
    "servidor_publico_estatutario_label": "servidor_pub_estat_label",
    "contribui_prev_trab_principal_label": "contrib_prev_princ_label",
    "servidor_publico_secundario_label": "servidor_pub_sec_label",
    "carteira_assinada_secundario_label": "carteira_ass_sec_label",
    "horas_habituais_todos_faixa_label": "horas_hab_todos_faixa_lbl",
    "motivo_fora_forca_trabalho2_label": "motivo_fora_forca2_label",
    "horas_habituais_principal_faixa_label": "horas_hab_princ_faixa_lbl",
    "horas_efetivas_principal_faixa_label": "horas_efet_princ_faixa_lbl",
    "log_renda_habitual_principal_winsor": "log_renda_hab_princ_winsor",
    "curso_mais_elevado_anterior_label": "curso_mais_elev_ant_label",
    "ultimo_ano_concluido_anterior_label": "ult_ano_concl_ant_label",
}
EDUCACAO_FROM_MICRODADOS_EXTRA_COLUMNS = [
    "ano",
    "trimestre",
    "id_uf",
    "sigla_uf",
    "capital",
    "rm_ride",
    "id_upa",
    "id_estrato",
    "id_domicilio",
    "id_pessoa",
    "V1008",
    "V1014",
    "V1016",
    "V1027",
    "V1029",
    "V1033",
    "posest_sxi",
    "VD3004",
]


def parse_flag(series, code_val, label_substring):
    numeric = pd.to_numeric(series, errors="coerce")
    by_code = numeric == code_val
    by_label = series.astype(str).str.contains(label_substring, case=False, na=False)
    return (by_code | by_label).astype("Int8")


def read_raw(dataset, year, use_sample=False):
    path = preferred_raw_path(dataset, year=year, use_sample=use_sample)
    return pd.read_csv(path, low_memory=False), path


def read_microdados_sample(input_path=None, year=None):
    path = Path(input_path).expanduser() if input_path else RAW_DIR / "PNADC_microdados_sample.csv"
    df = pd.read_csv(path, low_memory=False)
    if year is not None:
        if "ano" not in df.columns:
            raise ValueError(f"Cannot filter by year because {path} has no 'ano' column.")
        df = df.loc[pd.to_numeric(df["ano"], errors="coerce").eq(year)].copy()
    return df, path


def read_vd3004_raw(dataset, year, vd3004_value=DEFAULT_SUPERIOR_VALUES[0], input_path=None):
    path = Path(input_path).expanduser() if input_path else RAW_DIR / f"PNADC_{dataset}_VD3004_{vd3004_value}_{year}.csv"
    df = pd.read_csv(path, low_memory=False)
    if "ano" in df.columns:
        df = df.loc[pd.to_numeric(df["ano"], errors="coerce").eq(year)].copy()
    return df, path


def filter_vd3004(df, values=DEFAULT_SUPERIOR_VALUES):
    values = [int(value) for value in values]
    column = "VD3004" if "VD3004" in df.columns else "nivel_instrucao"
    if column not in df.columns:
        raise ValueError("Cannot filter by education because neither 'VD3004' nor 'nivel_instrucao' is present.")
    mask = pd.to_numeric(df[column], errors="coerce").isin(values)
    return df.loc[mask].copy()


def vd3004_output_tag(values=DEFAULT_SUPERIOR_VALUES):
    return "_".join(str(int(value)) for value in values)


def superior_sample_cleaned_path(dataset, values=DEFAULT_SUPERIOR_VALUES, year=None):
    tag = vd3004_output_tag(values)
    year_part = f"_{year}" if year is not None else ""
    if dataset == "microdados":
        return CLEANED_DIR / f"PNADC_limpo_VD3004_{tag}_sample{year_part}.csv"
    if dataset == "educacao":
        return CLEANED_DIR / f"PNADC_educacao_limpo_VD3004_{tag}_sample{year_part}.csv"
    raise ValueError(f"Unknown dataset: {dataset}")


def vd3004_cleaned_path(dataset, year, vd3004_value=DEFAULT_SUPERIOR_VALUES[0]):
    if dataset == "microdados":
        return CLEANED_DIR / f"PNADC_limpo_VD3004_{vd3004_value}_{year}.csv"
    if dataset == "educacao":
        return CLEANED_DIR / f"PNADC_educacao_limpo_VD3004_{vd3004_value}_{year}.csv"
    raise ValueError(f"Unknown dataset: {dataset}")


def prepare_for_stata(df):
    rename_map = {old: new for old, new in DTA_RENAME_MAP.items() if old in df.columns}
    out = df.rename(columns=rename_map).copy()
    too_long = [column for column in out.columns if len(column) > 32]
    if too_long:
        raise ValueError(f"These columns are too long for Stata variable names: {too_long}")

    variable_labels = {new: old for old, new in rename_map.items()}
    for column in out.columns:
        series = out[column]
        if pd.api.types.is_categorical_dtype(series):
            out[column] = series.astype(object).where(series.notna(), "")
        elif str(series.dtype).startswith(("Int", "UInt")):
            out[column] = series.astype("float64") if series.isna().any() else series.astype("int64")
        elif str(series.dtype) == "boolean":
            out[column] = series.astype("float64")
        elif pd.api.types.is_object_dtype(series):
            out[column] = series.astype(object).where(series.notna(), "")
    return out, variable_labels


def write_cleaned_output(df, output_path, output_format="csv"):
    output_format = output_format.lower()
    if output_format not in {"csv", "dta", "both"}:
        raise ValueError("output_format must be one of: csv, dta, both")

    written = []
    if output_format in {"csv", "both"}:
        df.to_csv(output_path, index=False)
        written.append(output_path)
    if output_format in {"dta", "both"}:
        dta_path = output_path.with_suffix(".dta")
        stata_df, variable_labels = prepare_for_stata(df)
        stata_df.to_stata(
            dta_path,
            write_index=False,
            version=118,
            variable_labels=variable_labels,
        )
        written.append(dta_path)
    return written


def keep_educacao_columns_from_microdados(df):
    raw_columns = EDUCACAO_FROM_MICRODADOS_EXTRA_COLUMNS + list(EDUCACAO_RENAME_MAP)
    columns = []
    for column in raw_columns:
        if column in df.columns and column not in columns:
            columns.append(column)
    return df.loc[:, columns].copy()


def coerce_numeric(df, columns):
    for column in columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")


def add_year_winsorized_column(df, source_col, output_col, lower_q=0.01, upper_q=0.99):
    if source_col not in df.columns:
        return df

    out = df.copy()
    out[source_col] = pd.to_numeric(out[source_col], errors="coerce")
    year_col = pd.to_numeric(out["ano"], errors="coerce") if "ano" in out.columns else pd.Series(np.nan, index=out.index)

    if year_col.notna().any():
        bounds = (
            pd.DataFrame({"ano": year_col, source_col: out[source_col]})
            .dropna(subset=["ano"])
            .groupby("ano")[source_col]
            .quantile([lower_q, upper_q])
            .unstack()
            .rename(columns={lower_q: "_winsor_low", upper_q: "_winsor_high"})
        )
        out = out.join(bounds, on="ano")
        out[output_col] = out[source_col].clip(lower=out["_winsor_low"], upper=out["_winsor_high"])
        out = out.drop(columns=["_winsor_low", "_winsor_high"])
    else:
        low = out[source_col].quantile(lower_q)
        high = out[source_col].quantile(upper_q)
        out[output_col] = out[source_col].clip(lower=low, upper=high)

    return out


def derive_microdados(df):
    coerce_numeric(df, INCOME_COLUMNS + HOURS_COLUMNS + ["idade", "peso", "anos_estudo", "num_entrevista"])

    if "condicao_ocupacao" in df.columns:
        df["ocupado"] = parse_flag(df["condicao_ocupacao"], 1, "ocupado")
        df["desocupado"] = parse_flag(df["condicao_ocupacao"], 2, "desocupado")

    if "condicao_forca_trabalho" in df.columns:
        df["na_pea"] = parse_flag(df["condicao_forca_trabalho"], 1, "forca de trabalho")

    if "servidor_publico_estatutario" in df.columns:
        df["servidor_publico"] = parse_flag(df["servidor_publico_estatutario"], 1, "sim")

    if "metodo_busca_emprego" in df.columns:
        numeric = pd.to_numeric(df["metodo_busca_emprego"], errors="coerce")
        flag = numeric == 5
        label_column = "metodo_busca_emprego_label"
        if label_column in df.columns:
            flag = flag | df[label_column].astype(str).str.contains("concurso", case=False, na=False)
        df["buscando_via_concurso"] = flag.astype("Int8")

    if "metodo_busca_emprego_v1" in df.columns:
        label_column = "metodo_busca_emprego_v1_label"
        if label_column in df.columns:
            flag_old = df[label_column].astype(str).str.contains("concurso", case=False, na=False)
        else:
            flag_old = pd.to_numeric(df["metodo_busca_emprego_v1"], errors="coerce").isin([5])
        if "buscando_via_concurso" in df.columns:
            df["buscando_via_concurso"] = (
                df["buscando_via_concurso"].astype(bool) | flag_old
            ).astype("Int8")
        else:
            df["buscando_via_concurso"] = flag_old.astype("Int8")

    if "posicao_emprego" in df.columns:
        pos = pd.to_numeric(df["posicao_emprego"], errors="coerce")
        # VD4009: 1=priv+cart, 2=priv-cart, 3=dom+cart, 4=dom-cart,
        #         5=pub+cart, 6=pub-cart, 7=militar/estatutário, 8=empregador, 9=conta-própria, 10=aux-familiar
        df["formal"] = pos.isin([1, 3, 5, 7]).astype("Int8")
        df["informal"] = pos.isin([2, 4, 6, 8, 9, 10]).astype("Int8")
        df["empregado_setor_priv"] = pos.isin([1, 2]).astype("Int8")
        df["empregado_setor_pub"] = pos.isin([5, 6, 7]).astype("Int8")
        df["conta_propria"] = (pos == 9).astype("Int8")
        df["empregador"] = (pos == 8).astype("Int8")
        df["trab_domestico"] = pos.isin([3, 4]).astype("Int8")
        df["trab_familiar_aux"] = (pos == 10).astype("Int8")

    if "idade" in df.columns:
        bins = [0, 14, 24, 34, 44, 54, 64, np.inf]
        labels = ["0-14", "15-24", "25-34", "35-44", "45-54", "55-64", "65+"]
        df["faixa_etaria"] = pd.cut(df["idade"], bins=bins, labels=labels, right=False)
        df["idade_trabalho"] = (df["idade"] >= 14).astype("Int8")

    income_column = next(
        (
            column
            for column in ["renda_habitual_principal", "renda_habitual_todos", "renda_habitual_total"]
            if column in df.columns and df[column].notna().any()
        ),
        None,
    )
    if income_column:
        minimum_wage = 1412
        positive = df[income_column].where(df[income_column] > 0)
        df["log_renda"] = np.log(positive)
        bins = [0, minimum_wage * 0.5, minimum_wage, minimum_wage * 2, minimum_wage * 3, minimum_wage * 5, np.inf]
        labels = ["< 0.5 SM", "0.5-1 SM", "1-2 SM", "2-3 SM", "3-5 SM", "> 5 SM"]
        df["faixa_salarial"] = pd.cut(df[income_column], bins=bins, labels=labels, right=False)

    if "renda_habitual_principal" in df.columns:
        df = add_year_winsorized_column(
            df,
            source_col="renda_habitual_principal",
            output_col="renda_habitual_principal_winsor",
            lower_q=0.01,
            upper_q=0.99,
        )
        positive_winsor = df["renda_habitual_principal_winsor"].where(df["renda_habitual_principal_winsor"] > 0)
        df["log_renda_habitual_principal_winsor"] = np.log(positive_winsor)

    return df


def derive_educacao(df):
    if "motivo_nao_frequenta_escola_atual" in df.columns:
        numeric = pd.to_numeric(df["motivo_nao_frequenta_escola_atual"], errors="coerce")
        flag = numeric == 8
        label_column = "motivo_nao_frequenta_escola_atual_label"
        if label_column in df.columns:
            flag = flag | df[label_column].astype(str).str.contains("concurso", case=False, na=False)
        df["estudando_concurso"] = flag.astype("Int8")

    if "motivo_nao_frequenta_escola" in df.columns:
        numeric = pd.to_numeric(df["motivo_nao_frequenta_escola"], errors="coerce")
        flag_old = numeric == 8
        label_column = "motivo_nao_frequenta_escola_label"
        if label_column in df.columns:
            flag_old = flag_old | df[label_column].astype(str).str.contains("concurso", case=False, na=False)
        if "estudando_concurso" in df.columns:
            df["estudando_concurso"] = (df["estudando_concurso"].astype(bool) | flag_old).astype("Int8")
        else:
            df["estudando_concurso"] = flag_old.astype("Int8")

    if "frequenta_pre_vestibular" in df.columns:
        df["esta_no_pre_vestibular"] = (
            df["frequenta_pre_vestibular"].astype(str).str.contains("sim|^1$", case=False, na=False)
        ).astype("Int8")

    if "frequenta_escola" in df.columns:
        df["na_escola"] = (
            df["frequenta_escola"].astype(str).str.contains("sim|^1$", case=False, na=False)
        ).astype("Int8")

    return df


def clean_microdados(year, use_sample=False):
    ensure_dir(CLEANED_DIR)
    df, source_path = read_raw("microdados", year, use_sample=use_sample)
    label_map = build_label_map(["microdados"])
    df = apply_label_columns(df, label_map)
    df = rename_with_labels(df, MICRO_RENAME_MAP)
    df = derive_microdados(df)

    output_path = cleaned_path("microdados", year, use_sample=use_sample)
    df.to_csv(output_path, index=False)
    return output_path, source_path, df.shape


def clean_microdados_superior_sample(
    year=None,
    vd3004_values=DEFAULT_SUPERIOR_VALUES,
    input_path=None,
    output_format="csv",
):
    ensure_dir(CLEANED_DIR)
    df, source_path = read_microdados_sample(input_path=input_path, year=year)
    df = filter_vd3004(df, vd3004_values)
    label_map = build_label_map(["microdados"])
    df = apply_label_columns(df, label_map)
    df = rename_with_labels(df, MICRO_RENAME_MAP)
    df = derive_microdados(df)

    output_path = superior_sample_cleaned_path("microdados", vd3004_values, year=year)
    output_paths = write_cleaned_output(df, output_path, output_format=output_format)
    return output_paths, source_path, df.shape


def clean_microdados_vd3004(year, vd3004_value=DEFAULT_SUPERIOR_VALUES[0], input_path=None, output_format="csv"):
    ensure_dir(CLEANED_DIR)
    df, source_path = read_vd3004_raw("microdados", year, vd3004_value=vd3004_value, input_path=input_path)
    df = filter_vd3004(df, [vd3004_value])
    label_map = build_label_map(["microdados"])
    df = apply_label_columns(df, label_map)
    df = rename_with_labels(df, MICRO_RENAME_MAP)
    df = derive_microdados(df)

    output_path = vd3004_cleaned_path("microdados", year, vd3004_value=vd3004_value)
    output_paths = write_cleaned_output(df, output_path, output_format=output_format)
    return output_paths, source_path, df.shape


def clean_educacao(year, use_sample=False):
    ensure_dir(CLEANED_DIR)
    df, source_path = read_raw("educacao", year, use_sample=use_sample)
    label_map = build_label_map(["microdados", "educacao"])
    df = apply_label_columns(df, label_map)
    df = rename_with_labels(df, EDUCACAO_RENAME_MAP)
    df = derive_educacao(df)

    output_path = cleaned_path("educacao", year, use_sample=use_sample)
    df.to_csv(output_path, index=False)
    return output_path, source_path, df.shape


def clean_educacao_superior_sample(
    year=None,
    vd3004_values=DEFAULT_SUPERIOR_VALUES,
    input_path=None,
    output_format="csv",
):
    ensure_dir(CLEANED_DIR)
    df, source_path = read_microdados_sample(input_path=input_path, year=year)
    df = filter_vd3004(df, vd3004_values)
    df = keep_educacao_columns_from_microdados(df)
    label_map = build_label_map(["microdados", "educacao"])
    df = apply_label_columns(df, label_map)
    df = rename_with_labels(df, {**EDUCACAO_RENAME_MAP, "VD3004": "nivel_instrucao"})
    df = derive_educacao(df)

    output_path = superior_sample_cleaned_path("educacao", vd3004_values, year=year)
    output_paths = write_cleaned_output(df, output_path, output_format=output_format)
    return output_paths, source_path, df.shape


def clean_educacao_vd3004(year, vd3004_value=DEFAULT_SUPERIOR_VALUES[0], input_path=None, output_format="csv"):
    ensure_dir(CLEANED_DIR)
    df, source_path = read_vd3004_raw("educacao", year, vd3004_value=vd3004_value, input_path=input_path)
    label_map = build_label_map(["microdados", "educacao"])
    df = apply_label_columns(df, label_map)
    df = rename_with_labels(df, EDUCACAO_RENAME_MAP)
    df = derive_educacao(df)

    output_path = vd3004_cleaned_path("educacao", year, vd3004_value=vd3004_value)
    output_paths = write_cleaned_output(df, output_path, output_format=output_format)
    return output_paths, source_path, df.shape
