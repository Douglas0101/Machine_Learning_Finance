import pandas as pd
import numpy as np
import logging
import os
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.feature_selection import VarianceThreshold

# Configuração de logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove linhas duplicadas do DataFrame.
    """
    initial_count = len(df)
    df = df.drop_duplicates()
    logging.info(f"Duplicatas removidas: {initial_count - len(df)} linhas.")
    return df


def handle_missing_data(df: pd.DataFrame,
                        numeric_impute_strategy: str = 'median',
                        categorical_impute_strategy: str = 'most_frequent') -> pd.DataFrame:
    """
    Trata dados missing:
      - Cria flags booleanas para indicar valores imputados.
      - Imputa colunas numéricas com a mediana (ou estratégia especificada)
        e colunas categóricas com a moda.
    """
    df = df.copy()
    # Criação de flags de imputação para colunas com missing
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            flag_col = col + "_imputed_flag"
            df[flag_col] = df[col].isnull().astype(int)
    logging.info("Flags de imputação criadas para colunas com dados missing.")

    # Identifica colunas numéricas e categóricas
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    # Imputação de dados numéricos
    if numeric_cols:
        num_imputer = SimpleImputer(strategy=numeric_impute_strategy)
        df[numeric_cols] = num_imputer.fit_transform(df[numeric_cols])
        logging.info("Colunas numéricas imputadas.")
    # Imputação de dados categóricos
    if categorical_cols:
        cat_imputer = SimpleImputer(strategy=categorical_impute_strategy, fill_value="missing")
        df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])
        logging.info("Colunas categóricas imputadas.")

    return df


def treat_outliers(df: pd.DataFrame, numeric_cols: list, method: str = "IQR", factor: float = 1.5) -> pd.DataFrame:
    """
    Trata outliers em colunas numéricas usando o método IQR.
    Valores abaixo de Q1 - factor*IQR e acima de Q3 + factor*IQR são clipados.
    """
    df = df.copy()
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR
        df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
    logging.info("Outliers tratados utilizando clipping pelo método IQR.")
    return df


def engineer_temporal_features(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """
    A partir de uma coluna de data, extrai features temporais:
      - Ano, mês, dia, dia da semana e hora.
    """
    df = df.copy()
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df[date_col + "_year"] = df[date_col].dt.year
        df[date_col + "_month"] = df[date_col].dt.month
        df[date_col + "_day"] = df[date_col].dt.day
        df[date_col + "_weekday"] = df[date_col].dt.weekday
        df[date_col + "_hour"] = df[date_col].dt.hour
        logging.info(f"Features temporais extraídas a partir da coluna '{date_col}'.")
    else:
        logging.warning(f"Coluna de data '{date_col}' não encontrada.")
    return df

def encode_categorical(df: pd.DataFrame, encoding_method: str = "onehot", onehot_threshold: int = 50) -> pd.DataFrame:
    """
    Codifica colunas categóricas.

    Para cada coluna categórica:
      - Se a cardinalidade (número de valores únicos) for menor ou igual a onehot_threshold,
        aplica One-Hot Encoding utilizando OneHotEncoder com sparse_output=True para manter a eficiência de memória.
      - Se a cardinalidade for maior que onehot_threshold, aplica Label Encoding.

    Args:
        df (pd.DataFrame): DataFrame de entrada.
        encoding_method (str): Método de codificação (neste exemplo, a estratégia mista é aplicada).
        onehot_threshold (int): Limiar para definir colunas de baixa cardinalidade.

    Returns:
        pd.DataFrame: DataFrame com colunas categóricas codificadas.
    """
    df = df.copy()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    low_card_cols = []
    high_card_cols = []

    # Separar colunas de baixa e alta cardinalidade
    for col in categorical_cols:
        unique_count = df[col].nunique()
        if unique_count <= onehot_threshold:
            low_card_cols.append(col)
        else:
            high_card_cols.append(col)

    logging.info(f"Colunas de baixa cardinalidade: {low_card_cols}")
    logging.info(f"Colunas de alta cardinalidade: {high_card_cols}")

    # Para colunas de baixa cardinalidade, aplicar OneHotEncoder com sparse_output=True
    if low_card_cols:
        from sklearn.preprocessing import OneHotEncoder  # Import garantido
        enc = OneHotEncoder(sparse_output=True, drop='first')
        encoded_sparse = enc.fit_transform(df[low_card_cols])
        encoded_cols = enc.get_feature_names_out(low_card_cols)
        encoded_df = pd.DataFrame.sparse.from_spmatrix(
            encoded_sparse,
            columns=encoded_cols,
            index=df.index
        )
        df = pd.concat([df.drop(columns=low_card_cols), encoded_df], axis=1)
        logging.info("One-Hot Encoding aplicado para colunas de baixa cardinalidade.")

    # Para colunas de alta cardinalidade, aplicar Label Encoding
    for col in high_card_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
    logging.info("Label Encoding aplicado para colunas de alta cardinalidade.")

    return df

def scale_numeric_features(df: pd.DataFrame, scaler=StandardScaler()) -> pd.DataFrame:
    """
    Aplica escalonamento em colunas numéricas utilizando o escalador fornecido (padrão: StandardScaler).
    """
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    logging.info(f"Escalonamento aplicado com {scaler.__class__.__name__}.")
    return df


def select_features_by_variance(df: pd.DataFrame, threshold: float = 0.0) -> pd.DataFrame:
    """
    Remove features com baixa variância utilizando VarianceThreshold.
    """
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    vt = VarianceThreshold(threshold=threshold)
    vt.fit(df[numeric_cols])
    cols_to_keep = [col for col, var in zip(numeric_cols, vt.variances_) if var > threshold]
    removed = set(numeric_cols) - set(cols_to_keep)
    logging.info(f"Features removidas por baixa variância: {removed}")
    # Mantém as colunas não numéricas inalteradas
    non_numeric_cols = [col for col in df.columns if col not in numeric_cols]
    return df[cols_to_keep + non_numeric_cols]


def preprocess_data(df: pd.DataFrame,
                    date_col: str = "date_transacao",
                    encoding_method: str = "onehot") -> pd.DataFrame:
    """
    Executa o pipeline de pré-processamento:
      1. Remove duplicatas.
      2. Trata dados missing e cria flags de imputação.
      3. Trata outliers em colunas numéricas.
      4. Extrai features temporais da coluna de data.
      5. Codifica features categóricas.
      6. Escalona features numéricas.
      7. Realiza seleção de features baseada em variância.

    Retorna o DataFrame pré-processado.
    """
    logging.info("Início do pré-processamento de dados.")
    try:
        df = remove_duplicates(df)
        df = handle_missing_data(df)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        df = treat_outliers(df, numeric_cols)
        df = engineer_temporal_features(df, date_col)
        df = encode_categorical(df, encoding_method)
        df = scale_numeric_features(df)
        df = select_features_by_variance(df)
        logging.info("Pré-processamento concluído com sucesso.")
    except Exception as e:
        logging.error(f"Erro durante o pré-processamento: {e}")
        raise
    return df


if __name__ == "__main__":
    # Exemplo de uso:
    # Suponha que exista um arquivo "dataset.csv" no diretório atual.
    try:
        input_path = "../raw/dataset_financeiro_simulado.csv"  # Ajuste conforme necessário
        if os.path.exists(input_path):
            df_input = pd.read_csv(input_path)
            # 'data_transacao' é o nome da coluna de data (ajuste conforme seu dataset)
            df_preprocessed = preprocess_data(df_input, date_col="data_transacao", encoding_method="onehot")
            df_preprocessed.to_csv("dataset_bank_preprocessado.csv", index=False)
            logging.info("Dataset pré-processado salvo como 'dataset_preprocessado.csv'.")
        else:
            logging.error(f"Arquivo {input_path} não encontrado.")
    except Exception as e:
        logging.error(f"Falha na execução do pipeline: {e}")
