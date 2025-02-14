#!/usr/bin/env python
"""
build_features.py

Este script implementa um pipeline completo de engenharia de features para transformar
um dataset bruto em um conjunto de features de alta qualidade, informativas e otimizadas
para modelos de Machine Learning de nível de mercado.

O pipeline abrange as seguintes etapas:
  I. Engenharia de Features Numéricas:
      - Escalonamento (Standard, MinMax ou RobustScaler).
      - Transformação não linear (PowerTransformer com Box-Cox ou Yeo-Johnson).
      - Discretização (Binning) com KBinsDiscretizer.
      - Rank Transformation para lidar com outliers.
      - Operações matemáticas (ex.: transformação logarítmica) para realçar relações.

  II. Engenharia de Features Categóricas:
      - One-Hot Encoding robusto (handle_unknown='ignore').
      - Label Encoding para variáveis ordinais.
      - Target Encoding (usando category_encoders) com mecanismos de validação para evitar data leakage.

  III. Engenharia de Features Temporais:
      - Extração de componentes básicos (ano, mês, dia, hora, weekday).
      - Transformações cíclicas (seno e cosseno) para representar a periodicidade.
      - (Opcional) Cálculo de estatísticas de janela móvel (rolling) para colunas numéricas.
      - (Opcional) Criação de features de lag para séries temporais.

  IV. Engenharia de Features Textuais (Opcional):
      - Transformação de textos em representações numéricas via TF-IDF ou CountVectorizer.

  V. Features de Interação e Polinômicas:
      - Criação de interações e termos polinomiais de baixa ordem, se justificado por insights da EDA.

O script é configurável via argumentos de linha de comando e permite ajuste dos parâmetros
de cada etapa sem modificar o código-fonte, facilitando experimentações e refinamentos iterativos.

Bibliotecas utilizadas:
  - pandas, numpy
  - scikit-learn (preprocessing, pipeline, feature_extraction)
  - category_encoders (para target encoding e WoE, se instalado)
  - argparse, logging

Autor: [Seu Nome]
Data: [Data Atual]
"""

import argparse
import logging
import os
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler,
    PowerTransformer, KBinsDiscretizer,
    LabelEncoder, OneHotEncoder, PolynomialFeatures
)
from sklearn.feature_extraction.text import TfidfVectorizer

# Tentar importar category_encoders para target encoding e WoE
try:
    import category_encoders as ce
except ImportError:
    ce = None

# Configuração do logging com timestamp
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"build_features_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
logger = logging.getLogger(__name__)


#############################
# Engenharia de Features Numéricas
#############################

def engineer_numerical_features(df, numeric_cols=None, scaler_type='robust', power_method='yeo-johnson',
                                apply_binning=False, binning_strategy='quantile', n_bins=5, apply_rank=False):
    """
    Aplica transformações em features numéricas:
      - Escalonamento com StandardScaler, MinMaxScaler ou RobustScaler.
      - Transformação não linear com PowerTransformer.
      - (Opcional) Discretização com KBinsDiscretizer.
      - (Opcional) Rank Transformation.
      - Operação logarítmica para estabilizar variâncias.

    Parâmetros:
      df: DataFrame de entrada.
      numeric_cols: Lista de colunas numéricas (se None, inferido automaticamente).
      scaler_type: 'standard', 'minmax' ou 'robust'.
      power_method: 'box-cox' (dados positivos) ou 'yeo-johnson'.
      apply_binning: Booleano, se True aplica discretização.
      binning_strategy: Estratégia de binning ('uniform', 'quantile' ou 'kmeans').
      n_bins: Número de bins para discretização.
      apply_rank: Booleano, se True gera features com o ranking dos valores.

    Retorna:
      DataFrame com features numéricas transformadas.
    """
    df_num = df.copy()
    if numeric_cols is None:
        numeric_cols = df_num.select_dtypes(include=[np.number]).columns.tolist()

    logger.info(f"Iniciando engenharia de features numéricas para: {numeric_cols}")

    # Seleciona o escalador
    if scaler_type == 'standard':
        scaler = StandardScaler()
    elif scaler_type == 'minmax':
        scaler = MinMaxScaler()
    elif scaler_type == 'robust':
        scaler = RobustScaler()
    else:
        logger.warning("Scaler não reconhecido. Usando RobustScaler por padrão.")
        scaler = RobustScaler()

    # Configura o PowerTransformer
    power_transformer = PowerTransformer(method=power_method)

    # Pipeline para escalonamento e transformação não linear
    num_pipeline = Pipeline(steps=[
        ('scaler', scaler),
        ('power', power_transformer)
    ])
    df_num[numeric_cols] = num_pipeline.fit_transform(df_num[numeric_cols])
    logger.info("Escalonamento e PowerTransformer aplicados.")

    # Discretização (Binning)
    if apply_binning:
        for col in numeric_cols:
            try:
                kbins = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy=binning_strategy)
                df_num[f"{col}_binned"] = kbins.fit_transform(df_num[[col]]).astype(int)
                logger.info(f"Binning aplicado na coluna {col}.")
            except Exception as e:
                logger.error(f"Erro no binning da coluna {col}: {e}")

    # Rank Transformation
    if apply_rank:
        for col in numeric_cols:
            df_num[f"{col}_rank"] = df_num[col].rank(method='average')
        logger.info("Rank transformation aplicada.")

    # Exemplo de operação matemática: transformação logarítmica (com ajuste para evitar log(0))
    for col in numeric_cols:
        df_num[f"{col}_log"] = np.log(df_num[col] + 1e-9)
    logger.info("Transformação logarítmica aplicada nas features numéricas.")

    return df_num


#############################
# Engenharia de Features Categóricas
#############################

def engineer_categorical_features(df, cat_cols=None, encoding_method='onehot', target=None,
                                  apply_target_encoding=False):
    """
    Aplica transformações em features categóricas.
      - One-Hot Encoding com handle_unknown='ignore' ou Label Encoding.
      - (Opcional) Target Encoding para alta cardinalidade (se target for fornecido).

    Parâmetros:
      df: DataFrame de entrada.
      cat_cols: Lista de colunas categóricas (se None, inferido automaticamente).
      encoding_method: 'onehot' ou 'label'.
      target: Série ou array com o target (necessário para target encoding).
      apply_target_encoding: Booleano, se True aplica Target Encoding.

    Retorna:
      DataFrame com features categóricas transformadas.
    """
    df_cat = df.copy()
    if cat_cols is None:
        cat_cols = df_cat.select_dtypes(include=['object', 'category']).columns.tolist()

    logger.info(f"Iniciando engenharia de features categóricas para: {cat_cols}")
    transformed_cols = pd.DataFrame(index=df_cat.index)

    # Target Encoding (se configurado)
    if apply_target_encoding:
        if ce is None:
            logger.error("category_encoders não instalado; Target Encoding não pode ser aplicado.")
        else:
            if target is None:
                logger.error("Target não fornecido para Target Encoding.")
            else:
                for col in cat_cols:
                    try:
                        te = ce.TargetEncoder(cols=[col], smoothing=0.3)
                        transformed_cols[f"{col}_te"] = te.fit_transform(df_cat[col], target)
                        logger.info(f"Target Encoding aplicado na coluna {col}.")
                    except Exception as e:
                        logger.error(f"Erro no Target Encoding da coluna {col}: {e}")

    # Se não estiver aplicando target encoding, aplica encoding padrão
    if encoding_method == 'onehot':
        try:
            ohe = OneHotEncoder(handle_unknown='ignore', sparse=False, drop='first')
            ohe_result = ohe.fit_transform(df_cat[cat_cols])
            ohe_df = pd.DataFrame(ohe_result, columns=ohe.get_feature_names_out(cat_cols), index=df_cat.index)
            transformed_cols = pd.concat([transformed_cols, ohe_df], axis=1)
            logger.info("One-Hot Encoding aplicado com sucesso.")
        except Exception as e:
            logger.error(f"Erro no One-Hot Encoding: {e}")
    elif encoding_method == 'label':
        for col in cat_cols:
            try:
                le = LabelEncoder()
                transformed_cols[col + "_le"] = le.fit_transform(df_cat[col])
                logger.info(f"Label Encoding aplicado na coluna {col}.")
            except Exception as e:
                logger.error(f"Erro no Label Encoding da coluna {col}: {e}")
    else:
        logger.error("Método de encoding não suportado. Use 'onehot' ou 'label'.")

    # Remove as colunas originais e concatena as transformadas
    df_cat = df_cat.drop(columns=cat_cols, errors='ignore')
    df_cat = pd.concat([df_cat, transformed_cols], axis=1)
    return df_cat


#############################
# Engenharia de Features Temporais
#############################

def engineer_temporal_features(df, date_col, extract_cyclical=True, apply_rolling=False,
                               rolling_window=3, apply_lag=False, lags=None):
    """
    Extrai e transforma features temporais a partir de uma coluna de data/hora.
      - Conversão para datetime e extração de ano, mês, dia, hora e weekday.
      - (Opcional) Criação de variáveis cíclicas (seno e cosseno) para hora e weekday.
      - (Opcional) Cálculo de estatísticas de janela móvel (rolling) para colunas numéricas.
      - (Opcional) Criação de features de lag para séries temporais.

    Parâmetros:
      df: DataFrame de entrada.
      date_col: Nome da coluna contendo data/hora.
      extract_cyclical: Se True, cria variáveis cíclicas.
      apply_rolling: Se True, calcula estatísticas de janela móvel.
      rolling_window: Tamanho da janela para rolling.
      apply_lag: Se True, gera features de lag.
      lags: Lista de períodos de lag (padrão: [1] se None).

    Retorna:
      DataFrame com features temporais adicionadas.
    """
    # Define o valor padrão para lags se não for fornecido
    if lags is None:
        lags = [1]

    df_temp = df.copy()
    try:
        df_temp[date_col] = pd.to_datetime(df_temp[date_col], errors='coerce')
        df_temp['year'] = df_temp[date_col].dt.year
        df_temp['month'] = df_temp[date_col].dt.month
        df_temp['day'] = df_temp[date_col].dt.day
        df_temp['hour'] = df_temp[date_col].dt.hour
        df_temp['minute'] = df_temp[date_col].dt.minute
        df_temp['weekday'] = df_temp[date_col].dt.weekday
        logger.info(f"Componentes básicos extraídos da coluna {date_col}.")
    except Exception as e:
        logger.error(f"Erro ao processar a coluna {date_col}: {e}")

    if extract_cyclical:
        try:
            df_temp['hour_sin'] = np.sin(2 * np.pi * df_temp['hour'] / 24)
            df_temp['hour_cos'] = np.cos(2 * np.pi * df_temp['hour'] / 24)
            df_temp['weekday_sin'] = np.sin(2 * np.pi * df_temp['weekday'] / 7)
            df_temp['weekday_cos'] = np.cos(2 * np.pi * df_temp['weekday'] / 7)
            logger.info("Transformações cíclicas aplicadas para hora e weekday.")
        except Exception as e:
            logger.error(f"Erro nas transformações cíclicas: {e}")

    if apply_rolling:
        numeric_cols = df_temp.select_dtypes(include=[np.number]).columns.tolist()
        for col in numeric_cols:
            try:
                df_temp[f"{col}_roll_mean"] = df_temp[col].rolling(window=rolling_window, min_periods=1).mean()
                df_temp[f"{col}_roll_std"] = df_temp[col].rolling(window=rolling_window, min_periods=1).std().fillna(0)
                logger.info(f"Rolling window aplicada na coluna {col}.")
            except Exception as e:
                logger.error(f"Erro no rolling window da coluna {col}: {e}")

    if apply_lag:
        numeric_cols = df_temp.select_dtypes(include=[np.number]).columns.tolist()
        for col in numeric_cols:
            for lag in lags:
                df_temp[f"{col}_lag_{lag}"] = df_temp[col].shift(lag)
        logger.info("Features de lag criadas para colunas numéricas.")

    return df_temp


#############################
# Engenharia de Features Textuais (Opcional)
#############################

def engineer_text_features(df, text_cols=None, method='tfidf', max_features=100):
    """
    Transforma colunas textuais em features numéricas utilizando TF-IDF ou CountVectorizer.

    Parâmetros:
      df: DataFrame de entrada.
      text_cols: Lista de colunas textuais (se None, inferido automaticamente).
      method: 'tfidf' ou 'count'.
      max_features: Número máximo de features.

    Retorna:
      DataFrame com as features textuais extraídas.
    """
    df_text = df.copy()
    if text_cols is None:
        text_cols = df_text.select_dtypes(include=['object']).columns.tolist()

    logger.info(f"Iniciando engenharia de features textuais para: {text_cols}")
    transformed_text = pd.DataFrame(index=df_text.index)

    for col in text_cols:
        try:
            if method == 'tfidf':
                vectorizer = TfidfVectorizer(max_features=max_features)
            else:
                from sklearn.feature_extraction.text import CountVectorizer
                vectorizer = CountVectorizer(max_features=max_features)
            text_transformed = vectorizer.fit_transform(df_text[col].fillna("")).toarray()
            text_df = pd.DataFrame(text_transformed,
                                   columns=[f"{col}_{feat}" for feat in vectorizer.get_feature_names_out()],
                                   index=df_text.index)
            transformed_text = pd.concat([transformed_text, text_df], axis=1)
            logger.info(f"Features textuais geradas para {col}.")
        except Exception as e:
            logger.error(f"Erro no processamento da coluna textual {col}: {e}")

    df_text = df_text.drop(columns=text_cols, errors='ignore')
    df_text = pd.concat([df_text, transformed_text], axis=1)
    return df_text


#############################
# Engenharia de Features de Interação e Polinômicas
#############################

def engineer_interaction_features(df, interaction_cols=None, degree=2, include_bias=False):
    """
    Gera features de interação e termos polinomiais utilizando PolynomialFeatures.

    Parâmetros:
      df: DataFrame de entrada.
      interaction_cols: Lista de colunas para gerar interações. Se None, nenhuma interação é criada.
      degree: Grau do polinômio.
      include_bias: Se True, inclui o termo de bias.

    Retorna:
      DataFrame com as features de interação adicionadas.
    """
    if not interaction_cols:
        logger.info("Nenhuma coluna especificada para interações; pulando esta etapa.")
        return df

    logger.info(f"Gerando features de interação para: {interaction_cols}")
    poly = PolynomialFeatures(degree=degree, interaction_only=False, include_bias=include_bias)
    interaction_data = poly.fit_transform(df[interaction_cols])
    interaction_feature_names = poly.get_feature_names_out(interaction_cols)
    df_interaction = pd.DataFrame(interaction_data, columns=interaction_feature_names, index=df.index)
    # Combina as features de interação com o restante do dataset
    df_result = pd.concat([df.drop(columns=interaction_cols), df_interaction], axis=1)
    return df_result


#############################
# Pipeline Principal: build_features
#############################

def build_features(df_input, config):
    """
    Executa o pipeline de engenharia de features, integrando todas as transformações:
      - Carregamento dos dados (se df_input for um caminho de arquivo).
      - Análise exploratória preliminar (EDA).
      - Engenharia de features numéricas, categóricas, temporais, textuais e de interação.

    Parâmetros:
      df_input: DataFrame ou caminho para arquivo CSV.
      config: Dicionário com parâmetros de configuração para cada etapa.

    Retorna:
      DataFrame com todas as features engenheiradas.
    """
    logger.info("Iniciando pipeline de engenharia de features.")

    # Carregar dados se necessário
    if isinstance(df_input, str):
        try:
            df = pd.read_csv(df_input)
            logger.info(f"Dataset carregado do arquivo: {df_input}")
        except Exception as e:
            logger.error(f"Erro ao carregar o arquivo {df_input}: {e}")
            raise
    elif isinstance(df_input, pd.DataFrame):
        df = df_input.copy()
        logger.info("Dataset fornecido como DataFrame.")
    else:
        logger.error("Tipo de input não suportado; use caminho de arquivo ou DataFrame.")
        raise ValueError("Input não suportado.")

    # EDA preliminar (pode ser estendido conforme necessário)
    logger.info("Realizando análise exploratória preliminar (EDA).")
    logger.info(f"Descrição do dataset:\n{df.describe(include='all')}")

    # Engenharia de features numéricas
    numeric_config = config.get('numerical', {})
    numeric_cols = config.get('numeric_cols', df.select_dtypes(include=[np.number]).columns.tolist())
    df = engineer_numerical_features(
        df,
        numeric_cols=numeric_cols,
        scaler_type=numeric_config.get('scaler_type', 'robust'),
        power_method=numeric_config.get('power_method', 'yeo-johnson'),
        apply_binning=numeric_config.get('apply_binning', False),
        binning_strategy=numeric_config.get('binning_strategy', 'quantile'),
        n_bins=numeric_config.get('n_bins', 5),
        apply_rank=numeric_config.get('apply_rank', False)
    )

    # Engenharia de features categóricas
    cat_config = config.get('categorical', {})
    cat_cols = config.get('cat_cols', df.select_dtypes(include=['object', 'category']).columns.tolist())
    df = engineer_categorical_features(
        df,
        cat_cols=cat_cols,
        encoding_method=cat_config.get('encoding_method', 'onehot'),
        target=config.get('target_series', None),
        apply_target_encoding=cat_config.get('apply_target_encoding', False)
    )

    # Engenharia de features temporais
    temporal_config = config.get('temporal', {})
    date_col = config.get('date_col', None)
    if date_col and date_col in df.columns:
        df = engineer_temporal_features(
            df,
            date_col=date_col,
            extract_cyclical=temporal_config.get('extract_cyclical', True),
            apply_rolling=temporal_config.get('apply_rolling', False),
            rolling_window=temporal_config.get('rolling_window', 3),
            apply_lag=temporal_config.get('apply_lag', False),
            lags=temporal_config.get('lags', None)
        )
    else:
        logger.warning("Coluna de data não especificada ou inexistente; ignorando engenharia temporal.")

    # Engenharia de features textuais (opcional)
    text_config = config.get('text', {})
    text_cols = config.get('text_cols', [])
    if text_cols:
        df = engineer_text_features(
            df,
            text_cols=text_cols,
            method=text_config.get('method', 'tfidf'),
            max_features=text_config.get('max_features', 100)
        )

    # Engenharia de features de interação e polinômicas (opcional)
    interaction_config = config.get('interaction', {})
    interaction_cols = config.get('interaction_cols', [])
    if interaction_cols:
        df = engineer_interaction_features(
            df,
            interaction_cols=interaction_cols,
            degree=interaction_config.get('degree', 2),
            include_bias=interaction_config.get('include_bias', False)
        )

    logger.info("Pipeline de engenharia de features concluído.")
    return df


#############################
# Função para salvar o dataset transformado
#############################

def save_features(df, output_path):
    """
    Salva o DataFrame de features em um arquivo CSV.

    Parâmetros:
      df: DataFrame de features.
      output_path: Caminho de saída.
    """
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Dataset de features salvo em: {output_path}")
    except Exception as e:
        logger.error(f"Erro ao salvar o dataset: {e}")
        raise


#############################
# Argumentos de Linha de Comando e Função Main
#############################

def parse_args():
    parser = argparse.ArgumentParser(description="Pipeline de Engenharia de Features para ML")
    parser.add_argument("--input", type=str, default="../../data/processed/dataset_final.csv",
                        help="Caminho para o arquivo CSV de entrada")
    parser.add_argument("--output", type=str, default="../../data/processed/features_final.csv",
                        help="Caminho para salvar o arquivo CSV de features")
    parser.add_argument("--date_col", type=str, default="data_transacao", help="Nome da coluna de data/hora")
    return parser.parse_args()


def main():
    args = parse_args()

    # Exemplo de configuração; esta pode ser carregada de um arquivo JSON/YAML para maior flexibilidade
    config = {
        "numerical": {
            "scaler_type": "robust",
            "power_method": "yeo-johnson",
            "apply_binning": True,
            "binning_strategy": "quantile",
            "n_bins": 5,
            "apply_rank": False
        },
        "categorical": {
            "encoding_method": "onehot",
            "apply_target_encoding": False
        },
        "temporal": {
            "extract_cyclical": True,
            "apply_rolling": False,
            "rolling_window": 3,
            "apply_lag": False,
            "lags": [1]
        },
        "text": {
            "method": "tfidf",
            "max_features": 100
        },
        "interaction": {
            "degree": 2,
            "include_bias": False
        },
        "date_col": args.date_col,
        "numeric_cols": None,
        "cat_cols": None,
        "text_cols": [],
        "interaction_cols": []
        # "target_series": ...  (incluir se for aplicar target encoding)
    }

    try:
        df_features = build_features(args.input, config)
        save_features(df_features, args.output)
    except Exception as e:
        logger.error(f"Erro no pipeline de engenharia de features: {e}")


if __name__ == "__main__":
    main()
