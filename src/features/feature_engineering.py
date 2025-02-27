"""
Módulo para engenharia de features do algoritmo de detecção de inadimplência.

Este módulo contém funções para criar, transformar e selecionar características
para melhorar o desempenho dos modelos de previsão de inadimplência.
"""

import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime
from typing import Dict, List, Any, Union, Tuple
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
import json

# Configurar logger
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Obter caminho da raiz do projeto
def get_project_root():
    """Retorna o caminho para a raiz do projeto."""
    # Assumindo que este script está em src/features/ ou src/data/
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Subir dois níveis para chegar à raiz do projeto
    project_root = os.path.abspath(os.path.join(script_dir, os.pardir, os.pardir))
    return project_root


def load_processed_data(data_dir: str, timestamp: str = None) -> Dict[str, pd.DataFrame]:
    """
    Carrega os conjuntos de dados processados.

    Args:
        data_dir: Diretório com os dados processados
        timestamp: Timestamp específico para carregar ou None para o mais recente

    Returns:
        Dictionary com DataFrames para treino, validação e teste
    """
    # Se data_dir não for absoluto, considerar relativo à raiz do projeto
    project_root = get_project_root()
    if not os.path.isabs(data_dir):
        data_dir = os.path.join(project_root, data_dir)

    logger.info(f"Carregando dados processados de: {data_dir}")

    # Encontrar metadados disponíveis
    metadata_files = [f for f in os.listdir(data_dir) if f.startswith('metadata_') and f.endswith('.json')]

    if not metadata_files:
        raise FileNotFoundError(f"Nenhum arquivo de metadados encontrado em {data_dir}")

    # Selecionar o arquivo de metadados correto
    if timestamp:
        metadata_file = f"metadata_{timestamp}.json"
        if metadata_file not in metadata_files:
            raise FileNotFoundError(f"Metadados para timestamp {timestamp} não encontrados")
    else:
        # Ordenar por timestamp (assumindo formato padrão)
        metadata_files.sort(reverse=True)
        metadata_file = metadata_files[0]
        logger.info(f"Usando metadados mais recentes: {metadata_file}")

    # Carregar metadados
    with open(os.path.join(data_dir, metadata_file), 'r') as f:
        metadata = json.load(f)

    # Extrair timestamp dos metadados
    timestamp_to_load = metadata.get('timestamp', metadata_file.replace('metadata_', '').replace('.json', ''))

    # Carregar arquivos de dados
    train_file = os.path.join(data_dir, f"train_{timestamp_to_load}.csv")
    val_file = os.path.join(data_dir, f"val_{timestamp_to_load}.csv")
    test_file = os.path.join(data_dir, f"test_{timestamp_to_load}.csv")

    # Verificar se os arquivos existem
    if not all(os.path.exists(f) for f in [train_file, val_file, test_file]):
        raise FileNotFoundError(f"Arquivos de dados para timestamp {timestamp_to_load} não encontrados")

    # Carregar dados
    df_train = pd.read_csv(train_file)
    df_val = pd.read_csv(val_file)
    df_test = pd.read_csv(test_file)

    logger.info(f"Dados carregados com sucesso:")
    logger.info(f"  Treino: {df_train.shape[0]} linhas, {df_train.shape[1]} colunas")
    logger.info(f"  Validação: {df_val.shape[0]} linhas, {df_val.shape[1]} colunas")
    logger.info(f"  Teste: {df_test.shape[0]} linhas, {df_test.shape[1]} colunas")

    return {
        'train': df_train,
        'val': df_val,
        'test': df_test,
        'metadata': metadata
    }


def handle_missing_values(X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame,
                         missing_config: Dict[str, Any] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
    """
    Trata valores ausentes nos conjuntos de dados.

    Args:
        X_train: Features de treino
        X_val: Features de validação
        X_test: Features de teste
        missing_config: Configuração de tratamento de valores ausentes

    Returns:
        Features tratadas e dicionário com imputadores
    """
    logger.info("Tratando valores ausentes...")

    # Configurações padrão
    if missing_config is None:
        missing_config = {
            'numeric_strategy': 'mean',  # 'mean', 'median', 'most_frequent', 'constant'
            'categorical_strategy': 'most_frequent',  # 'most_frequent', 'constant'
            'constant_value_numeric': 0,
            'constant_value_categorical': 'missing',
            'add_indicator': False  # Modificado para False para evitar o problema
        }

    # Separar features numéricas e categóricas
    numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

    # Imputadores
    imputadores = {}

    # Tratar features numéricas
    if numeric_features:
        logger.info(f"Tratando {len(numeric_features)} features numéricas com estratégia '{missing_config['numeric_strategy']}'")

        # Verificar quais colunas realmente têm valores ausentes
        missing_mask = X_train[numeric_features].isnull().any()
        num_missing_cols = missing_mask.sum()

        if num_missing_cols > 0:
            logger.info(f"  {num_missing_cols} colunas numéricas têm valores ausentes")

            # Criar o imputador sem add_indicator
            numeric_imputer = SimpleImputer(
                strategy=missing_config['numeric_strategy'],
                fill_value=missing_config['constant_value_numeric'] if missing_config['numeric_strategy'] == 'constant' else None
            )

            # Ajustar e transformar
            X_train_numeric = numeric_imputer.fit_transform(X_train[numeric_features])
            X_val_numeric = numeric_imputer.transform(X_val[numeric_features])
            X_test_numeric = numeric_imputer.transform(X_test[numeric_features])

            # Guardar imputador
            imputadores['numeric'] = numeric_imputer

            # Criar DataFrames
            X_train_numeric_df = pd.DataFrame(X_train_numeric, columns=numeric_features, index=X_train.index)
            X_val_numeric_df = pd.DataFrame(X_val_numeric, columns=numeric_features, index=X_val.index)
            X_test_numeric_df = pd.DataFrame(X_test_numeric, columns=numeric_features, index=X_test.index)

            # Criar indicadores manualmente se necessário
            if missing_config['add_indicator']:
                for col in numeric_features:
                    if X_train[col].isnull().any():
                        indicator_name = f"{col}_missing"
                        X_train_numeric_df[indicator_name] = X_train[col].isnull().astype(int)
                        X_val_numeric_df[indicator_name] = X_val[col].isnull().astype(int)
                        X_test_numeric_df[indicator_name] = X_test[col].isnull().astype(int)
        else:
            logger.info("  Nenhuma coluna numérica tem valores ausentes")
            X_train_numeric_df = X_train[numeric_features].copy()
            X_val_numeric_df = X_val[numeric_features].copy()
            X_test_numeric_df = X_test[numeric_features].copy()
    else:
        X_train_numeric_df = pd.DataFrame(index=X_train.index)
        X_val_numeric_df = pd.DataFrame(index=X_val.index)
        X_test_numeric_df = pd.DataFrame(index=X_test.index)

    # Tratar features categóricas
    if categorical_features:
        logger.info(f"Tratando {len(categorical_features)} features categóricas com estratégia '{missing_config['categorical_strategy']}'")

        # Verificar quais colunas realmente têm valores ausentes
        missing_mask = X_train[categorical_features].isnull().any()
        cat_missing_cols = missing_mask.sum()

        if cat_missing_cols > 0:
            logger.info(f"  {cat_missing_cols} colunas categóricas têm valores ausentes")

            # Criar o imputador sem add_indicator
            categorical_imputer = SimpleImputer(
                strategy=missing_config['categorical_strategy'],
                fill_value=missing_config['constant_value_categorical'] if missing_config['categorical_strategy'] == 'constant' else None
            )

            # Converter para objeto (SimpleImputer não trabalha com tipo 'category')
            X_train_cat = X_train[categorical_features].astype('object')
            X_val_cat = X_val[categorical_features].astype('object')
            X_test_cat = X_test[categorical_features].astype('object')

            # Ajustar e transformar
            X_train_cat_imp = categorical_imputer.fit_transform(X_train_cat)
            X_val_cat_imp = categorical_imputer.transform(X_val_cat)
            X_test_cat_imp = categorical_imputer.transform(X_test_cat)

            # Guardar imputador
            imputadores['categorical'] = categorical_imputer

            # Criar DataFrames
            X_train_cat_df = pd.DataFrame(X_train_cat_imp, columns=categorical_features, index=X_train.index)
            X_val_cat_df = pd.DataFrame(X_val_cat_imp, columns=categorical_features, index=X_val.index)
            X_test_cat_df = pd.DataFrame(X_test_cat_imp, columns=categorical_features, index=X_test.index)

            # Criar indicadores manualmente se necessário
            if missing_config['add_indicator']:
                for col in categorical_features:
                    if X_train[col].isnull().any():
                        indicator_name = f"{col}_missing"
                        X_train_cat_df[indicator_name] = X_train[col].isnull().astype(int)
                        X_val_cat_df[indicator_name] = X_val[col].isnull().astype(int)
                        X_test_cat_df[indicator_name] = X_test[col].isnull().astype(int)
        else:
            logger.info("  Nenhuma coluna categórica tem valores ausentes")
            X_train_cat_df = X_train[categorical_features].copy()
            X_val_cat_df = X_val[categorical_features].copy()
            X_test_cat_df = X_test[categorical_features].copy()
    else:
        X_train_cat_df = pd.DataFrame(index=X_train.index)
        X_val_cat_df = pd.DataFrame(index=X_val.index)
        X_test_cat_df = pd.DataFrame(index=X_test.index)

    # Combinar DataFrames numéricos e categóricos
    X_train_imp = pd.concat([X_train_numeric_df, X_train_cat_df], axis=1)
    X_val_imp = pd.concat([X_val_numeric_df, X_val_cat_df], axis=1)
    X_test_imp = pd.concat([X_test_numeric_df, X_test_cat_df], axis=1)

    # Estatísticas finais
    logger.info(f"Imputação concluída:")
    logger.info(f"  Treino: {X_train_imp.shape[1] - X_train.shape[1]} novas features de indicadores criadas")

    # Verificar colunas com valores ausentes
    missing_cols_train = X_train_imp.columns[X_train_imp.isnull().any()].tolist()
    if missing_cols_train:
        logger.warning(f"Ainda há valores ausentes após imputação: {missing_cols_train}")
    else:
        logger.info("  Não há mais valores ausentes nos dados")

    return X_train_imp, X_val_imp, X_test_imp, imputadores


def encode_categorical_features(X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame,
                               encoding_config: Dict[str, Any] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
    """
    Codifica features categóricas.

    Args:
        X_train: Features de treino
        X_val: Features de validação
        X_test: Features de teste
        encoding_config: Configuração de codificação

    Returns:
        Features codificadas e dicionário com codificadores
    """
    logger.info("Codificando features categóricas...")

    # Configurações padrão
    if encoding_config is None:
        encoding_config = {
            'method': 'onehot',  # 'onehot', 'label', 'target', 'count', 'binary'
            'max_categories': 10,  # Máximo de categorias para one-hot
            'handle_unknown': 'ignore',  # 'error', 'ignore'
            'drop': 'first',  # None, 'first', 'if_binary'
        }

    # Identificar features categóricas
    categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

    if not categorical_features:
        logger.info("Nenhuma feature categórica encontrada para codificar")
        return X_train, X_val, X_test, {}

    logger.info(f"Codificando {len(categorical_features)} features categóricas com método '{encoding_config['method']}'")

    # Preparar DataFrames de resultado
    X_train_encoded = X_train.drop(columns=categorical_features).copy()
    X_val_encoded = X_val.drop(columns=categorical_features).copy()
    X_test_encoded = X_test.drop(columns=categorical_features).copy()

    # Dicionário para armazenar codificadores
    encoders = {}

    # One-Hot Encoding
    if encoding_config['method'] == 'onehot':
        from sklearn.preprocessing import OneHotEncoder

        # Para cada feature categórica
        for feature in categorical_features:
            unique_vals = X_train[feature].nunique()

            # Se tiver muitas categorias, emitir aviso
            if unique_vals > encoding_config['max_categories']:
                logger.warning(f"Feature '{feature}' tem {unique_vals} categorias > {encoding_config['max_categories']} (limite)")

            # Criar e ajustar encoder
            encoder = OneHotEncoder(
                sparse_output=False,  # Alterado de sparse=False para sparse_output=False (sklearn atualizado)
                handle_unknown=encoding_config['handle_unknown'],
                drop=encoding_config['drop']
            )

            # Ajustar no conjunto de treino
            feature_data = X_train[[feature]].fillna('missing')  # Preencher valores ausentes para evitar erros
            encoder.fit(feature_data)

            # Transformar conjuntos
            train_encoded = encoder.transform(X_train[[feature]].fillna('missing'))
            val_encoded = encoder.transform(X_val[[feature]].fillna('missing'))
            test_encoded = encoder.transform(X_test[[feature]].fillna('missing'))

            # Criar nomes de colunas para as categorias
            if encoding_config['drop'] == 'first' and len(encoder.categories_[0]) > 0:
                # Gera nomes para todas as categorias exceto a primeira
                feature_names = [f"{feature}_{cat}" for cat in encoder.categories_[0][1:]]
            else:
                # Gera nomes para todas as categorias
                feature_names = [f"{feature}_{cat}" for cat in encoder.categories_[0]]

            # Verificação de segurança para garantir que o número de colunas bate com os dados transformados
            if len(feature_names) != train_encoded.shape[1]:
                logger.warning(f"Número de colunas geradas ({len(feature_names)}) não corresponde ao output do encoder ({train_encoded.shape[1]}). Ajustando.")
                # Criar nomes genéricos se necessário
                feature_names = [f"{feature}_cat{i}" for i in range(train_encoded.shape[1])]

            # Criar DataFrames
            train_encoded_df = pd.DataFrame(train_encoded, columns=feature_names, index=X_train.index)
            val_encoded_df = pd.DataFrame(val_encoded, columns=feature_names, index=X_val.index)
            test_encoded_df = pd.DataFrame(test_encoded, columns=feature_names, index=X_test.index)

            # Adicionar ao DataFrame de resultado
            X_train_encoded = pd.concat([X_train_encoded, train_encoded_df], axis=1)
            X_val_encoded = pd.concat([X_val_encoded, val_encoded_df], axis=1)
            X_test_encoded = pd.concat([X_test_encoded, test_encoded_df], axis=1)

            # Guardar encoder
            encoders[feature] = encoder

    # Label Encoding
    elif encoding_config['method'] == 'label':
        from sklearn.preprocessing import LabelEncoder

        for feature in categorical_features:
            encoder = LabelEncoder()

            # Preencher valores ausentes
            X_train_feature = X_train[feature].fillna('missing')
            X_val_feature = X_val[feature].fillna('missing')
            X_test_feature = X_test[feature].fillna('missing')

            # Converter para string para garantir compatibilidade
            X_train_feature = X_train_feature.astype(str)
            X_val_feature = X_val_feature.astype(str)
            X_test_feature = X_test_feature.astype(str)

            # Ajustar no conjunto de treino
            X_train_encoded[feature] = encoder.fit_transform(X_train_feature)

            # Lidar com valores não vistos no treino
            X_val_feature_encoded = []
            for val in X_val_feature:
                try:
                    X_val_feature_encoded.append(encoder.transform([val])[0])
                except ValueError:
                    # Valor não visto no treino, usar -1 ou outro valor para indicar
                    X_val_feature_encoded.append(-1)
            X_val_encoded[feature] = X_val_feature_encoded

            X_test_feature_encoded = []
            for val in X_test_feature:
                try:
                    X_test_feature_encoded.append(encoder.transform([val])[0])
                except ValueError:
                    # Valor não visto no treino, usar -1 ou outro valor para indicar
                    X_test_feature_encoded.append(-1)
            X_test_encoded[feature] = X_test_feature_encoded

            # Guardar encoder
            encoders[feature] = encoder

    # Método não implementado ou Target Encoding
    else:
        logger.warning(f"Método de encoding '{encoding_config['method']}' não está completamente implementado. Usando one-hot encoding.")

        # Usar one-hot como fallback
        from sklearn.preprocessing import OneHotEncoder

        for feature in categorical_features:
            # Criar e ajustar encoder com configurações seguras
            encoder = OneHotEncoder(
                sparse_output=False,
                handle_unknown='ignore',
                drop='if_binary'
            )

            # Ajustar no conjunto de treino com tratamento de NaN
            feature_data = X_train[[feature]].fillna('missing')
            encoder.fit(feature_data)

            # Transformar conjuntos
            train_encoded = encoder.transform(X_train[[feature]].fillna('missing'))
            val_encoded = encoder.transform(X_val[[feature]].fillna('missing'))
            test_encoded = encoder.transform(X_test[[feature]].fillna('missing'))

            # Criar nomes de colunas para as categorias
            if encoder.drop and len(encoder.categories_[0]) > 0:
                feature_names = [f"{feature}_{cat}" for cat in encoder.categories_[0][1:]]
            else:
                feature_names = [f"{feature}_{cat}" for cat in encoder.categories_[0]]

            # Verificação de segurança
            if len(feature_names) != train_encoded.shape[1]:
                feature_names = [f"{feature}_cat{i}" for i in range(train_encoded.shape[1])]

            # Criar DataFrames
            train_encoded_df = pd.DataFrame(train_encoded, columns=feature_names, index=X_train.index)
            val_encoded_df = pd.DataFrame(val_encoded, columns=feature_names, index=X_val.index)
            test_encoded_df = pd.DataFrame(test_encoded, columns=feature_names, index=X_test.index)

            # Adicionar ao DataFrame de resultado
            X_train_encoded = pd.concat([X_train_encoded, train_encoded_df], axis=1)
            X_val_encoded = pd.concat([X_val_encoded, val_encoded_df], axis=1)
            X_test_encoded = pd.concat([X_test_encoded, test_encoded_df], axis=1)

            # Guardar encoder
            encoders[feature] = encoder

    logger.info(f"Codificação concluída. Novas dimensões:")
    logger.info(f"  Treino: {X_train.shape[1]} -> {X_train_encoded.shape[1]} features")

    return X_train_encoded, X_val_encoded, X_test_encoded, encoders


def scale_features(X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame,
                  scaling_config: Dict[str, Any] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
    """
    Escala features numéricas.

    Args:
        X_train: Features de treino
        X_val: Features de validação
        X_test: Features de teste
        scaling_config: Configuração de escala

    Returns:
        Features escaladas e dicionário com escaladores
    """
    logger.info("Escalando features numéricas...")

    # Configurações padrão
    if scaling_config is None:
        scaling_config = {
            'method': 'standard',  # 'standard', 'minmax', 'robust', 'none'
            'feature_range': (0, 1),  # Para MinMaxScaler
        }

    # Se não escalar, retornar os dados originais
    if scaling_config['method'] == 'none':
        logger.info("Nenhum método de escala selecionado. Retornando dados originais.")
        return X_train, X_val, X_test, {}

    # Identificar features numéricas
    numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()

    if not numeric_features:
        logger.info("Nenhuma feature numérica encontrada para escalar")
        return X_train, X_val, X_test, {}

    logger.info(f"Escalando {len(numeric_features)} features numéricas com método '{scaling_config['method']}'")

    # Inicializar o escalador adequado
    if scaling_config['method'] == 'standard':
        scaler = StandardScaler()
    elif scaling_config['method'] == 'minmax':
        scaler = MinMaxScaler(feature_range=scaling_config['feature_range'])
    elif scaling_config['method'] == 'robust':
        scaler = RobustScaler()
    else:
        logger.warning(f"Método de escala '{scaling_config['method']}' não reconhecido. Usando StandardScaler.")
        scaler = StandardScaler()

    # Ajustar no conjunto de treino
    scaler.fit(X_train[numeric_features])

    # Transformar conjuntos
    X_train_scaled_array = scaler.transform(X_train[numeric_features])
    X_val_scaled_array = scaler.transform(X_val[numeric_features])
    X_test_scaled_array = scaler.transform(X_test[numeric_features])

    # Criar DataFrames com nomes de colunas preservados
    X_train_scaled = X_train.copy()
    X_val_scaled = X_val.copy()
    X_test_scaled = X_test.copy()

    # Substituir colunas numéricas pelas escaladas
    X_train_scaled[numeric_features] = X_train_scaled_array
    X_val_scaled[numeric_features] = X_val_scaled_array
    X_test_scaled[numeric_features] = X_test_scaled_array

    logger.info(f"Escala concluída. Range de valores (min-max) após escala:")
    for feature in numeric_features[:5]:  # Mostrar apenas as 5 primeiras para não poluir o log
        logger.info(f"  {feature}: [{X_train_scaled[feature].min():.2f}, {X_train_scaled[feature].max():.2f}]")

    if len(numeric_features) > 5:
        logger.info(f"  ... e mais {len(numeric_features) - 5} features")

    return X_train_scaled, X_val_scaled, X_test_scaled, {'scaler': scaler, 'numeric_features': numeric_features}


def create_interaction_features(X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame,
                              interaction_config: Dict[str, Any] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Cria features de interação entre pares de features.

    Args:
        X_train: Features de treino
        X_val: Features de validação
        X_test: Features de teste
        interaction_config: Configuração para criação de interações

    Returns:
        Features com interações adicionadas
    """
    logger.info("Criando features de interação...")

    # Configurações padrão
    if interaction_config is None:
        interaction_config = {
            'max_features': 5,  # Máximo de features para criar interações (para evitar explosão)
            'operations': ['multiply', 'divide', 'add', 'subtract'],  # Operações para criar interações
            'feature_selection': 'importance',  # 'all', 'correlation', 'importance'
            'top_correlated': 10,  # Número de features mais correlacionadas para usar
            'important_features': None,  # Lista de features importantes (se None, usará todas ou baseada em correlação)
        }

    # Identificar features numéricas
    numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()

    if len(numeric_features) < 2:
        logger.info("Menos de 2 features numéricas encontradas. Não é possível criar interações.")
        return X_train, X_val, X_test

    # Selecionar features para criar interações
    features_to_use = numeric_features

    # Limitar número de features para evitar explosão combinatória
    if len(features_to_use) > interaction_config['max_features']:
        if interaction_config['feature_selection'] == 'importance' and interaction_config['important_features']:
            # Usar lista de features importantes fornecida
            important_set = set(interaction_config['important_features'])
            features_to_use = [f for f in numeric_features if f in important_set]
            features_to_use = features_to_use[:interaction_config['max_features']]
            logger.info(f"Usando {len(features_to_use)} features importantes para criar interações")

        elif interaction_config['feature_selection'] == 'correlation':
            # Usar features mais correlacionadas com a variável alvo (não implementado aqui)
            logger.warning("Seleção por correlação não implementada. Usando as primeiras max_features.")
            features_to_use = numeric_features[:interaction_config['max_features']]

        else:
            # Usar as primeiras max_features
            logger.info(f"Limitando a {interaction_config['max_features']} features para criar interações")
            features_to_use = numeric_features[:interaction_config['max_features']]

    logger.info(f"Criando interações entre {len(features_to_use)} features com operações: {interaction_config['operations']}")

    # Inicializar DataFrames de resultado
    X_train_interactions = X_train.copy()
    X_val_interactions = X_val.copy()
    X_test_interactions = X_test.copy()

    # Criar todas as combinações de pares de features
    from itertools import combinations
    feature_pairs = list(combinations(features_to_use, 2))

    # Contadores para estatísticas
    operations_count = {op: 0 for op in interaction_config['operations']}
    skipped_count = 0

    # Para cada par de features, criar interações
    for f1, f2 in feature_pairs:
        for operation in interaction_config['operations']:
            # Nome da nova feature
            if operation == 'multiply':
                new_feature = f"{f1}*{f2}"
                try:
                    X_train_interactions[new_feature] = X_train[f1] * X_train[f2]
                    X_val_interactions[new_feature] = X_val[f1] * X_val[f2]
                    X_test_interactions[new_feature] = X_test[f1] * X_test[f2]
                    operations_count[operation] += 1
                except Exception as e:
                    logger.warning(f"Erro ao criar feature {new_feature}: {str(e)}")
                    skipped_count += 1

            elif operation == 'divide':
                new_feature = f"{f1}/{f2}"
                try:
                    # Evitar divisão por zero
                    X_train_interactions[new_feature] = X_train[f1] / (X_train[f2] + 1e-6)
                    X_val_interactions[new_feature] = X_val[f1] / (X_val[f2] + 1e-6)
                    X_test_interactions[new_feature] = X_test[f1] / (X_test[f2] + 1e-6)

                    # Substituir infinito e NaN
                    X_train_interactions[new_feature].replace([np.inf, -np.inf], np.nan, inplace=True)
                    X_val_interactions[new_feature].replace([np.inf, -np.inf], np.nan, inplace=True)
                    X_test_interactions[new_feature].replace([np.inf, -np.inf], np.nan, inplace=True)

                    X_train_interactions[new_feature].fillna(X_train_interactions[new_feature].mean(), inplace=True)
                    X_val_interactions[new_feature].fillna(X_train_interactions[new_feature].mean(), inplace=True)
                    X_test_interactions[new_feature].fillna(X_train_interactions[new_feature].mean(), inplace=True)

                    operations_count[operation] += 1
                except Exception as e:
                    logger.warning(f"Erro ao criar feature {new_feature}: {str(e)}")
                    skipped_count += 1

            elif operation == 'add':
                new_feature = f"{f1}+{f2}"
                try:
                    X_train_interactions[new_feature] = X_train[f1] + X_train[f2]
                    X_val_interactions[new_feature] = X_val[f1] + X_val[f2]
                    X_test_interactions[new_feature] = X_test[f1] + X_test[f2]
                    operations_count[operation] += 1
                except Exception as e:
                    logger.warning(f"Erro ao criar feature {new_feature}: {str(e)}")
                    skipped_count += 1

            elif operation == 'subtract':
                new_feature = f"{f1}-{f2}"
                try:
                    X_train_interactions[new_feature] = X_train[f1] - X_train[f2]
                    X_val_interactions[new_feature] = X_val[f1] - X_val[f2]
                    X_test_interactions[new_feature] = X_test[f1] - X_test[f2]
                    operations_count[operation] += 1
                except Exception as e:
                    logger.warning(f"Erro ao criar feature {new_feature}: {str(e)}")
                    skipped_count += 1

    # Estatísticas
    total_created = sum(operations_count.values())
    logger.info(f"Criação de interações concluída:")
    logger.info(f"  Total de features criadas: {total_created}")
    logger.info(f"  Features puladas por erros: {skipped_count}")
    logger.info(f"  Por operação: {operations_count}")
    logger.info(f"  Dimensões finais: {X_train_interactions.shape[1]} features")

    return X_train_interactions, X_val_interactions, X_test_interactions


def select_best_features(X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, X_test: pd.DataFrame,
                        selection_config: Dict[str, Any] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Seleciona as melhores features baseado em importância.

    Args:
        X_train: Features de treino
        y_train: Target de treino
        X_val: Features de validação
        X_test: Features de teste
        selection_config: Configuração para seleção de features

    Returns:
        Features selecionadas e lista com nomes das features mantidas
    """
    logger.info("Selecionando melhores features...")

    # Configurações padrão
    if selection_config is None:
        selection_config = {
            'method': 'kbest',  # 'kbest', 'model_based', 'none'
            'k': 20,  # Número de features para selecionar
            'score_func': 'f_classif',  # 'f_classif', 'mutual_info_classif'
            'min_features': 5,  # Mínimo de features para manter
        }

    # Se não selecionar, retornar os dados originais
    if selection_config['method'] == 'none':
        logger.info("Nenhum método de seleção escolhido. Mantendo todas as features.")
        return X_train, X_val, X_test, X_train.columns.tolist()

    # Número de features para selecionar
    k = min(selection_config['k'], X_train.shape[1])
    k = max(k, selection_config['min_features'])  # Garantir pelo menos min_features

    try:
        # Ajustar seletor
        if selection_config['method'] == 'kbest':
            # Escolher função de pontuação
            if selection_config['score_func'] == 'f_classif':
                score_func = f_classif
            else:  # score_func == 'mutual_info_classif'
                score_func = mutual_info_classif

            logger.info(f"Usando SelectKBest com {selection_config['score_func']} para selecionar {k} features")

            # Criar seletor
            selector = SelectKBest(score_func=score_func, k=k)

            # Ajustar no conjunto de treino
            # Verificar se há valores infinitos ou NaN
            if X_train.isnull().any().any() or np.isinf(X_train.to_numpy()).any():
                logger.warning("Detectados valores NaN ou infinitos. Preenchendo com valores seguros.")
                X_train_clean = X_train.copy()
                X_train_clean.replace([np.inf, -np.inf], np.nan, inplace=True)
                X_train_clean.fillna(X_train_clean.mean().fillna(0), inplace=True)

                selector.fit(X_train_clean, y_train)
            else:
                selector.fit(X_train, y_train)

            # Obter máscaras de seleção
            mask = selector.get_support()

            # Obter nomes das features selecionadas
            selected_features = X_train.columns[mask].tolist()

        elif selection_config['method'] == 'model_based':
            logger.warning("Seleção baseada em modelo não implementada nesta versão. Usando SelectKBest.")

            # Usar SelectKBest como fallback
            selector = SelectKBest(score_func=f_classif, k=k)

            # Limpar dados antes do ajuste
            X_train_clean = X_train.copy()
            X_train_clean.replace([np.inf, -np.inf], np.nan, inplace=True)
            X_train_clean.fillna(X_train_clean.mean().fillna(0), inplace=True)

            selector.fit(X_train_clean, y_train)
            mask = selector.get_support()
            selected_features = X_train.columns[mask].tolist()

        else:
            logger.warning(f"Método de seleção '{selection_config['method']}' não reconhecido. Mantendo todas as features.")
            return X_train, X_val, X_test, X_train.columns.tolist()

    except Exception as e:
        logger.error(f"Erro na seleção de features: {str(e)}")
        logger.info("Mantendo todas as features devido ao erro.")
        return X_train, X_val, X_test, X_train.columns.tolist()

    # Se nenhuma feature foi selecionada, manter todas
    if not selected_features:
        logger.warning("Nenhuma feature selecionada. Mantendo todas as features.")
        return X_train, X_val, X_test, X_train.columns.tolist()

    # Filtrar DataFrames
    X_train_selected = X_train[selected_features]
    X_val_selected = X_val[selected_features]
    X_test_selected = X_test[selected_features]

    # Mostrar features selecionadas
    logger.info(f"Seleção concluída. {len(selected_features)} features selecionadas de {X_train.shape[1]}.")

    if len(selected_features) <= 10:
        logger.info(f"Features selecionadas: {selected_features}")
    else:
        logger.info(f"Top 10 features selecionadas: {selected_features[:10]} ...")

    return X_train_selected, X_val_selected, X_test_selected, selected_features


def feature_engineering_pipeline(data_dir: str, output_dir: str = None, timestamp: str = None,
                                missing_config: Dict[str, Any] = None,
                                encoding_config: Dict[str, Any] = None,
                                scaling_config: Dict[str, Any] = None,
                                interaction_config: Dict[str, Any] = None,
                                selection_config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Pipeline completo de engenharia de features.

    Args:
        data_dir: Diretório com os dados processados
        output_dir: Diretório para salvar os dados com features engenheiradas
        timestamp: Timestamp específico para carregar ou None para o mais recente
        missing_config: Configuração para tratamento de valores ausentes
        encoding_config: Configuração para codificação de features categóricas
        scaling_config: Configuração para escala de features numéricas
        interaction_config: Configuração para criação de interações
        selection_config: Configuração para seleção de features

    Returns:
        Dicionário com dados e metadados
    """
    # 1. Configurar diretório de saída
    project_root = get_project_root()

    if output_dir is None:
        output_dir = os.path.join(project_root, 'data', 'interim')
    elif not os.path.isabs(output_dir):
        # Se o caminho não for absoluto, considerar relativo à raiz do projeto
        output_dir = os.path.join(project_root, output_dir)

    os.makedirs(output_dir, exist_ok=True)

    # Timestamp para nomeação dos arquivos
    current_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info(f"Iniciando pipeline de engenharia de features com timestamp {current_timestamp}")

    # 2. Carregar dados processados
    data_dict = load_processed_data(data_dir, timestamp)

    # 3. Separar conjuntos
    df_train = data_dict['train']
    df_val = data_dict['val']
    df_test = data_dict['test']
    metadata = data_dict['metadata']

    # 4. Identificar a variável alvo
    target_col = metadata.get('target_column')
    if not target_col:
        # Tentar encontrar no dataset
        target_options = ['Inadimplente', 'inadimplente', 'Target', 'target', 'Default', 'default', 'Risco_Inadimplencia']
        for col in target_options:
            if col in df_train.columns:
                target_col = col
                break

    if not target_col:
        raise ValueError("Não foi possível identificar a variável alvo")

    logger.info(f"Variável alvo identificada: {target_col}")

    # 5. Separar features e target
    X_train = df_train.drop(columns=[target_col])
    y_train = df_train[target_col]

    X_val = df_val.drop(columns=[target_col])
    y_val = df_val[target_col]

    X_test = df_test.drop(columns=[target_col])
    y_test = df_test[target_col]

    # Estatísticas da variável alvo
    logger.info(f"Distribuição da variável alvo (% positivos):")
    logger.info(f"  Treino: {y_train.mean():.2%}")
    logger.info(f"  Validação: {y_val.mean():.2%}")
    logger.info(f"  Teste: {y_test.mean():.2%}")

    # 6. Tratar valores ausentes
    X_train_imp, X_val_imp, X_test_imp, imputadores = handle_missing_values(
        X_train, X_val, X_test, missing_config
    )

    # 7. Codificar features categóricas
    X_train_enc, X_val_enc, X_test_enc, encoders = encode_categorical_features(
        X_train_imp, X_val_imp, X_test_imp, encoding_config
    )

    # 8. Escalar features numéricas
    X_train_scaled, X_val_scaled, X_test_scaled, scalers = scale_features(
        X_train_enc, X_val_enc, X_test_enc, scaling_config
    )

    # 9. Criar features de interação
    X_train_interactions, X_val_interactions, X_test_interactions = create_interaction_features(
        X_train_scaled, X_val_scaled, X_test_scaled, interaction_config
    )

    # 10. Selecionar melhores features
    X_train_selected, X_val_selected, X_test_selected, selected_features = select_best_features(
        X_train_interactions, y_train, X_val_interactions, X_test_interactions, selection_config
    )

    # 11. Recriar DataFrames completos com target para salvar
    df_train_engineered = pd.concat([X_train_selected, y_train], axis=1)
    df_val_engineered = pd.concat([X_val_selected, y_val], axis=1)
    df_test_engineered = pd.concat([X_test_selected, y_test], axis=1)

    # 12. Salvar resultados
    train_file = os.path.join(output_dir, f"train_engineered_{current_timestamp}.csv")
    val_file = os.path.join(output_dir, f"val_engineered_{current_timestamp}.csv")
    test_file = os.path.join(output_dir, f"test_engineered_{current_timestamp}.csv")

    df_train_engineered.to_csv(train_file, index=False)
    df_val_engineered.to_csv(val_file, index=False)
    df_test_engineered.to_csv(test_file, index=False)

    logger.info(f"Dados com features engenheiradas salvos em:")
    logger.info(f"  Treino: {train_file}")
    logger.info(f"  Validação: {val_file}")
    logger.info(f"  Teste: {test_file}")

    # 13. Criar e salvar metadados
    pipeline_metadata = {
        'timestamp': current_timestamp,
        'original_timestamp': metadata.get('timestamp'),
        'target_column': target_col,
        'original_features': X_train.shape[1],
        'engineered_features': X_train_selected.shape[1],
        'selected_features': selected_features,
        'train_size': X_train_selected.shape[0],
        'val_size': X_val_selected.shape[0],
        'test_size': X_test_selected.shape[0],
    }

    metadata_file = os.path.join(output_dir, f"feature_engineering_metadata_{current_timestamp}.json")

    with open(metadata_file, 'w') as f:
        json.dump(pipeline_metadata, f, indent=4, default=str)

    logger.info(f"Metadados salvos em {metadata_file}")

    # 14. Retornar dicionário com dados e metadados
    result = {
        'train': df_train_engineered,
        'val': df_val_engineered,
        'test': df_test_engineered,
        'X_train': X_train_selected,
        'y_train': y_train,
        'X_val': X_val_selected,
        'y_val': y_val,
        'X_test': X_test_selected,
        'y_test': y_test,
        'target_column': target_col,
        'metadata': pipeline_metadata,
        'transformers': {
            'imputadores': imputadores,
            'encoders': encoders,
            'scalers': scalers,
            'selected_features': selected_features
        }
    }

    logger.info("Pipeline de engenharia de features concluído com sucesso!")

    return result


# Função principal para executar a engenharia de features
if __name__ == "__main__":
    import argparse

    # Obter a raiz do projeto para caminhos padrão
    project_root = get_project_root()
    default_input = os.path.join('data', 'processed')
    default_output = os.path.join('data', 'interim')

    parser = argparse.ArgumentParser(description='Engenharia de features para modelo de inadimplência.')
    parser.add_argument('--input', type=str, default=default_input,
                        help='Diretório com dados processados (relativo à raiz do projeto)')
    parser.add_argument('--output', type=str, default=default_output,
                        help='Diretório para salvar dados com features engenheiradas (relativo à raiz do projeto)')
    parser.add_argument('--timestamp', type=str, help='Timestamp específico para carregar dados processados')

    # Configurações para cada etapa
    parser.add_argument('--no-imputation', action='store_true', help='Pular etapa de imputação')
    parser.add_argument('--no-encoding', action='store_true', help='Pular etapa de codificação categórica')
    parser.add_argument('--no-scaling', action='store_true', help='Pular etapa de escala')
    parser.add_argument('--no-interactions', action='store_true', help='Pular etapa de criação de interações')
    parser.add_argument('--no-selection', action='store_true', help='Pular etapa de seleção de features')

    args = parser.parse_args()

    # Configurar pipelines com base nos argumentos
    missing_config = None if not args.no_imputation else {'method': 'none'}
    encoding_config = None if not args.no_encoding else {'method': 'none'}
    scaling_config = None if not args.no_scaling else {'method': 'none'}
    interaction_config = None if not args.no_interactions else {'max_features': 0}
    selection_config = None if not args.no_selection else {'method': 'none'}

    try:
        # Executar pipeline
        result = feature_engineering_pipeline(
            data_dir=args.input,
            output_dir=args.output,
            timestamp=args.timestamp,
            missing_config=missing_config,
            encoding_config=encoding_config,
            scaling_config=scaling_config,
            interaction_config=interaction_config,
            selection_config=selection_config
        )

        print(f"Engenharia de features concluída! Dados salvos em {os.path.join(project_root, args.output)}")
    except Exception as e:
        print(f"Erro durante a engenharia de features: {str(e)}")
        import traceback
        traceback.print_exc()