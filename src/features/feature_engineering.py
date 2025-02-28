"""
Módulo para engenharia de features do algoritmo de detecção de inadimplência.
Versão otimizada para processadores multicore com foco em uso eficiente de CPU.
"""

import pandas as pd
import numpy as np
import os
import logging
import gc
import time
from datetime import datetime
from typing import Dict, List, Any, Union, Tuple
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
import json
from joblib import Parallel, delayed, parallel_backend
import multiprocessing
import warnings

# Suprimir avisos desnecessários
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Configurar logger
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Definir constantes para otimização
# Número de núcleos para processamento paralelo (deixar 1 livre para o sistema)
N_JOBS = max(1, multiprocessing.cpu_count() - 1)
# Tamanho de lote para processamento
BATCH_SIZE = 5000

# Obter caminho da raiz do projeto
def get_project_root():
    """Retorna o caminho para a raiz do projeto."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, os.pardir, os.pardir))
    return project_root


def load_processed_data(data_dir: str, timestamp: str = None) -> Dict[str, pd.DataFrame]:
    """
    Carrega os conjuntos de dados processados com otimização de memória.
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

    # Primeiro, analisar os tipos de coluna para otimizar uso de memória
    # Apenas amostrar algumas linhas para detectar tipos
    dtypes_sample = pd.read_csv(train_file, nrows=1000)

    # Determinar tipos para cada coluna para uso mais eficiente de memória
    optimized_dtypes = {}
    for col in dtypes_sample.columns:
        # Preservar colunas categóricas como object para processamento posterior
        if dtypes_sample[col].dtype == 'object':
            optimized_dtypes[col] = 'object'
        # Usar tipos mais eficientes para dados numéricos
        elif dtypes_sample[col].dtype.kind in 'iuf':
            if dtypes_sample[col].nunique() < 10 and 'int' in str(dtypes_sample[col].dtype):
                optimized_dtypes[col] = 'int8'  # Para colunas com poucos valores únicos
            elif 'int' in str(dtypes_sample[col].dtype):
                optimized_dtypes[col] = 'int32'
            else:
                optimized_dtypes[col] = 'float32'

    # Carregar dados com dtypes otimizados e chunks para leitura eficiente
    try:
        logger.info("Carregando conjunto de treino...")
        df_train = pd.read_csv(train_file, dtype=optimized_dtypes, chunksize=BATCH_SIZE)
        df_train = pd.concat(df_train, ignore_index=True)

        logger.info("Carregando conjunto de validação...")
        df_val = pd.read_csv(val_file, dtype=optimized_dtypes, chunksize=BATCH_SIZE)
        df_val = pd.concat(df_val, ignore_index=True)

        logger.info("Carregando conjunto de teste...")
        df_test = pd.read_csv(test_file, dtype=optimized_dtypes, chunksize=BATCH_SIZE)
        df_test = pd.concat(df_test, ignore_index=True)
    except Exception as e:
        # Fallback para carregamento padrão se houver erros
        logger.warning(f"Erro ao carregar com tipos otimizados: {str(e)}")
        logger.warning("Usando carregamento padrão...")

        df_train = pd.read_csv(train_file)
        df_val = pd.read_csv(val_file)
        df_test = pd.read_csv(test_file)

    # Forçar coleta de lixo após carregamento
    gc.collect()

    logger.info(f"Dados carregados com sucesso:")
    logger.info(f"  Treino: {df_train.shape[0]} linhas, {df_train.shape[1]} colunas")
    logger.info(f"  Validação: {df_val.shape[0]} linhas, {df_val.shape[1]} colunas")
    logger.info(f"  Teste: {df_test.shape[0]} linhas, {df_test.shape[1]} colunas")
    logger.info(f"  Uso de memória: {df_train.memory_usage(deep=True).sum() / 1024**2:.2f} MB (treino)")

    return {
        'train': df_train,
        'val': df_val,
        'test': df_test,
        'metadata': metadata
    }


def _process_batch_missing_values(batch, col, imputer, is_categorical=False):
    """Processa um lote para imputação de valores ausentes"""
    if is_categorical:
        batch_data = batch[col].astype('object')
    else:
        batch_data = batch[col].astype('float32')

    # Aplicar imputador ao lote
    batch_imputed = imputer.transform(batch_data.values.reshape(-1, 1)).flatten()

    return batch_imputed


def handle_missing_values(X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame,
                         missing_config: Dict[str, Any] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
    """
    Trata valores ausentes nos conjuntos de dados com processamento paralelo.
    """
    start_time = time.time()
    logger.info("Tratando valores ausentes...")

    # Configurações padrão
    if missing_config is None:
        missing_config = {
            'numeric_strategy': 'mean',  # 'mean', 'median', 'most_frequent', 'constant'
            'categorical_strategy': 'most_frequent',  # 'most_frequent', 'constant'
            'constant_value_numeric': 0,
            'constant_value_categorical': 'missing',
            'add_indicator': False  # Desativado para evitar problemas
        }

    # Estatísticas sobre valores ausentes
    missing_percent = X_train.isnull().mean().mean() * 100
    logger.info(f"Percentual de valores ausentes: {missing_percent:.2f}%")

    # Verificar se há valores ausentes para processar
    if missing_percent < 0.1:
        logger.info("Poucos valores ausentes detectados. Usando método rápido...")
        # Método rápido para poucos valores ausentes
        X_train_processed = X_train.fillna(X_train.mean(numeric_only=True))
        X_val_processed = X_val.fillna(X_train.mean(numeric_only=True))
        X_test_processed = X_test.fillna(X_train.mean(numeric_only=True))

        # Preencher colunas não numéricas com a moda
        for col in X_train.select_dtypes(exclude=['number']).columns:
            if X_train[col].isnull().any():
                fill_value = X_train[col].mode()[0]
                X_train_processed[col] = X_train[col].fillna(fill_value)
                X_val_processed[col] = X_val[col].fillna(fill_value)
                X_test_processed[col] = X_test[col].fillna(fill_value)

        logger.info(f"Imputação rápida concluída em {time.time() - start_time:.2f} segundos")
        return X_train_processed, X_val_processed, X_test_processed, {}

    # Cópias para trabalhar com os dados
    X_train_processed = X_train.copy()
    X_val_processed = X_val.copy()
    X_test_processed = X_test.copy()

    # Separar features numéricas e categóricas
    numeric_features = X_train.select_dtypes(include=['int64', 'int32', 'int16', 'int8', 'float64', 'float32']).columns.tolist()
    categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

    # Imputadores
    imputadores = {}

    # Tratar features numéricas com valores ausentes
    num_missing_cols = [col for col in numeric_features if X_train[col].isnull().any()]
    if num_missing_cols:
        logger.info(f"  Processando {len(num_missing_cols)} colunas numéricas com valores ausentes...")

        # Processar cada coluna separadamente para evitar problemas de memória
        with parallel_backend('loky', n_jobs=N_JOBS):
            for col in num_missing_cols:
                # Criar imputador para a coluna
                numeric_imputer = SimpleImputer(
                    strategy=missing_config['numeric_strategy'],
                    fill_value=missing_config['constant_value_numeric'] if missing_config['numeric_strategy'] == 'constant' else None
                )

                # Ajustar no conjunto de treino
                numeric_imputer.fit(X_train[col].values.reshape(-1, 1))

                # Transformar valores
                X_train_processed[col] = numeric_imputer.transform(X_train[col].values.reshape(-1, 1)).flatten()
                X_val_processed[col] = numeric_imputer.transform(X_val[col].values.reshape(-1, 1)).flatten()
                X_test_processed[col] = numeric_imputer.transform(X_test[col].values.reshape(-1, 1)).flatten()

                # Guardar imputador
                imputadores[col] = numeric_imputer

    # Tratar features categóricas com valores ausentes
    cat_missing_cols = [col for col in categorical_features if X_train[col].isnull().any()]
    if cat_missing_cols:
        logger.info(f"  Processando {len(cat_missing_cols)} colunas categóricas com valores ausentes...")

        # Método mais simples e eficiente para colunas categóricas
        for col in cat_missing_cols:
            # Encontrar o valor mais frequente
            if X_train[col].isnull().mean() < 0.5:  # Se menos da metade dos valores são nulos
                most_freq = X_train[col].mode()[0]
                X_train_processed[col] = X_train[col].fillna(most_freq)
                X_val_processed[col] = X_val[col].fillna(most_freq)
                X_test_processed[col] = X_test[col].fillna(most_freq)
            else:
                # Se muitos valores são nulos, criar uma categoria especial
                X_train_processed[col] = X_train[col].fillna("MISSING")
                X_val_processed[col] = X_val[col].fillna("MISSING")
                X_test_processed[col] = X_test[col].fillna("MISSING")

    # Verificar colunas com valores ausentes
    missing_cols_train = X_train_processed.columns[X_train_processed.isnull().any()].tolist()
    if missing_cols_train:
        logger.warning(f"Ainda há valores ausentes após imputação: {missing_cols_train}")
        # Forçar preenchimento de qualquer NaN restante
        X_train_processed = X_train_processed.fillna(0)
        X_val_processed = X_val_processed.fillna(0)
        X_test_processed = X_test_processed.fillna(0)
    else:
        logger.info("  Não há mais valores ausentes nos dados")

    # Forçar coleta de lixo
    gc.collect()

    logger.info(f"Imputação de valores ausentes concluída em {time.time() - start_time:.2f} segundos")
    return X_train_processed, X_val_processed, X_test_processed, imputadores


def _process_categorical_feature(X_train, X_val, X_test, feature, method='label'):
    """Processa uma feature categórica para codificação (label ou onehot)"""
    start_time = time.time()
    result = {}

    # Preencher valores ausentes para garantir
    X_train_feature = X_train[[feature]].fillna('missing')
    X_val_feature = X_val[[feature]].fillna('missing')
    X_test_feature = X_test[[feature]].fillna('missing')

    if method == 'label':
        # Label encoding é mais eficiente em memória
        encoder = LabelEncoder()
        # Ajustar no conjunto de treino
        encoder.fit(X_train_feature[feature].astype(str))

        # Transformar conjuntos
        train_encoded = pd.Series(encoder.transform(X_train_feature[feature].astype(str)), index=X_train_feature.index)

        # Lidar com valores não vistos
        val_encoded = X_val_feature[feature].astype(str).map(
            lambda x: next((i for i, c in enumerate(encoder.classes_) if c == x), -1)
        )
        test_encoded = X_test_feature[feature].astype(str).map(
            lambda x: next((i for i, c in enumerate(encoder.classes_) if c == x), -1)
        )

        # Substituir valores não vistos (-1)
        if (val_encoded == -1).any():
            val_encoded = val_encoded.replace(-1, train_encoded.median())
        if (test_encoded == -1).any():
            test_encoded = test_encoded.replace(-1, train_encoded.median())

        result = {
            'feature': feature,
            'encoder': encoder,
            'method': 'label',
            'train_encoded': train_encoded.values,
            'val_encoded': val_encoded.values,
            'test_encoded': test_encoded.values,
            'output_columns': [feature]
        }

    elif method == 'onehot':
        # One-hot encoding
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

        # Ajustar no conjunto de treino
        encoder.fit(X_train_feature)

        # Transformar conjuntos
        train_encoded = encoder.transform(X_train_feature)
        val_encoded = encoder.transform(X_val_feature)
        test_encoded = encoder.transform(X_test_feature)

        # Criar nomes de colunas
        if hasattr(encoder, 'get_feature_names_out'):
            output_columns = encoder.get_feature_names_out([feature]).tolist()
        else:
            # Para versões antigas do sklearn
            output_columns = [f"{feature}_{cat}" for cat in encoder.categories_[0]]
            if encoder.drop and len(output_columns) > 0:
                output_columns = output_columns[1:]

        result = {
            'feature': feature,
            'encoder': encoder,
            'method': 'onehot',
            'train_encoded': train_encoded,
            'val_encoded': val_encoded,
            'test_encoded': test_encoded,
            'output_columns': output_columns
        }

    return result


def encode_categorical_features(X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame,
                               encoding_config: Dict[str, Any] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
    """
    Codifica features categóricas com processamento paralelo.
    """
    start_time = time.time()
    logger.info("Codificando features categóricas...")

    # Configurações padrão
    if encoding_config is None:
        encoding_config = {
            'method': 'label',  # 'label' é mais eficiente, 'onehot' pode ser melhor para ML
            'max_categories': 10,  # Máximo de categorias para one-hot
            'handle_unknown': 'ignore',
            'drop': 'first',
        }

    # Identificar features categóricas
    categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

    if not categorical_features:
        logger.info("Nenhuma feature categórica encontrada para codificar")
        return X_train, X_val, X_test, {}

    logger.info(f"Codificando {len(categorical_features)} features categóricas com método '{encoding_config['method']}'")

    # Se houver muitas colunas categóricas, usar label encoding forçado
    if len(categorical_features) > 20 and encoding_config['method'] == 'onehot':
        logger.warning(f"Muitas features categóricas ({len(categorical_features)}). Forçando label encoding para eficiência.")
        encoding_config['method'] = 'label'

    # Preparar DataFrames de resultado - converter para float32 para eficiência de memória
    # Remover todas as colunas categóricas
    numeric_columns = X_train.select_dtypes(exclude=['object', 'category']).columns
    X_train_encoded = X_train[numeric_columns].copy().astype('float32')
    X_val_encoded = X_val[numeric_columns].copy().astype('float32')
    X_test_encoded = X_test[numeric_columns].copy().astype('float32')

    # Processar features categóricas em paralelo
    encoding_method = encoding_config['method']

    # Determinar o método para cada feature baseado no número de categorias
    feature_methods = {}
    for feature in categorical_features:
        n_unique = X_train[feature].nunique()
        if n_unique <= encoding_config['max_categories'] and encoding_method == 'onehot':
            feature_methods[feature] = 'onehot'
        else:
            # Se muitas categorias, usar label encoding mesmo se método for onehot
            if n_unique > encoding_config['max_categories'] and encoding_method == 'onehot':
                logger.info(f"Feature '{feature}' tem {n_unique} categorias > {encoding_config['max_categories']}. Usando label encoding.")
            feature_methods[feature] = 'label'

    # Processar em paralelo
    with parallel_backend('loky', n_jobs=min(N_JOBS, len(categorical_features))):
        results = Parallel()(
            delayed(_process_categorical_feature)(X_train, X_val, X_test, feature, feature_methods[feature])
            for feature in categorical_features
        )

    # Dicionário para armazenar codificadores
    encoders = {}

    # Reconstruir o DataFrame com as features codificadas
    for result in results:
        feature = result['feature']
        method = result['method']
        encoders[feature] = result['encoder']

        if method == 'label':
            # Para label encoding, adicionar uma coluna
            X_train_encoded[feature] = result['train_encoded']
            X_val_encoded[feature] = result['val_encoded']
            X_test_encoded[feature] = result['test_encoded']
        else:
            # Para one-hot encoding, adicionar múltiplas colunas
            for i, col_name in enumerate(result['output_columns']):
                X_train_encoded[col_name] = result['train_encoded'][:, i]
                X_val_encoded[col_name] = result['val_encoded'][:, i]
                X_test_encoded[col_name] = result['test_encoded'][:, i]

    # Verificar e converter tipos
    for col in X_train_encoded.columns:
        if X_train_encoded[col].dtype != 'float32':
            X_train_encoded[col] = X_train_encoded[col].astype('float32')
            X_val_encoded[col] = X_val_encoded[col].astype('float32')
            X_test_encoded[col] = X_test_encoded[col].astype('float32')

    # Forçar coleta de lixo
    gc.collect()

    logger.info(f"Codificação concluída em {time.time() - start_time:.2f} segundos. Novas dimensões:")
    logger.info(f"  Treino: {X_train.shape[1]} -> {X_train_encoded.shape[1]} features")

    return X_train_encoded, X_val_encoded, X_test_encoded, encoders


def scale_features(X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame,
                  scaling_config: Dict[str, Any] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
    """
    Escala features numéricas usando processamento em lotes.
    """
    start_time = time.time()
    logger.info("Escalando features numéricas...")

    # Configurações padrão
    if scaling_config is None:
        scaling_config = {
            'method': 'standard',  # 'standard', 'minmax', 'robust', 'none'
            'feature_range': (0, 1),  # Para MinMaxScaler
            'batch_size': BATCH_SIZE  # Tamanho do lote para processamento em lotes
        }

    # Se não escalar, retornar os dados originais
    if scaling_config['method'] == 'none':
        logger.info("Nenhum método de escala selecionado. Retornando dados originais.")
        return X_train, X_val, X_test, {}

    # Verificar se há features numéricas
    if X_train.shape[1] == 0:
        logger.info("Nenhuma feature numérica encontrada para escalar")
        return X_train, X_val, X_test, {}

    # IMPORTANTE: Criar cópias e garantir que todas as colunas são float32
    X_train_scaled = X_train.copy().astype('float32')
    X_val_scaled = X_val.copy().astype('float32')
    X_test_scaled = X_test.copy().astype('float32')

    # Colunas para escalar
    numeric_features = X_train_scaled.columns.tolist()

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

    # Ajustar scaler nos dados de treino
    # Isso pode ser feito em chunks para grandes datasets
    batch_size = scaling_config.get('batch_size', BATCH_SIZE)
    n_samples = X_train_scaled.shape[0]

    if n_samples > batch_size:
        logger.info(f"Usando processamento em lotes para ajustar o escalador (tamanho do lote: {batch_size})")

        # Inicializar arrays para calcular média e variância incrementalmente
        if scaling_config['method'] == 'standard':
            # Para StandardScaler, calculamos incrementalmente
            n_features = len(numeric_features)
            mean_sum = np.zeros(n_features)
            var_sum = np.zeros(n_features)
            n_total = 0

            # Calcular estatísticas em lotes
            for i in range(0, n_samples, batch_size):
                end = min(i + batch_size, n_samples)
                batch = X_train_scaled.iloc[i:end]
                batch_mean = batch.mean()
                batch_var = batch.var()
                batch_size_actual = end - i

                # Atualizar estatísticas
                mean_sum += batch_mean * batch_size_actual
                var_sum += batch_var * batch_size_actual
                n_total += batch_size_actual

            # Calcular média e variância finais
            means = mean_sum / n_total
            variances = var_sum / n_total

            # Ajustar o escalador com as estatísticas calculadas
            scaler.mean_ = means.values
            scaler.var_ = variances.values
            scaler.scale_ = np.sqrt(scaler.var_)
            scaler.n_features_in_ = n_features
            scaler.n_samples_seen_ = n_total
        else:
            # Para outros escaladores, ajustar no conjunto completo
            # Isso pode usar mais memória, mas é necessário para MinMaxScaler e RobustScaler
            scaler.fit(X_train_scaled)
    else:
        # Para datasets menores, ajustar no conjunto completo
        scaler.fit(X_train_scaled)

    # Transformar em lotes
    logger.info(f"Transformando dados em lotes...")

    # Função para transformar um lote de dados
    def transform_batch(df, scaler, batch_size=BATCH_SIZE):
        result = df.copy()
        for i in range(0, df.shape[0], batch_size):
            end = min(i + batch_size, df.shape[0])
            batch_idx = df.index[i:end]
            batch_data = df.loc[batch_idx]
            result.loc[batch_idx] = scaler.transform(batch_data)
        return result

    # Transformar os dados em paralelo usando os diferentes cores
    with parallel_backend('loky', n_jobs=N_JOBS):
        # Dividir em chunks para processamento paralelo
        train_chunks = [X_train_scaled.iloc[i:i+batch_size] for i in range(0, n_samples, batch_size)]
        val_chunks = [X_val_scaled.iloc[i:i+batch_size] for i in range(0, X_val_scaled.shape[0], batch_size)]
        test_chunks = [X_test_scaled.iloc[i:i+batch_size] for i in range(0, X_test_scaled.shape[0], batch_size)]

        # Transformar chunks em paralelo
        train_transformed = Parallel()(delayed(scaler.transform)(chunk) for chunk in train_chunks)
        val_transformed = Parallel()(delayed(scaler.transform)(chunk) for chunk in val_chunks)
        test_transformed = Parallel()(delayed(scaler.transform)(chunk) for chunk in test_chunks)

        # Reconstruir DataFrames
        for i, chunk_transformed in enumerate(train_transformed):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_samples)
            X_train_scaled.iloc[start_idx:end_idx] = chunk_transformed

        for i, chunk_transformed in enumerate(val_transformed):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, X_val_scaled.shape[0])
            X_val_scaled.iloc[start_idx:end_idx] = chunk_transformed

        for i, chunk_transformed in enumerate(test_transformed):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, X_test_scaled.shape[0])
            X_test_scaled.iloc[start_idx:end_idx] = chunk_transformed

    # Forçar coleta de lixo após processamento
    gc.collect()

    logger.info(f"Escala concluída em {time.time() - start_time:.2f} segundos")

    return X_train_scaled, X_val_scaled, X_test_scaled, {'scaler': scaler, 'numeric_features': numeric_features}


def create_interaction_features(X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame,
                              interaction_config: Dict[str, Any] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Cria features de interação entre pares de features com foco em eficiência de CPU.
    """
    start_time = time.time()
    logger.info("Criando features de interação...")

    # Configurações padrão
    if interaction_config is None:
        interaction_config = {
            'max_features': 3,  # Máximo de features para criar interações (para evitar explosão)
            'operations': ['multiply'],  # Operações para criar interações
            'feature_selection': 'all',  # 'all', 'correlation', 'importance'
            'top_correlated': 10,  # Número de features mais correlacionadas para usar
            'important_features': None,  # Lista de features importantes
        }

    # Garantir que a chave 'feature_selection' exista no dicionário
    if 'feature_selection' not in interaction_config:
        interaction_config['feature_selection'] = 'all'

    # Identificar features numéricas
    numeric_features = X_train.columns.tolist()

    if len(numeric_features) < 2:
        logger.info("Menos de 2 features numéricas encontradas. Não é possível criar interações.")
        return X_train, X_val, X_test

    # Selecionar features para criar interações
    features_to_use = numeric_features

    # Limitar número de features para evitar explosão combinatória
    max_features = min(interaction_config['max_features'], 5)  # Limitar para no máximo 5 features para evitar explosão

    if len(features_to_use) > max_features:
        if interaction_config['feature_selection'] == 'importance' and interaction_config['important_features']:
            # Usar lista de features importantes fornecida
            important_set = set(interaction_config['important_features'])
            features_to_use = [f for f in numeric_features if f in important_set]
            features_to_use = features_to_use[:max_features]
            logger.info(f"Usando {len(features_to_use)} features importantes para criar interações")
        else:
            # Selecionar features aleatoriamente para evitar viés
            np.random.seed(42)  # Para reprodutibilidade
            features_to_use = np.random.choice(numeric_features, size=max_features, replace=False).tolist()
            logger.info(f"Selecionando aleatoriamente {max_features} features para criar interações")

    logger.info(f"Criando interações entre {len(features_to_use)} features com operações: {interaction_config['operations']}")

    # Inicializar DataFrames de resultado - reutilizamos os originais para economia de memória
    X_train_interactions = X_train.copy()
    X_val_interactions = X_val.copy()
    X_test_interactions = X_test.copy()

    # Gerar pares de features
    from itertools import combinations
    feature_pairs = list(combinations(features_to_use, 2))

    # Limitar número de pares para evitar explosão
    max_pairs = 10  # Limitar para no máximo 10 pares
    if len(feature_pairs) > max_pairs:
        logger.info(f"Limitando para {max_pairs} pares de features (de {len(feature_pairs)} possíveis)")
        feature_pairs = feature_pairs[:max_pairs]

    # Contadores para estatísticas
    operations_count = {op: 0 for op in interaction_config['operations']}
    created_features = []
    skipped_count = 0

    # Função para criar uma feature de interação de forma otimizada
    def create_interaction(X_train, X_val, X_test, f1, f2, operation):
        if operation == 'multiply':
            new_feature = f"{f1}*{f2}"
            try:
                X_train[new_feature] = X_train[f1] * X_train[f2]
                X_val[new_feature] = X_val[f1] * X_val[f2]
                X_test[new_feature] = X_test[f1] * X_test[f2]
                return new_feature, True
            except Exception as e:
                logger.warning(f"Erro ao criar feature {new_feature}: {str(e)}")
                return None, False
        elif operation == 'add':
            new_feature = f"{f1}+{f2}"
            try:
                X_train[new_feature] = X_train[f1] + X_train[f2]
                X_val[new_feature] = X_val[f1] + X_val[f2]
                X_test[new_feature] = X_test[f1] + X_test[f2]
                return new_feature, True
            except Exception as e:
                logger.warning(f"Erro ao criar feature {new_feature}: {str(e)}")
                return None, False
        return None, False

    # Criar interações
    logger.info(f"Criando {len(feature_pairs) * len(interaction_config['operations'])} possíveis interações...")

    # Processamento sequencial é mais eficiente para criação de features
    # por não exigir overhead de comunicação entre processos
    for f1, f2 in feature_pairs:
        for operation in interaction_config['operations']:
            new_feature, success = create_interaction(
                X_train_interactions, X_val_interactions, X_test_interactions, f1, f2, operation
            )
            if success:
                created_features.append(new_feature)
                operations_count[operation] += 1
            else:
                skipped_count += 1

            # Forçar coleta de lixo a cada 10 features para evitar vazamento de memória
            if len(created_features) % 10 == 0:
                gc.collect()

    # Estatísticas
    total_created = sum(operations_count.values())
    logger.info(f"Criação de interações concluída em {time.time() - start_time:.2f} segundos:")
    logger.info(f"  Total de features criadas: {total_created}")
    logger.info(f"  Features puladas por erros: {skipped_count}")
    logger.info(f"  Dimensões finais: {X_train_interactions.shape[1]} features")

    # Forçar coleta de lixo
    gc.collect()

    return X_train_interactions, X_val_interactions, X_test_interactions


def select_best_features(X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, X_test: pd.DataFrame,
                        selection_config: Dict[str, Any] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Seleciona as melhores features com foco em CPU.
    """
    start_time = time.time()
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

    # Se o número de features for pequeno, pular seleção
    if X_train.shape[1] <= selection_config['k']:
        logger.info(f"Número de features ({X_train.shape[1]}) é menor que k ({selection_config['k']}). Mantendo todas.")
        return X_train, X_val, X_test, X_train.columns.tolist()

    # Número de features para selecionar
    k = min(selection_config['k'], X_train.shape[1])
    k = max(k, selection_config['min_features'])  # Garantir pelo menos min_features

    try:
        # Trabalhar com uma cópia limpa dos dados
        X_train_clean = X_train.copy()

        # Substituir valores problemáticos
        X_train_clean.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Método rápido de imputação para valores NaN
        for col in X_train_clean.columns:
            if X_train_clean[col].isnull().any():
                X_train_clean[col] = X_train_clean[col].fillna(X_train_clean[col].median())

        # Converter para float32 para economia de memória
        X_train_clean = X_train_clean.astype('float32')

        # Ajustar seletor
        if selection_config['method'] == 'kbest':
            # Escolher função de pontuação
            if selection_config['score_func'] == 'f_classif':
                score_func = f_classif
            else:  # 'mutual_info_classif'
                score_func = mutual_info_classif

            logger.info(f"Usando SelectKBest com {selection_config['score_func']} para selecionar {k} features")

            with parallel_backend('loky', n_jobs=1):  # Parallelismo não ajuda aqui
                # Criar seletor - processamento em um único core para evitar overhead
                selector = SelectKBest(score_func=score_func, k=k)

                # Ajustar no conjunto de treino
                selector.fit(X_train_clean, y_train)

            # Obter máscaras de seleção
            mask = selector.get_support()

            # Obter nomes das features selecionadas
            selected_features = X_train.columns[mask].tolist()
        else:
            logger.warning(f"Método de seleção '{selection_config['method']}' não suportado. Mantendo todas as features.")
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
    logger.info(f"Seleção concluída em {time.time() - start_time:.2f} segundos. {len(selected_features)} features selecionadas de {X_train.shape[1]}.")

    # Forçar coleta de lixo
    gc.collect()

    return X_train_selected, X_val_selected, X_test_selected, selected_features


def feature_engineering_pipeline(data_dir: str, output_dir: str = None, timestamp: str = None,
                                missing_config: Dict[str, Any] = None,
                                encoding_config: Dict[str, Any] = None,
                                scaling_config: Dict[str, Any] = None,
                                interaction_config: Dict[str, Any] = None,
                                selection_config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Pipeline completo de engenharia de features otimizado para CPU.
    """
    # Inicia contagem de tempo
    total_start_time = time.time()

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
    logger.info(f"Utilizando {N_JOBS} núcleos de CPU para processamento")

    try:
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

        # Coletar lixo antes das operações de processamento
        gc.collect()

        # 6. Tratar valores ausentes
        X_train, X_val, X_test, imputadores = handle_missing_values(
            X_train, X_val, X_test, missing_config
        )
        gc.collect()  # Liberar memória

        # 7. Codificar features categóricas
        X_train, X_val, X_test, encoders = encode_categorical_features(
            X_train, X_val, X_test, encoding_config
        )
        gc.collect()  # Liberar memória

        # 8. Escalar features numéricas
        X_train, X_val, X_test, scalers = scale_features(
            X_train, X_val, X_test, scaling_config
        )
        gc.collect()  # Liberar memória

        # 9. Criar features de interação
        X_train, X_val, X_test = create_interaction_features(
            X_train, X_val, X_test, interaction_config
        )
        gc.collect()  # Liberar memória

        # 10. Selecionar melhores features
        X_train, X_val, X_test, selected_features = select_best_features(
            X_train, y_train, X_val, X_test, selection_config
        )
        gc.collect()  # Liberar memória

        # 11. Recriar DataFrames completos com target para salvar
        logger.info("Preparando dados para salvar...")

        # Usar valores do índice original para preservar ordem
        df_train_engineered = pd.DataFrame(index=X_train.index)
        df_val_engineered = pd.DataFrame(index=X_val.index)
        df_test_engineered = pd.DataFrame(index=X_test.index)

        # Adicionar features selecionadas
        for col in X_train.columns:
            df_train_engineered[col] = X_train[col]
            df_val_engineered[col] = X_val[col]
            df_test_engineered[col] = X_test[col]

        # Adicionar target
        df_train_engineered[target_col] = y_train
        df_val_engineered[target_col] = y_val
        df_test_engineered[target_col] = y_test

        # 12. Salvar resultados em lotes para economia de memória
        train_file = os.path.join(output_dir, f"train_engineered_{current_timestamp}.csv")
        val_file = os.path.join(output_dir, f"val_engineered_{current_timestamp}.csv")
        test_file = os.path.join(output_dir, f"test_engineered_{current_timestamp}.csv")

        logger.info("Salvando conjunto de treino...")
        df_train_engineered.to_csv(train_file, index=False)

        logger.info("Salvando conjunto de validação...")
        df_val_engineered.to_csv(val_file, index=False)

        logger.info("Salvando conjunto de teste...")
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
            'original_features': len(X_train.columns),
            'engineered_features': len(selected_features),
            'selected_features': selected_features,
            'train_size': X_train.shape[0],
            'val_size': X_val.shape[0],
            'test_size': X_test.shape[0],
            'elapsed_time_seconds': time.time() - total_start_time,
            'cores_used': N_JOBS
        }

        metadata_file = os.path.join(output_dir, f"feature_engineering_metadata_{current_timestamp}.json")

        with open(metadata_file, 'w') as f:
            json.dump(pipeline_metadata, f, indent=4, default=str)

        logger.info(f"Metadados salvos em {metadata_file}")

        # Limpar objetos grandes antes de retornar
        del df_train_engineered, df_val_engineered, df_test_engineered
        gc.collect()

        # 14. Retornar dicionário com dados e metadados
        total_time = time.time() - total_start_time
        logger.info(f"Pipeline de engenharia de features concluído com sucesso em {total_time:.2f} segundos!")

        return {
            'train_file': train_file,
            'val_file': val_file,
            'test_file': test_file,
            'target_column': target_col,
            'selected_features': selected_features,
            'metadata': pipeline_metadata,
            'elapsed_time': total_time
        }

    except Exception as e:
        logger.error(f"Erro no pipeline de engenharia de features: {str(e)}")
        # Forçar liberação de memória antes de sair
        gc.collect()
        import traceback
        traceback.print_exc()
        raise


# Função principal para executar a engenharia de features
if __name__ == "__main__":
    import argparse

    # Obter a raiz do projeto para caminhos padrão
    project_root = get_project_root()
    default_input = os.path.join('data', 'processed')
    default_output = os.path.join('data', 'processed')

    parser = argparse.ArgumentParser(description='Engenharia de features para modelo de inadimplência (Otimizado para CPU).')
    parser.add_argument('--input', type=str, default=default_input,
                        help='Diretório com dados processados (relativo à raiz do projeto)')
    parser.add_argument('--output', type=str, default=default_output,
                        help='Diretório para salvar dados com features engenheiradas (relativo à raiz do projeto)')
    parser.add_argument('--timestamp', type=str, help='Timestamp específico para carregar dados processados')
    parser.add_argument('--cores', type=int, default=N_JOBS,
                        help=f'Número de cores a serem usados (padrão: {N_JOBS})')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE,
                        help=f'Tamanho do lote para processamento (padrão: {BATCH_SIZE})')

    # Configurações para cada etapa
    parser.add_argument('--no-imputation', action='store_true', help='Pular etapa de imputação')
    parser.add_argument('--no-encoding', action='store_true', help='Pular etapa de codificação categórica')
    parser.add_argument('--no-scaling', action='store_true', help='Pular etapa de escala')
    parser.add_argument('--no-interactions', action='store_true', help='Pular etapa de criação de interações')
    parser.add_argument('--no-selection', action='store_true', help='Pular etapa de seleção de features')
    parser.add_argument('--label-encoding', action='store_true', help='Forçar uso de label encoding em vez de one-hot')

    args = parser.parse_args()

    # Atualizar constantes globais
    if args.cores > 0:
        N_JOBS = args.cores
    if args.batch_size > 0:
        BATCH_SIZE = args.batch_size

    # Configurar pipelines com base nos argumentos
    missing_config = None if not args.no_imputation else {'method': 'none'}

    encoding_config = None
    if args.no_encoding:
        encoding_config = {'method': 'none'}
    elif args.label_encoding:
        encoding_config = {'method': 'label', 'max_categories': 10}

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
        print(f"Tempo total: {result['elapsed_time']:.2f} segundos")
    except Exception as e:
        print(f"Erro durante a engenharia de features: {str(e)}")
        import traceback
        traceback.print_exc()