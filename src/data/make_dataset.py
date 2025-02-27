"""
Módulo para processamento de dados do algoritmo de detecção de inadimplência.

Este módulo contém funções para carregar, limpar, transformar e dividir os dados
para treinamento e avaliação de modelos de inadimplência bancária.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import logging
from datetime import datetime
from typing import Tuple, Dict, Any, List, Union

# Configurar logger
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


# Obter caminho da raiz do projeto
def get_project_root():
    """Retorna o caminho para a raiz do projeto."""
    # Assumindo que este script está em src/data/
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Subir dois níveis para chegar à raiz do projeto
    project_root = os.path.abspath(os.path.join(script_dir, os.pardir, os.pardir))
    return project_root


def load_data(file_path: str) -> pd.DataFrame:
    """
    Carrega os dados brutos de um arquivo CSV ou Excel.

    Args:
        file_path: Caminho para o arquivo de dados

    Returns:
        DataFrame com os dados brutos

    Raises:
        FileNotFoundError: Se o arquivo não for encontrado
        ValueError: Se o formato do arquivo não for suportado
    """
    logger.info(f"Carregando dados de: {file_path}")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Arquivo não encontrado: {file_path}")

    # Verificar extensão do arquivo
    _, ext = os.path.splitext(file_path)

    try:
        if ext.lower() == '.csv':
            # Tentar diferentes encodings e delimitadores
            try:
                df = pd.read_csv(file_path, encoding='utf-8')
            except UnicodeDecodeError:
                df = pd.read_csv(file_path, encoding='latin1')
            except pd.errors.ParserError:
                # Se falhar, tentar com delimitador diferente
                df = pd.read_csv(file_path, encoding='utf-8', sep=';')

        elif ext.lower() in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path)
        else:
            raise ValueError(f"Formato de arquivo não suportado: {ext}")

        # Verificar se os dados foram carregados corretamente
        if df.empty:
            raise ValueError("O arquivo não contém dados")

        logger.info(f"Dados carregados com sucesso: {df.shape[0]} linhas, {df.shape[1]} colunas")

        # Informações básicas sobre o dataset
        logger.info(f"Colunas no dataset: {df.columns.tolist()}")
        logger.info(f"Tipos de dados: {df.dtypes.value_counts().to_dict()}")

        # Verificar se a coluna alvo está presente
        target_options = ['Inadimplente', 'inadimplente', 'Target', 'target', 'Default', 'default']
        target_present = any(col in df.columns for col in target_options)

        if not target_present:
            logger.warning("Aviso: Não foi encontrada coluna com nome sugestivo de variável alvo")

        return df

    except Exception as e:
        logger.error(f"Erro ao carregar dados: {str(e)}")
        raise


def clean_data(df: pd.DataFrame, config: Dict[str, Any] = None) -> pd.DataFrame:
    """
    Realiza a limpeza inicial dos dados.

    Args:
        df: DataFrame com os dados brutos
        config: Dicionário com configurações para limpeza dos dados

    Returns:
        DataFrame com os dados limpos
    """
    logger.info("Iniciando limpeza de dados...")

    # Configurações padrão se não forem fornecidas
    if config is None:
        config = {
            'remove_duplicates': True,
            'handle_missing': True,
            'missing_threshold': 0.8,  # Remove colunas com mais de 80% de valores ausentes
            'convert_types': True,
            'datetime_columns': [],  # Lista de colunas para converter para datetime
            'numeric_columns': [],  # Lista de colunas para converter para numérico
            'categorical_columns': []  # Lista de colunas para converter para categórico
        }

    # Cópia para não modificar o original
    df_clean = df.copy()

    # Remover duplicatas
    if config['remove_duplicates']:
        n_before = df_clean.shape[0]
        df_clean = df_clean.drop_duplicates()
        n_after = df_clean.shape[0]

        if n_before > n_after:
            logger.info(f"Removidas {n_before - n_after} linhas duplicadas")

    # Verificar valores ausentes
    missing_stats = df_clean.isnull().mean().sort_values(ascending=False)
    missing_cols = missing_stats[missing_stats > 0]

    if not missing_cols.empty:
        logger.info("Colunas com valores ausentes:")
        for col, pct in missing_cols.items():
            logger.info(f"  - {col}: {pct * 100:.2f}%")

    # Remover colunas com muitos valores ausentes
    if config['handle_missing']:
        cols_to_drop = missing_stats[missing_stats > config['missing_threshold']].index.tolist()

        if cols_to_drop:
            logger.info(
                f"Removendo colunas com mais de {config['missing_threshold'] * 100}% de valores ausentes: {cols_to_drop}")
            df_clean = df_clean.drop(columns=cols_to_drop)

    # Converter tipos de dados
    if config['convert_types']:
        # Converter colunas datetime
        for col in config['datetime_columns']:
            if col in df_clean.columns:
                try:
                    df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')
                    logger.info(f"Coluna {col} convertida para datetime")
                except Exception as e:
                    logger.warning(f"Erro ao converter coluna {col} para datetime: {str(e)}")

        # Converter colunas numéricas
        for col in config['numeric_columns']:
            if col in df_clean.columns:
                try:
                    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
                    logger.info(f"Coluna {col} convertida para numérico")
                except Exception as e:
                    logger.warning(f"Erro ao converter coluna {col} para numérico: {str(e)}")

        # Converter colunas categóricas
        for col in config['categorical_columns']:
            if col in df_clean.columns:
                try:
                    df_clean[col] = df_clean[col].astype('category')
                    logger.info(f"Coluna {col} convertida para categórico")
                except Exception as e:
                    logger.warning(f"Erro ao converter coluna {col} para categórico: {str(e)}")

    # Verificar faixas inválidas em variáveis importantes
    numeric_cols = df_clean.select_dtypes(include=['int64', 'float64']).columns

    for col in numeric_cols:
        if 'score' in col.lower() and col + '_original' not in df_clean.columns:
            # Score de crédito geralmente está entre 0 e 1000
            if df_clean[col].max() > 1000 or df_clean[col].min() < 0:
                logger.warning(f"Coluna {col} pode ter valores fora da faixa esperada para score de crédito")

        elif 'idade' in col.lower() and col + '_original' not in df_clean.columns:
            # Idade geralmente está entre 18 e 100
            if df_clean[col].max() > 100 or df_clean[col].min() < 18:
                logger.warning(f"Coluna {col} pode ter valores fora da faixa esperada para idade")

    logger.info(f"Limpeza concluída. Dataset final: {df_clean.shape[0]} linhas, {df_clean.shape[1]} colunas")

    return df_clean


def identify_column_types(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Identifica automaticamente os tipos de colunas no DataFrame.

    Args:
        df: DataFrame com os dados

    Returns:
        Dicionário com listas de colunas por tipo
    """
    logger.info("Identificando tipos de colunas...")

    # Inicializar dicionário
    column_types = {
        'id_columns': [],
        'datetime_columns': [],
        'numeric_columns': [],
        'categorical_columns': [],
        'binary_columns': [],
        'text_columns': [],
        'target_column': None
    }

    # Identificar possíveis colunas alvo
    target_names = ['Inadimplente', 'inadimplente', 'Target', 'target', 'Default', 'default']
    for col in target_names:
        if col in df.columns:
            column_types['target_column'] = col
            break

    # Identificar colunas por nome e tipo
    for col in df.columns:
        col_lower = col.lower()

        # Skip target column
        if col == column_types['target_column']:
            continue

        # ID columns
        if ('id' in col_lower or 'código' in col_lower or 'codigo' in col_lower) and df[col].nunique() > df.shape[
            0] * 0.9:
            column_types['id_columns'].append(col)

        # Datetime columns
        elif ('data' in col_lower or 'date' in col_lower or 'dt_' in col_lower) or pd.api.types.is_datetime64_any_dtype(
                df[col]):
            column_types['datetime_columns'].append(col)

        # Binary columns
        elif df[col].nunique() == 2:
            column_types['binary_columns'].append(col)

        # Categorical columns (menos de 20% de valores únicos)
        elif (pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_categorical_dtype(df[col])) and df[
            col].nunique() < df.shape[0] * 0.2:
            column_types['categorical_columns'].append(col)

        # Text columns (more than 100 unique values and object type)
        elif pd.api.types.is_object_dtype(df[col]) and df[col].nunique() > 100:
            column_types['text_columns'].append(col)

        # Numeric columns
        elif pd.api.types.is_numeric_dtype(df[col]):
            column_types['numeric_columns'].append(col)

        # Default to categorical for other object types
        elif pd.api.types.is_object_dtype(df[col]):
            column_types['categorical_columns'].append(col)

    # Log de resultados
    for col_type, cols in column_types.items():
        if col_type == 'target_column':
            logger.info(f"Coluna alvo identificada: {cols}")
        else:
            logger.info(f"{col_type}: {len(cols)} colunas")

    return column_types


def prepare_target_variable(df: pd.DataFrame, target_config: Dict[str, Any] = None) -> Tuple[pd.DataFrame, str]:
    """
    Prepara a variável alvo para modelagem de inadimplência.

    Args:
        df: DataFrame com os dados
        target_config: Configurações para a variável alvo

    Returns:
        DataFrame com a variável alvo preparada e nome da coluna alvo
    """
    logger.info("Preparando variável alvo...")

    # Configurações padrão
    if target_config is None:
        target_config = {
            'target_col': None,  # Nome da coluna alvo, se já existir
            'create_target': True,  # Se True, cria uma variável alvo se não existir
            'days_overdue_threshold': 30,  # Dias de atraso para considerar inadimplente
            'default_indicators': ['Inadimplente', 'Em Atraso', 'Default'],  # Valores que indicam inadimplência
            'use_risk_score': False,  # Se True, usa score de risco para criar a variável alvo
            'risk_score_threshold': 600,  # Threshold para considerar inadimplente baseado no score
            'target_name': 'Inadimplente'  # Nome da nova coluna alvo
        }

    df_result = df.copy()

    # Verificar se a variável alvo já existe
    target_options = ['Inadimplente', 'inadimplente', 'Target', 'target', 'Default', 'default',
                      'Risco_Inadimplencia', 'risco_inadimplencia', 'Status_Emprestimo', 'status_emprestimo']
    existing_target = None

    if target_config['target_col'] and target_config['target_col'] in df.columns:
        existing_target = target_config['target_col']
    else:
        for col in target_options:
            if col in df.columns:
                existing_target = col
                break

    # Caso especial: Se encontrarmos Risco_Inadimplencia, usar ela diretamente
    if any(col.lower() == 'risco_inadimplencia' for col in df.columns):
        risk_col = next(col for col in df.columns if col.lower() == 'risco_inadimplencia')
        existing_target = risk_col
        logger.info(f"Usando coluna '{risk_col}' como variável alvo")

    # Se já existe uma variável alvo e não queremos criar nova
    if existing_target and not target_config['create_target']:
        logger.info(f"Usando variável alvo existente: {existing_target}")

        # Verificar se é binária e ajustar se necessário
        if df_result[existing_target].nunique() > 2:
            logger.warning(f"Variável alvo {existing_target} tem mais de 2 valores únicos. Convertendo para binária.")

            # Tentar converter para binária baseado nos valores
            if df_result[existing_target].dtype == 'object':
                # Padronizar valores para garantir consistência
                default_map = {}
                for val in df_result[existing_target].unique():
                    val_lower = str(val).lower()
                    if any(ind.lower() in val_lower for ind in target_config['default_indicators']):
                        default_map[val] = 1
                    else:
                        default_map[val] = 0

                df_result[existing_target] = df_result[existing_target].map(default_map)
                logger.info(f"Variável alvo convertida para binária: {dict(df_result[existing_target].value_counts())}")
            else:
                # Se for numérica, usar threshold para binarizar
                threshold = df_result[existing_target].median()
                df_result[existing_target] = (df_result[existing_target] > threshold).astype(int)
                logger.info(
                    f"Variável alvo numérica binarizada com threshold {threshold}: {dict(df_result[existing_target].value_counts())}")

        return df_result, existing_target

    # Criar nova variável alvo se necessário
    if target_config['create_target']:
        logger.info("Criando nova variável alvo...")
        target_name = target_config['target_name']

        # Estratégia 1: Baseada em dias de atraso
        dias_atraso_cols = [col for col in df.columns if 'dias_atraso' in col.lower() or 'days_overdue' in col.lower()]

        if dias_atraso_cols:
            logger.info(f"Criando variável alvo baseada em dias de atraso: {dias_atraso_cols[0]}")
            dias_col = dias_atraso_cols[0]
            df_result[target_name] = (df_result[dias_col] > target_config['days_overdue_threshold']).astype(int)
            logger.info(f"Variável alvo criada: {dict(df_result[target_name].value_counts())}")
            return df_result, target_name

        # Estratégia 2: Baseada em status de empréstimo/pagamento
        status_cols = [col for col in df.columns if 'status' in col.lower() or 'situacao' in col.lower()]

        if status_cols:
            logger.info(f"Criando variável alvo baseada em status: {status_cols[0]}")
            status_col = status_cols[0]

            # Mapear valores para binário
            default_map = {}
            for val in df_result[status_col].unique():
                val_lower = str(val).lower()
                if any(ind.lower() in val_lower for ind in target_config['default_indicators']):
                    default_map[val] = 1
                else:
                    default_map[val] = 0

            df_result[target_name] = df_result[status_col].map(default_map)
            logger.info(f"Variável alvo criada: {dict(df_result[target_name].value_counts())}")
            return df_result, target_name

        # Estratégia 3: Baseada em score de risco
        if target_config['use_risk_score']:
            score_cols = [col for col in df.columns if 'score' in col.lower() or 'pontuacao' in col.lower()]

            if score_cols:
                logger.info(f"Criando variável alvo baseada em score de risco: {score_cols[0]}")
                score_col = score_cols[0]

                # Verificar se score mais alto é melhor ou pior
                # Geralmente, score de crédito mais alto é melhor (menor risco)
                df_result[target_name] = (df_result[score_col] < target_config['risk_score_threshold']).astype(int)
                logger.info(f"Variável alvo criada: {dict(df_result[target_name].value_counts())}")
                return df_result, target_name

        # Se não conseguiu criar, avisar
        logger.warning("Não foi possível criar uma variável alvo automaticamente")
        logger.warning("Crie a variável alvo manualmente ou forneça mais informações")

        # Criar uma variável alvo aleatória (apenas para demonstração)
        logger.warning("Criando variável alvo ALEATÓRIA apenas para demonstração")
        df_result[target_name] = np.random.binomial(1, 0.15, size=df_result.shape[0])
        logger.info(f"Variável alvo aleatória criada: {dict(df_result[target_name].value_counts())}")
        return df_result, target_name

    # Se chegou aqui, não tem variável alvo
    logger.error("Não foi possível identificar ou criar uma variável alvo")
    return df_result, None


def split_data(df: pd.DataFrame, target_col: str, split_config: Dict[str, Any] = None) -> Dict[
    str, Union[pd.DataFrame, np.ndarray]]:
    """
    Divide os dados em conjuntos de treino, validação e teste.

    Args:
        df: DataFrame com os dados preparados
        target_col: Nome da coluna alvo
        split_config: Configurações para divisão dos dados

    Returns:
        Dicionário com os conjuntos de dados divididos
    """
    logger.info("Dividindo dados em conjuntos de treino, validação e teste...")

    # Configurações padrão
    if split_config is None:
        split_config = {
            'test_size': 0.2,
            'val_size': 0.1,  # Em relação ao conjunto de treino
            'stratify': True,
            'random_state': 42,
            'temporal_split': False,
            'temporal_col': None,
        }

    # Verificar se a coluna alvo existe
    if target_col not in df.columns:
        raise ValueError(f"Coluna alvo {target_col} não encontrada no DataFrame")

    # Separar features e target
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Verificar se há colunas de ID para remover
    id_cols = [col for col in X.columns if 'id' in col.lower() and X[col].nunique() == X.shape[0]]
    if id_cols:
        logger.info(f"Removendo colunas de ID para divisão: {id_cols}")
        X = X.drop(columns=id_cols)

    # Divisão temporal
    if split_config['temporal_split'] and split_config['temporal_col']:
        temporal_col = split_config['temporal_col']
        if temporal_col not in df.columns:
            logger.warning(f"Coluna temporal {temporal_col} não encontrada. Usando divisão aleatória.")
            split_config['temporal_split'] = False
        else:
            logger.info(f"Usando divisão temporal baseada na coluna {temporal_col}")

            # Ordenar por data
            sorted_idx = np.argsort(df[temporal_col].values)
            X_sorted = X.iloc[sorted_idx].reset_index(drop=True)
            y_sorted = y.iloc[sorted_idx].reset_index(drop=True)

            # Calcular pontos de corte
            n_samples = X_sorted.shape[0]
            test_size = int(n_samples * split_config['test_size'])
            val_size = int((n_samples - test_size) * split_config['val_size'])

            # Separar conjuntos
            X_train = X_sorted.iloc[:(n_samples - test_size - val_size)]
            y_train = y_sorted.iloc[:(n_samples - test_size - val_size)]

            X_val = X_sorted.iloc[(n_samples - test_size - val_size):(n_samples - test_size)]
            y_val = y_sorted.iloc[(n_samples - test_size - val_size):(n_samples - test_size)]

            X_test = X_sorted.iloc[(n_samples - test_size):]
            y_test = y_sorted.iloc[(n_samples - test_size):]

            logger.info(f"Divisão temporal concluída:")
            logger.info(f"  Train: {X_train.shape[0]} exemplos ({X_train.shape[0] / n_samples:.1%})")
            logger.info(f"  Validation: {X_val.shape[0]} exemplos ({X_val.shape[0] / n_samples:.1%})")
            logger.info(f"  Test: {X_test.shape[0]} exemplos ({X_test.shape[0] / n_samples:.1%})")

            # Verificar distribuição da variável alvo por conjunto
            train_dist = y_train.mean()
            val_dist = y_val.mean()
            test_dist = y_test.mean()

            logger.info(f"Distribuição da variável alvo (% positivos):")
            logger.info(f"  Train: {train_dist:.2%}")
            logger.info(f"  Validation: {val_dist:.2%}")
            logger.info(f"  Test: {test_dist:.2%}")

            # Verificar drift entre conjuntos
            if abs(train_dist - test_dist) > 0.05:
                logger.warning(
                    f"Possível drift temporal na variável alvo: diferença train-test de {abs(train_dist - test_dist):.3f}")

            result = {
                'X_train': X_train,
                'y_train': y_train,
                'X_val': X_val,
                'y_val': y_val,
                'X_test': X_test,
                'y_test': y_test,
                'id_cols': id_cols
            }

            return result

    # Divisão estratificada ou aleatória
    if split_config['stratify']:
        stratify_y = y
        logger.info("Usando divisão estratificada para manter a distribuição da variável alvo")
    else:
        stratify_y = None
        logger.info("Usando divisão aleatória simples")

    # Primeiro, separar teste
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=split_config['test_size'],
        random_state=split_config['random_state'],
        stratify=stratify_y
    )

    # Depois, separar validação
    if split_config['stratify']:
        stratify_temp = y_temp
    else:
        stratify_temp = None

    val_ratio = split_config['val_size'] / (1 - split_config['test_size'])

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_ratio,
        random_state=split_config['random_state'],
        stratify=stratify_temp
    )

    # Log resultados
    logger.info(f"Divisão concluída:")
    logger.info(f"  Train: {X_train.shape[0]} exemplos ({X_train.shape[0] / df.shape[0]:.1%})")
    logger.info(f"  Validation: {X_val.shape[0]} exemplos ({X_val.shape[0] / df.shape[0]:.1%})")
    logger.info(f"  Test: {X_test.shape[0]} exemplos ({X_test.shape[0] / df.shape[0]:.1%})")

    # Verificar distribuição da variável alvo por conjunto
    train_dist = y_train.mean()
    val_dist = y_val.mean()
    test_dist = y_test.mean()

    logger.info(f"Distribuição da variável alvo (% positivos):")
    logger.info(f"  Train: {train_dist:.2%}")
    logger.info(f"  Validation: {val_dist:.2%}")
    logger.info(f"  Test: {test_dist:.2%}")

    result = {
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'X_test': X_test,
        'y_test': y_test,
        'id_cols': id_cols
    }

    return result


def process_data_pipeline(
        input_file: str,
        output_dir: str = None,
        clean_config: Dict[str, Any] = None,
        target_config: Dict[str, Any] = None,
        split_config: Dict[str, Any] = None,
        save_intermediates: bool = False
) -> Dict[str, Union[pd.DataFrame, np.ndarray, str]]:
    """
    Pipeline completo de processamento de dados.

    Args:
        input_file: Caminho para o arquivo de dados brutos
        output_dir: Diretório para salvar os dados processados
        clean_config: Configurações para limpeza dos dados
        target_config: Configurações para a variável alvo
        split_config: Configurações para divisão dos dados
        save_intermediates: Se True, salva os dados intermediários

    Returns:
        Dicionário com os conjuntos de dados processados e metadados
    """
    # 1. Configurar diretório de saída
    project_root = get_project_root()

    if output_dir is None:
        output_dir = os.path.join(project_root, 'data', 'processed')
    elif not os.path.isabs(output_dir):
        # Se o caminho não for absoluto, considerar relativo à raiz do projeto
        output_dir = os.path.join(project_root, output_dir)

    os.makedirs(output_dir, exist_ok=True)

    # Timestamp para nomeação dos arquivos
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info(f"Iniciando pipeline de processamento de dados com timestamp {timestamp}")

    # 2. Carregar dados
    # Se o input_file não for absoluto, considerar relativo à raiz do projeto
    if not os.path.isabs(input_file):
        input_file = os.path.join(project_root, input_file)

    df_raw = load_data(input_file)

    if save_intermediates:
        raw_output = os.path.join(output_dir, f"raw_data_{timestamp}.csv")
        df_raw.to_csv(raw_output, index=False)
        logger.info(f"Dados brutos salvos em {raw_output}")

    # 3. Identificar tipos de colunas
    column_types = identify_column_types(df_raw)

    # 4. Preparar variável alvo
    # Verificar se há uma coluna que possa ser usada como alvo
    default_target_config = {
        'target_col': column_types['target_column'],
        'create_target': True,
        'days_overdue_threshold': 30,
        'default_indicators': ['Inadimplente', 'Em Atraso', 'Default'],
        'use_risk_score': False,
        'risk_score_threshold': 600,
        'target_name': 'Inadimplente'
    }

    # Se encontrar uma coluna que parece ser a variável alvo, usar ela
    if 'risco_inadimplencia' in [col.lower() for col in df_raw.columns]:
        logger.info("Usando coluna 'Risco_Inadimplencia' como variável alvo")
        col_name = next(col for col in df_raw.columns if col.lower() == 'risco_inadimplencia')
        default_target_config['target_col'] = col_name
        default_target_config['create_target'] = False

    # Usar configuração fornecida ou a padrão
    if target_config is None:
        target_config = default_target_config

    df_target, target_col = prepare_target_variable(df_raw, target_config)

    # Atualizar configuração de limpeza com os tipos identificados
    if clean_config is None:
        clean_config = {
            'remove_duplicates': True,
            'handle_missing': True,
            'missing_threshold': 0.8,
            'convert_types': True,
            'datetime_columns': column_types['datetime_columns'],
            'numeric_columns': column_types['numeric_columns'],
            'categorical_columns': column_types['categorical_columns']
        }

    # 5. Limpar dados
    df_clean = clean_data(df_target, clean_config)

    if save_intermediates:
        clean_output = os.path.join(output_dir, f"clean_data_{timestamp}.csv")
        df_clean.to_csv(clean_output, index=False)
        logger.info(f"Dados limpos salvos em {clean_output}")

    # 6. Atualizar configuração de divisão com coluna temporal
    if split_config is None:
        # Verificar se há coluna temporal para usar na divisão
        temporal_col = None
        if column_types['datetime_columns']:
            temporal_col = column_types['datetime_columns'][0]

        split_config = {
            'test_size': 0.2,
            'val_size': 0.1,
            'stratify': True,
            'random_state': 42,
            'temporal_split': temporal_col is not None,
            'temporal_col': temporal_col
        }

    # 7. Dividir dados
    split_data_dict = split_data(df_clean, target_col, split_config)

    # 8. Salvar dados processados
    if save_intermediates or True:  # Sempre salvar os dados finais
        # Criar arquivo metadata
        metadata = {
            'timestamp': timestamp,
            'input_file': input_file,
            'target_column': target_col,
            'train_size': split_data_dict['X_train'].shape[0],
            'val_size': split_data_dict['X_val'].shape[0],
            'test_size': split_data_dict['X_test'].shape[0],
            'train_distribution': float(split_data_dict['y_train'].mean()),
            'val_distribution': float(split_data_dict['y_val'].mean()),
            'test_distribution': float(split_data_dict['y_test'].mean()),
            'column_types': {k: v if not isinstance(v, list) else v for k, v in column_types.items()},
            'features': split_data_dict['X_train'].columns.tolist(),
            'removed_id_columns': split_data_dict['id_cols']
        }

        # Salvar metadados
        metadata_file = os.path.join(output_dir, f"metadata_{timestamp}.json")
        import json
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=4, default=str)

        logger.info(f"Metadados salvos em {metadata_file}")

        # Salvar conjuntos de dados
        train_file = os.path.join(output_dir, f"train_{timestamp}.csv")
        val_file = os.path.join(output_dir, f"val_{timestamp}.csv")
        test_file = os.path.join(output_dir, f"test_{timestamp}.csv")

        # Recriar os DataFrames completos para salvar
        df_train = pd.concat([split_data_dict['X_train'], split_data_dict['y_train']], axis=1)
        df_val = pd.concat([split_data_dict['X_val'], split_data_dict['y_val']], axis=1)
        df_test = pd.concat([split_data_dict['X_test'], split_data_dict['y_test']], axis=1)

        df_train.to_csv(train_file, index=False)
        df_val.to_csv(val_file, index=False)
        df_test.to_csv(test_file, index=False)

        logger.info(f"Conjunto de treino salvo em {train_file}")
        logger.info(f"Conjunto de validação salvo em {val_file}")
        logger.info(f"Conjunto de teste salvo em {test_file}")

    # 9. Retornar dicionário com dados processados e metadados
    result = {
        'X_train': split_data_dict['X_train'],
        'y_train': split_data_dict['y_train'],
        'X_val': split_data_dict['X_val'],
        'y_val': split_data_dict['y_val'],
        'X_test': split_data_dict['X_test'],
        'y_test': split_data_dict['y_test'],
        'target_column': target_col,
        'timestamp': timestamp,
        'metadata': metadata if 'metadata' in locals() else None
    }

    logger.info("Pipeline de processamento de dados concluído com sucesso!")

    return result


# Função principal para executar o processamento de dados
if __name__ == "__main__":
    import argparse

    # Obter a raiz do projeto para caminhos padrão
    project_root = get_project_root()
    default_input = os.path.join('data', 'raw', 'dataset_bancario.csv')
    default_output = os.path.join('data', 'processed')

    parser = argparse.ArgumentParser(description='Processamento de dados para modelo de inadimplência.')
    parser.add_argument('--input', type=str, default=default_input,
                        help='Caminho para o arquivo de dados de entrada (relativo à raiz do projeto)')
    parser.add_argument('--output', type=str, default=default_output,
                        help='Diretório para salvar os dados processados (relativo à raiz do projeto)')
    parser.add_argument('--save-intermediates', action='store_true', help='Salvar dados intermediários')
    parser.add_argument('--target-col', type=str, help='Nome da coluna alvo (opcional)')

    args = parser.parse_args()

    # Configurar target_config se uma coluna alvo foi especificada
    target_config = None
    if args.target_col:
        target_config = {
            'target_col': args.target_col,
            'create_target': False  # Não criar se já foi especificada
        }

    try:
        # Executar pipeline
        result = process_data_pipeline(
            input_file=args.input,
            output_dir=args.output,
            target_config=target_config,
            save_intermediates=args.save_intermediates
        )

        print(f"Processamento concluído! Dados salvos em {os.path.join(project_root, args.output)}")
    except Exception as e:
        print(f"Erro durante o processamento: {str(e)}")
        import traceback

        traceback.print_exc()