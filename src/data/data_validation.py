"""
Módulo para validação de dados do algoritmo de detecção de inadimplência.

Este módulo contém funções para verificar a qualidade, integridade e adequação dos
dados antes do processamento principal, detectando problemas potenciais e garantindo
que os dados estejam em conformidade com os requisitos do modelo.
"""

import pandas as pd
import numpy as np
import os
import logging
import json
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
from scipy import stats
from datetime import datetime
import importlib.util

# Configurar logger
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


@dataclass
class ValidationResult:
    """Classe para armazenar os resultados da validação."""
    is_valid: bool
    error_messages: List[str]
    warning_messages: List[str]
    validation_details: Dict[str, Any]


# Função para obter a raiz do projeto
def get_project_root():
    """Retorna o caminho para a raiz do projeto."""
    # Assumindo que este script está em src/data/
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Subir dois níveis para chegar à raiz do projeto
    project_root = os.path.abspath(os.path.join(script_dir, os.pardir, os.pardir))
    return project_root


# Função auxiliar para importar um módulo pelo caminho do arquivo
def import_module_from_path(module_name, file_path):
    """Importa um módulo pelo caminho do arquivo."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_schema(df: pd.DataFrame, schema_config: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Valida se o DataFrame está em conformidade com o esquema esperado.

    Args:
        df: DataFrame a ser validado
        schema_config: Configuração do esquema esperado (colunas, tipos, etc.)

    Returns:
        Tupla (is_valid, error_messages)
    """
    is_valid = True
    error_messages = []

    # Verificar colunas obrigatórias
    required_columns = schema_config.get('required_columns', [])
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        is_valid = False
        error_messages.append(f"Colunas obrigatórias ausentes: {missing_columns}")

    # Verificar tipos de dados
    column_types = schema_config.get('column_types', {})
    for col, expected_type in column_types.items():
        if col in df.columns:
            # Mapear tipos esperados para tipos do pandas
            type_map = {
                'numeric': ['int64', 'int32', 'int16', 'int8', 'float64', 'float32'],
                'categorical': ['object', 'category'],
                'datetime': ['datetime64']
            }

            actual_type = str(df[col].dtype)
            if expected_type in type_map:
                if actual_type not in type_map[expected_type]:
                    is_valid = False
                    error_messages.append(f"Tipo incorreto para coluna {col}: esperado {expected_type}, encontrado {actual_type}")
            else:
                # Tipo específico (não mapeado)
                if expected_type != actual_type:
                    is_valid = False
                    error_messages.append(f"Tipo incorreto para coluna {col}: esperado {expected_type}, encontrado {actual_type}")

    # Verificar número mínimo de linhas
    min_rows = schema_config.get('min_rows', 0)
    if len(df) < min_rows:
        is_valid = False
        error_messages.append(f"Número insuficiente de linhas: {len(df)} (mínimo: {min_rows})")

    return is_valid, error_messages


def validate_missing_values(df: pd.DataFrame, threshold_config: Dict[str, Any]) -> Tuple[bool, List[str], Dict[str, Any]]:
    """
    Valida a quantidade de valores ausentes no DataFrame.

    Args:
        df: DataFrame a ser validado
        threshold_config: Configuração com thresholds para valores ausentes

    Returns:
        Tupla (is_valid, error_messages, validation_details)
    """
    is_valid = True
    error_messages = []
    validation_details = {}

    # Calcular percentual de valores ausentes por coluna
    missing_percent = df.isnull().mean() * 100
    validation_details['missing_percent_by_column'] = missing_percent.to_dict()

    # Percentual total de valores ausentes
    total_missing_percent = df.isnull().values.mean() * 100
    validation_details['total_missing_percent'] = total_missing_percent

    # Verificar threshold global
    global_threshold = threshold_config.get('global_threshold', 50.0)
    if total_missing_percent > global_threshold:
        is_valid = False
        error_messages.append(
            f"Percentual total de valores ausentes ({total_missing_percent:.2f}%) "
            f"excede o threshold global ({global_threshold:.2f}%)"
        )

    # Verificar thresholds por coluna
    column_thresholds = threshold_config.get('column_thresholds', {})

    # Verificar colunas explicitamente configuradas
    for col, threshold in column_thresholds.items():
        if col in df.columns:
            col_missing = missing_percent[col]
            if col_missing > threshold:
                is_valid = False
                error_messages.append(
                    f"Coluna '{col}' tem {col_missing:.2f}% de valores ausentes, "
                    f"excedendo o threshold de {threshold:.2f}%"
                )

    # Verificar colunas críticas
    critical_columns = threshold_config.get('critical_columns', [])
    critical_threshold = threshold_config.get('critical_threshold', 5.0)

    for col in critical_columns:
        if col in df.columns:
            col_missing = missing_percent[col]
            if col_missing > critical_threshold:
                is_valid = False
                error_messages.append(
                    f"Coluna crítica '{col}' tem {col_missing:.2f}% de valores ausentes, "
                    f"excedendo o threshold crítico de {critical_threshold:.2f}%"
                )

    # Número de colunas com mais de 50% de valores ausentes
    high_missing_cols = missing_percent[missing_percent > 50].index.tolist()
    validation_details['high_missing_columns'] = high_missing_cols
    validation_details['high_missing_count'] = len(high_missing_cols)

    return is_valid, error_messages, validation_details


def validate_outliers(df: pd.DataFrame, outlier_config: Dict[str, Any]) -> Tuple[bool, List[str], Dict[str, Any]]:
    """
    Valida a presença de outliers em colunas numéricas.

    Args:
        df: DataFrame a ser validado
        outlier_config: Configuração para detecção de outliers

    Returns:
        Tupla (is_valid, error_messages, validation_details)
    """
    is_valid = True
    error_messages = []
    validation_details = {'outliers_by_column': {}}

    # Método de detecção de outliers
    method = outlier_config.get('method', 'zscore')
    threshold = outlier_config.get('threshold', 3.0)  # Z-score ou IQR
    max_outlier_percent = outlier_config.get('max_outlier_percent', 5.0)

    # Colunas a verificar
    columns_to_check = outlier_config.get('columns', df.select_dtypes(include=['number']).columns.tolist())

    # Remover colunas ausentes da lista
    columns_to_check = [col for col in columns_to_check if col in df.columns]

    for col in columns_to_check:
        # Pular colunas não numéricas
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue

        # Pegar valores não nulos
        values = df[col].dropna()

        # Pular se não houver valores suficientes
        if len(values) < 10:
            continue

        outliers = []

        if method == 'zscore':
            # Método Z-score
            z_scores = stats.zscore(values, nan_policy='omit')
            outliers = values[np.abs(z_scores) > threshold]
        elif method == 'iqr':
            # Método IQR (Interquartile Range)
            Q1 = values.quantile(0.25)
            Q3 = values.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            outliers = values[(values < lower_bound) | (values > upper_bound)]

        # Calcular percentual de outliers
        outlier_percent = 100 * len(outliers) / len(values)
        validation_details['outliers_by_column'][col] = {
            'outlier_count': len(outliers),
            'outlier_percent': outlier_percent,
            'min_value': values.min(),
            'max_value': values.max(),
            'mean': values.mean(),
            'median': values.median()
        }

        # Verificar se excede o threshold
        if outlier_percent > max_outlier_percent:
            error_messages.append(
                f"Coluna '{col}' tem {outlier_percent:.2f}% de outliers, "
                f"excedendo o máximo permitido de {max_outlier_percent:.2f}%"
            )
            is_valid = False

    # Total de colunas com outliers acima do limite
    outlier_columns = [col for col, details in validation_details['outliers_by_column'].items()
                      if details['outlier_percent'] > max_outlier_percent]
    validation_details['outlier_columns'] = outlier_columns
    validation_details['outlier_column_count'] = len(outlier_columns)

    return is_valid, error_messages, validation_details


def validate_business_rules(df: pd.DataFrame, rules_config: Dict[str, Any]) -> Tuple[bool, List[str], Dict[str, Any]]:
    """
    Valida se os dados estão de acordo com regras de negócio específicas.

    Args:
        df: DataFrame a ser validado
        rules_config: Configuração com regras de negócio

    Returns:
        Tupla (is_valid, error_messages, validation_details)
    """
    is_valid = True
    error_messages = []
    validation_details = {'rule_violations': {}}

    # Validar faixas de valores
    range_rules = rules_config.get('value_ranges', {})
    for col, range_config in range_rules.items():
        if col not in df.columns:
            continue

        min_val = range_config.get('min')
        max_val = range_config.get('max')

        violations = 0
        records = len(df)

        if min_val is not None and max_val is not None:
            violations = ((df[col] < min_val) | (df[col] > max_val)).sum()
        elif min_val is not None:
            violations = (df[col] < min_val).sum()
        elif max_val is not None:
            violations = (df[col] > max_val).sum()

        violation_percent = 100 * violations / records if records > 0 else 0

        validation_details['rule_violations'][f"{col}_range"] = {
            'violation_count': int(violations),
            'violation_percent': violation_percent,
            'rule': f"{min_val if min_val is not None else '-inf'} <= {col} <= {max_val if max_val is not None else 'inf'}"
        }

        # Verificar threshold de violação
        threshold = range_config.get('threshold', 1.0)  # % máximo de violações permitido
        if violation_percent > threshold:
            error_messages.append(
                f"Coluna '{col}' tem {violation_percent:.2f}% de valores fora da faixa permitida "
                f"({min_val if min_val is not None else '-inf'} a {max_val if max_val is not None else 'inf'}), "
                f"excedendo o threshold de {threshold:.2f}%"
            )
            is_valid = False

    # Validar regras de relacionamento entre variáveis
    relation_rules = rules_config.get('variable_relations', [])
    for rule in relation_rules:
        # Formato esperado: {"col1": "coluna1", "col2": "coluna2", "relation": "col1 <= col2", "threshold": 1.0}
        col1 = rule.get('col1')
        col2 = rule.get('col2')
        relation = rule.get('relation')
        threshold = rule.get('threshold', 1.0)  # % máximo de violações permitido

        if col1 not in df.columns or col2 not in df.columns:
            continue

        violations = 0
        records = len(df)

        if relation == 'col1 <= col2':
            violations = (df[col1] > df[col2]).sum()
        elif relation == 'col1 >= col2':
            violations = (df[col1] < df[col2]).sum()
        elif relation == 'col1 < col2':
            violations = (df[col1] >= df[col2]).sum()
        elif relation == 'col1 > col2':
            violations = (df[col1] <= df[col2]).sum()
        elif relation == 'col1 == col2':
            violations = (df[col1] != df[col2]).sum()
        elif relation == 'col1 != col2':
            violations = (df[col1] == df[col2]).sum()

        violation_percent = 100 * violations / records if records > 0 else 0

        validation_details['rule_violations'][f"{col1}_{relation}_{col2}"] = {
            'violation_count': int(violations),
            'violation_percent': violation_percent,
            'rule': f"{col1} {relation.replace('col1', '').replace('col2', '')} {col2}"
        }

        if violation_percent > threshold:
            rule_text = relation.replace('col1', col1).replace('col2', col2)
            error_messages.append(
                f"Relação '{rule_text}' tem {violation_percent:.2f}% de violações, "
                f"excedendo o threshold de {threshold:.2f}%"
            )
            is_valid = False

    return is_valid, error_messages, validation_details


def detect_data_drift(
    current_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    drift_config: Dict[str, Any]
) -> Tuple[bool, List[str], Dict[str, Any]]:
    """
    Detecta drift nos dados em relação a um conjunto de referência.

    Args:
        current_df: DataFrame atual
        reference_df: DataFrame de referência
        drift_config: Configuração para detecção de drift

    Returns:
        Tupla (is_valid, error_messages, validation_details)
    """
    is_valid = True
    error_messages = []
    validation_details = {'drift_by_column': {}}

    # Método de detecção de drift
    method = drift_config.get('method', 'ks')
    threshold = drift_config.get('threshold', 0.05)  # p-value para rejeitar hipótese nula

    # Colunas a verificar
    columns_to_check = drift_config.get('columns', [])

    # Se não há colunas explícitas, usar todas as numéricas
    if not columns_to_check:
        columns_to_check = current_df.select_dtypes(include=['number']).columns.tolist()

    # Remover colunas ausentes da lista
    columns_to_check = [col for col in columns_to_check if col in current_df.columns and col in reference_df.columns]

    drift_detected_columns = []

    for col in columns_to_check:
        # Pular colunas não numéricas
        if not pd.api.types.is_numeric_dtype(current_df[col]) or not pd.api.types.is_numeric_dtype(reference_df[col]):
            continue

        # Pegar valores não nulos
        current_values = current_df[col].dropna()
        reference_values = reference_df[col].dropna()

        # Pular se não houver valores suficientes
        if len(current_values) < 10 or len(reference_values) < 10:
            continue

        # Calcular estatísticas
        current_mean = current_values.mean()
        reference_mean = reference_values.mean()
        current_std = current_values.std()
        reference_std = reference_values.std()
        mean_diff_percent = 100 * abs(current_mean - reference_mean) / max(abs(reference_mean), 1e-10)

        validation_details['drift_by_column'][col] = {
            'current_mean': current_mean,
            'reference_mean': reference_mean,
            'current_std': current_std,
            'reference_std': reference_std,
            'mean_diff_percent': mean_diff_percent
        }

        if method == 'ks':
            # Teste de Kolmogorov-Smirnov
            ks_statistic, p_value = stats.ks_2samp(current_values, reference_values)
            validation_details['drift_by_column'][col]['ks_statistic'] = ks_statistic
            validation_details['drift_by_column'][col]['p_value'] = p_value

            if p_value < threshold:
                drift_detected_columns.append(col)
                error_messages.append(
                    f"Drift detectado na coluna '{col}': p-value = {p_value:.4f} < {threshold} (KS test). "
                    f"Diferença na média: {mean_diff_percent:.2f}%"
                )
                is_valid = False
        elif method == 'mean_diff':
            # Diferença percentual na média
            max_diff = drift_config.get('max_mean_diff_percent', 10.0)
            validation_details['drift_by_column'][col]['max_allowed_diff'] = max_diff

            if mean_diff_percent > max_diff:
                drift_detected_columns.append(col)
                error_messages.append(
                    f"Drift detectado na coluna '{col}': diferença na média = {mean_diff_percent:.2f}% > {max_diff}%. "
                    f"Current: {current_mean:.2f}, Reference: {reference_mean:.2f}"
                )
                is_valid = False

    validation_details['drift_detected_columns'] = drift_detected_columns
    validation_details['drift_column_count'] = len(drift_detected_columns)

    return is_valid, error_messages, validation_details


def validate_integrity(df: pd.DataFrame, integrity_config: Dict[str, Any]) -> Tuple[bool, List[str], Dict[str, Any]]:
    """
    Valida a integridade dos dados, incluindo duplicatas e inconsistências.

    Args:
        df: DataFrame a ser validado
        integrity_config: Configuração para validação de integridade

    Returns:
        Tupla (is_valid, error_messages, validation_details)
    """
    is_valid = True
    error_messages = []
    validation_details = {}

    # Verificar duplicatas
    check_duplicates = integrity_config.get('check_duplicates', True)
    duplicate_threshold = integrity_config.get('duplicate_threshold', 1.0)  # % máximo de duplicatas permitido

    if check_duplicates:
        # Verificar duplicatas em todo o DataFrame
        duplicate_rows = df.duplicated().sum()
        duplicate_percent = 100 * duplicate_rows / len(df) if len(df) > 0 else 0

        validation_details['duplicate_rows'] = duplicate_rows
        validation_details['duplicate_percent'] = duplicate_percent

        if duplicate_percent > duplicate_threshold:
            error_messages.append(
                f"Percentual de linhas duplicadas ({duplicate_percent:.2f}%) "
                f"excede o threshold de {duplicate_threshold:.2f}%"
            )
            is_valid = False

        # Verificar duplicatas por chaves específicas
        key_columns = integrity_config.get('key_columns', [])
        if key_columns:
            # Verificar se todas as colunas existem
            missing_key_columns = [col for col in key_columns if col not in df.columns]
            if missing_key_columns:
                error_messages.append(f"Colunas de chave ausentes: {missing_key_columns}")
            else:
                # Verificar duplicatas nas colunas de chave
                duplicate_keys = df.duplicated(subset=key_columns).sum()
                duplicate_keys_percent = 100 * duplicate_keys / len(df) if len(df) > 0 else 0

                validation_details['duplicate_keys'] = duplicate_keys
                validation_details['duplicate_keys_percent'] = duplicate_keys_percent
                validation_details['key_columns'] = key_columns

                if duplicate_keys_percent > duplicate_threshold:
                    error_messages.append(
                        f"Percentual de chaves duplicadas ({duplicate_keys_percent:.2f}%) "
                        f"excede o threshold de {duplicate_threshold:.2f}%"
                    )
                    is_valid = False

    # Verificar integridade referencial
    ref_integrity = integrity_config.get('referential_integrity', [])
    for ref in ref_integrity:
        source_column = ref.get('source_column')
        reference_data = ref.get('reference_data')
        reference_column = ref.get('reference_column')

        # Pular se alguma informação estiver faltando
        if not source_column or not reference_data or not reference_column:
            continue

        # Verificar se a coluna existe
        if source_column not in df.columns:
            error_messages.append(f"Coluna de origem '{source_column}' não encontrada")
            continue

        # Carregar dados de referência se for um path
        ref_values = reference_data
        if isinstance(reference_data, str) and os.path.exists(reference_data):
            try:
                ref_df = pd.read_csv(reference_data)
                if reference_column in ref_df.columns:
                    ref_values = set(ref_df[reference_column].dropna().unique())
                else:
                    error_messages.append(f"Coluna de referência '{reference_column}' não encontrada em {reference_data}")
                    continue
            except Exception as e:
                error_messages.append(f"Erro ao carregar dados de referência: {str(e)}")
                continue

        # Verificar valores que não estão na referência
        source_values = set(df[source_column].dropna().unique())
        invalid_values = [val for val in source_values if val not in ref_values]

        validation_details[f'{source_column}_integrity'] = {
            'invalid_values': invalid_values[:10],  # Limitar para não ficar muito grande
            'invalid_count': len(invalid_values),
            'total_unique': len(source_values),
            'invalid_percent': 100 * len(invalid_values) / len(source_values) if source_values else 0
        }

        # Verificar threshold
        threshold = ref.get('threshold', 0.0)  # % máximo de valores inválidos permitido
        if len(invalid_values) > 0 and (len(invalid_values) / len(source_values) * 100 > threshold):
            error_messages.append(
                f"Coluna '{source_column}' tem {len(invalid_values)} valores inválidos "
                f"({100 * len(invalid_values) / len(source_values):.2f}%), "
                f"excedendo o threshold de {threshold:.2f}%"
            )
            is_valid = False

    return is_valid, error_messages, validation_details


def generate_validation_report(validation_result: ValidationResult, df: pd.DataFrame, output_dir: str = None) -> str:
    """
    Gera um relatório detalhado da validação com codificação UTF-8 correta.

    Args:
        validation_result: Resultado da validação
        df: DataFrame validado
        output_dir: Diretório para salvar o relatório

    Returns:
        Caminho para o relatório gerado
    """
    # Configurar diretório de saída
    if output_dir is None:
        output_dir = os.getcwd()
    os.makedirs(output_dir, exist_ok=True)

    # Timestamp para o nome do arquivo
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = os.path.join(output_dir, f"validation_report_{timestamp}.html")

    # Criar relatório HTML
    html_content = [
        "<!DOCTYPE html>",
        "<html>",
        "<head>",
        "    <meta charset=\"UTF-8\">",  # Especificar codificação UTF-8
        "    <title>Relatório de Validação de Dados</title>",
        "    <style>",
        "        body { font-family: Arial, sans-serif; margin: 20px; }",
        "        h1, h2, h3 { color: #333366; }",
        "        .section { margin-bottom: 30px; }",
        "        .status-valid { color: green; }",
        "        .status-invalid { color: red; }",
        "        .error { color: red; margin-left: 20px; }",
        "        .warning { color: orange; margin-left: 20px; }",
        "        table { border-collapse: collapse; width: 100%; }",
        "        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }",
        "        th { background-color: #f2f2f2; }",
        "        tr:nth-child(even) { background-color: #f9f9f9; }",
        "    </style>",
        "</head>",
        "<body>",
        f"    <h1>Relatório de Validação de Dados</h1>",
        f"    <p><strong>Data:</strong> {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}</p>",
        f"    <p><strong>Status:</strong> <span class=\"{'status-valid' if validation_result.is_valid else 'status-invalid'}\">"
        f"{'Válido' if validation_result.is_valid else 'Inválido'}</span></p>",
        f"    <p><strong>Registros analisados:</strong> {len(df)}</p>",
        f"    <p><strong>Colunas analisadas:</strong> {len(df.columns)}</p>",
    ]

    # Adicionar seção de erros
    if validation_result.error_messages:
        html_content.extend([
            "    <div class=\"section\">",
            "        <h2>Erros encontrados</h2>",
            "        <ul>"
        ])
        for error in validation_result.error_messages:
            html_content.append(f"            <li class=\"error\">{error}</li>")
        html_content.append("        </ul>")
        html_content.append("    </div>")

    # Adicionar seção de avisos
    if validation_result.warning_messages:
        html_content.extend([
            "    <div class=\"section\">",
            "        <h2>Avisos</h2>",
            "        <ul>"
        ])
        for warning in validation_result.warning_messages:
            html_content.append(f"            <li class=\"warning\">{warning}</li>")
        html_content.append("        </ul>")
        html_content.append("    </div>")

    # Adicionar detalhes da validação
    html_content.extend([
        "    <div class=\"section\">",
        "        <h2>Detalhes da validação</h2>",
    ])

    # Adicionar tabela de estatísticas básicas
    html_content.extend([
        "        <h3>Estatísticas básicas</h3>",
        "        <table>",
        "            <tr><th>Coluna</th><th>Tipo</th><th>Não nulos</th><th>Valores únicos</th><th>Min</th><th>Max</th><th>Média</th><th>Desvio padrão</th></tr>"
    ])

    for col in df.columns:
        col_type = str(df[col].dtype)
        non_null = df[col].count()
        unique = df[col].nunique()

        # Estatísticas para colunas numéricas
        if pd.api.types.is_numeric_dtype(df[col]):
            min_val = df[col].min() if non_null > 0 else "N/A"
            max_val = df[col].max() if non_null > 0 else "N/A"
            mean = f"{df[col].mean():.2f}" if non_null > 0 else "N/A"
            std = f"{df[col].std():.2f}" if non_null > 0 else "N/A"
        else:
            min_val = "N/A"
            max_val = "N/A"
            mean = "N/A"
            std = "N/A"

        html_content.append(f"            <tr><td>{col}</td><td>{col_type}</td><td>{non_null}</td><td>{unique}</td><td>{min_val}</td><td>{max_val}</td><td>{mean}</td><td>{std}</td></tr>")

    html_content.append("        </table>")

    # Adicionar detalhes específicos da validação
    detail_sections = {
        'missing_percent_by_column': 'Valores ausentes por coluna',
        'outliers_by_column': 'Outliers por coluna',
        'rule_violations': 'Violações de regras de negócio',
        'drift_by_column': 'Drift detectado por coluna'
    }

    for key, title in detail_sections.items():
        if key in validation_result.validation_details:
            data = validation_result.validation_details[key]
            if data:
                html_content.extend([
                    f"        <h3>{title}</h3>",
                    "        <table>",
                ])

                # Cabeçalho da tabela
                if key == 'missing_percent_by_column':
                    html_content.append("            <tr><th>Coluna</th><th>% Ausente</th></tr>")
                    for col, percent in data.items():
                        html_content.append(f"            <tr><td>{col}</td><td>{percent:.2f}%</td></tr>")
                elif key == 'outliers_by_column':
                    html_content.append("            <tr><th>Coluna</th><th>Contagem de outliers</th><th>% Outliers</th><th>Min</th><th>Max</th><th>Média</th><th>Mediana</th></tr>")
                    for col, details in data.items():
                        html_content.append(
                            f"            <tr><td>{col}</td><td>{details['outlier_count']}</td>"
                            f"<td>{details['outlier_percent']:.2f}%</td><td>{details['min_value']:.2f}</td>"
                            f"<td>{details['max_value']:.2f}</td><td>{details['mean']:.2f}</td>"
                            f"<td>{details['median']:.2f}</td></tr>"
                        )
                elif key == 'rule_violations':
                    html_content.append("            <tr><th>Regra</th><th>Contagem de violações</th><th>% Violações</th><th>Descrição</th></tr>")
                    for rule, details in data.items():
                        html_content.append(
                            f"            <tr><td>{rule}</td><td>{details['violation_count']}</td>"
                            f"<td>{details['violation_percent']:.2f}%</td><td>{details['rule']}</td></tr>"
                        )
                elif key == 'drift_by_column':
                    html_content.append("            <tr><th>Coluna</th><th>Média atual</th><th>Média referência</th><th>% Diferença</th><th>p-value (se disponível)</th></tr>")
                    for col, details in data.items():
                        p_value = details.get('p_value', 'N/A')
                        p_value = f"{p_value:.4f}" if p_value != 'N/A' else p_value
                        html_content.append(
                            f"            <tr><td>{col}</td><td>{details['current_mean']:.2f}</td>"
                            f"<td>{details['reference_mean']:.2f}</td><td>{details['mean_diff_percent']:.2f}%</td>"
                            f"<td>{p_value}</td></tr>"
                        )

                html_content.append("        </table>")

    # Fechar HTML
    html_content.extend([
        "    </div>",
        "</body>",
        "</html>"
    ])

    # Escrever arquivo com codificação UTF-8 explícita
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(html_content))

    logger.info(f"Relatório de validação gerado: {report_file}")
    return report_file


def validate_data(
    df: pd.DataFrame,
    config_file: str = None,
    reference_df: pd.DataFrame = None,
    output_dir: str = None
) -> ValidationResult:
    """
    Executa todas as validações de dados conforme configuração.

    Args:
        df: DataFrame a ser validado
        config_file: Caminho para arquivo de configuração de validação
        reference_df: DataFrame de referência para detecção de drift
        output_dir: Diretório para salvar relatórios de validação

    Returns:
        Resultado da validação
    """
    start_time = datetime.now()
    logger.info(f"Iniciando validação de dados com {len(df)} registros e {len(df.columns)} colunas")

    is_valid = True
    all_error_messages = []
    all_warning_messages = []
    validation_details = {}

    # Carregar configuração
    if config_file and os.path.exists(config_file):
        with open(config_file, 'r') as f:
            config = json.load(f)
    else:
        # Configuração padrão
        config = {
            'schema': {
                'required_columns': [],
                'column_types': {},
                'min_rows': 10
            },
            'missing_values': {
                'global_threshold': 30.0,
                'column_thresholds': {},
                'critical_columns': [],
                'critical_threshold': 5.0
            },
            'outliers': {
                'method': 'zscore',
                'threshold': 3.0,
                'max_outlier_percent': 5.0,
                'columns': []
            },
            'business_rules': {
                'value_ranges': {},
                'variable_relations': []
            },
            'data_drift': {
                'method': 'ks',
                'threshold': 0.05,
                'columns': []
            },
            'integrity': {
                'check_duplicates': True,
                'duplicate_threshold': 1.0,
                'key_columns': [],
                'referential_integrity': []
            }
        }

        logger.info("Usando configuração padrão para validação")

    # Executar validações

    # 1. Validar esquema
    schema_config = config.get('schema', {})
    schema_valid, schema_errors = validate_schema(df, schema_config)
    is_valid = is_valid and schema_valid
    all_error_messages.extend(schema_errors)

    # 2. Validar valores ausentes
    missing_config = config.get('missing_values', {})
    missing_valid, missing_errors, missing_details = validate_missing_values(df, missing_config)
    is_valid = is_valid and missing_valid
    all_error_messages.extend(missing_errors)
    validation_details.update(missing_details)

    # 3. Validar outliers
    outlier_config = config.get('outliers', {})
    outliers_valid, outlier_errors, outlier_details = validate_outliers(df, outlier_config)
    is_valid = is_valid and outliers_valid
    all_error_messages.extend(outlier_errors)
    validation_details.update(outlier_details)

    # 4. Validar regras de negócio
    rules_config = config.get('business_rules', {})
    rules_valid, rules_errors, rules_details = validate_business_rules(df, rules_config)
    is_valid = is_valid and rules_valid
    all_error_messages.extend(rules_errors)
    validation_details.update(rules_details)

    # 5. Validar integridade
    integrity_config = config.get('integrity', {})
    integrity_valid, integrity_errors, integrity_details = validate_integrity(df, integrity_config)
    is_valid = is_valid and integrity_valid
    all_error_messages.extend(integrity_errors)
    validation_details.update(integrity_details)

    # 6. Detectar drift (se houver dados de referência)
    if reference_df is not None:
        drift_config = config.get('data_drift', {})
        drift_valid, drift_errors, drift_details = detect_data_drift(df, reference_df, drift_config)
        is_valid = is_valid and drift_valid
        all_error_messages.extend(drift_errors)
        validation_details.update(drift_details)

    # Construir resultado da validação
    validation_result = ValidationResult(
        is_valid=is_valid,
        error_messages=all_error_messages,
        warning_messages=all_warning_messages,
        validation_details=validation_details
    )

    # Gerar relatório
    if output_dir:
        report_file = generate_validation_report(validation_result, df, output_dir)
        validation_details['report_file'] = report_file

    # Log de resultado
    elapsed_time = (datetime.now() - start_time).total_seconds()
    if is_valid:
        logger.info(f"Validação concluída com SUCESSO em {elapsed_time:.2f} segundos")
    else:
        logger.warning(f"Validação concluída com ERROS em {elapsed_time:.2f} segundos. {len(all_error_messages)} erros encontrados.")

    return validation_result


def load_and_validate_data(
    file_path: str,
    config_file: str = None,
    reference_file: str = None,
    output_dir: str = None
) -> Tuple[pd.DataFrame, ValidationResult]:
    """
    Carrega e valida dados de um arquivo.

    Args:
        file_path: Caminho para o arquivo de dados
        config_file: Caminho para arquivo de configuração de validação
        reference_file: Caminho para arquivo de dados de referência (para detecção de drift)
        output_dir: Diretório para salvar relatórios de validação

    Returns:
        Tupla (DataFrame, Resultado da validação)
    """
    # Carregar função load_data do make_dataset.py
    project_root = get_project_root()
    dataset_module_path = os.path.join(project_root, 'src', 'data', 'make_dataset.py')

    try:
        # Importar usando importlib
        dataset_module = import_module_from_path('make_dataset', dataset_module_path)
        load_data = dataset_module.load_data
    except Exception as e:
        logger.warning(f"Não foi possível importar load_data de make_dataset.py: {str(e)}")
        logger.warning("Usando função de carregamento interna simplificada")

        # Função de carregamento simplificada
        def load_data(path):
            ext = os.path.splitext(path)[1].lower()
            if ext == '.csv':
                return pd.read_csv(path)
            elif ext in ['.xlsx', '.xls']:
                return pd.read_excel(path)
            else:
                raise ValueError(f"Formato de arquivo não suportado: {ext}")

    # Carregar dados
    logger.info(f"Carregando dados de {file_path}")
    df = load_data(file_path)

    # Carregar dados de referência (se fornecido)
    reference_df = None
    if reference_file and os.path.exists(reference_file):
        try:
            logger.info(f"Carregando dados de referência de {reference_file}")
            reference_df = load_data(reference_file)
        except Exception as e:
            logger.warning(f"Erro ao carregar dados de referência: {str(e)}")

    # Validar dados
    validation_result = validate_data(df, config_file, reference_df, output_dir)

    return df, validation_result


# Função principal para executar a validação de dados
if __name__ == "__main__":
    import argparse

    # Obter a raiz do projeto para caminhos padrão
    project_root = get_project_root()

    # Definir caminhos padrão
    default_input = os.path.join(project_root, 'data', 'raw', 'dataset_bancario.csv')
    default_output = os.path.join(project_root, 'reports', 'validation')

    parser = argparse.ArgumentParser(description='Validação de dados para modelo de inadimplência.')
    parser.add_argument('--input', type=str, default=default_input,
                       help=f'Caminho para o arquivo de dados de entrada (padrão: {default_input})')
    parser.add_argument('--config', type=str,
                       help='Caminho para arquivo de configuração de validação')
    parser.add_argument('--reference', type=str,
                       help='Caminho para arquivo de dados de referência (para detecção de drift)')
    parser.add_argument('--output', type=str, default=default_output,
                       help=f'Diretório para salvar relatórios de validação (padrão: {default_output})')

    args = parser.parse_args()

    # Verificar se o arquivo de entrada existe
    if not os.path.exists(args.input):
        print(f"ERRO: Arquivo de entrada '{args.input}' não encontrado.")
        print(f"Verifique o caminho ou especifique um arquivo existente com --input.")
        exit(1)

    try:
        # Executar validação
        df, result = load_and_validate_data(
            file_path=args.input,
            config_file=args.config,
            reference_file=args.reference,
            output_dir=args.output
        )

        if result.is_valid:
            print(f"Validação concluída com SUCESSO. Conjunto de dados válido.")
            exit(0)
        else:
            print(f"Validação concluída com ERROS. {len(result.error_messages)} erros encontrados.")
            for i, error in enumerate(result.error_messages[:5], 1):
                print(f"{i}. {error}")
            if len(result.error_messages) > 5:
                print(f"... e mais {len(result.error_messages) - 5} erros.")
            exit(1)
    except Exception as e:
        print(f"Erro durante a validação: {str(e)}")
        import traceback
        traceback.print_exc()
        exit(2)