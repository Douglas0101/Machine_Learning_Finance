import logging
from datetime import datetime
from typing import Union, Optional, List, Dict, Any
import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, RobustScaler, PowerTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold, SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE  # Para balanceamento de classes

# Configuração de logging com timestamp no nome do arquivo
log_filename = f"make_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_filename)
    ]
)
logger = logging.getLogger(__name__)

# Seed para reprodutibilidade
np.random.seed(42)

# =============================================================================
# 1. Ingestão e Validação de Dados
# =============================================================================

def load_data(source: Union[str, pd.DataFrame],
              file_type: str = 'csv',
              chunksize: Optional[int] = None,
              **kwargs) -> pd.DataFrame:
    """Carrega dados de diversas fontes"""
    try:
        if isinstance(source, pd.DataFrame):
            df = source.copy()
        elif file_type == 'csv':
            if chunksize:
                chunks = pd.read_csv(source, chunksize=chunksize, **kwargs)
                df = pd.concat(chunks, ignore_index=True)
            else:
                df = pd.read_csv(source, **kwargs)
        else:
            raise ValueError("Tipo de fonte não suportado.")
        logger.info(f"Dataset carregado com {df.shape[0]} linhas e {df.shape[1]} colunas.")
        return df
    except Exception as e:
        logger.error(f"Erro ao carregar dados: {e}")
        raise

def validate_schema(df: pd.DataFrame, expected_columns: Dict[str, Any]) -> None:
    """Valida o esquema e tipos de dados"""
    missing = [col for col in expected_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Colunas ausentes: {missing}")
    for col, expected_type in expected_columns.items():
        if not np.issubdtype(df[col].dtype, expected_type):
            logger.warning(f"A coluna '{col}' possui dtype {df[col].dtype}; esperado {expected_type}.")
    logger.info("Validação de esquema concluída.")

# =============================================================================
# 2. Imputação Avançada de Dados Faltantes
# =============================================================================

def advanced_missing_imputation(df: pd.DataFrame,
                                numeric_strategy: str = 'iterative',
                                categorical_strategy: str = 'mode') -> pd.DataFrame:
    """Trata dados faltantes com imputação avançada"""
    df_imputed = df.copy()

    # Imputação numérica
    num_cols = df_imputed.select_dtypes(include=[np.number]).columns.tolist()
    if num_cols:
        if numeric_strategy == 'iterative':
            imputer = IterativeImputer(random_state=42)
        elif numeric_strategy == 'knn':
            imputer = KNNImputer()
        else:
            raise ValueError("numeric_strategy não suportado.")
        df_imputed[num_cols] = imputer.fit_transform(df_imputed[num_cols])

    # Imputação categórica
    cat_cols = df_imputed.select_dtypes(include=['object', 'category']).columns.tolist()
    for col in cat_cols:
        mode_val = df_imputed[col].mode().iloc[0]
        df_imputed[col] = df_imputed[col].fillna(mode_val)

    logger.info("Imputação de missing concluída.")
    return df_imputed

# =============================================================================
# 3. Detecção e Tratamento de Outliers
# =============================================================================

def handle_outliers(df: pd.DataFrame,
                    num_cols: List[str],
                    method: str = 'IQR',
                    factor: float = 1.5) -> pd.DataFrame:
    """Detecta e trata outliers em colunas numéricas"""
    df_out = df.copy()
    if method.upper() == 'IQR':
        for col in num_cols:
            Q1 = df_out[col].quantile(0.25)
            Q3 = df_out[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - factor * IQR
            upper_bound = Q3 + factor * IQR
            df_out[col] = df_out[col].clip(lower=lower_bound, upper=upper_bound)
            logger.info(f"Outliers tratados na coluna '{col}' com clipping via IQR.")
    else:
        raise ValueError("Método de tratamento de outliers não suportado.")
    return df_out

# =============================================================================
# 4. Codificação e Escalonamento de Features
# =============================================================================

def encode_and_scale_features(df: pd.DataFrame,
                              ordinal_cols: Optional[List[str]] = None,
                              cat_max_cardinality: int = 50,
                              exclude_cols: Optional[List[str]] = None) -> pd.DataFrame:
    """Aplica codificação e escalonamento nas features"""
    if ordinal_cols is None:
        ordinal_cols = []
    if exclude_cols is None:
        exclude_cols = []

    df_excluded = df[exclude_cols].copy() if exclude_cols else pd.DataFrame(index=df.index)
    df_to_transform = df.drop(columns=exclude_cols) if exclude_cols else df.copy()

    numeric_cols = df_to_transform.select_dtypes(include=[np.number]).columns.tolist()
    all_cat_cols = df_to_transform.select_dtypes(include=['object', 'category']).columns.tolist()
    non_ordinal_cols = [col for col in all_cat_cols if col not in ordinal_cols]

    filtered_cat_cols = [col for col in non_ordinal_cols if df_to_transform[col].nunique() <= cat_max_cardinality]

    num_transformer = Pipeline(steps=[
        ('scaler', RobustScaler()),
        ('power', PowerTransformer(method='yeo-johnson'))
    ])
    cat_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', num_transformer, numeric_cols),
        ('cat', cat_transformer, filtered_cat_cols)
    ])

    processed_array = preprocessor.fit_transform(df_to_transform)
    df_processed = pd.DataFrame(processed_array, columns=preprocessor.get_feature_names_out(), index=df_to_transform.index)

    if ordinal_cols:
        df_ordinal = df_to_transform[ordinal_cols].apply(lambda col: LabelEncoder().fit_transform(col))
        df_processed = pd.concat([df_processed, df_ordinal], axis=1)

    df_final = pd.concat([df_processed, df_excluded], axis=1)
    return df_final

# =============================================================================
# 5. Seleção de Features
# =============================================================================

def feature_selection(df: pd.DataFrame,
                      target: str,
                      model_estimator: Any = RandomForestClassifier(random_state=42),
                      var_threshold: float = 0.0) -> pd.DataFrame:
    """Realiza seleção de features com base em variância e modelos"""
    X = df.drop(target, axis=1)
    y = df[target]

    vt = VarianceThreshold(threshold=var_threshold)
    X_vt = vt.fit_transform(X)
    selected_columns = X.columns[vt.get_support()]
    X_vt_df = pd.DataFrame(X_vt, columns=selected_columns, index=X.index)

    sfm = SelectFromModel(model_estimator, threshold='median')
    sfm.fit(X_vt_df, y)
    final_columns = X_vt_df.columns[sfm.get_support()]

    df_selected = X_vt_df[final_columns].copy()
    df_selected[target] = y.values
    return df_selected

# =============================================================================
# 6. Balanceamento de Classes
# =============================================================================

def balance_classes(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """Balanceia as classes do target usando SMOTE"""
    X = df.drop(target, axis=1)
    y = df[target]

    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)

    df_balanced = pd.DataFrame(X_res, columns=X.columns)
    df_balanced[target] = y_res
    return df_balanced

# =============================================================================
# 7. Pipeline Principal: make_dataset
# =============================================================================

def make_dataset(source: Union[str, pd.DataFrame],
                 target_column: str,
                 expected_schema: Optional[Dict[str, Any]] = None,
                 file_type: str = 'csv',
                 chunksize: Optional[int] = None) -> pd.DataFrame:
    """Pipeline principal para transformar o dataset em um formato pronto para modelagem"""
    df = load_data(source, file_type=file_type, chunksize=chunksize)

    if expected_schema:
        validate_schema(df, expected_schema)

    df = advanced_missing_imputation(df)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df = handle_outliers(df, numeric_cols, method='IQR', factor=1.5)

    df = encode_and_scale_features(df, exclude_cols=[target_column])

    df = feature_selection(df, target=target_column)
    df = balance_classes(df, target=target_column)

    logger.info(f"Dataset final shape: {df.shape}")
    return df

# =============================================================================
# Execução Principal
# =============================================================================

def main() -> None:
    data_source = "../../data/raw/dataset_financeiro_simulado.csv"
    expected_schema = {
        # Definir o esquema esperado das colunas
    }

    target_column = "flag_fraude"

    try:
        dataset_final = make_dataset(
            source=data_source,
            target_column=target_column,
            expected_schema=expected_schema,
            file_type='csv'
        )
        output_path = "../../data/processed/dataset_bank_make.csv"
        dataset_final.to_csv(output_path, index=False)
        logger.info(f"Dataset final salvo em: {output_path}")
    except Exception as e:
        logger.error(f"Erro no pipeline make_dataset: {e}")

if __name__ == '__main__':
    main()
