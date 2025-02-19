import logging
from datetime import datetime
from typing import Union, Optional, List, Dict, Any

import numpy as np
import pandas as pd

# Scikit-learn imports (PCA é utilizado em PCAToDataFrame)
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin

import matplotlib.pyplot as plt

# Imbalanced-learn: SMOTE para oversampling
from imblearn.over_sampling import SMOTE

# -----------------------------------------------------------------------------
# CONFIGURAÇÃO DE LOG E SEED
# -----------------------------------------------------------------------------
log_filename = f"make_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler(log_filename)]
)
logger = logging.getLogger(__name__)
np.random.seed(42)


# -----------------------------------------------------------------------------
# FUNÇÕES DE INGESTÃO E VALIDAÇÃO
# -----------------------------------------------------------------------------
def load_data(source: Union[str, pd.DataFrame],
              file_type: str = 'csv',
              chunksize: Optional[int] = None,
              **kwargs) -> pd.DataFrame:
    """Carrega dados de um arquivo CSV ou DataFrame."""
    try:
        if isinstance(source, pd.DataFrame):
            df = source.copy()
        elif file_type.lower() == 'csv':
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


def validate_schema(df: pd.DataFrame, expected_schema: Dict[str, Any]) -> None:
    """Valida se as colunas esperadas estão presentes e se os tipos são compatíveis."""
    missing_cols = [col for col in expected_schema if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Colunas ausentes: {missing_cols}")
    for col, expected_type in expected_schema.items():
        if col in df.columns and not np.issubdtype(df[col].dtype, expected_type):
            logger.warning(f"Coluna '{col}' possui dtype {df[col].dtype}; esperado {expected_type}.")
    logger.info("Validação de esquema concluída.")


# -----------------------------------------------------------------------------
# VERIFICAÇÃO DA DEPENDÊNCIA DO PACOTE category_encoders
# -----------------------------------------------------------------------------
try:
    from category_encoders.hashing import HashingEncoder

    HASHING_ENCODER_AVAILABLE = True
    logger.info("Pacote 'category_encoders' disponível. Utilizando HashingEncoder para encoding categórico.")
except ImportError:
    HASHING_ENCODER_AVAILABLE = False
    logger.warning("Pacote 'category_encoders' não encontrado. Para utilizar o HashingEncoder, instale-o via:\n"
                   "  pip install category_encoders\n  ou\n  conda install -c conda-forge category_encoders\n"
                   "Utilizando fallback: OneHotEncoder para encoding categórico.")


    class HashingEncoder(BaseEstimator, TransformerMixin):
        """
        Fallback: utiliza OneHotEncoder para encoding categórico.
        Essa implementação não realiza hashing real, mas garante a continuidade do pipeline.
        """

        def __init__(self, n_components: int = 8, return_df: bool = True):
            self.n_components = n_components
            self.return_df = return_df
            self.encoder = OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False)

        def fit(self, X, y=None):
            return self.encoder.fit(X, y)

        def transform(self, X):
            out = self.encoder.transform(X)
            if self.return_df:
                return pd.DataFrame(out, columns=[f"hash_{i}" for i in range(out.shape[1])])
            return out

        def get_feature_names_out(self, input_features=None):
            return self.encoder.get_feature_names_out(input_features) if input_features is not None else None


# -----------------------------------------------------------------------------
# TRANSFORMER: PCAToDataFrame
# -----------------------------------------------------------------------------
class PCAToDataFrame(BaseEstimator, TransformerMixin):
    """
    Envolve um objeto PCA e converte sua saída (numpy.ndarray) em um pandas.DataFrame.
    As colunas são nomeadas como PC1, PC2, etc.
    """

    def __init__(self, n_components=0.95, **kwargs):
        self.n_components = n_components
        self.kwargs = kwargs
        self.pca = PCA(n_components=self.n_components, **self.kwargs)

    def fit(self, X, y=None):
        self.pca.fit(X)
        return self

    def transform(self, X):
        X_pca = self.pca.transform(X)
        n_components = X_pca.shape[1]
        columns = [f"PC{i + 1}" for i in range(n_components)]
        if isinstance(X, pd.DataFrame):
            return pd.DataFrame(X_pca, columns=columns, index=X.index)
        else:
            return pd.DataFrame(X_pca, columns=columns)

    def get_feature_names_out(self, input_features=None):
        n_components = self.pca.n_components_
        return [f"PC{i + 1}" for i in range(n_components)]


# -----------------------------------------------------------------------------
# TRANSFORMER: BooleanToIntTransformer
# -----------------------------------------------------------------------------
class BooleanToIntTransformer(BaseEstimator, TransformerMixin):
    """
    Converte arrays booleanos para inteiros (0, 1).
    Se o input for um DataFrame, converte as colunas booleanas.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            df = X.copy()
            bool_cols = df.select_dtypes(include=['bool']).columns
            df[bool_cols] = df[bool_cols].astype(int)
            return df.values
        else:
            X = np.array(X)
            if X.dtype == np.bool_:
                return X.astype(int)
            return X

    def get_feature_names_out(self, input_features=None):
        return input_features if input_features is not None else None


# -----------------------------------------------------------------------------
# TRANSFORMER: DateTimeFeatureExtractor
# -----------------------------------------------------------------------------
class DateTimeFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Converte colunas de data (strings) em datetime e extrai features numéricas:
      - _year, _month, _day, _hour.
    A comparação é feita de forma case-insensitive; se drop_original=True, as colunas originais são removidas.
    """

    def __init__(self, datetime_cols: Optional[List[str]] = None, drop_original: bool = True):
        self.datetime_cols = datetime_cols
        self.drop_original = drop_original
        self._dt_cols_fit = None

    def fit(self, X, y=None):
        df = pd.DataFrame(X).copy()
        if self.datetime_cols is None:
            self._dt_cols_fit = [col for col in df.columns if 'data' in col.lower() or 'dt' in col.lower()]
        else:
            dt_lower = [d.lower() for d in self.datetime_cols]
            self._dt_cols_fit = [col for col in df.columns if col.lower() in dt_lower]
        logger.info(f"DateTimeFeatureExtractor identificou as colunas: {self._dt_cols_fit}")
        return self

    def transform(self, X):
        df = pd.DataFrame(X).copy()
        for col in self._dt_cols_fit:
            df[col] = pd.to_datetime(df[col], errors='coerce')
            df[col + "_year"] = df[col].dt.year
            df[col + "_month"] = df[col].dt.month
            df[col + "_day"] = df[col].dt.day
            df[col + "_hour"] = df[col].dt.hour
        if self.drop_original:
            df.drop(columns=self._dt_cols_fit, inplace=True)
            logger.info(f"Colunas originais {self._dt_cols_fit} removidas após extração.")
        return df

    def get_feature_names_out(self, input_features=None):
        return input_features if input_features is not None else None


# -----------------------------------------------------------------------------
# TRANSFORMER: OutlierClipper
# -----------------------------------------------------------------------------
class OutlierClipper(BaseEstimator, TransformerMixin):
    """
    Realiza clipping de outliers em colunas numéricas utilizando o método IQR.
    """

    def __init__(self, factor: float = 1.5):
        self.factor = factor
        self.lower_bounds_ = {}
        self.upper_bounds_ = {}
        self.feature_names_in_ = None

    def fit(self, X, y=None):
        df = pd.DataFrame(X).copy()
        self.feature_names_in_ = df.columns.tolist()
        for col in self.feature_names_in_:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            self.lower_bounds_[col] = Q1 - self.factor * IQR
            self.upper_bounds_[col] = Q3 + self.factor * IQR
        return self

    def transform(self, X):
        df = pd.DataFrame(X, columns=self.feature_names_in_).copy()
        for col in self.feature_names_in_:
            lb = self.lower_bounds_[col]
            ub = self.upper_bounds_[col]
            df[col] = df[col].clip(lower=lb, upper=ub)
        return df.values

    def get_feature_names_out(self, input_features=None):
        return input_features if input_features is not None else self.feature_names_in_


# -----------------------------------------------------------------------------
# TRANSFORMER: LogTransformer
# -----------------------------------------------------------------------------
class LogTransformer(BaseEstimator, TransformerMixin):
    """
    Aplica transformação logarítmica (log1p) para estabilizar variâncias e reduzir o impacto de valores extremos.
    Antes de aplicar np.log1p, os valores são convertidos para float e negativos são ajustados para 0.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.array(X, dtype=np.float64)
        X_clipped = np.clip(X, 0, None)
        return np.log1p(X_clipped)

    def get_feature_names_out(self, input_features=None):
        return input_features if input_features is not None else None


# -----------------------------------------------------------------------------
# TRANSFORMER: DynamicColumnSelector
# -----------------------------------------------------------------------------
class DynamicColumnSelector(BaseEstimator, TransformerMixin):
    """
    Separa colunas numéricas, categóricas e ordinais e aplica os sub-pipelines correspondentes.
    Possui parâmetros para processar todas as colunas categóricas e excluir colunas indesejadas.
    """

    def __init__(self,
                 numeric_pipeline: Pipeline,
                 ordinal_pipeline: Pipeline,
                 cat_pipeline: Pipeline,
                 ordinal_cols: List[str],
                 cat_max_cardinality: int,
                 process_all_categorical: bool = True,
                 exclude_cols: Optional[List[str]] = None):
        self.numeric_pipeline = numeric_pipeline
        self.ordinal_pipeline = ordinal_pipeline
        self.cat_pipeline = cat_pipeline
        self.ordinal_cols = ordinal_cols
        self.cat_max_cardinality = cat_max_cardinality
        self.process_all_categorical = process_all_categorical
        self.exclude_cols = exclude_cols if exclude_cols is not None else []
        self.column_transformer_ = None

    def fit(self, X, y=None):
        df = pd.DataFrame(X).copy()
        if self.exclude_cols:
            df = df.drop(columns=self.exclude_cols, errors='ignore')
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        ordinal_cols = [col for col in self.ordinal_cols if col in df.columns]
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        cat_cols = [c for c in cat_cols if c not in ordinal_cols]
        if self.process_all_categorical:
            cat_low_card_cols = cat_cols
        else:
            cat_low_card_cols = [c for c in cat_cols if df[c].nunique(dropna=True) <= self.cat_max_cardinality]
        transformers = []
        if numeric_cols:
            transformers.append(("numeric", self.numeric_pipeline, numeric_cols))
        if ordinal_cols:
            transformers.append(("ordinal", self.ordinal_pipeline, ordinal_cols))
        if cat_low_card_cols:
            transformers.append(("categorical", self.cat_pipeline, cat_low_card_cols))
        from sklearn.compose import ColumnTransformer
        self.column_transformer_ = ColumnTransformer(transformers=transformers, remainder="drop")
        self.column_transformer_.fit(df, y)
        return self

    def transform(self, X):
        df = pd.DataFrame(X).copy()
        if self.exclude_cols:
            df = df.drop(columns=self.exclude_cols, errors='ignore')
        transformed_array = self.column_transformer_.transform(df)
        col_names = self.column_transformer_.get_feature_names_out()
        if col_names is None:
            return pd.DataFrame(transformed_array, index=df.index)
        return pd.DataFrame(transformed_array, columns=col_names, index=df.index)


# -----------------------------------------------------------------------------
# FUNÇÃO: build_preprocessing_pipeline
# -----------------------------------------------------------------------------
def build_preprocessing_pipeline(
        datetime_cols: Optional[List[str]] = None,
        drop_original_datetime: bool = True,
        numeric_imputer: str = 'iterative',
        cat_imputer_strategy: str = 'most_frequent',
        ordinal_cols: Optional[List[str]] = None,
        cat_max_cardinality: int = 50,
        process_all_categorical: bool = True,
        exclude_cols: Optional[List[str]] = None,
        apply_pca_flag: bool = True,
        pca_components: float = 0.95,
        hashing_n_components: int = 8
) -> Pipeline:
    """
    Constrói o pipeline de pré-processamento com as seguintes etapas:
      1. Extração de features de data/hora.
      2. Seleção dinâmica de colunas e aplicação dos sub-pipelines:
         - Numéricas: imputação, clipping, conversão de booleanos para inteiros, transformação logarítmica (com tratamento de valores inválidos) e escalonamento.
         - Ordinais e categóricas: imputação e encoding.
           -> Se category_encoders estiver disponível, utiliza HashingEncoder; caso contrário, utiliza OneHotEncoder.
      3. (Opcional) PCA para redução de dimensionalidade, com saída convertida para DataFrame.
    """
    dt_extractor = DateTimeFeatureExtractor(datetime_cols=datetime_cols,
                                            drop_original=drop_original_datetime)
    if numeric_imputer == 'iterative':
        num_imputer = IterativeImputer(random_state=42)
    elif numeric_imputer == 'knn':
        num_imputer = KNNImputer()
    else:
        raise ValueError("numeric_imputer deve ser 'iterative' ou 'knn'.")
    cat_imputer = SimpleImputer(strategy=cat_imputer_strategy)

    # Pipeline numérico
    numeric_pipeline = Pipeline(steps=[
        ('num_imputer', num_imputer),
        ('outlier_clipper', OutlierClipper(factor=1.5)),
        ('bool_to_int', BooleanToIntTransformer()),
        ('log_transform', LogTransformer()),
        ('final_imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Pipeline ordinal
    ordinal_pipeline = Pipeline(steps=[
        ('cat_imputer', cat_imputer),
        ('ordinal_enc', OrdinalEncoder())
    ])

    # Pipeline categórico: utiliza HashingEncoder se disponível; caso contrário, OneHotEncoder.
    if HASHING_ENCODER_AVAILABLE:
        cat_pipeline = Pipeline(steps=[
            ('cat_imputer', cat_imputer),
            ('hashing', HashingEncoder(n_components=hashing_n_components, return_df=True))
        ])
    else:
        cat_pipeline = Pipeline(steps=[
            ('cat_imputer', cat_imputer),
            ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False))
        ])

    if ordinal_cols is None:
        ordinal_cols = []

    col_selector = DynamicColumnSelector(
        numeric_pipeline=numeric_pipeline,
        ordinal_pipeline=ordinal_pipeline,
        cat_pipeline=cat_pipeline,
        ordinal_cols=ordinal_cols,
        cat_max_cardinality=cat_max_cardinality,
        process_all_categorical=process_all_categorical,
        exclude_cols=exclude_cols
    )

    steps = [
        ('date_time_extractor', dt_extractor),
        ('dynamic_col_selector', col_selector)
    ]
    if apply_pca_flag:
        steps.append(('pca', PCAToDataFrame(n_components=pca_components)))
    pipeline = Pipeline(steps=steps)
    return pipeline


# -----------------------------------------------------------------------------
# FUNÇÃO: make_dataset
# -----------------------------------------------------------------------------
def make_dataset(
        source: Union[str, pd.DataFrame],
        target_column: str,
        expected_schema: Optional[Dict[str, Any]] = None,
        file_type: str = 'csv',
        test_size: float = 0.2,
        random_state: int = 42,
        datetime_cols: Optional[List[str]] = None,
        drop_original_datetime: bool = True,
        numeric_imputer: str = 'iterative',
        cat_imputer_strategy: str = 'most_frequent',
        ordinal_cols: Optional[List[str]] = None,
        cat_max_cardinality: int = 50,
        process_all_categorical: bool = True,
        exclude_cols: Optional[List[str]] = None,
        apply_pca_flag: bool = True,
        pca_components: float = 0.95,
        hashing_n_components: int = 8
) -> Dict[str, pd.DataFrame]:
    """
    Carrega, valida e processa o dataset, dividindo-o em treino e teste.
    Retorna um dicionário com: X_train, X_test, y_train e y_test.
    """
    df = load_data(source, file_type=file_type)
    if expected_schema:
        validate_schema(df, expected_schema)
    if target_column not in df.columns:
        raise ValueError(f"Coluna alvo '{target_column}' não encontrada.")
    y = df[target_column].copy()
    X = df.drop(columns=[target_column]).copy()
    stratify_param = y if y.nunique() < 10 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify_param
    )
    logger.info(f"Antes do SMOTE - Treino: {X_train.shape}, Target distribution: {y_train.value_counts().to_dict()}")
    logger.info(f"Teste: {X_test.shape}, Target distribution: {y_test.value_counts().to_dict()}")

    # Aplicar o pipeline de pré-processamento
    pipeline = build_preprocessing_pipeline(
        datetime_cols=datetime_cols,
        drop_original_datetime=drop_original_datetime,
        numeric_imputer=numeric_imputer,
        cat_imputer_strategy=cat_imputer_strategy,
        ordinal_cols=ordinal_cols,
        cat_max_cardinality=cat_max_cardinality,
        process_all_categorical=process_all_categorical,
        exclude_cols=exclude_cols,
        apply_pca_flag=apply_pca_flag,
        pca_components=pca_components,
        hashing_n_components=hashing_n_components
    )
    pipeline.fit(X_train, y_train)
    X_train_proc = pipeline.transform(X_train)
    X_test_proc = pipeline.transform(X_test)

    # Aplicar SMOTE somente no conjunto de treinamento
    smote = SMOTE(random_state=random_state)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_proc, y_train)
    logger.info(
        f"Após SMOTE - Treino: {X_train_resampled.shape}, Target distribution: {pd.Series(y_train_resampled).value_counts().to_dict()}")

    # Converter o resultado de SMOTE para DataFrame, preservando as colunas do pipeline
    X_train_resampled = pd.DataFrame(X_train_resampled, columns=X_train_proc.columns)

    logger.info(f"Dataset final processado -> Treino: {X_train_resampled.shape}, Teste: {X_test_proc.shape}")
    return {
        'X_train': X_train_resampled,
        'X_test': X_test_proc,
        'y_train': pd.Series(y_train_resampled),
        'y_test': y_test
    }


# -----------------------------------------------------------------------------
# FUNÇÃO: main
# -----------------------------------------------------------------------------
def main() -> None:
    data_source = "../../data/raw/dataset_financeiro_simulado.csv"
    expected_schema = {
        # Exemplo: ajuste conforme seu dataset
        # "id_transacao": np.integer,
        # "valor_transacao": np.number,
        # "flag_fraude": np.bool_,
    }
    target_column = "flag_fraude"
    try:
        datasets = make_dataset(
            source=data_source,
            target_column=target_column,
            expected_schema=expected_schema,
            file_type='csv',
            test_size=0.2,
            random_state=42,
            datetime_cols=["data_transacao"],
            drop_original_datetime=True,
            numeric_imputer='iterative',
            cat_imputer_strategy='most_frequent',
            ordinal_cols=[],  # Adicione se houver colunas ordinais
            cat_max_cardinality=50,
            process_all_categorical=True,
            exclude_cols=[],  # Liste colunas a excluir, se houver
            apply_pca_flag=True,
            pca_components=0.95,
            hashing_n_components=8
        )
        # Salvando os datasets processados; os outputs são DataFrames.
        datasets['X_train'].to_csv("../../data/processed/X_train_processed.csv", index=True)
        datasets['X_test'].to_csv("../../data/processed/X_test_processed.csv", index=True)
        datasets['y_train'].to_csv("../../data/processed/y_train.csv", index=True)
        datasets['y_test'].to_csv("../../data/processed/y_test.csv", index=True)
        logger.info("Datasets processados salvos com sucesso.")

        # Visualizações gráficas:
        # 1. Scatter plot dos dois primeiros componentes PCA (se disponíveis)
        X_train_df = datasets['X_train']
        pca_cols = [col for col in X_train_df.columns if col.startswith("PC")]
        if len(pca_cols) >= 2:
            plt.figure(figsize=(8, 6))
            plt.scatter(X_train_df[pca_cols[0]], X_train_df[pca_cols[1]], alpha=0.5)
            plt.xlabel(pca_cols[0])
            plt.ylabel(pca_cols[1])
            plt.title("Scatter Plot dos dois primeiros Componentes PCA")
            plt.show()
        # 2. Histograma da distribuição do target (Treino)
        plt.figure(figsize=(8, 6))
        datasets['y_train'].astype(int).hist(bins=30)
        plt.xlabel("Target")
        plt.ylabel("Frequência")
        plt.title("Histograma do Target (Treino)")
        plt.show()

    except Exception as e:
        logger.error(f"Erro no pipeline make_dataset: {e}")


if __name__ == '__main__':
    main()
