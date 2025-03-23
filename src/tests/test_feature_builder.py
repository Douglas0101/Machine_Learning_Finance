"""
Módulo para construção e transformação de features para modelos de predição de inadimplência.
Inclui processamento de variáveis numéricas, categóricas, temporais e criação de features
derivadas específicas para o contexto de crédito.
"""

import os
import pandas as pd
import numpy as np
import joblib
import yaml
import logging
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import NotFittedError

# Configurar logger
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def get_project_root():
    """Retorna o caminho para a raiz do projeto."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, os.pardir, os.pardir))
    return project_root


class WoETransformer(BaseEstimator, TransformerMixin):
    """
    Transformador para codificação Weight of Evidence (WoE).
    Substitui categorias por seu valor de WoE, que mede a força da relação
    com a variável target.
    """

    def __init__(self, smooth: float = 0.5, min_samples: int = 30):
        """
        Inicializa o transformador WoE.

        Args:
            smooth: Fator de suavização para evitar divisão por zero e valores extremos
            min_samples: Número mínimo de amostras para uma categoria ser considerada
        """
        self.smooth = smooth
        self.min_samples = min_samples
        self.woe_maps = {}
        self.default_woe = {}
        self.iv_values = {}
        self._fitted = False

    def fit(self, X, y):
        """
        Calcula o WoE e IV para cada categoria de cada variável.

        Args:
            X: DataFrame com as variáveis categóricas
            y: Series com o target (0 ou 1)

        Returns:
            self
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        # Verificar que y é binário
        unique_y = np.unique(y)
        if len(unique_y) != 2:
            raise ValueError(f"y deve ser binário, mas contém {len(unique_y)} valores distintos")

        # Total de eventos (1) e não-eventos (0)
        total_events = y.sum()
        total_non_events = len(y) - total_events

        # Eventos e não-eventos por variável categórica
        for col in X.columns:
            woe_dict = {}
            iv = 0

            # Contar eventos e não-eventos por categoria
            counts = pd.crosstab(X[col], y)

            # Renomear colunas se necessário
            if 0 not in counts.columns:
                counts[0] = 0
            if 1 not in counts.columns:
                counts[1] = 0

            # Calcular WoE e IV para cada categoria
            for category in counts.index:
                non_events = counts.loc[category, 0]
                events = counts.loc[category, 1]

                # Verificar se há amostras suficientes
                total_in_category = non_events + events
                if total_in_category < self.min_samples:
                    logger.warning(f"Categoria '{category}' na variável '{col}' tem menos do que {self.min_samples} amostras. Usando WoE global.")
                    continue

                # Calcular proporções com suavização
                event_rate = (events + self.smooth) / (total_events + self.smooth * len(counts))
                non_event_rate = (non_events + self.smooth) / (total_non_events + self.smooth * len(counts))

                # Calcular WoE
                woe = np.log(non_event_rate / event_rate)
                woe_dict[category] = woe

                # Calcular contribuição para o IV
                iv_contribution = (non_event_rate - event_rate) * woe
                iv += iv_contribution

            # Calcular WoE padrão (média ponderada)
            default_woe = 0
            total_weight = 0

            for category, woe in woe_dict.items():
                count = counts.loc[category].sum()
                default_woe += woe * count
                total_weight += count

            if total_weight > 0:
                default_woe /= total_weight

            # Armazenar resultados
            self.woe_maps[col] = woe_dict
            self.default_woe[col] = default_woe
            self.iv_values[col] = iv

        self._fitted = True
        return self

    def transform(self, X):
        """
        Transforma as categorias em seus valores de WoE.

        Args:
            X: DataFrame com as variáveis categóricas

        Returns:
            DataFrame com os valores transformados
        """
        if not self._fitted:
            raise NotFittedError("WoETransformer não foi treinado. Chame fit() antes de transform().")

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        X_woe = X.copy()

        for col in X.columns:
            if col in self.woe_maps:
                # Transformar cada valor pelo seu WoE correspondente
                woe_map = self.woe_maps[col]
                default_value = self.default_woe[col]

                # Aplicar mapeamento, usando valor padrão para categorias não vistas
                X_woe[col] = X[col].map(lambda x: woe_map.get(x, default_value))

        return X_woe

    def fit_transform(self, X, y=None, **fit_params):
        """
        Ajusta o transformador aos dados e retorna os dados transformados.

        Args:
            X: DataFrame com as variáveis categóricas
            y: Series com o target (0 ou 1)

        Returns:
            DataFrame com os valores transformados
        """
        return self.fit(X, y).transform(X)

    def get_feature_names_out(self, input_features=None):
        """
        Retorna nomes das features após transformação.

        Args:
            input_features: Nomes das features de entrada (opcional)

        Returns:
            Lista de nomes das features
        """
        if not self._fitted:
            raise NotFittedError("WoETransformer não foi treinado. Chame fit() antes de get_feature_names_out().")

        # Se input_features não for fornecido, usar as chaves do woe_maps
        if input_features is None:
            if self.woe_maps:
                return np.array(list(self.woe_maps.keys()))
            else:
                raise ValueError("input_features não fornecido e woe_maps vazio")

        # WoE mantém os mesmos nomes de features, apenas transforma os valores
        return np.array(input_features)

    def __sklearn_is_fitted__(self):
        """Método exigido pela sklearn para verificar se o estimador foi ajustado."""
        return self._fitted

    def get_iv_values(self) -> Dict[str, float]:
        """
        Retorna os valores de Information Value para cada variável.
        Valores de IV podem ser interpretados como:
        < 0.02: não preditiva
        0.02-0.1: fraca
        0.1-0.3: média
        0.3-0.5: forte
        > 0.5: muito forte (possível overfit)

        Returns:
            Dicionário com os valores de IV por coluna
        """
        if not self._fitted:
            raise NotFittedError("WoETransformer não foi treinado. Chame fit() antes de get_iv_values().")

        return self.iv_values


class AgeCalculator(BaseEstimator, TransformerMixin):
    """
    Transformador para calcular idade a partir de data de nascimento.
    """

    def __init__(self, date_col: str, reference_date: Optional[str] = None, output_col: str = 'idade'):
        """
        Inicializa o transformador de idade.

        Args:
            date_col: Nome da coluna com a data de nascimento
            reference_date: Data de referência para cálculo (formato 'YYYY-MM-DD').
                          Se None, usa a data atual no momento da transformação.
            output_col: Nome da coluna de saída com a idade calculada
        """
        self.date_col = date_col
        self.reference_date = reference_date
        self.output_col = output_col
        self._fitted = False

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'AgeCalculator':
        """
        Não faz nada, apenas retorna self. Incluído para compatibilidade com API sklearn.
        """
        self._fitted = True
        self.input_features_ = X.columns.tolist()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula a idade com base na data de nascimento.

        Args:
            X: DataFrame contendo a coluna de data de nascimento

        Returns:
            DataFrame com a coluna de idade adicionada
        """
        X_transformed = X.copy()

        # Converter coluna para datetime se não for
        if not pd.api.types.is_datetime64_dtype(X[self.date_col]):
            X_transformed[self.date_col] = pd.to_datetime(X[self.date_col], errors='coerce')

        # Determinar data de referência
        reference_date = pd.to_datetime(self.reference_date) if self.reference_date else pd.Timestamp.now()

        # Calcular idade em anos
        X_transformed[self.output_col] = (reference_date - X_transformed[self.date_col]).dt.days / 365.25
        X_transformed[self.output_col] = X_transformed[self.output_col].astype(int)

        return X_transformed

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Ajusta o transformador e aplica a transformação.

        Args:
            X: DataFrame contendo a coluna de data de nascimento
            y: Ignorado, incluído para compatibilidade

        Returns:
            DataFrame com a coluna de idade adicionada
        """
        return self.fit(X, y).transform(X)

    def get_feature_names_out(self, input_features=None):
        """
        Retorna nomes das features após transformação.

        Args:
            input_features: Nomes das features de entrada (opcional)

        Returns:
            Lista de nomes das features
        """
        if not self._fitted:
            # Para compatibilidade, vamos apenas retornar os nomes das colunas de entrada e saída
            if input_features is not None:
                all_features = list(input_features)
                if self.output_col not in all_features:
                    all_features.append(self.output_col)
                return np.array(all_features)
            else:
                return np.array([self.date_col, self.output_col])

        # Se temos colunas armazenadas do fit
        if hasattr(self, 'input_features_'):
            all_columns = list(self.input_features_)
            if self.output_col not in all_columns:
                all_columns.append(self.output_col)
            return np.array(all_columns)

        # Fallback
        return np.array([self.date_col, self.output_col])

    def __sklearn_is_fitted__(self):
        """Método exigido pela sklearn para verificar se o estimador foi ajustado."""
        return self._fitted


class DateFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Extrai características a partir de colunas de data.
    """

    def __init__(self, date_cols: List[str],
                 extract_year: bool = True,
                 extract_month: bool = True,
                 extract_day: bool = True,
                 extract_dow: bool = True,
                 extract_quarter: bool = True,
                 extract_weekend: bool = True,
                 drop_original: bool = False):
        """
        Inicializa o extrator de características de data.

        Args:
            date_cols: Lista de colunas de data
            extract_year: Se deve extrair o ano
            extract_month: Se deve extrair o mês
            extract_day: Se deve extrair o dia
            extract_dow: Se deve extrair o dia da semana
            extract_quarter: Se deve extrair o trimestre
            extract_weekend: Se deve extrair flag de fim de semana
            drop_original: Se deve remover as colunas originais
        """
        self.date_cols = date_cols
        self.extract_year = extract_year
        self.extract_month = extract_month
        self.extract_day = extract_day
        self.extract_dow = extract_dow
        self.extract_quarter = extract_quarter
        self.extract_weekend = extract_weekend
        self.drop_original = drop_original
        self._fitted = False
        self.output_features_ = []

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'DateFeatureExtractor':
        """
        Prepara o extrator analisando o DataFrame.

        Args:
            X: DataFrame com colunas de data
            y: Ignorado, incluído para compatibilidade

        Returns:
            self
        """
        # Armazenar colunas de entrada
        self.input_features_ = X.columns.tolist()

        # Determinar colunas de saída
        self.output_features_ = []

        for col in self.date_cols:
            # Verificar se a coluna existe
            if col not in X.columns:
                logger.warning(f"Coluna de data '{col}' não encontrada no DataFrame.")
                continue

            # Adicionar colunas originais, se não forem removidas
            if not self.drop_original:
                self.output_features_.append(col)

            # Adicionar colunas derivadas
            if self.extract_year:
                self.output_features_.append(f'{col}_ano')

            if self.extract_month:
                self.output_features_.append(f'{col}_mes')

            if self.extract_day:
                self.output_features_.append(f'{col}_dia')

            if self.extract_dow:
                self.output_features_.append(f'{col}_dia_semana')

            if self.extract_quarter:
                self.output_features_.append(f'{col}_trimestre')

            if self.extract_weekend:
                self.output_features_.append(f'{col}_fim_semana')

        # Adicionar outras colunas que não são dates
        for col in X.columns:
            if col not in self.date_cols and col not in self.output_features_:
                self.output_features_.append(col)

        self._fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Extrai características de data e adiciona ao DataFrame.

        Args:
            X: DataFrame contendo as colunas de data

        Returns:
            DataFrame com características extraídas
        """
        X_transformed = X.copy()

        for col in self.date_cols:
            # Verificar se a coluna existe
            if col not in X_transformed.columns:
                logger.warning(f"Coluna de data '{col}' não encontrada no DataFrame.")
                continue

            # Converter para datetime se não for
            if not pd.api.types.is_datetime64_dtype(X_transformed[col]):
                X_transformed[col] = pd.to_datetime(X_transformed[col], errors='coerce')

            # Extrair características
            if self.extract_year:
                X_transformed[f'{col}_ano'] = X_transformed[col].dt.year

            if self.extract_month:
                X_transformed[f'{col}_mes'] = X_transformed[col].dt.month

            if self.extract_day:
                X_transformed[f'{col}_dia'] = X_transformed[col].dt.day

            if self.extract_dow:
                X_transformed[f'{col}_dia_semana'] = X_transformed[col].dt.dayofweek

            if self.extract_quarter:
                X_transformed[f'{col}_trimestre'] = X_transformed[col].dt.quarter

            if self.extract_weekend:
                X_transformed[f'{col}_fim_semana'] = X_transformed[col].dt.dayofweek.isin([5, 6]).astype(int)

            # Remover coluna original se solicitado
            if self.drop_original:
                X_transformed = X_transformed.drop(columns=[col])

        return X_transformed

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Ajusta e transforma as colunas de data.

        Args:
            X: DataFrame contendo as colunas de data
            y: Ignorado, incluído para compatibilidade

        Returns:
            DataFrame com características extraídas
        """
        return self.fit(X, y).transform(X)

    def get_feature_names_out(self, input_features=None):
        """
        Retorna nomes das features após transformação.

        Args:
            input_features: Nomes das features de entrada (opcional)

        Returns:
            Lista de nomes das features
        """
        if not self._fitted:
            # Para compatibilidade, vamos apenas retornar os nomes das colunas derivadas
            # mesmo se não estiver ajustado
            output_features = []

            # Se input_features não foi fornecido, usar date_cols
            date_cols = self.date_cols
            if input_features is not None:
                # Filtrar apenas as colunas que estão em date_cols
                date_cols = [col for col in input_features if col in self.date_cols]

                # Adicionar outras colunas que não são dates
                non_date_cols = [col for col in input_features if col not in self.date_cols]
                output_features.extend(non_date_cols)

            for col in date_cols:
                # Adicionar colunas originais, se não forem removidas
                if not self.drop_original:
                    output_features.append(col)

                # Adicionar colunas derivadas
                if self.extract_year:
                    output_features.append(f'{col}_ano')

                if self.extract_month:
                    output_features.append(f'{col}_mes')

                if self.extract_day:
                    output_features.append(f'{col}_dia')

                if self.extract_dow:
                    output_features.append(f'{col}_dia_semana')

                if self.extract_quarter:
                    output_features.append(f'{col}_trimestre')

                if self.extract_weekend:
                    output_features.append(f'{col}_fim_semana')

            return np.array(output_features)

        # Retornar colunas de saída determinadas durante o fit
        if hasattr(self, 'output_features_'):
            return np.array(self.output_features_)

        # Não deveria chegar aqui
        return np.array([])

    def __sklearn_is_fitted__(self):
        """Método exigido pela sklearn para verificar se o estimador foi ajustado."""
        return self._fitted


class OutlierHandler(BaseEstimator, TransformerMixin):
    """
    Lida com outliers em variáveis numéricas usando diferentes métodos.
    """

    def __init__(self, method: str = 'winsorize', quantile_range: Tuple[float, float] = (0.01, 0.99),
                 std_threshold: float = 3.0):
        """
        Inicializa o tratador de outliers.

        Args:
            method: Método para tratar outliers ('winsorize', 'clip', 'quantile', 'std')
            quantile_range: Range de quantis para winsorização ou recorte (min, max)
            std_threshold: Número de desvios padrão para método std
        """
        self.method = method
        self.quantile_range = quantile_range
        self.std_threshold = std_threshold
        self.limits = {}
        self._fitted = False

        # Validar método
        valid_methods = ['winsorize', 'clip', 'quantile', 'std']
        if self.method not in valid_methods:
            raise ValueError(f"Método '{method}' não reconhecido. Use um de: {valid_methods}")

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'OutlierHandler':
        """
        Calcula os limites para tratamento de outliers.

        Args:
            X: DataFrame com variáveis numéricas
            y: Ignorado, incluído para compatibilidade

        Returns:
            self
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        # Armazenar colunas de entrada para usar em get_feature_names_out
        self.input_features_ = X.columns.tolist()

        for col in X.columns:
            if not np.issubdtype(X[col].dtype, np.number):
                logger.warning(f"Coluna '{col}' não é numérica e será ignorada.")
                continue

            if self.method == 'winsorize' or self.method == 'clip':
                q_low, q_high = X[col].quantile(self.quantile_range)
                self.limits[col] = (q_low, q_high)

            elif self.method == 'quantile':
                q_low, q_high = X[col].quantile(self.quantile_range)
                self.limits[col] = (q_low, q_high)

            elif self.method == 'std':
                mean = X[col].mean()
                std = X[col].std()
                self.limits[col] = (
                    mean - self.std_threshold * std,
                    mean + self.std_threshold * std
                )

        self._fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Aplica o tratamento de outliers.

        Args:
            X: DataFrame com variáveis numéricas

        Returns:
            DataFrame com outliers tratados
        """
        if not self._fitted:
            raise NotFittedError("OutlierHandler não foi treinado. Chame fit() antes de transform().")

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        X_transformed = X.copy()

        for col in X.columns:
            if col not in self.limits:
                continue

            lower_limit, upper_limit = self.limits[col]

            if self.method == 'winsorize' or self.method == 'clip':
                X_transformed[col] = X_transformed[col].clip(lower_limit, upper_limit)

            elif self.method == 'quantile':
                # Substituir outliers por NaN
                mask = (X_transformed[col] < lower_limit) | (X_transformed[col] > upper_limit)
                X_transformed.loc[mask, col] = np.nan

            elif self.method == 'std':
                # Substituir outliers por NaN
                mask = (X_transformed[col] < lower_limit) | (X_transformed[col] > upper_limit)
                X_transformed.loc[mask, col] = np.nan

        return X_transformed

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Ajusta aos dados e aplica o tratamento de outliers.

        Args:
            X: DataFrame com variáveis numéricas
            y: Ignorado, incluído para compatibilidade

        Returns:
            DataFrame com outliers tratados
        """
        return self.fit(X, y).transform(X)

    def get_feature_names_out(self, input_features=None):
        """
        Retorna nomes das features após transformação.

        Args:
            input_features: Nomes das features de entrada (opcional)

        Returns:
            Lista de nomes das features
        """
        if not self._fitted:
            raise NotFittedError("OutlierHandler não foi treinado. Chame fit() antes de get_feature_names_out().")

        # Se input_features não for fornecido, usar as colunas do DataFrame usado no fit
        if input_features is None:
            if hasattr(self, 'input_features_'):
                return np.array(self.input_features_)
            else:
                raise ValueError("input_features não fornecido e não disponível do fit")

        # Caso contrário, retornar os input_features (OutlierHandler não muda os nomes)
        return np.array(input_features)

    def __sklearn_is_fitted__(self):
        """Método exigido pela sklearn para verificar se o estimador foi ajustado."""
        return self._fitted


class FeatureBuilder:
    """
    Constrói, organiza e transforma features para modelos de crédito.
    Gerencia todo o processo de preparação de dados, incluindo imputação,
    tratamento de outliers, encoding de variáveis categóricas, e extração
    de features avançadas.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Inicializa o construtor de features.

        Args:
            config_path: Caminho para arquivo de configuração YAML (opcional)
        """
        self.config = self._load_config(config_path)
        self.preprocessor = None
        self.feature_names = None
        self.transformers = {}
        self._fitted = False

    def _load_config(self, config_path: Optional[str]) -> Dict:
        """
        Carrega configurações do arquivo YAML.

        Args:
            config_path: Caminho para arquivo de configuração

        Returns:
            Dicionário com configurações
        """
        default_config = {
            'numerical': {
                'scaling': 'standard',
                'impute_strategy': 'median',
                'handle_outliers': True,
                'outlier_method': 'winsorize'
            },
            'categorical': {
                'encoding': 'onehot',
                'impute_strategy': 'most_frequent',
                'max_categories': 10
            },
            'dates': {
                'extract_features': True,
                'extract_year': True,
                'extract_month': True,
                'extract_day': True,
                'extract_dow': True,
                'extract_quarter': True,
                'extract_weekend': True
            }
        }

        if config_path is None:
            return default_config

        try:
            with open(config_path, 'r') as file:
                user_config = yaml.safe_load(file)

            # Mesclar com configurações padrão
            for section in default_config:
                if section in user_config:
                    default_config[section].update(user_config[section])

            # Adicionar novas seções
            for section in user_config:
                if section not in default_config:
                    default_config[section] = user_config[section]

            return default_config

        except Exception as e:
            logger.warning(f"Erro ao carregar configurações: {str(e)}. Usando configurações padrão.")
            return default_config

    def add_numerical_features(self, df: pd.DataFrame, cols: List[str],
                              scaling: Optional[str] = None,
                              impute_strategy: Optional[str] = None,
                              handle_outliers: Optional[bool] = None,
                              outlier_method: Optional[str] = None) -> 'FeatureBuilder':
        """
        Adiciona colunas numéricas para processamento.

        Args:
            df: DataFrame para identificar os tipos de dados
            cols: Lista de colunas numéricas
            scaling: Método de scaling ('standard', 'minmax', 'robust', None)
            impute_strategy: Estratégia de imputação ('mean', 'median', 'constant', None)
            handle_outliers: Se deve tratar outliers
            outlier_method: Método para tratar outliers ('winsorize', 'clip', 'quantile', 'std')

        Returns:
            self
        """
        # Usar configurações do arquivo se não especificado
        scaling = scaling or self.config['numerical']['scaling']
        impute_strategy = impute_strategy or self.config['numerical']['impute_strategy']

        if handle_outliers is None:
            handle_outliers = self.config['numerical']['handle_outliers']

        outlier_method = outlier_method or self.config['numerical']['outlier_method']

        # Verificar tipos de colunas e filtrar apenas numéricas
        numerical_cols = []
        for col in cols:
            if col not in df.columns:
                logger.warning(f"Coluna '{col}' não encontrada no DataFrame.")
                continue

            if np.issubdtype(df[col].dtype, np.number):
                numerical_cols.append(col)
            else:
                logger.warning(f"Coluna '{col}' não é numérica e será ignorada.")

        # Criar passos do pipeline
        steps = []

        # Adicionar tratamento de outliers
        if handle_outliers:
            steps.append(('outlier_handler', OutlierHandler(method=outlier_method)))

        # Adicionar imputação
        if impute_strategy:
            steps.append(('imputer', SimpleImputer(strategy=impute_strategy)))

        # Adicionar scaling
        if scaling == 'standard':
            steps.append(('scaler', StandardScaler()))
        elif scaling == 'minmax':
            steps.append(('scaler', MinMaxScaler()))
        elif scaling == 'robust':
            steps.append(('scaler', RobustScaler()))
        elif scaling is not None:
            logger.warning(f"Método de scaling '{scaling}' não reconhecido. Nenhum scaling será aplicado.")

        # Criar pipeline
        if steps:
            self.transformers['numerical'] = {
                'columns': numerical_cols,
                'pipeline': Pipeline(steps)
            }

        return self

    def add_categorical_features(self, df: pd.DataFrame, cols: List[str],
                                encoding: Optional[str] = None,
                                impute_strategy: Optional[str] = None,
                                max_categories: Optional[int] = None,
                                handle_rare: bool = True,
                                rare_threshold: float = 0.01) -> 'FeatureBuilder':
        """
        Adiciona colunas categóricas para processamento.

        Args:
            df: DataFrame para identificar os tipos de dados
            cols: Lista de colunas categóricas
            encoding: Método de encoding ('onehot', 'ordinal', 'label', 'woe')
            impute_strategy: Estratégia de imputação ('most_frequent', 'constant', None)
            max_categories: Número máximo de categorias para onehot encoding
            handle_rare: Se deve agrupar categorias raras
            rare_threshold: Limiar de frequência para categorias raras

        Returns:
            self
        """
        # Usar configurações do arquivo se não especificado
        encoding = encoding or self.config['categorical']['encoding']
        impute_strategy = impute_strategy or self.config['categorical']['impute_strategy']
        max_categories = max_categories or self.config['categorical']['max_categories']

        # Verificar tipos de colunas e filtrar
        categorical_cols = []
        for col in cols:
            if col not in df.columns:
                logger.warning(f"Coluna '{col}' não encontrada no DataFrame.")
                continue

            # Colunas categóricas ou de objeto
            if isinstance(df[col].dtype, pd.CategoricalDtype) or pd.api.types.is_object_dtype(df[col]):
                categorical_cols.append(col)
            else:
                # Se é numérica mas com poucos valores únicos, considerar categórica
                if pd.api.types.is_numeric_dtype(df[col]) and df[col].nunique() <= max_categories:
                    categorical_cols.append(col)
                    logger.info(f"Coluna numérica '{col}' com {df[col].nunique()} valores únicos será tratada como categórica.")
                else:
                    logger.warning(f"Coluna '{col}' não é categórica e será ignorada.")

        # Criar passos do pipeline
        steps = []

        # Adicionar imputação
        if impute_strategy:
            steps.append(('imputer', SimpleImputer(strategy=impute_strategy, fill_value='MISSING')))

        # Adicionar encoding
        if encoding == 'onehot':
            encoder_params = {
                'handle_unknown': 'ignore',
                'sparse_output': False
            }

            if max_categories is not None and max_categories > 0:
                encoder_params['max_categories'] = max_categories
                encoder_params['min_frequency'] = rare_threshold if handle_rare else None

            steps.append(('encoder', OneHotEncoder(**encoder_params)))

        elif encoding == 'ordinal':
            steps.append(('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)))

        elif encoding == 'label':
            steps.append(('encoder', LabelEncoder()))

        elif encoding == 'woe':
            # WoE requer o target para fit - será adicionado depois
            steps.append(('encoder', WoETransformer()))

        elif encoding is not None:
            logger.warning(f"Método de encoding '{encoding}' não reconhecido. Nenhum encoding será aplicado.")

        # Criar pipeline
        if steps:
            self.transformers['categorical'] = {
                'columns': categorical_cols,
                'pipeline': Pipeline(steps)
            }

        return self

    def add_date_features(self, df: pd.DataFrame, cols: List[str],
                          extract_year: Optional[bool] = None,
                          extract_month: Optional[bool] = None,
                          extract_day: Optional[bool] = None,
                          extract_dow: Optional[bool] = None,
                          extract_quarter: Optional[bool] = None,
                          extract_weekend: Optional[bool] = None,
                          drop_original: bool = False) -> 'FeatureBuilder':
        """
        Adiciona colunas de data para extração de features.

        Args:
            df: DataFrame para identificar os tipos de dados
            cols: Lista de colunas de data
            extract_year: Se deve extrair o ano
            extract_month: Se deve extrair o mês
            extract_day: Se deve extrair o dia
            extract_dow: Se deve extrair o dia da semana
            extract_quarter: Se deve extrair o trimestre
            extract_weekend: Se deve extrair flag de fim de semana
            drop_original: Se deve remover as colunas originais

        Returns:
            self
        """
        # Usar configurações do arquivo se não especificado
        date_config = self.config.get('dates', {})
        extract_year = extract_year if extract_year is not None else date_config.get('extract_year', True)
        extract_month = extract_month if extract_month is not None else date_config.get('extract_month', True)
        extract_day = extract_day if extract_day is not None else date_config.get('extract_day', True)
        extract_dow = extract_dow if extract_dow is not None else date_config.get('extract_dow', True)
        extract_quarter = extract_quarter if extract_quarter is not None else date_config.get('extract_quarter', True)
        extract_weekend = extract_weekend if extract_weekend is not None else date_config.get('extract_weekend', True)

        # Verificar colunas de data
        date_cols = []
        for col in cols:
            if col not in df.columns:
                logger.warning(f"Coluna '{col}' não encontrada no DataFrame.")
                continue

            # Tentar converter para datetime
            try:
                pd.to_datetime(df[col])
                date_cols.append(col)
            except:
                logger.warning(f"Coluna '{col}' não pode ser convertida para datetime e será ignorada.")

        # Criar extrator de features de data
        if date_cols:
            self.transformers['dates'] = {
                'columns': date_cols,
                'transformer': DateFeatureExtractor(
                    date_cols=date_cols,
                    extract_year=extract_year,
                    extract_month=extract_month,
                    extract_day=extract_day,
                    extract_dow=extract_dow,
                    extract_quarter=extract_quarter,
                    extract_weekend=extract_weekend,
                    drop_original=drop_original
                )
            }

        return self

    def add_interaction_features(self, interaction_pairs: List[Tuple[str, str]]) -> 'FeatureBuilder':
        """
        Adiciona pares de colunas para criar features de interação.

        Args:
            interaction_pairs: Lista de tuplas (col1, col2) de colunas a interagir

        Returns:
            self
        """
        if interaction_pairs:
            self.transformers['interactions'] = {
                'pairs': interaction_pairs
            }

        return self

    def add_credit_specific_features(self, df: pd.DataFrame,
                                     income_col: Optional[str] = None,
                                     debt_cols: Optional[List[str]] = None,
                                     late_payment_cols: Optional[List[str]] = None,
                                     credit_utilization_cols: Optional[Tuple[str, str]] = None,
                                     age_col: Optional[str] = None) -> 'FeatureBuilder':
        """
        Adiciona features específicas para análise de crédito.

        Args:
            df: DataFrame para verificar colunas
            income_col: Coluna de renda
            debt_cols: Colunas de dívidas
            late_payment_cols: Colunas de pagamentos em atraso
            credit_utilization_cols: Tupla (col_usado, col_limite) para cálculo de utilização de crédito
            age_col: Coluna de data de nascimento para cálculo de idade

        Returns:
            self
        """
        credit_features = {}

        # Verificar renda
        if income_col and income_col in df.columns:
            credit_features['income_col'] = income_col

        # Verificar dívidas
        if debt_cols:
            valid_debt_cols = [col for col in debt_cols if col in df.columns]
            if valid_debt_cols:
                credit_features['debt_cols'] = valid_debt_cols

        # Verificar pagamentos em atraso
        if late_payment_cols:
            valid_late_cols = [col for col in late_payment_cols if col in df.columns]
            if valid_late_cols:
                credit_features['late_payment_cols'] = valid_late_cols

        # Verificar utilização de crédito
        if credit_utilization_cols:
            used_col, limit_col = credit_utilization_cols
            if used_col in df.columns and limit_col in df.columns:
                credit_features['credit_utilization'] = (used_col, limit_col)

        # Verificar idade
        if age_col and age_col in df.columns:
            try:
                # Testar se pode ser convertido para datetime
                pd.to_datetime(df[age_col])
                credit_features['age_col'] = age_col
            except:
                logger.warning(f"Coluna de idade '{age_col}' não pode ser convertida para datetime.")

        if credit_features:
            self.transformers['credit_specific'] = credit_features

        return self

    def _create_credit_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cria features específicas de crédito.

        Args:
            df: DataFrame com os dados

        Returns:
            DataFrame com features adicionadas
        """
        if 'credit_specific' not in self.transformers:
            return df

        result = df.copy()
        credit_config = self.transformers['credit_specific']

        # Calcular DTI (Debt-to-Income)
        if 'income_col' in credit_config and 'debt_cols' in credit_config:
            income_col = credit_config['income_col']
            debt_cols = credit_config['debt_cols']

            # Somar todas as dívidas
            result['total_debt'] = result[debt_cols].sum(axis=1)

            # Calcular DTI (razão dívida/renda)
            result['debt_to_income'] = result['total_debt'] / result[income_col].replace(0, np.nan)
            result['debt_to_income'] = result['debt_to_income'].fillna(0).clip(0, 10)  # Limitar outliers

        # Calcular score de pagamentos em atraso
        if 'late_payment_cols' in credit_config:
            late_cols = credit_config['late_payment_cols']

            # Criar indicador de pagamentos em atraso
            for col in late_cols:
                # Assumindo que valores positivos indicam atraso
                result[f'{col}_flag'] = (result[col] > 0).astype(int)

            # Criar score total de atraso (soma de todos os flags)
            result['late_payment_score'] = result[[f'{col}_flag' for col in late_cols]].sum(axis=1)

        # Calcular utilização de crédito
        if 'credit_utilization' in credit_config:
            used_col, limit_col = credit_config['credit_utilization']

            result['credit_utilization_ratio'] = result[used_col] / result[limit_col].replace(0, np.nan)
            result['credit_utilization_ratio'] = result['credit_utilization_ratio'].fillna(0).clip(0, 1)

            # Criar flag para alta utilização
            result['high_utilization_flag'] = (result['credit_utilization_ratio'] > 0.7).astype(int)

        # Calcular idade
        if 'age_col' in credit_config:
            age_col = credit_config['age_col']

            # Usar transformer para calcular idade
            age_transformer = AgeCalculator(date_col=age_col)
            result = age_transformer.transform(result)

            # Criar faixas etárias
            age_bins = [0, 25, 35, 45, 55, 65, 120]
            age_labels = ['<25', '25-34', '35-44', '45-54', '55-64', '65+']

            result['age_group'] = pd.cut(result['idade'], bins=age_bins, labels=age_labels, right=False)

        return result

    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cria features de interação entre pares de variáveis.

        Args:
            df: DataFrame com os dados

        Returns:
            DataFrame com features de interação adicionadas
        """
        if 'interactions' not in self.transformers:
            return df

        result = df.copy()
        interactions = self.transformers['interactions']['pairs']

        for col1, col2 in interactions:
            if col1 not in result.columns or col2 not in result.columns:
                logger.warning(f"Colunas de interação '{col1}' ou '{col2}' não encontradas.")
                continue

            # Verifica se ambas são numéricas
            if np.issubdtype(result[col1].dtype, np.number) and np.issubdtype(result[col2].dtype, np.number):
                # Multiplicação
                result[f'{col1}_X_{col2}'] = result[col1] * result[col2]

                # Soma
                result[f'{col1}_+_{col2}'] = result[col1] + result[col2]

                # Razão (com tratamento para divisão por zero)
                result[f'{col1}_/_{col2}'] = result[col1] / result[col2].replace(0, np.nan)
                result[f'{col1}_/_{col2}'] = result[f'{col1}_/_{col2}'].fillna(0)

            # Se uma é categórica e a outra numérica
            elif (np.issubdtype(result[col1].dtype, np.number) and
                  (isinstance(result[col2].dtype, pd.CategoricalDtype) or pd.api.types.is_object_dtype(result[col2]))):
                # Criar uma feature por categoria
                for category in result[col2].dropna().unique():
                    result[f'{col1}_if_{col2}_{category}'] = result[col1] * (result[col2] == category).astype(int)

            elif (np.issubdtype(result[col2].dtype, np.number) and
                  (isinstance(result[col1].dtype, pd.CategoricalDtype) or pd.api.types.is_object_dtype(result[col1]))):
                # Criar uma feature por categoria
                for category in result[col1].dropna().unique():
                    result[f'{col2}_if_{col1}_{category}'] = result[col2] * (result[col1] == category).astype(int)

            # Se ambas são categóricas, cria uma nova feature combinada
            elif ((isinstance(result[col1].dtype, pd.CategoricalDtype) or pd.api.types.is_object_dtype(result[col1])) and
                  (isinstance(result[col2].dtype, pd.CategoricalDtype) or pd.api.types.is_object_dtype(result[col2]))):
                result[f'{col1}_{col2}_combined'] = result[col1].astype(str) + "_" + result[col2].astype(str)

        return result

    def build_preprocessor(self) -> ColumnTransformer:
        """
        Constrói o preprocessador de dados combinando todos os transformadores.

        Returns:
            ColumnTransformer configurado com todos os transformadores
        """
        if not self.transformers:
            raise ValueError("Nenhum transformador configurado. Adicione features antes de construir o preprocessador.")

        transformers = []

        # Adicionar transformadores para variáveis numéricas
        if 'numerical' in self.transformers:
            num_config = self.transformers['numerical']
            transformers.append(
                ('numerical', num_config['pipeline'], num_config['columns'])
            )

        # Adicionar transformadores para variáveis categóricas
        if 'categorical' in self.transformers:
            cat_config = self.transformers['categorical']
            transformers.append(
                ('categorical', cat_config['pipeline'], cat_config['columns'])
            )

        # Criar preprocessador
        self.preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder='drop'  # Remover colunas não especificadas
        )

        return self.preprocessor

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'FeatureBuilder':
        """
        Ajusta todos os transformadores aos dados.

        Args:
            X: DataFrame com os dados
            y: Series com o target (necessário para WoE encoding)

        Returns:
            self
        """
        logger.info("Ajustando transformadores de features...")

        # Armazenar cópia do DataFrame original para uso futuro
        X_transformed = X.copy()

        # Aplicar transformações de data
        if 'dates' in self.transformers:
            try:
                date_transformer = self.transformers['dates']['transformer']
                X_transformed = date_transformer.fit_transform(X_transformed, y)
                logger.info(f"Transformador de datas aplicado: {len(self.transformers['dates']['columns'])} colunas processadas")
            except Exception as e:
                logger.error(f"Erro ao aplicar transformações de data: {str(e)}")

        # Aplicar transformações específicas de crédito
        try:
            X_transformed = self._create_credit_features(X_transformed)
            logger.info("Features específicas de crédito criadas")
        except Exception as e:
            logger.error(f"Erro ao criar features específicas de crédito: {str(e)}")

        # Aplicar transformações de interação
        try:
            X_transformed = self._create_interaction_features(X_transformed)
            logger.info("Features de interação criadas")
        except Exception as e:
            logger.error(f"Erro ao criar features de interação: {str(e)}")

        # Construir e ajustar preprocessador principal
        if not self.preprocessor:
            try:
                self.build_preprocessor()
                logger.info("Preprocessador construído")
            except Exception as e:
                logger.error(f"Erro ao construir preprocessador: {str(e)}")
                raise

        # Verificar tratamento especial para WoE encoding
        if 'categorical' in self.transformers:
            pipeline = self.transformers['categorical']['pipeline']
            for step_name, step in pipeline.steps:
                if isinstance(step, WoETransformer) and y is None:
                    raise ValueError("Target (y) é necessário para WoE encoding.")

        # Ajustar preprocessador
        if self.preprocessor:
            try:
                self.preprocessor.fit(X_transformed, y)
                logger.info("Preprocessador ajustado aos dados")
            except Exception as e:
                logger.error(f"Erro ao ajustar preprocessador: {str(e)}")
                raise

        # Definir que o FeatureBuilder foi ajustado
        self._fitted = True

        # Extrair nomes das features
        try:
            self._extract_feature_names(X_transformed)
            if self.feature_names:
                logger.info(f"Extraídos {len(self.feature_names)} nomes de features")
            else:
                logger.warning("Nenhum nome de feature extraído")
        except Exception as e:
            logger.warning(f"Erro ao extrair nomes de features: {str(e)}")
            # Criar nomes genéricos como fallback
            try:
                X_sample = X_transformed.iloc[:1]
                X_final = self.preprocessor.transform(X_sample)
                num_features = X_final.shape[1]
                self.feature_names = [f"feature_{i}" for i in range(num_features)]
                logger.info(f"Criados {len(self.feature_names)} nomes genéricos para as features")
            except Exception as e2:
                logger.error(f"Erro ao criar nomes genéricos: {str(e2)}")
                self.feature_names = []

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforma os dados aplicando todos os transformadores.

        Args:
            X: DataFrame com os dados

        Returns:
            DataFrame transformado
        """
        if not self.preprocessor:
            raise ValueError("Preprocessador não construído. Chame fit() antes de transform().")

        if not self._fitted:
            raise ValueError("FeatureBuilder não foi ajustado. Chame fit() antes de transform().")

        logger.info("Aplicando transformações nos dados...")

        # Fazer cópia para não modificar o DataFrame original
        X_transformed = X.copy()

        # Aplicar transformações de data
        if 'dates' in self.transformers:
            try:
                X_transformed = self.transformers['dates']['transformer'].transform(X_transformed)
                logger.debug("Transformações de data aplicadas")
            except Exception as e:
                logger.warning(f"Erro ao aplicar transformações de data: {str(e)}")

        # Aplicar transformações específicas de crédito
        try:
            X_transformed = self._create_credit_features(X_transformed)
            logger.debug("Features específicas de crédito criadas")
        except Exception as e:
            logger.warning(f"Erro ao criar features específicas de crédito: {str(e)}")

        # Aplicar transformações de interação
        try:
            X_transformed = self._create_interaction_features(X_transformed)
            logger.debug("Features de interação criadas")
        except Exception as e:
            logger.warning(f"Erro ao criar features de interação: {str(e)}")

        # Aplicar preprocessador principal
        try:
            final_data = self.preprocessor.transform(X_transformed)
            logger.debug("Preprocessador aplicado")
        except Exception as e:
            logger.error(f"Erro ao aplicar preprocessador: {str(e)}")
            raise

        # Converter para DataFrame
        if isinstance(final_data, np.ndarray):
            try:
                # Usar os nomes de features extraídos durante o fit
                if self.feature_names and len(self.feature_names) == final_data.shape[1]:
                    result = pd.DataFrame(final_data, columns=self.feature_names, index=X.index)
                else:
                    # Se o número de features não corresponde, usar nomes genéricos
                    logger.warning(f"Incompatibilidade no número de features: {len(self.feature_names)} nomes vs {final_data.shape[1]} colunas")
                    result = pd.DataFrame(
                        final_data,
                        columns=[f"feature_{i}" for i in range(final_data.shape[1])],
                        index=X.index
                    )
                logger.info(f"Dados transformados: {result.shape[0]} linhas, {result.shape[1]} colunas")
                return result
            except Exception as e:
                logger.error(f"Erro ao converter array para DataFrame: {str(e)}")
                # Retornar o array numpy como fallback
                return final_data
        else:
            # Já é um pandas DataFrame ou similar
            return final_data

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Ajusta todos os transformadores e transforma os dados.

        Args:
            X: DataFrame com os dados
            y: Series com o target (necessário para WoE encoding)

        Returns:
            DataFrame transformado
        """
        try:
            # Ajustar transformadores
            self.fit(X, y)

            # Transformar dados
            return self.transform(X)
        except Exception as e:
            logger.error(f"Erro em fit_transform: {str(e)}")
            raise

    def _extract_feature_names(self, X: pd.DataFrame) -> None:
        """
        Extrai os nomes das features após transformação.

        Args:
            X: DataFrame de exemplo
        """
        # Se o preprocessador não foi construído, não podemos extrair nomes
        if not self.preprocessor:
            logger.warning("Preprocessador não construído. Nomes de features não podem ser extraídos.")
            self.feature_names = []
            return

        # Se o preprocessador não foi ajustado, nomes não podem ser extraídos
        if not hasattr(self.preprocessor, 'transformers_'):
            logger.warning("Preprocessador não ajustado. Nomes de features não podem ser extraídos.")
            self.feature_names = []
            return

        # Tentar usar get_feature_names_out diretamente no preprocessador
        try:
            self.feature_names = self.preprocessor.get_feature_names_out().tolist()
            logger.info(f"Extraídos {len(self.feature_names)} nomes de features via get_feature_names_out()")
            return
        except Exception as e:
            logger.debug(f"Erro ao usar get_feature_names_out() no preprocessador: {str(e)}")
            logger.debug("Tentando extrair nomes manualmente.")

        # Método manual para extrair nomes de features
        try:
            # Para verificação: aplicar transformação e verificar número de colunas
            X_sample = X.iloc[:1]
            X_transformed = self.preprocessor.transform(X_sample)
            num_features = X_transformed.shape[1]

            # Criar nomes genéricos
            self.feature_names = [f"feature_{i}" for i in range(num_features)]
            logger.info(f"Criados {len(self.feature_names)} nomes genéricos para as features")

        except Exception as e:
            logger.warning(f"Erro ao extrair nomes de features: {str(e)}")
            self.feature_names = []

    def save(self, path: str) -> None:
        """
        Salva o objeto FeatureBuilder em disco.

        Args:
            path: Caminho para salvar
        """
        if not self.preprocessor:
            raise ValueError("Preprocessador não construído. Chame fit() antes de save().")

        # Garantir que o diretório existe
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Salvar objeto
        joblib.dump(self, path)
        logger.info(f"FeatureBuilder salvo em: {path}")

    @staticmethod
    def load(path: str) -> 'FeatureBuilder':
        """
        Carrega um objeto FeatureBuilder do disco.

        Args:
            path: Caminho para carregar

        Returns:
            Objeto FeatureBuilder
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Arquivo não encontrado: {path}")

        feature_builder = joblib.load(path)
        logger.info(f"FeatureBuilder carregado de: {path}")

        return feature_builder

    def get_feature_importances(self, model) -> pd.DataFrame:
        """
        Extrai as importâncias das features para um modelo treinado.

        Args:
            model: Modelo treinado com feature_importances_ ou get_feature_importances()

        Returns:
            DataFrame com importâncias das features
        """
        if not self.feature_names:
            raise ValueError("Nomes das features não estão disponíveis. Chame fit() primeiro.")

        # Obter importâncias
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'get_feature_importances'):
            importances = model.get_feature_importances()
        elif hasattr(model, 'coef_'):
            importances = abs(model.coef_[0]) if len(model.coef_.shape) > 1 else abs(model.coef_)
        else:
            raise ValueError("Modelo não suporta extração de importâncias de features.")

        # Garantir mesmo tamanho
        if len(importances) != len(self.feature_names):
            logger.warning(f"Número de importâncias ({len(importances)}) difere do número de features ({len(self.feature_names)}).")
            # Truncar para o menor tamanho
            min_len = min(len(importances), len(self.feature_names))
            importances = importances[:min_len]
            feature_names = self.feature_names[:min_len]
        else:
            feature_names = self.feature_names

        # Criar DataFrame e ordenar
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        })

        importance_df = importance_df.sort_values('importance', ascending=False).reset_index(drop=True)

        return importance_df

    def get_iv_values(self) -> pd.DataFrame:
        """
        Retorna Information Value (IV) das variáveis categóricas se tiver usado WoE encoding.

        Returns:
            DataFrame com valores de IV para cada variável
        """
        if 'categorical' not in self.transformers:
            raise ValueError("Transformador categórico não encontrado.")

        pipeline = self.transformers['categorical']['pipeline']
        for step_name, step in pipeline.steps:
            if isinstance(step, WoETransformer):
                iv_values = step.get_iv_values()

                # Criar DataFrame
                iv_df = pd.DataFrame({
                    'feature': list(iv_values.keys()),
                    'information_value': list(iv_values.values())
                })

                # Adicionar interpretação
                conditions = [
                    iv_df['information_value'] < 0.02,
                    iv_df['information_value'] < 0.1,
                    iv_df['information_value'] < 0.3,
                    iv_df['information_value'] < 0.5,
                    iv_df['information_value'] >= 0.5
                ]
                choices = [
                    'Não preditiva',
                    'Fraca',
                    'Média',
                    'Forte',
                    'Muito forte (possível overfit)'
                ]
                iv_df['interpretacao'] = np.select(conditions, choices, default='Desconhecida')

                # Ordenar
                iv_df = iv_df.sort_values('information_value', ascending=False).reset_index(drop=True)

                return iv_df

        raise ValueError("WoETransformer não encontrado no pipeline categórico.")


def main():
    """
    Demonstração de uso da classe FeatureBuilder.
    Esta função principal executa quando o script é chamado diretamente.
    """
    import argparse
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

    # Configurar parser de argumentos
    parser = argparse.ArgumentParser(description='Demonstração de construção de features')
    parser.add_argument('--data_path', type=str, help='Caminho para dados CSV (opcional)')
    parser.add_argument('--config_path', type=str, help='Caminho para arquivo de configuração YAML (opcional)')
    parser.add_argument('--demo', action='store_true', default=True, help='Executar demonstração com dados sintéticos')

    args = parser.parse_args()

    if args.data_path:
        # Usar dados reais
        logger.info(f"Carregando dados de: {args.data_path}")
        try:
            data = pd.read_csv(args.data_path)

            # Identificar coluna target
            if 'target' in data.columns:
                y = data['target']
                X = data.drop('target', axis=1)
            elif 'inadimplente' in data.columns:
                y = data['inadimplente']
                X = data.drop('inadimplente', axis=1)
            elif 'default' in data.columns:
                y = data['default']
                X = data.drop('default', axis=1)
            else:
                logger.warning("Coluna target não identificada. Assumindo última coluna como target.")
                y = data.iloc[:, -1]
                X = data.iloc[:, :-1]

            logger.info(f"Dados carregados: {X.shape[0]} amostras, {X.shape[1]} features")
        except Exception as e:
            logger.error(f"Erro ao carregar dados: {str(e)}")
            return

    elif args.demo:
        # Criar dados sintéticos
        logger.info("Gerando dados sintéticos para demonstração...")

        # Gerar features numéricas
        X_num, y = make_classification(n_samples=1000, n_features=7, n_informative=5,
                                       n_redundant=2, n_classes=2, weights=[0.8, 0.2],
                                       random_state=42)

        # Converter para DataFrame
        X_num = pd.DataFrame(X_num, columns=[f'num_{i}' for i in range(X_num.shape[1])])

        # Adicionar features categóricas
        X_cat = pd.DataFrame({
            'cat_1': np.random.choice(['A', 'B', 'C', 'D'], size=1000),
            'cat_2': np.random.choice(['Alto', 'Médio', 'Baixo'], size=1000),
            'cat_3': np.random.choice(['SP', 'RJ', 'MG', 'RS', 'PR'], size=1000)
        })

        # Adicionar datas
        start_date = pd.Timestamp('2020-01-01')
        dates = [start_date + pd.Timedelta(days=np.random.randint(0, 365*2)) for _ in range(1000)]

        X_date = pd.DataFrame({
            'data_cadastro': dates,
            'data_nascimento': [pd.Timestamp('1970-01-01') + pd.Timedelta(days=np.random.randint(365*18, 365*70)) for _ in range(1000)]
        })

        # Adicionar features específicas de crédito
        X_credit = pd.DataFrame({
            'renda': np.random.gamma(5, 1000, 1000),
            'divida_cartao': np.random.gamma(2, 500, 1000),
            'divida_emprestimo': np.random.gamma(3, 800, 1000),
            'limite_credito': np.random.gamma(5, 2000, 1000),
            'saldo_utilizado': np.random.gamma(4, 1000, 1000),
            'dias_atraso_30d': np.random.randint(0, 5, 1000),
            'dias_atraso_60d': np.random.randint(0, 3, 1000),
            'dias_atraso_90d': np.random.randint(0, 2, 1000)
        })

        # Combinar todas as features
        X = pd.concat([X_num, X_cat, X_date, X_credit], axis=1)
        y = pd.Series(y, name='inadimplente')

        logger.info(f"Dados sintéticos gerados: {X.shape[0]} amostras, {X.shape[1]} features")

    else:
        logger.error("Nenhum dado fornecido ou modo de demonstração desativado.")
        parser.print_help()
        return

    # Dividir em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Iniciar construtor de features
    feature_builder = FeatureBuilder(config_path=args.config_path)

    # Adicionar features
    feature_builder.add_numerical_features(
        X_train,
        cols=[col for col in X_train.columns if col.startswith('num_') or col in
              ['renda', 'divida_cartao', 'divida_emprestimo', 'limite_credito', 'saldo_utilizado']],
        scaling='standard',
        handle_outliers=True
    )

    feature_builder.add_categorical_features(
        X_train,
        cols=[col for col in X_train.columns if col.startswith('cat_')],
        encoding='onehot'
    )

    feature_builder.add_date_features(
        X_train,
        cols=[col for col in X_train.columns if col.startswith('data_')],
        extract_year=True,
        extract_month=True,
        extract_day=False
    )

    feature_builder.add_credit_specific_features(
        X_train,
        income_col='renda',
        debt_cols=['divida_cartao', 'divida_emprestimo'],
        late_payment_cols=['dias_atraso_30d', 'dias_atraso_60d', 'dias_atraso_90d'],
        credit_utilization_cols=('saldo_utilizado', 'limite_credito'),
        age_col='data_nascimento'
    )

    feature_builder.add_interaction_features([
        ('renda', 'cat_2'),
        ('divida_cartao', 'divida_emprestimo')
    ])

    # Ajustar e transformar dados
    logger.info("Aplicando transformações nos dados de treino...")
    X_train_transformed = feature_builder.fit_transform(X_train, y_train)

    logger.info("Aplicando transformações nos dados de teste...")
    X_test_transformed = feature_builder.transform(X_test)

    # Mostrar resultados
    logger.info(f"Dimensões originais do treino: {X_train.shape}")
    logger.info(f"Dimensões após transformação: {X_train_transformed.shape}")

    # Salvar feature builder
    output_dir = os.path.join(get_project_root(), 'models', 'feature_builders')
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"feature_builder_{timestamp}.joblib")

    feature_builder.save(output_path)
    logger.info(f"FeatureBuilder salvo em: {output_path}")

    print(f"\nDemonstração concluída com sucesso!")
    print(f"Features transformadas: {X_train_transformed.shape[1]}")
    print(f"FeatureBuilder salvo em: {output_path}")


if __name__ == "__main__":
    main()