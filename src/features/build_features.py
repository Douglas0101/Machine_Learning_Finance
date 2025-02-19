import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
import logging

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineering:
    def __init__(self, data: pd.DataFrame):
        self.data = data

    def load_and_validate_data(self, file_path: str) -> pd.DataFrame:
        """
        Carrega o dataset de um arquivo CSV e valida os dados.
        """
        logger.info(f"Carregando o dataset de {file_path}...")
        data = pd.read_csv(file_path)
        logger.info(f"Dataset carregado com {data.shape[0]} linhas e {data.shape[1]} colunas.")
        return data

    def advanced_preprocessing(self) -> pd.DataFrame:
        """
        Aplica pré-processamento avançado, como imputação de valores ausentes, remoção de outliers, e transformações.
        """
        logger.info("Iniciando pré-processamento avançado...")

        # Identificar colunas numéricas e categóricas
        numeric_features = self.data.select_dtypes(include=['number']).columns
        categorical_features = self.data.select_dtypes(exclude=['number']).columns

        # Criar imputadores
        numeric_imputer = SimpleImputer(strategy='mean')  # Imputação pela média para numéricas
        categorical_imputer = SimpleImputer(strategy='most_frequent')  # Imputação pela mais frequente para categóricas

        # Criar ColumnTransformer para aplicar imputadores seletivamente
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_imputer, numeric_features),  # Imputar numéricas com numeric_imputer
                ('cat', categorical_imputer, categorical_features)  # Imputar categóricas com categorical_imputer
            ])

        # Aplicar ColumnTransformer para imputação
        self.data = pd.DataFrame(preprocessor.fit_transform(self.data), columns=self.data.columns)

        logger.info("Imputação de valores ausentes realizada para variáveis numéricas e categóricas.")
        return self.data

    def feature_engineering(self) -> pd.DataFrame:
        """
        Cria novas features, como extração de componentes temporais e aplicação de PCA para redução de dimensionalidade.
        """
        logger.info("Iniciando a engenharia de features...")

        # Exemplo de criação de novas features (componentes temporais a partir de data)
        if 'data_transacao' in self.data.columns:
            self.data['dia_semana_transacao'] = pd.to_datetime(self.data['data_transacao']).dt.dayofweek
            self.data['mes_transacao'] = pd.to_datetime(self.data['data_transacao']).dt.month

        # Verificar o número de características numéricas disponíveis
        numeric_features = self.data.select_dtypes(include=[np.number]).columns
        n_features = len(numeric_features)
        logger.info(f"Número de características numéricas disponíveis: {n_features}")

        # Ajustar n_components do PCA para um valor válido
        n_components = min(5,
                           n_features)  # Definir n_components como o mínimo entre 5 e o número de características disponíveis
        logger.info(f"Aplicando PCA com n_components={n_components}...")

        pca = PCA(n_components=n_components)
        pca_features = pca.fit_transform(self.data[numeric_features])
        pca_df = pd.DataFrame(pca_features, columns=[f'pca_{i}' for i in range(pca_features.shape[1])])

        # Concatenar as novas features de PCA
        self.data = pd.concat([self.data, pca_df], axis=1)

        logger.info(f"Redução de dimensionalidade com PCA aplicada. {pca_features.shape[1]} componentes gerados.")
        return self.data

    def select_features(self, target_column: str) -> (pd.DataFrame, pd.Series):
        """
        Seleciona as features mais relevantes e separa a variável alvo.
        """
        logger.info(f"Selecionando features e separando a coluna alvo '{target_column}'...")

        if target_column not in self.data.columns:
            raise KeyError(f"Coluna '{target_column}' não encontrada no DataFrame.")

        X = self.data.drop(columns=[target_column])
        y = self.data[target_column]
        logger.info("Features e alvo selecionados com sucesso.")

        return X, y

    def run_pipeline(self, file_path: str, target_column: str) -> (pd.DataFrame, pd.Series):
        """
        Executa o pipeline de engenharia de features completo.
        """
        # Carregar dados e validar
        self.data = self.load_and_validate_data(file_path)

        # Pré-processamento avançado
        self.data = self.advanced_preprocessing()

        # Engenharia de features
        self.data = self.feature_engineering()

        # Seleção de features
        X, y = self.select_features(target_column)

        return X, y


# Exemplo de uso:
if __name__ == "__main__":
    file_path = "../../data/raw/dataset_financeiro_simulado.csv"
    target_column = "flag_fraude"  # Substitua pelo nome correto da sua coluna alvo

    # Criar instância da classe e rodar pipeline
    feature_engineering = FeatureEngineering(None)
    X, y = feature_engineering.run_pipeline(file_path, target_column)

    # Exibir as primeiras linhas do dataset processado
    logger.info("Dados processados com sucesso.")
    logger.info(f"X (Features): {X.head()}")
    logger.info(f"y (Alvo): {y.head()}")
