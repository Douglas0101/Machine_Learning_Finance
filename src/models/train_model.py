"""
Módulo para treinamento e avaliação de modelos de predição de inadimplência.
Implementa múltiplos modelos (Logística, LightGBM, XGBoost e Ensemble) com foco
em métricas de negócio e interpretabilidade.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
import warnings
from datetime import datetime
import re
from typing import Optional

# Modelos
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
import xgboost as xgb

# Técnicas de balanceamento
from imblearn.over_sampling import SMOTE

# Métricas e validação
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, confusion_matrix,
    average_precision_score, roc_curve
)

# Configurar logger
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Suprimir avisos
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Obter caminho da raiz do projeto
def get_project_root():
    """Retorna o caminho para a raiz do projeto."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, os.pardir, os.pardir))
    return project_root

class FeatureEngineer:
    """
    Classe para engenharia de features específicas para modelos de inadimplência.
    """

    def __init__(self):
        self.feature_map = {}
        self.categorical_features = []
        self.generated_features = []
        self.selected_features = None

    def _handle_date_columns(self, df):
        """
        Converte colunas de data em features numéricas úteis.

        Args:
            df: DataFrame com dados originais

        Returns:
            DataFrame com colunas de data transformadas
        """
        # Cópia para evitar SettingWithCopyWarning
        result_df = df.copy()

        # Detectar possíveis colunas de data
        date_pattern = r'\d{4}-\d{2}-\d{2}'
        potential_date_cols = []

        for col in result_df.columns:
            # Verificar coluna pelo nome
            if any(date_term in col.lower() for date_term in ['data', 'date', 'dt_', 'nasc']):
                potential_date_cols.append(col)
            # Verificar se os primeiros valores se parecem com datas
            elif result_df[col].dtype == 'object':
                sample_values = result_df[col].dropna().head(10).astype(str)
                if any(bool(re.match(date_pattern, str(val))) for val in sample_values):
                    potential_date_cols.append(col)

        if potential_date_cols:
            logger.info(f"Detectadas {len(potential_date_cols)} possíveis colunas de data: {potential_date_cols}")

        date_features_created = 0

        # Converter colunas de data
        for col in potential_date_cols:
            try:
                # Converter para datetime
                result_df[col] = pd.to_datetime(result_df[col], errors='coerce')

                # Extrair features úteis
                col_prefix = f"{col}_"
                result_df[col_prefix + 'year'] = result_df[col].dt.year
                result_df[col_prefix + 'month'] = result_df[col].dt.month
                result_df[col_prefix + 'day'] = result_df[col].dt.day
                result_df[col_prefix + 'dayofweek'] = result_df[col].dt.dayofweek

                # Adicionar as novas colunas à lista de features geradas
                new_date_cols = [col_prefix + suffix for suffix in ['year', 'month', 'day', 'dayofweek']]
                self.generated_features.extend(new_date_cols)
                date_features_created += len(new_date_cols)

                # Calcular idade (se parece ser data de nascimento)
                if any(term in col.lower() for term in ['nasc', 'birth', 'nascimento']):
                    today = pd.Timestamp.now()
                    result_df[col_prefix + 'age'] = ((today - result_df[col]).dt.days / 365.25).astype('float32')
                    self.generated_features.append(col_prefix + 'age')
                    date_features_created += 1

                # Calcular tempo decorrido para outras datas
                else:
                    reference_date = pd.Timestamp.now()
                    result_df[col_prefix + 'days_since'] = ((reference_date - result_df[col]).dt.days).astype('float32')
                    self.generated_features.append(col_prefix + 'days_since')
                    date_features_created += 1

                # Remover coluna original de data
                result_df = result_df.drop(columns=[col])

                logger.info(f"Processada coluna de data '{col}', criadas {len(new_date_cols) + 1} features numéricas")

            except Exception as e:
                logger.warning(f"Erro ao processar coluna de data {col}: {str(e)}")

        if date_features_created > 0:
            logger.info(f"Total de features de data criadas: {date_features_created}")

        return result_df

    def fit_transform(self, df: pd.DataFrame, target_col: Optional[str] = None) -> pd.DataFrame:
        """
        Cria features adicionais relevantes para predição de inadimplência.

        Args:
            df: DataFrame com dados originais
            target_col: Nome da coluna alvo (opcional)

        Returns:
            DataFrame com novas features
        """
        logger.info("Realizando engenharia de features...")

        result_df = df.copy()

        # Armazenar nomes de colunas originais
        original_columns = result_df.columns.tolist()
        if target_col in original_columns:
            original_columns.remove(target_col)

        # 1. Tratar colunas de data (NOVA ETAPA)
        result_df = self._handle_date_columns(result_df)

        # 2. Identificar colunas categóricas e numéricas
        numeric_cols = result_df.select_dtypes(include=['number']).columns.tolist()
        if target_col in numeric_cols:
            numeric_cols.remove(target_col)

        categorical_cols = result_df.select_dtypes(include=['object', 'category']).columns.tolist()

        # 3. Criar features de razão (para variáveis financeiras)
        financial_keywords = ['renda', 'salario', 'valor', 'montante', 'saldo', 'limite',
                             'parcela', 'divida', 'emprestimo', 'pagamento', 'credito']

        financial_cols = []
        for col in numeric_cols:
            if any(keyword in col.lower() for keyword in financial_keywords):
                financial_cols.append(col)

        # Criar razões entre variáveis financeiras (quando fazem sentido)
        if len(financial_cols) >= 2:
            logger.info(f"Criando features de razão para {len(financial_cols)} colunas financeiras...")

            # Evitar zeros no denominador
            for col in financial_cols:
                if (result_df[col] == 0).any():
                    epsilon = 0.001  # Valor pequeno para evitar divisão por zero
                    result_df[col] = result_df[col].replace(0, epsilon)

            # Criar features de razão
            created_ratios = 0
            for i, col1 in enumerate(financial_cols):
                for col2 in financial_cols[i+1:]:
                    # Evitar criar razões sem sentido financeiro
                    # Exemplo: não faz sentido dividir renda por idade
                    if 'renda' in col1.lower() and 'divida' in col2.lower():
                        ratio_name = f'razao_{col2}_{col1}'
                        result_df[ratio_name] = result_df[col2] / result_df[col1]
                        self.generated_features.append(ratio_name)
                        created_ratios += 1

                    elif 'divida' in col1.lower() and 'renda' in col2.lower():
                        ratio_name = f'razao_{col1}_{col2}'
                        result_df[ratio_name] = result_df[col1] / result_df[col2]
                        self.generated_features.append(ratio_name)
                        created_ratios += 1

                    # Razões envolvendo limite de crédito
                    elif 'limite' in col1.lower() and ('gasto' in col2.lower() or 'uso' in col2.lower()):
                        ratio_name = f'utilizacao_{col2}_{col1}'
                        result_df[ratio_name] = result_df[col2] / result_df[col1]
                        self.generated_features.append(ratio_name)
                        created_ratios += 1

            logger.info(f"Criadas {created_ratios} features de razão financeira.")

        # 4. Features específicas para inadimplência

        # Indicador de pagamento mínimo
        payment_cols = [col for col in numeric_cols if 'pagamento' in col.lower()]
        min_payment_cols = [col for col in numeric_cols if 'min' in col.lower() and 'pagamento' in col.lower()]

        if payment_cols and min_payment_cols:
            for pay_col in payment_cols:
                for min_col in min_payment_cols:
                    col_name = f'pagamento_apenas_minimo_{pay_col}'
                    # Tolerância de 10% acima do mínimo
                    result_df[col_name] = ((result_df[pay_col] >= result_df[min_col]) &
                                          (result_df[pay_col] <= result_df[min_col] * 1.1)).astype(int)
                    self.generated_features.append(col_name)

        # Utilização de crédito (se existirem colunas relevantes)
        if any('limite' in col.lower() for col in numeric_cols) and any('uso' in col.lower() or 'gasto' in col.lower() for col in numeric_cols):
            limite_cols = [col for col in numeric_cols if 'limite' in col.lower()]
            gasto_cols = [col for col in numeric_cols if 'uso' in col.lower() or 'gasto' in col.lower()]

            for limite_col in limite_cols:
                for gasto_col in gasto_cols:
                    col_name = f'utilizacao_credito_{gasto_col}_{limite_col}'
                    result_df[col_name] = result_df[gasto_col] / result_df[limite_col].replace(0, 0.001)
                    # Limitar a valores razoáveis (0-100)
                    result_df[col_name] = result_df[col_name].clip(0, 100)
                    self.generated_features.append(col_name)

        # 5. Codificar variáveis categóricas (se necessário)
        if categorical_cols:
            self.categorical_features = categorical_cols

            # Armazenar mapeamentos para uso posterior
            for col in categorical_cols:
                # Usar apenas encoding ordinal para não aumentar demais a dimensionalidade
                categories = result_df[col].unique()
                mapping = {cat: i for i, cat in enumerate(categories)}
                self.feature_map[col] = mapping

                # Aplicar encoding
                result_df[f'{col}_encoded'] = result_df[col].map(mapping)
                self.generated_features.append(f'{col}_encoded')

        # 6. Criar indicadores para valores extremos (outliers)
        for col in numeric_cols:
            # Calcular percentis
            p1 = result_df[col].quantile(0.01)
            p99 = result_df[col].quantile(0.99)

            # Criar indicadores
            col_name_low = f'{col}_muito_baixo'
            result_df[col_name_low] = (result_df[col] <= p1).astype(int)
            self.generated_features.append(col_name_low)

            col_name_high = f'{col}_muito_alto'
            result_df[col_name_high] = (result_df[col] >= p99).astype(int)
            self.generated_features.append(col_name_high)

        # 7. Seleção de features (opcional)
        # Se tivermos o target, podemos selecionar features com base na correlação
        if target_col and target_col in result_df.columns:
            # Calcular correlação com o target
            corrs = []
            for col in result_df.columns:
                if col != target_col and pd.api.types.is_numeric_dtype(result_df[col]):
                    corr = result_df[col].corr(result_df[target_col])
                    corrs.append((col, abs(corr)))

            # Selecionar top features
            corrs.sort(key=lambda x: x[1], reverse=True)
            top_n = min(50, len(corrs))  # Limitar a 50 features
            self.selected_features = [col for col, _ in corrs[:top_n]]

            logger.info(f"Selecionadas {len(self.selected_features)} features baseadas em correlação.")

        logger.info(f"Engenharia de features concluída. Criadas {len(self.generated_features)} novas features.")
        return result_df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aplica as transformações em novos dados.

        Args:
            df: DataFrame para transformar

        Returns:
            DataFrame transformado
        """
        if not self.feature_map and not self.generated_features:
            raise ValueError("O FeatureEngineer não foi ajustado. Execute fit_transform() primeiro.")

        result_df = df.copy()

        # Tratar colunas de data
        result_df = self._handle_date_columns(result_df)

        # Aplicar mapeamentos categóricos
        for col, mapping in self.feature_map.items():
            if col in result_df.columns:
                result_df[f'{col}_encoded'] = result_df[col].map(mapping)
                # Tratar valores não vistos no treinamento
                result_df[f'{col}_encoded'] = result_df[f'{col}_encoded'].fillna(-1)

        # Recriar as features que temos que calcular
        # (Idealmente, este código deveria ser refatorado para evitar duplicação da lógica do fit_transform)

        # Aplicar seleção de features (se disponível)
        if self.selected_features:
            available_features = [col for col in self.selected_features if col in result_df.columns]
            return result_df[available_features]

        return result_df

    def save(self, filepath):
        """Salva o engenheiro de features"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)
        logger.info(f"FeatureEngineer salvo em: {filepath}")

    @classmethod
    def load(cls, filepath):
        """Carrega um engenheiro de features salvo"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Arquivo não encontrado: {filepath}")
        return joblib.load(filepath)

class ModelTrainer:
    """
    Classe para treinar, avaliar e salvar modelos de predição de inadimplência.
    """

    def __init__(self, model_dir=None, eval_dir=None, default_threshold=0.5, cost_fn_ratio=5.0):
        """
        Inicializa o treinador de modelos.

        Args:
            model_dir: Diretório para salvar modelos treinados
            eval_dir: Diretório para salvar avaliações
            default_threshold: Limiar padrão para classificação (0.5)
            cost_fn_ratio: Custo relativo de falsos negativos vs. falsos positivos
        """
        self.models = {}
        self.feature_importances = {}
        self.evaluation_results = {}
        self.thresholds = {}
        self.default_threshold = default_threshold
        self.cost_fn_ratio = cost_fn_ratio
        self.best_model_name = None

        # Configurar diretórios para salvar resultados
        project_root = get_project_root()
        self.model_dir = model_dir or os.path.join(project_root, 'models', 'trained_models')
        self.eval_dir = eval_dir or os.path.join(project_root, 'reports', 'model_evaluation')

        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.eval_dir, exist_ok=True)

        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def train_logistic_regression(self, X_train, y_train, name="LogisticRegression", **kwargs):
        """
        Treina um modelo de regressão logística.

        Args:
            X_train: Features de treinamento
            y_train: Target de treinamento
            name: Nome do modelo (para referência)
            **kwargs: Parâmetros adicionais para LogisticRegression

        Returns:
            self
        """
        logger.info(f"Treinando modelo {name}...")

        # Verificar se há colunas não numéricas
        non_numeric_cols = self._check_non_numeric_cols(X_train)
        if non_numeric_cols:
            logger.warning(f"Detectadas {len(non_numeric_cols)} colunas não numéricas. Removendo-as para o treinamento.")
            X_train = X_train.drop(columns=non_numeric_cols)

        # Parâmetros padrão
        params = {
            'penalty': 'l2',
            'C': 1.0,
            'class_weight': 'balanced',
            'random_state': 42,
            'max_iter': 1000,
            'solver': 'liblinear'
        }

        # Atualizar com parâmetros fornecidos
        params.update(kwargs)

        # Criar e treinar modelo
        model = LogisticRegression(**params)
        model.fit(X_train, y_train)

        # Armazenar modelo e importância de features
        self.models[name] = model

        # Calcular importância de features (coeficientes absolutos)
        importance = np.abs(model.coef_[0])
        feature_importance = pd.DataFrame({
            'Feature': X_train.columns,
            'Importance': importance
        })
        self.feature_importances[name] = feature_importance.sort_values('Importance', ascending=False)

        logger.info(f"Modelo {name} treinado com sucesso.")
        return self

    def _check_non_numeric_cols(self, df):
        """
        Verifica e identifica colunas não numéricas em um DataFrame.

        Args:
            df: DataFrame a verificar

        Returns:
            Lista de colunas não numéricas
        """
        non_numeric_cols = []
        for col in df.columns:
            # Verificar se a coluna é não numérica
            if not pd.api.types.is_numeric_dtype(df[col]):
                non_numeric_cols.append(col)
            # Ou se contém valores não convertíveis para float
            elif pd.api.types.is_object_dtype(df[col]):
                try:
                    df[col].astype('float')
                except (ValueError, TypeError):
                    non_numeric_cols.append(col)

        return non_numeric_cols

    def train_lightgbm(self, X_train, y_train, X_val=None, y_val=None, name="LightGBM", **kwargs):
        """
        Treina um modelo LightGBM.

        Args:
            X_train: Features de treinamento
            y_train: Target de treinamento
            X_val: Features de validação (opcional)
            y_val: Target de validação (opcional)
            name: Nome do modelo
            **kwargs: Parâmetros adicionais para LGBMClassifier

        Returns:
            self
        """
        logger.info(f"Treinando modelo {name}...")

        # Verificar se há colunas não numéricas
        non_numeric_cols = self._check_non_numeric_cols(X_train)
        if non_numeric_cols:
            logger.warning(f"Detectadas {len(non_numeric_cols)} colunas não numéricas. Removendo-as para o treinamento.")
            X_train = X_train.drop(columns=non_numeric_cols)
            if X_val is not None:
                X_val = X_val.drop(columns=[col for col in non_numeric_cols if col in X_val.columns])

        # Parâmetros padrão
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'learning_rate': 0.05,
            'n_estimators': 200,
            'max_depth': 7,
            'num_leaves': 31,
            'class_weight': 'balanced',
            'random_state': 42,
            'verbosity': -1,
            'importance_type': 'gain'
        }

        # Atualizar com parâmetros fornecidos
        params.update(kwargs)

        # Criar modelo
        model = lgb.LGBMClassifier(**params)

        # Configurar validação (se fornecida)
        fit_params = {}
        if X_val is not None and y_val is not None:
            fit_params = {
                'eval_set': [(X_val, y_val)],
                'eval_metric': 'auc',
                'early_stopping_rounds': 50,
                'verbose': 100
            }

        # Treinar modelo
        model.fit(X_train, y_train, **fit_params)

        # Armazenar modelo e importância de features
        self.models[name] = model

        # Calcular importância de features
        importance = model.feature_importances_
        feature_importance = pd.DataFrame({
            'Feature': X_train.columns,
            'Importance': importance
        })
        self.feature_importances[name] = feature_importance.sort_values('Importance', ascending=False)

        logger.info(f"Modelo {name} treinado com sucesso.")
        return self

    def train_xgboost(self, X_train, y_train, X_val=None, y_val=None, name="XGBoost", **kwargs):
        """
        Treina um modelo XGBoost.

        Args:
            X_train: Features de treinamento
            y_train: Target de treinamento
            X_val: Features de validação (opcional)
            y_val: Target de validação (opcional)
            name: Nome do modelo
            **kwargs: Parâmetros adicionais para XGBClassifier

        Returns:
            self
        """
        logger.info(f"Treinando modelo {name}...")

        # Verificar se há colunas não numéricas
        non_numeric_cols = self._check_non_numeric_cols(X_train)
        if non_numeric_cols:
            logger.warning(f"Detectadas {len(non_numeric_cols)} colunas não numéricas. Removendo-as para o treinamento.")
            X_train = X_train.drop(columns=non_numeric_cols)
            if X_val is not None:
                X_val = X_val.drop(columns=[col for col in non_numeric_cols if col in X_val.columns])

        # Parâmetros padrão
        params = {
            'objective': 'binary:logistic',
            'learning_rate': 0.05,
            'n_estimators': 200,
            'max_depth': 6,
            'min_child_weight': 1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'scale_pos_weight': 2,  # Para balancear classes
            'random_state': 42,
            'verbosity': 0
        }

        # Atualizar com parâmetros fornecidos
        params.update(kwargs)

        # Criar modelo
        model = xgb.XGBClassifier(**params)

        # Configurar validação (se fornecida)
        fit_params = {}
        if X_val is not None and y_val is not None:
            fit_params = {
                'eval_set': [(X_val, y_val)],
                'eval_metric': 'auc',
                'early_stopping_rounds': 50,
                'verbose': 100
            }

        # Treinar modelo
        model.fit(X_train, y_train, **fit_params)

        # Armazenar modelo e importância de features
        self.models[name] = model

        # Calcular importância de features
        importance = model.feature_importances_
        feature_importance = pd.DataFrame({
            'Feature': X_train.columns,
            'Importance': importance
        })
        self.feature_importances[name] = feature_importance.sort_values('Importance', ascending=False)

        logger.info(f"Modelo {name} treinado com sucesso.")
        return self

    def train_random_forest(self, X_train, y_train, name="RandomForest", **kwargs):
        """
        Treina um modelo Random Forest.

        Args:
            X_train: Features de treinamento
            y_train: Target de treinamento
            name: Nome do modelo
            **kwargs: Parâmetros adicionais para RandomForestClassifier

        Returns:
            self
        """
        logger.info(f"Treinando modelo {name}...")

        # Verificar se há colunas não numéricas
        non_numeric_cols = self._check_non_numeric_cols(X_train)
        if non_numeric_cols:
            logger.warning(f"Detectadas {len(non_numeric_cols)} colunas não numéricas. Removendo-as para o treinamento.")
            X_train = X_train.drop(columns=non_numeric_cols)

        # Parâmetros padrão
        params = {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'class_weight': 'balanced',
            'random_state': 42,
            'n_jobs': -1,
            'verbose': 0
        }

        # Atualizar com parâmetros fornecidos
        params.update(kwargs)

        # Criar e treinar modelo
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)

        # Armazenar modelo e importância de features
        self.models[name] = model

        # Calcular importância de features
        importance = model.feature_importances_
        feature_importance = pd.DataFrame({
            'Feature': X_train.columns,
            'Importance': importance
        })
        self.feature_importances[name] = feature_importance.sort_values('Importance', ascending=False)

        logger.info(f"Modelo {name} treinado com sucesso.")
        return self

    def train_ensemble(self, base_models=None):
        """
        Cria um modelo ensemble a partir dos modelos já treinados.

        Args:
            base_models: Lista de nomes dos modelos a incluir no ensemble (se None, usa todos)

        Returns:
            self
        """
        if not self.models:
            raise ValueError("Nenhum modelo foi treinado ainda. Treine alguns modelos base primeiro.")

        if base_models is None:
            base_models = list(self.models.keys())

        # Verificar se todos os modelos base existem
        for name in base_models:
            if name not in self.models:
                raise ValueError(f"Modelo base '{name}' não encontrado.")

        logger.info(f"Criando modelo ensemble com {len(base_models)} modelos base: {base_models}")

        # Criar classe do modelo ensemble
        class EnsembleModel:
            def __init__(self, models, model_weights=None):
                self.models = models
                # Se não houver pesos, usar pesos iguais
                if model_weights is None:
                    self.weights = [1/len(models)] * len(models)
                else:
                    # Normalizar pesos
                    total_weight = sum(model_weights)
                    self.weights = [w/total_weight for w in model_weights]

                self.threshold = 0.5

            def predict_proba(self, X):
                """Combina probabilidades de todos os modelos base"""
                probas = np.zeros((X.shape[0], 2))

                for i, (name, model) in enumerate(self.models.items()):
                    model_proba = model.predict_proba(X)
                    probas += self.weights[i] * model_proba

                # Normalizar (garantir que soma = 1)
                row_sums = probas.sum(axis=1)
                probas = probas / row_sums[:, np.newaxis]

                return probas

            def predict(self, X):
                """Faz predições com base no threshold definido"""
                probas = self.predict_proba(X)
                return (probas[:, 1] >= self.threshold).astype(int)

        # Coletar modelos base
        base_model_dict = {name: self.models[name] for name in base_models}

        # Definir pesos com base nas performances (se já avaliados)
        weights = None
        if self.evaluation_results:
            # Usar AUC como peso
            weights = []
            for name in base_models:
                if name in self.evaluation_results and 'auc' in self.evaluation_results[name]:
                    weights.append(self.evaluation_results[name]['auc'])
                else:
                    weights.append(1.0)  # Peso padrão

        # Criar e armazenar o modelo ensemble
        ensemble = EnsembleModel(base_model_dict, weights)
        self.models['Ensemble'] = ensemble

        logger.info("Modelo ensemble criado com sucesso.")
        return self

    def optimize_threshold(self, model_name, X_val, y_val, optimize_for_business=True):
        """
        Otimiza o threshold de classificação para um modelo específico.

        Args:
            model_name: Nome do modelo
            X_val: Features de validação
            y_val: Target de validação
            optimize_for_business: Se True, otimiza para custo de negócio, senão para F1

        Returns:
            Threshold otimizado
        """
        if model_name not in self.models:
            raise ValueError(f"Modelo '{model_name}' não encontrado.")

        model = self.models[model_name]
        logger.info(f"Otimizando threshold para modelo '{model_name}'...")

        # Verificar se há colunas não numéricas em X_val
        non_numeric_cols = self._check_non_numeric_cols(X_val)
        if non_numeric_cols:
            logger.warning(f"Detectadas {len(non_numeric_cols)} colunas não numéricas em X_val. Removendo-as.")
            X_val = X_val.drop(columns=non_numeric_cols)

        # Obter probabilidades
        y_proba = model.predict_proba(X_val)[:, 1]

        if optimize_for_business:
            # Otimizar para custo de negócio
            thresholds = np.linspace(0.01, 0.99, 99)
            best_threshold = 0.5
            min_cost = float('inf')

            for thresh in thresholds:
                y_pred = (y_proba >= thresh).astype(int)

                # Calcular matriz de confusão
                tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()

                # Calcular custo (FP + cost_ratio * FN)
                cost = fp + self.cost_fn_ratio * fn

                if cost < min_cost:
                    min_cost = cost
                    best_threshold = thresh

            logger.info(f"Threshold otimizado para custo de negócio: {best_threshold:.4f}")
            logger.info(f"Custo relativo: {min_cost}")

        else:
            # Otimizar para F1-score
            precision, recall, thresholds = precision_recall_curve(y_val, y_proba)

            # Adicionar threshold = 1.0 (ausente por padrão)
            thresholds = np.append(thresholds, 1.0)

            # Calcular F1 para cada threshold
            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
            best_idx = np.argmax(f1_scores)
            best_threshold = thresholds[best_idx]

            logger.info(f"Threshold otimizado para F1-score: {best_threshold:.4f}")
            logger.info(f"F1-score máximo: {f1_scores[best_idx]:.4f}")

        # Armazenar threshold otimizado
        self.thresholds[model_name] = best_threshold

        # Se o modelo tiver atributo "threshold", atualizar
        if hasattr(model, 'threshold'):
            model.threshold = best_threshold

        return best_threshold

    def evaluate_model(self, model_name, X_test, y_test, threshold=None):
        """
        Avalia um modelo específico no conjunto de teste.

        Args:
            model_name: Nome do modelo
            X_test: Features de teste
            y_test: Target de teste
            threshold: Threshold para classificação (se None, usa o otimizado ou padrão)

        Returns:
            Dicionário com métricas de avaliação
        """
        if model_name not in self.models:
            raise ValueError(f"Modelo '{model_name}' não encontrado.")

        model = self.models[model_name]
        logger.info(f"Avaliando modelo '{model_name}' no conjunto de teste...")

        # Verificar se há colunas não numéricas em X_test
        non_numeric_cols = self._check_non_numeric_cols(X_test)
        if non_numeric_cols:
            logger.warning(f"Detectadas {len(non_numeric_cols)} colunas não numéricas em X_test. Removendo-as.")
            X_test = X_test.drop(columns=non_numeric_cols)

        # Determinar threshold
        if threshold is None:
            threshold = self.thresholds.get(model_name, self.default_threshold)

        # Obter probabilidades e predições
        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= threshold).astype(int)

        # Calcular métricas
        acc = (y_pred == y_test).mean()
        auc = roc_auc_score(y_test, y_proba)
        avg_precision = average_precision_score(y_test, y_proba)

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        # Métricas de negócio
        aprovacao_rate = (tp + fp) / (tn + fp + fn + tp)
        inadimplencia_portfolio = fn / (tp + fn) if (tp + fn) > 0 else 0

        # Custo de negócio (FP + cost_ratio * FN)
        business_cost = fp + self.cost_fn_ratio * fn
        normalized_cost = business_cost / len(y_test)

        # Armazenar resultados
        results = {
            'model_name': model_name,
            'threshold': threshold,
            'accuracy': acc,
            'precision': precision,
            'recall': recall,
            'specificity': specificity,
            'f1_score': f1,
            'auc': auc,
            'avg_precision': avg_precision,
            'tn': tn,
            'fp': fp,
            'fn': fn,
            'tp': tp,
            'aprovacao_rate': aprovacao_rate,
            'inadimplencia_portfolio': inadimplencia_portfolio,
            'business_cost': business_cost,
            'normalized_cost': normalized_cost,
            'y_proba': y_proba,
            'y_pred': y_pred
        }

        self.evaluation_results[model_name] = results

        # Imprimir relatório
        logger.info("\nRelatório de Avaliação:")
        logger.info(f"Modelo: {model_name}")
        logger.info(f"Threshold: {threshold:.4f}")
        logger.info(f"Acurácia: {acc:.4f}")
        logger.info(f"AUC-ROC: {auc:.4f}")
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"Recall/Sensitivity: {recall:.4f}")
        logger.info(f"Specificity: {specificity:.4f}")
        logger.info(f"F1-score: {f1:.4f}")
        logger.info("\nMatriz de Confusão:")
        logger.info(f"TN: {tn}, FP: {fp}")
        logger.info(f"FN: {fn}, TP: {tp}")
        logger.info("\nMétricas de Negócio:")
        logger.info(f"Taxa de Aprovação: {aprovacao_rate:.2%}")
        logger.info(f"Taxa de Inadimplência no Portfolio: {inadimplencia_portfolio:.2%}")
        logger.info(f"Custo de Negócio Normalizado: {normalized_cost:.4f}")

        # Gerar visualizações
        self._plot_evaluation(model_name, y_test, y_proba, y_pred)

        return results

    def evaluate_all_models(self, X_test, y_test):
        """
        Avalia todos os modelos treinados.

        Args:
            X_test: Features de teste
            y_test: Target de teste

        Returns:
            DataFrame com métricas de todos os modelos
        """
        if not self.models:
            raise ValueError("Nenhum modelo foi treinado ainda.")

        results = []

        for name in self.models:
            self.evaluate_model(name, X_test, y_test)

            # Extrair métricas relevantes
            metrics = {
                'Modelo': name,
                'AUC': self.evaluation_results[name]['auc'],
                'F1-Score': self.evaluation_results[name]['f1_score'],
                'Precision': self.evaluation_results[name]['precision'],
                'Recall': self.evaluation_results[name]['recall'],
                'Taxa de Aprovação': self.evaluation_results[name]['aprovacao_rate'],
                'Taxa de Inadimplência': self.evaluation_results[name]['inadimplencia_portfolio'],
                'Custo de Negócio': self.evaluation_results[name]['normalized_cost']
            }

            results.append(metrics)

        # Criar DataFrame com resultados
        results_df = pd.DataFrame(results)

        # Encontrar melhor modelo (menor custo de negócio)
        best_model_idx = results_df['Custo de Negócio'].idxmin()
        self.best_model_name = results_df.loc[best_model_idx, 'Modelo']

        logger.info(f"\n>>> Melhor modelo: {self.best_model_name}")
        logger.info(f"Custo de Negócio: {results_df.loc[best_model_idx, 'Custo de Negócio']:.4f}")

        # Salvar resultados
        results_file = os.path.join(self.eval_dir, f"model_comparison_{self.timestamp}.csv")
        results_df.to_csv(results_file, index=False)
        logger.info(f"Resultados comparativos salvos em: {results_file}")

        # Criar visualização comparativa
        self._plot_model_comparison(results_df)

        return results_df

    def _plot_evaluation(self, model_name, y_test, y_proba, y_pred):
        """
        Gera visualizações para avaliação do modelo.

        Args:
            model_name: Nome do modelo
            y_test: Target de teste
            y_proba: Probabilidades preditas
            y_pred: Classes preditas
        """
        # Criar diretório para gráficos
        plots_dir = os.path.join(self.eval_dir, 'plots', model_name)
        os.makedirs(plots_dir, exist_ok=True)

        # 1. Curva ROC
        plt.figure(figsize=(10, 8))
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        plt.plot(fpr, tpr, label=f'AUC = {roc_auc_score(y_test, y_proba):.4f}')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('Taxa de Falsos Positivos')
        plt.ylabel('Taxa de Verdadeiros Positivos')
        plt.title(f'Curva ROC - {model_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(plots_dir, f'roc_curve_{self.timestamp}.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # 2. Matriz de Confusão
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.xlabel('Predito')
        plt.ylabel('Real')
        plt.title(f'Matriz de Confusão - {model_name}')
        plt.savefig(os.path.join(plots_dir, f'confusion_matrix_{self.timestamp}.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # 3. Distribuição de Probabilidades
        plt.figure(figsize=(12, 6))
        sns.histplot(pd.DataFrame({
            'Probabilidade': y_proba,
            'Real': y_test
        }), x='Probabilidade', hue='Real', bins=50, kde=True)
        threshold = self.thresholds.get(model_name, self.default_threshold)
        plt.axvline(x=threshold, color='red', linestyle='--', label=f'Threshold = {threshold:.4f}')
        plt.title(f'Distribuição de Probabilidades por Classe - {model_name}')
        plt.legend()
        plt.savefig(os.path.join(plots_dir, f'prob_distribution_{self.timestamp}.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # 4. Precision-Recall Curve
        plt.figure(figsize=(10, 8))
        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        plt.plot(recall, precision)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Curva Precision-Recall - {model_name}')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(plots_dir, f'precision_recall_{self.timestamp}.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # 5. Feature Importance (se disponível)
        if model_name in self.feature_importances:
            importance_df = self.feature_importances[model_name]
            top_n = min(20, len(importance_df))

            plt.figure(figsize=(12, 8))
            sns.barplot(x='Importance', y='Feature', data=importance_df.head(top_n))
            plt.title(f'Top {top_n} Features Mais Importantes - {model_name}')
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f'feature_importance_{self.timestamp}.png'), dpi=300, bbox_inches='tight')
            plt.close()

    def _plot_model_comparison(self, results_df):
        """
        Gera visualizações comparativas entre modelos.

        Args:
            results_df: DataFrame com resultados dos modelos
        """
        # 1. Comparação de métricas
        metrics = ['AUC', 'F1-Score', 'Precision', 'Recall']
        plt.figure(figsize=(12, 8))
        results_df.set_index('Modelo')[metrics].plot(kind='bar')
        plt.title('Comparação de Métricas de Classificação')
        plt.ylabel('Valor')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.eval_dir, f'metrics_comparison_{self.timestamp}.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # 2. Comparação de custo de negócio
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Modelo', y='Custo de Negócio', data=results_df)
        plt.title('Comparação de Custo de Negócio')
        plt.xticks(rotation=45)
        plt.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.eval_dir, f'business_cost_comparison_{self.timestamp}.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # 3. Comparação de taxas de aprovação e inadimplência
        plt.figure(figsize=(12, 6))
        results_df.set_index('Modelo')[['Taxa de Aprovação', 'Taxa de Inadimplência']].plot(kind='bar')
        plt.title('Taxas de Aprovação vs. Inadimplência')
        plt.ylabel('Taxa (%)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.eval_dir, f'approval_vs_default_{self.timestamp}.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def save_models(self):
        """
        Salva todos os modelos treinados.

        Returns:
            Dicionário com caminhos dos modelos salvos
        """
        if not self.models:
            raise ValueError("Nenhum modelo foi treinado ainda.")

        saved_paths = {}

        for name, model in self.models.items():
            # Criar caminho do arquivo
            file_path = os.path.join(self.model_dir, f"{name}_{self.timestamp}.joblib")

            # Salvar modelo
            joblib.dump(model, file_path)
            saved_paths[name] = file_path

            logger.info(f"Modelo '{name}' salvo em: {file_path}")

        # Salvar métricas e thresholds para referência
        metadata = {
            'timestamp': self.timestamp,
            'models': list(self.models.keys()),
            'thresholds': self.thresholds,
            'best_model': self.best_model_name,
            'cost_fn_ratio': self.cost_fn_ratio
        }

        metadata_file = os.path.join(self.model_dir, f"model_metadata_{self.timestamp}.json")
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=4)

        logger.info(f"Metadados dos modelos salvos em: {metadata_file}")

        # Salvar o melhor modelo separadamente
        if self.best_model_name:
            best_model_path = os.path.join(self.model_dir, f"best_model_{self.timestamp}.joblib")
            joblib.dump(self.models[self.best_model_name], best_model_path)
            logger.info(f"Melhor modelo ({self.best_model_name}) salvo separadamente em: {best_model_path}")

        return saved_paths

    @classmethod
    def load_model(cls, model_path):
        """
        Carrega um modelo salvo.

        Args:
            model_path: Caminho do modelo

        Returns:
            Modelo carregado
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Arquivo não encontrado: {model_path}")

        try:
            model = joblib.load(model_path)
            logger.info(f"Modelo carregado de: {model_path}")
            return model
        except Exception as e:
            logger.error(f"Erro ao carregar modelo: {str(e)}")
            raise

def validate_numeric_data(df):
    """
    Verifica se todas as colunas são numéricas e gera um relatório dos problemas.

    Args:
        df: DataFrame a verificar

    Returns:
        Tuple (is_valid, non_numeric_cols, problematic_values)
    """
    non_numeric_cols = []
    problematic_values = {}

    for col in df.columns:
        # Verificar se a coluna é não numérica
        if not pd.api.types.is_numeric_dtype(df[col]):
            non_numeric_cols.append(col)
            # Coletar amostra de valores problemáticos
            problematic_values[col] = df[col].head(3).tolist()
        # Ou verificar se contém valores não convertíveis para float
        elif pd.api.types.is_object_dtype(df[col]):
            try:
                df[col].astype('float')
            except (ValueError, TypeError):
                non_numeric_cols.append(col)
                problematic_values[col] = df[col].head(3).tolist()

    is_valid = len(non_numeric_cols) == 0

    if not is_valid:
        logger.warning(f"Detectadas {len(non_numeric_cols)} colunas não numéricas: {non_numeric_cols}")
        logger.warning("Exemplos de valores problemáticos:")
        for col, values in problematic_values.items():
            logger.warning(f"  {col}: {values}")

    return is_valid, non_numeric_cols, problematic_values

def load_and_prepare_data(processed_dir, timestamp=None, target_col='Inadimplente', handle_imbalance=True):
    """
    Carrega e prepara dados para modelagem.

    Args:
        processed_dir: Diretório com dados processados
        timestamp: Timestamp específico para carregar (se None, usa o mais recente)
        target_col: Nome da coluna alvo
        handle_imbalance: Se True, aplica técnicas de balanceamento de classes

    Returns:
        Conjuntos de dados preparados para treinamento
    """
    logger.info("Carregando e preparando dados para modelagem...")

    # Obter caminho absoluto do diretório
    project_root = get_project_root()
    if not os.path.isabs(processed_dir):
        processed_dir = os.path.join(project_root, processed_dir)

    # Se timestamp não for fornecido, pegar o mais recente
    if timestamp is None:
        # Procurar por arquivos de metadata para identificar timestamps disponíveis
        meta_files = [f for f in os.listdir(processed_dir) if f.startswith('metadata_') and f.endswith('.json')]
        if not meta_files:
            # Tentar outros formatos
            meta_files = [f for f in os.listdir(processed_dir) if f.startswith('metadata_')]

        if not meta_files:
            raise FileNotFoundError(f"Nenhum arquivo de metadados encontrado em {processed_dir}")

        # Ordenar por timestamp (assumindo formato metadata_YYYYMMDD_HHMMSS.*)
        meta_files.sort(reverse=True)
        timestamp = meta_files[0].replace('metadata_', '').split('.')[0]

    logger.info(f"Usando timestamp: {timestamp}")

    # Carregar arquivos de dados
    train_file = os.path.join(processed_dir, f"train_{timestamp}.csv")
    val_file = os.path.join(processed_dir, f"val_{timestamp}.csv")
    test_file = os.path.join(processed_dir, f"test_{timestamp}.csv")

    # Verificar se arquivos existem
    if not os.path.exists(train_file):
        # Tentar encontrar em diretórios alternativos
        alternatives = [
            os.path.join(project_root, 'data/processed'),
            os.path.join(project_root, 'data/interim'),
            os.path.join(project_root, 'data')
        ]

        for alt_dir in alternatives:
            alt_file = os.path.join(alt_dir, f"train_{timestamp}.csv")
            if os.path.exists(alt_file):
                logger.info(f"Arquivos encontrados em diretório alternativo: {alt_dir}")
                train_file = os.path.join(alt_dir, f"train_{timestamp}.csv")
                val_file = os.path.join(alt_dir, f"val_{timestamp}.csv")
                test_file = os.path.join(alt_dir, f"test_{timestamp}.csv")
                break
        else:
            raise FileNotFoundError(f"Arquivo de treino não encontrado: {train_file}")

    # Carregar dados
    logger.info(f"Carregando dados de: {train_file}")
    df_train = pd.read_csv(train_file)

    logger.info(f"Carregando dados de: {val_file}")
    df_val = pd.read_csv(val_file)

    logger.info(f"Carregando dados de: {test_file}")
    df_test = pd.read_csv(test_file)

    # Verificar se target_col existe
    if target_col not in df_train.columns:
        # Tentar identificar coluna alvo
        possible_targets = ['inadimplente', 'target', 'default', 'risco_inadimplencia', 'Risco_Inadimplencia']
        for col in df_train.columns:
            if col.lower() in possible_targets:
                target_col = col
                logger.info(f"Coluna alvo identificada: {target_col}")
                break
        else:
            raise ValueError(f"Coluna alvo '{target_col}' não encontrada e não foi possível identificar automaticamente.")

    # Separar features e target
    X_train = df_train.drop(columns=[target_col])
    y_train = df_train[target_col]

    X_val = df_val.drop(columns=[target_col])
    y_val = df_val[target_col]

    X_test = df_test.drop(columns=[target_col])
    y_test = df_test[target_col]

    # Remover colunas que possam causar vazamento de dados
    columns_to_exclude = [
        'ID_Cliente', 'Nome', 'CPF', 'Email', 'Telefone', 'Data_Referencia',
        'Nome_Completo', 'RG', 'CEP', 'Endereco'
    ]

    for col in columns_to_exclude:
        if col in X_train.columns:
            logger.info(f"Removendo coluna potencial de vazamento: {col}")
            X_train = X_train.drop(columns=[col])
            X_val = X_val.drop(columns=[col])
            X_test = X_test.drop(columns=[col])

    # Verificar tipos de dados
    # Converter colunas categóricas para 'category'
    categorical_cols = []
    for col in X_train.columns:
        if X_train[col].dtype == 'object':
            X_train[col] = X_train[col].astype('category')
            X_val[col] = X_val[col].astype('category')
            X_test[col] = X_test[col].astype('category')
            categorical_cols.append(col)

    # Resumo dos dados
    logger.info("\nResumo dos dados:")
    logger.info(f"Conjunto de Treino: {X_train.shape[0]} exemplos, {X_train.shape[1]} features")
    logger.info(f"Conjunto de Validação: {X_val.shape[0]} exemplos, {X_val.shape[1]} features")
    logger.info(f"Conjunto de Teste: {X_test.shape[0]} exemplos, {X_test.shape[1]} features")
    logger.info(f"Colunas categóricas: {len(categorical_cols)}")

    # Verificar desbalanceamento
    train_pos_rate = y_train.mean()
    val_pos_rate = y_val.mean()
    test_pos_rate = y_test.mean()

    logger.info("\nDistribuição da variável alvo:")
    logger.info(f"- Treino: {train_pos_rate:.2%} positivos")
    logger.info(f"- Validação: {val_pos_rate:.2%} positivos")
    logger.info(f"- Teste: {test_pos_rate:.2%} positivos")

    # Tratar desbalanceamento (se necessário)
    if handle_imbalance and train_pos_rate < 0.3:  # Classe minoritária < 30%
        logger.info("\nAplicando técnica de balanceamento SMOTE...")

        # Criar SMOTE
        smote = SMOTE(random_state=42)

        # Aplicar SMOTE (apenas no conjunto de treino)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

        # Atualizar conjunto de treino
        X_train = X_train_resampled
        y_train = y_train_resampled

        logger.info(f"Conjunto de treino após SMOTE: {X_train.shape[0]} exemplos")
        logger.info(f"Nova distribuição no treino: {y_train.mean():.2%} positivos")

    # Aplicar engenharia de features
    logger.info("\nAplicando engenharia de features...")
    feature_engineer = FeatureEngineer()

    # Combinar dados para engenharia
    X_combined = pd.concat([X_train, X_val, X_test])
    y_combined = pd.concat([y_train, y_val, y_test])

    # Aplicar transformações
    X_combined_transformed = feature_engineer.fit_transform(X_combined, target_col)

    # Dividir novamente
    X_train_transformed = X_combined_transformed.iloc[:len(X_train)]
    X_val_transformed = X_combined_transformed.iloc[len(X_train):len(X_train)+len(X_val)]
    X_test_transformed = X_combined_transformed.iloc[len(X_train)+len(X_val):]

    logger.info(f"Dimensões após engenharia de features:")
    logger.info(f"- X_train: {X_train_transformed.shape}")
    logger.info(f"- X_val: {X_val_transformed.shape}")
    logger.info(f"- X_test: {X_test_transformed.shape}")

    # Salvar feature engineer para uso futuro
    feature_engineer_dir = os.path.join(project_root, 'models', 'preprocessing')
    os.makedirs(feature_engineer_dir, exist_ok=True)

    feature_engineer_path = os.path.join(feature_engineer_dir, f"feature_engineer_{timestamp}.joblib")
    feature_engineer.save(feature_engineer_path)

    return {
        'X_train': X_train_transformed,
        'y_train': y_train,
        'X_val': X_val_transformed,
        'y_val': y_val,
        'X_test': X_test_transformed,
        'y_test': y_test,
        'feature_engineer': feature_engineer,
        'timestamp': timestamp
    }

def train_all_models(data_dict):
    """
    Treina e avalia todos os modelos recomendados.

    Args:
        data_dict: Dicionário com dados preparados

    Returns:
        Trainer com modelos treinados
    """
    # Extrair dados
    X_train = data_dict['X_train']
    y_train = data_dict['y_train']
    X_val = data_dict['X_val']
    y_val = data_dict['y_val']
    X_test = data_dict['X_test']
    y_test = data_dict['y_test']
    timestamp = data_dict.get('timestamp', datetime.now().strftime("%Y%m%d_%H%M%S"))

    # Criar trainer
    trainer = ModelTrainer()

    # 1. Treinar Regressão Logística
    logger.info("\n" + "="*80)
    logger.info("Treinando Regressão Logística...")
    logger.info("="*80)

    # Verificar e remover colunas não numéricas
    is_valid, non_numeric_cols, _ = validate_numeric_data(X_train)

    if not is_valid:
        logger.warning("Removendo colunas não numéricas antes do treinamento.")
        X_train_numeric = X_train.drop(columns=non_numeric_cols)
        X_val_numeric = X_val.drop(columns=[col for col in non_numeric_cols if col in X_val.columns])
        X_test_numeric = X_test.drop(columns=[col for col in non_numeric_cols if col in X_test.columns])

        logger.info(f"Dimensões após remoção de colunas não numéricas:")
        logger.info(f"- X_train: {X_train_numeric.shape}")
        logger.info(f"- X_val: {X_val_numeric.shape}")
        logger.info(f"- X_test: {X_test_numeric.shape}")

        trainer.train_logistic_regression(
            X_train_numeric, y_train,
            name="LogisticRegression",
            C=1.0,
            class_weight='balanced'
        )
    else:
        trainer.train_logistic_regression(
            X_train, y_train,
            name="LogisticRegression",
            C=1.0,
            class_weight='balanced'
        )

    # 2. Treinar Random Forest
    logger.info("\n" + "="*80)
    logger.info("Treinando Random Forest...")
    logger.info("="*80)

    trainer.train_random_forest(
        X_train_numeric if not is_valid else X_train,
        y_train,
        name="RandomForest",
        n_estimators=100,
        max_depth=10,
        class_weight='balanced'
    )

    # 3. Treinar LightGBM
    logger.info("\n" + "="*80)
    logger.info("Treinando LightGBM...")
    logger.info("="*80)

    trainer.train_lightgbm(
        X_train_numeric if not is_valid else X_train,
        y_train,
        X_val_numeric if not is_valid else X_val,
        y_val,
        name="LightGBM",
        learning_rate=0.05,
        n_estimators=200,
        max_depth=7,
        num_leaves=31
    )

    # 4. Treinar XGBoost
    logger.info("\n" + "="*80)
    logger.info("Treinando XGBoost...")
    logger.info("="*80)

    trainer.train_xgboost(
        X_train_numeric if not is_valid else X_train,
        y_train,
        X_val_numeric if not is_valid else X_val,
        y_val,
        name="XGBoost",
        learning_rate=0.05,
        n_estimators=200,
        max_depth=6,
        scale_pos_weight=2
    )

    # 5. Criar Ensemble
    logger.info("\n" + "="*80)
    logger.info("Criando Ensemble...")
    logger.info("="*80)

    trainer.train_ensemble()

    # 6. Otimizar thresholds
    logger.info("\n" + "="*80)
    logger.info("Otimizando thresholds...")
    logger.info("="*80)

    for name in trainer.models:
        trainer.optimize_threshold(
            name,
            X_val_numeric if not is_valid and name in ["LogisticRegression", "RandomForest", "LightGBM", "XGBoost"] else X_val,
            y_val,
            optimize_for_business=True
        )

    # 7. Avaliar modelos
    logger.info("\n" + "="*80)
    logger.info("Avaliando modelos...")
    logger.info("="*80)

    results_df = trainer.evaluate_all_models(
        X_test_numeric if not is_valid else X_test,
        y_test
    )

    # 8. Salvar modelos
    logger.info("\n" + "="*80)
    logger.info("Salvando modelos...")
    logger.info("="*80)

    trainer.save_models()

    return trainer, results_df

def main():
    """
    Função principal para treinamento e avaliação de modelos de inadimplência.
    """
    # Configurar caminho para dados processados
    project_root = get_project_root()
    processed_dir = os.path.join(project_root, 'data', 'processed')

    # Carregar e preparar dados
    logger.info("\n" + "="*80)
    logger.info("CARREGANDO E PREPARANDO DADOS")
    logger.info("="*80)

    try:
        data_dict = load_and_prepare_data(processed_dir, handle_imbalance=True)
    except Exception as e:
        logger.error(f"Erro ao carregar dados: {e}")
        logger.info("Tentando diretório alternativo (data/interim)...")

        try:
            data_dict = load_and_prepare_data(os.path.join(project_root, 'data', 'interim'), handle_imbalance=True)
        except Exception as e:
            logger.error(f"Erro ao carregar dados alternativos: {e}")
            raise

    # Treinar e avaliar modelos
    logger.info("\n" + "="*80)
    logger.info("TREINANDO E AVALIANDO MODELOS")
    logger.info("="*80)

    trainer, results = train_all_models(data_dict)

    # Exibir resumo final
    logger.info("\n" + "="*80)
    logger.info("RESUMO FINAL")
    logger.info("="*80)

    logger.info("\nResultados dos modelos:")
    logger.info(results.to_string())

    # Melhor modelo
    best_model_name = trainer.best_model_name
    if best_model_name:
        logger.info(f"\nMelhor modelo: {best_model_name}")
        logger.info(f"Métricas do melhor modelo:")
        metrics = trainer.evaluation_results[best_model_name]
        logger.info(f"- AUC: {metrics['auc']:.4f}")
        logger.info(f"- F1-Score: {metrics['f1_score']:.4f}")
        logger.info(f"- Precision: {metrics['precision']:.4f}")
        logger.info(f"- Recall: {metrics['recall']:.4f}")
        logger.info(f"- Taxa de Aprovação: {metrics['aprovacao_rate']:.2%}")
        logger.info(f"- Taxa de Inadimplência no Portfolio: {metrics['inadimplencia_portfolio']:.2%}")
        logger.info(f"- Custo de Negócio Normalizado: {metrics['normalized_cost']:.4f}")

    logger.info("\nTreinamento e avaliação concluídos com sucesso!")
    logger.info(f"Resultados salvos em: {trainer.eval_dir}")
    logger.info(f"Modelos salvos em: {trainer.model_dir}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Treinamento de modelos para predição de inadimplência")
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Diretório com dados processados (padrão: data/processed)')
    parser.add_argument('--timestamp', type=str, default=None,
                        help='Timestamp dos dados a serem usados')
    parser.add_argument('--no_balance', action='store_true',
                        help='Desativar balanceamento de classes')

    args = parser.parse_args()

    try:
        main()
    except Exception as e:
        logger.error(f"Erro durante a execução: {e}")
        import traceback
        traceback.print_exc()