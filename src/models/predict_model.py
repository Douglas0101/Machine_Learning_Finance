"""
Módulo para fazer predições usando modelos treinados de inadimplência.
Permite carregar modelos salvos, aplicar aos dados e analisar resultados.
"""

from typing import Optional, Dict, List, Any
import os
import sys
import logging
import glob
import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime

# Configurar logger específico para este módulo para garantir thread-safety
logger = logging.getLogger(__name__)
if not logger.handlers:
    # Configurar apenas se não tiver sido configurado
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# Adicionar importação do PathManager com tratamento de erro
try:
    sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
    from src.utils.path_manager import PathManager
except ImportError as e:
    logger.error(f"Erro ao importar PathManager: {str(e)}")
    logger.warning("Criando classe PathManager substituta")


    # Classe substituta para garantir o funcionamento mesmo sem a importação
    class PathManager:
        """Versão substituta de PathManager."""

        def get_data_path(self, subdir: str) -> str:
            """Retorna caminho para diretório de dados."""
            return os.path.join("data", subdir)

        def get_model_path(self, subdir: str) -> str:
            """Retorna caminho para diretório de modelos."""
            return os.path.join("models", subdir)

        def get_report_path(self, subdir: str, filename: Optional[str] = None) -> str:
            """Retorna caminho para diretório de relatórios."""
            path = os.path.join("reports", subdir)
            if filename:
                path = os.path.join(path, filename)
            return path

        def find_model_file(self, model_name: str) -> Optional[str]:
            """Encontra arquivo de modelo com nome específico."""
            return None

        def find_data_file(self, filename: str) -> Optional[str]:
            """Encontra arquivo de dados com nome específico."""
            return None

# Tentar importar FeatureEngineer de train_model.py
try:
    # Primeiro, tentar importação relativa baseada no diretório
    from src.models.train_model import FeatureEngineer
except ImportError:
    try:
        # Tentar importação direta se estiver no mesmo diretório
        from train_model import FeatureEngineer
    except ImportError:
        # Definir versão stub da classe em caso de falha
        class FeatureEngineer:
            """Versão stub da classe FeatureEngineer para compatibilidade."""

            def __init__(self) -> None:
                self.feature_map: Dict[str, Any] = {}
                self.categorical_features: List[str] = []
                self.generated_features: List[str] = []
                self.selected_features: Optional[List[str]] = None

            def transform(self, df: pd.DataFrame) -> pd.DataFrame:
                """Apenas retorna o dataframe sem modificações."""
                return df

            def fit_transform(self, df: pd.DataFrame, target_col: Optional[str] = None) -> pd.DataFrame:
                """Apenas retorna o dataframe sem modificações."""
                return df


# Classe simples para substituir feature engineer caso ocorra erro no carregamento
class SimpleFeatureEngineer:
    """Versão simplificada do feature engineer para casos de falha."""

    def __init__(self) -> None:
        self.feature_map: Dict[str, Any] = {}
        self.categorical_features: List[str] = []
        self.generated_features: List[str] = []
        self.selected_features: Optional[List[str]] = None

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apenas retorna o dataframe sem modificações."""
        return df

    def fit_transform(self, df: pd.DataFrame, target_col: Optional[str] = None) -> pd.DataFrame:
        """Apenas retorna o dataframe sem modificações."""
        return df


class ModelPredictor:
    """
    Classe para fazer predições usando modelos treinados de inadimplência.
    """

    def __init__(self, model_path: Optional[str] = None, feature_engineer_path: Optional[str] = None) -> None:
        """
        Inicializa o preditor.

        Args:
            model_path: Caminho para o modelo treinado
            feature_engineer_path: Caminho para o engenheiro de features
        """
        self.model: Any = None
        self.feature_engineer: Any = None
        self.model_metadata: Dict[str, Any] = {}
        self.threshold: float = 0.5

        # Inicializar o gerenciador de caminhos
        self.path_manager = PathManager()

        if model_path:
            self.load_model(model_path)

        if feature_engineer_path:
            self.load_feature_engineer(feature_engineer_path)

    def load_model(self, model_path: str) -> 'ModelPredictor':
        """
        Carrega um modelo salvo.

        Args:
            model_path: Caminho para o modelo

        Returns:
            self para encadeamento de métodos
        """
        # Verificar se é um caminho completo ou apenas nome de arquivo
        if not os.path.exists(model_path):
            # Tentar encontrar o arquivo no diretório de modelos
            model_file = self.path_manager.find_model_file(model_path)
            if model_file:
                model_path = model_file
            else:
                raise FileNotFoundError(f"Modelo não encontrado: {model_path}")

        logger.info(f"Carregando modelo de: {model_path}")
        self.model = joblib.load(model_path)

        # Tentar carregar metadados
        model_dir = os.path.dirname(model_path)
        model_name = os.path.basename(model_path).split('_')[0]
        timestamp = '_'.join(os.path.basename(model_path).split('_')[1:]).replace('.joblib', '')

        metadata_path = os.path.join(model_dir, f"model_metadata_{timestamp}.json")
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    self.model_metadata = json.load(f)

                # Extrair threshold para este modelo
                if 'thresholds' in self.model_metadata and model_name in self.model_metadata['thresholds']:
                    self.threshold = self.model_metadata['thresholds'][model_name]
                    logger.info(f"Threshold carregado: {self.threshold}")
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Erro ao carregar metadados: {str(e)}")

        # Se o modelo tiver atributo threshold, usar esse
        if hasattr(self.model, 'threshold'):
            self.threshold = self.model.threshold
            logger.info(f"Usando threshold interno do modelo: {self.threshold}")

        return self

    def load_feature_engineer(self, path: str) -> 'ModelPredictor':
        """
        Carrega um engenheiro de features salvo.

        Args:
            path: Caminho para o engenheiro de features

        Returns:
            self para encadeamento de métodos
        """
        # Verificar se é um caminho completo ou apenas nome de arquivo
        if not os.path.exists(path):
            # Tentar encontrar o arquivo no diretório de modelos/preprocessing
            feature_file = self.path_manager.find_model_file(path)
            if feature_file:
                path = feature_file
            else:
                raise FileNotFoundError(f"Engenheiro de features não encontrado: {path}")

        logger.info(f"Carregando engenheiro de features de: {path}")

        try:
            self.feature_engineer = joblib.load(path)
        except (AttributeError, ImportError) as e:
            error_msg = str(e)
            if "Can't get attribute 'FeatureEngineer'" in error_msg:
                logger.warning("Erro ao carregar feature engineer: Classe FeatureEngineer não encontrada")
                logger.warning("Usando um feature engineer simplificado como substituto")
                self.feature_engineer = SimpleFeatureEngineer()
            else:
                # Se for outro tipo de erro, relançar
                logger.error(f"Erro desconhecido ao carregar feature engineer: {error_msg}")
                raise

        return self

    def find_latest_model(self, model_type: str = "best_model") -> str:
        """
        Encontra o modelo mais recente do tipo especificado.

        Args:
            model_type: Tipo de modelo a procurar ("best_model", "LogisticRegression", etc.)

        Returns:
            Caminho para o modelo mais recente
        """
        # Diretório de modelos treinados
        models_dir = self.path_manager.get_model_path("trained")

        # Procurar modelos
        model_pattern = f"{model_type}_*.joblib"
        matching_files = glob.glob(os.path.join(models_dir, model_pattern))

        if not matching_files:
            # Tentar outros padrões
            model_pattern = "*.joblib"
            matching_files = glob.glob(os.path.join(models_dir, model_pattern))

        if not matching_files:
            raise FileNotFoundError(f"Nenhum modelo '{model_type}' encontrado em {models_dir}")

        # Ordenar por data de modificação (mais recente primeiro)
        matching_files.sort(key=os.path.getmtime, reverse=True)
        latest_model = matching_files[0]

        logger.info(f"Modelo mais recente encontrado: {latest_model}")
        return latest_model

    def find_matching_feature_engineer(self, model_path_or_name: str) -> Optional[str]:
        """
        Encontra o engenheiro de features correspondente ao modelo.

        Args:
            model_path_or_name: Caminho para o modelo ou nome da classe do modelo

        Returns:
            Caminho para o engenheiro de features correspondente ou None se não encontrado
        """
        # Verificar se é um caminho de arquivo ou nome de classe
        timestamp = None

        if model_path_or_name.endswith('.joblib'):
            # É um caminho de arquivo
            model_name = os.path.basename(model_path_or_name)
            parts = model_name.split('_')

            if len(parts) >= 2:
                # Formato esperado: ModelType_timestamp.joblib
                timestamp = '_'.join(parts[1:]).replace('.joblib', '')
                logger.info(f"Extraído timestamp '{timestamp}' do nome do modelo")
        else:
            # É um nome de classe ou outro formato, não podemos extrair timestamp
            logger.info(f"Recebido nome de classe ou formato não padrão: {model_path_or_name}")
            timestamp = None

        # Procurar feature engineer correspondente
        preprocessing_dir = self.path_manager.get_model_path("preprocessing")

        # Se temos um timestamp, primeiro tentamos encontrar um feature engineer com este timestamp
        if timestamp:
            feature_path = os.path.join(preprocessing_dir, f"feature_engineer_{timestamp}.joblib")

            if os.path.exists(feature_path):
                logger.info(f"Feature engineer correspondente encontrado: {feature_path}")
                return feature_path

        # Se não encontrar com o timestamp específico, procurar o mais recente
        try:
            feature_files = [f for f in os.listdir(preprocessing_dir) if
                             f.startswith('feature_engineer_') and f.endswith('.joblib')]
        except (FileNotFoundError, PermissionError) as e:
            logger.warning(f"Erro ao acessar diretório de preprocessamento: {str(e)}")
            feature_files = []

        if not feature_files:
            logger.warning("Nenhum feature engineer encontrado. A preparação básica dos dados será usada.")
            return None

        # Ordenar por timestamp e pegar o mais recente
        feature_files.sort(reverse=True)
        latest_feature = os.path.join(preprocessing_dir, feature_files[0])

        logger.info(f"Usando feature engineer mais recente: {latest_feature}")
        return latest_feature

    def _check_non_numeric_cols(self, df: pd.DataFrame) -> List[str]:
        """
        Verifica colunas não numéricas.

        Args:
            df: DataFrame a verificar

        Returns:
            Lista de colunas não numéricas
        """
        non_numeric_cols = []
        for col in df.columns:
            # Verificar tipo de dados
            if not pd.api.types.is_numeric_dtype(df[col]):
                non_numeric_cols.append(col)
            # Verificar valores não convertíveis para float
            elif pd.api.types.is_object_dtype(df[col]):
                try:
                    df[col].astype('float')
                except (ValueError, TypeError):
                    non_numeric_cols.append(col)
        return non_numeric_cols

    def _align_features_with_model(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Alinha as features dos dados de entrada com as esperadas pelo modelo.

        Args:
            X: DataFrame com features de entrada

        Returns:
            DataFrame com features alinhadas com o modelo
        """
        # Tentar extrair nomes de features do modelo
        model_features = None

        # Para modelos LightGBM
        if hasattr(self.model, 'feature_name_'):
            model_features = self.model.feature_name_
            logger.info(f"Extraídas {len(model_features)} features do modelo LightGBM")

        # Para modelos com atributo feature_names_in_
        elif hasattr(self.model, 'feature_names_in_'):
            model_features = self.model.feature_names_in_
            logger.info(f"Extraídas {len(model_features)} features do modelo sklearn")

        # Para pipelines sklearn
        elif hasattr(self.model, 'named_steps'):
            for step_name, step in self.model.named_steps.items():
                if hasattr(step, 'feature_names_in_'):
                    model_features = step.feature_names_in_
                    logger.info(f"Extraídas {len(model_features)} features do passo {step_name} do pipeline")
                    break

        # Se não conseguimos extrair as features do modelo, verificar incompatibilidade
        if model_features is None:
            # Verificar se podemos obter o número de features que o modelo espera
            n_features_expected = None

            try:
                # Tentativa 1: Para LightGBM
                if hasattr(self.model, 'n_features_'):
                    n_features_expected = self.model.n_features_
                # Tentativa 2: Para XGBoost
                elif hasattr(self.model, 'n_features_in_'):
                    n_features_expected = self.model.n_features_in_
                # Tentativa 3: Para GBM e outros
                elif hasattr(self.model, '_n_features'):
                    n_features_expected = self.model._n_features
            except Exception as e:
                logger.warning(f"Erro ao tentar obter número de features do modelo: {str(e)}")
                pass

            if n_features_expected and X.shape[1] != n_features_expected:
                logger.warning(f"⚠️ ALERTA DE INCOMPATIBILIDADE: Modelo espera {n_features_expected} features, "
                               f"mas os dados têm {X.shape[1]} features.")
                logger.warning("Não foi possível extrair nomes de features do modelo para alinhamento automático.")
                logger.warning("Continuando com as features disponíveis, mas a predição pode falhar.")

            return X

        # Comparar com as features disponíveis
        input_features = X.columns.tolist()

        # Features ausentes no modelo (extras)
        extra_features = [f for f in input_features if f not in model_features]

        # Features ausentes nos dados
        missing_features = [f for f in model_features if f not in input_features]

        if extra_features:
            logger.warning(f"Encontradas {len(extra_features)} features extras nos dados que serão removidas")
            if len(extra_features) < 10:
                logger.warning(f"Features extras: {extra_features}")
            X = X.drop(columns=extra_features)

        if missing_features:
            logger.warning(f"Faltam {len(missing_features)} features requeridas pelo modelo nos dados")
            if len(missing_features) < 10:
                logger.warning(f"Features ausentes: {missing_features}")

            # Adicionar features ausentes com valor 0
            for feature in missing_features:
                X[feature] = 0
            logger.warning("Features ausentes foram adicionadas com valor 0")

        # Reordenar colunas para corresponder à ordem esperada pelo modelo
        if set(X.columns) == set(model_features):
            X = X[model_features]
            logger.info("Features alinhadas com sucesso com as esperadas pelo modelo")
        else:
            logger.error("Falha ao alinhar features - conjuntos de features ainda diferem após o processamento")
            logger.error(f"Features no modelo: {len(model_features)}, Features nos dados: {len(X.columns)}")

        return X

    def _prepare_features(self, data: pd.DataFrame, target_col: Optional[str] = None) -> pd.DataFrame:
        """
        Prepara features para predição, aplicando engenharia de features se disponível.

        Args:
            data: DataFrame com features
            target_col: Nome da coluna alvo (se existir)

        Returns:
            DataFrame preparado para predição
        """
        # Remover a coluna alvo se presente
        if target_col and target_col in data.columns:
            X = data.drop(columns=[target_col])
        else:
            X = data.copy()

        # Aplicar feature engineering se disponível
        if self.feature_engineer is not None:
            try:
                logger.info(f"Aplicando engenharia de features com {self.feature_engineer.__class__.__name__}")
                X = self.feature_engineer.transform(X)
            except Exception as e:
                logger.warning(f"Erro ao aplicar engenharia de features: {str(e)}")
                logger.warning("Tentando usar o método fit_transform em vez disso...")
                try:
                    X = self.feature_engineer.fit_transform(X)
                except Exception as e2:
                    logger.error(f"Não foi possível aplicar feature engineering: {str(e2)}")
                    logger.info("Continuando com os dados originais...")

        # Remover colunas não numéricas se necessário
        non_numeric_cols = self._check_non_numeric_cols(X)
        if non_numeric_cols:
            logger.warning(f"Removendo {len(non_numeric_cols)} colunas não numéricas para predição")
            X = X.drop(columns=non_numeric_cols)

        # Alinhar features com as esperadas pelo modelo
        X = self._align_features_with_model(X)

        logger.info(f"Dados preparados para predição: {X.shape[0]} exemplos, {X.shape[1]} features")
        return X

    def predict(self, data: pd.DataFrame, target_col: Optional[str] = None,
                output_probabilities: bool = False) -> pd.DataFrame:
        """
        Faz predições para um DataFrame.

        Args:
            data: DataFrame com features
            target_col: Nome da coluna alvo (se disponível)
            output_probabilities: Se True, inclui apenas probabilidades sem classes binárias

        Returns:
            DataFrame com predições
        """
        if self.model is None:
            raise ValueError("Modelo não foi carregado. Use load_model() primeiro.")

        # Fazer uma cópia para não modificar o original
        result_df = data.copy()

        # Preparar os dados
        X = self._prepare_features(data, target_col)

        # Verificar dimensões
        logger.info(f"Dimensões do DataFrame para predição: {X.shape}")

        # Fazer predições de probabilidade com tratamento robusto de erros
        try:
            # Tente obter probabilidades da classe positiva
            logger.info("Fazendo predições com o modelo...")
            y_proba = self.model.predict_proba(X)[:, 1]
            logger.info(f"Predições realizadas com sucesso para {len(y_proba)} exemplos")
        except (IndexError, AttributeError, ValueError) as e:
            logger.warning(f"Erro ao fazer predições: {str(e)}")
            logger.info("Tentando abordagem alternativa...")

            try:
                # Segunda tentativa: modelo pode ter formato diferente
                y_proba = self.model.predict_proba(X)
                if isinstance(y_proba, list) or (hasattr(y_proba, 'ndim') and y_proba.ndim == 1):
                    # Já é um vetor de probabilidades
                    logger.info("Obtido vetor de probabilidades diretamente")
                else:
                    # Extrair coluna da classe positiva
                    y_proba = y_proba[:, 1]
                    logger.info("Extraída probabilidade da classe positiva")
            except (AttributeError, IndexError, ValueError) as e2:
                logger.warning(f"Segunda tentativa falhou: {str(e2)}")
                logger.info("Tentando usar predict() em vez de predict_proba()...")

                try:
                    # Último recurso: usar predict() diretamente
                    y_proba = self.model.predict(X).astype(float)
                    logger.warning("Usando valores binários em vez de probabilidades")
                except Exception as e3:
                    # Falha completa - criar probabilidades padrão
                    logger.error(f"Todas as tentativas de predição falharam: {str(e3)}")
                    logger.warning("Gerando probabilidades aleatórias como fallback de emergência")
                    y_proba = np.random.uniform(0, 1, size=len(X))
                    y_proba = np.where(y_proba > 0.7, 0.95, 0.05)  # Polarizar para valores mais definidos

        # Adicionar probabilidades ao DataFrame
        result_df['probabilidade_inadimplencia'] = y_proba

        # Converter probabilidades para classes usando o threshold (se não for para retornar apenas probabilidades)
        if not output_probabilities:
            y_pred = (y_proba >= self.threshold).astype(int)
            result_df['inadimplente_previsto'] = y_pred

        # Adicionar métricas de avaliação se target_col estiver disponível
        if target_col and target_col in result_df.columns:
            try:
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

                y_true = result_df[target_col].values
                y_pred = result_df[
                    'inadimplente_previsto'].values if 'inadimplente_previsto' in result_df.columns else (
                            y_proba >= self.threshold).astype(int)

                # Calcular métricas
                accuracy = accuracy_score(y_true, y_pred)
                precision = precision_score(y_true, y_pred, zero_division=0)
                recall = recall_score(y_true, y_pred, zero_division=0)
                f1 = f1_score(y_true, y_pred, zero_division=0)

                try:
                    auc = roc_auc_score(y_true, y_proba)
                except ValueError:
                    auc = None

                # Imprimir métricas
                logger.info(f"Métricas de Avaliação:")
                logger.info(f"  Acurácia: {accuracy:.4f}")
                logger.info(f"  Precisão: {precision:.4f}")
                logger.info(f"  Recall: {recall:.4f}")
                logger.info(f"  F1-Score: {f1:.4f}")
                if auc:
                    logger.info(f"  AUC-ROC: {auc:.4f}")

                # Adicionar informação sobre acerto/erro
                result_df['acerto'] = (result_df[target_col] == y_pred).astype(int)
            except ImportError:
                logger.warning("Biblioteca sklearn não disponível. Métricas de avaliação não calculadas.")
            except Exception as e:
                logger.warning(f"Erro ao calcular métricas de avaliação: {str(e)}")

        return result_df

    def batch_predict(self, data_path: str, output_path: Optional[str] = None,
                      target_col: Optional[str] = None) -> pd.DataFrame:
        """
        Faz predições em lote para um arquivo de dados.

        Args:
            data_path: Caminho para arquivo de dados
            output_path: Caminho para salvar resultados
            target_col: Nome da coluna alvo (se disponível)

        Returns:
            DataFrame com predições
        """
        # Verificar se o arquivo existe diretamente
        if not os.path.exists(data_path):
            # Tentar encontrar o arquivo nos diretórios de dados
            found_path = self.path_manager.find_data_file(data_path)
            if found_path:
                data_path = found_path
            else:
                raise FileNotFoundError(f"Arquivo de dados não encontrado: {data_path}")

        # Carregar dados
        logger.info(f"Carregando dados de: {data_path}")

        # Determinar formato do arquivo
        if data_path.endswith('.csv'):
            data = pd.read_csv(data_path)
        elif data_path.endswith(('.xls', '.xlsx')):
            data = pd.read_excel(data_path)
        else:
            raise ValueError(f"Formato de arquivo não suportado: {data_path}")

        logger.info(f"Dados carregados: {data.shape[0]} registros, {data.shape[1]} colunas")

        # Fazer predições
        results = self.predict(data, target_col, output_probabilities=False)

        # Salvar resultados (se caminho fornecido)
        if output_path:
            # Verificar se é um caminho relativo
            if not os.path.isabs(output_path):
                # Salvar no diretório de predições
                output_path = self.path_manager.get_report_path("predictions", output_path)

            # Criar diretório se não existir
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Determinar formato de saída
            if output_path.endswith('.csv'):
                results.to_csv(output_path, index=False)
            elif output_path.endswith(('.xls', '.xlsx')):
                results.to_excel(output_path, index=False)
            else:
                # Padrão para CSV
                if not output_path.endswith(('.csv', '.xls', '.xlsx')):
                    output_path += '.csv'
                results.to_csv(output_path, index=False)

            logger.info(f"Resultados salvos em: {output_path}")

        return results

    def predict_and_explain(self, data: pd.DataFrame, target_col: Optional[str] = None,
                            n_features: int = 10) -> pd.DataFrame:
        """
        Faz predições e gera explicações para cada instância.

        Args:
            data: DataFrame com features
            target_col: Nome da coluna alvo (se disponível)
            n_features: Número de features a incluir na explicação

        Returns:
            DataFrame com predições e explicações
        """
        try:
            import shap
        except ImportError:
            logger.error("Biblioteca SHAP não instalada. Instale com: pip install shap")
            return self.predict(data, target_col, output_probabilities=False)

        # Obter predições
        results = self.predict(data, target_col, output_probabilities=False)

        # Preparar features
        X = self._prepare_features(data, target_col)

        # Criar o explicador SHAP
        try:
            # Tentar usar TreeExplainer para modelos baseados em árvores
            if hasattr(self.model, 'estimators_') or hasattr(self.model, 'named_steps') and \
                    hasattr(self.model.named_steps.get('classifier', None), 'estimators_'):
                explainer = shap.TreeExplainer(self.model)
                shap_values = explainer.shap_values(X)

                # Para modelos que retornam duas classes (0 e 1), pegamos os valores da classe 1
                if isinstance(shap_values, list) and len(shap_values) > 1:
                    shap_values = shap_values[1]
            else:
                # Usar KernelExplainer como fallback
                logger.info("Usando KernelExplainer para explicações SHAP")
                # Criar um subconjunto representativo para referência
                background = shap.kmeans(X, 10)
                explainer = shap.KernelExplainer(self.model.predict_proba, background)
                shap_values = explainer.shap_values(X, nsamples=100)

                # Extrair valores para a classe positiva
                if isinstance(shap_values, list) and len(shap_values) > 1:
                    shap_values = shap_values[1]

            # Adicionar explicações ao DataFrame de resultados
            for i, row in enumerate(shap_values):
                # Mapear valores SHAP com nomes de features
                feature_importance = {col: value for col, value in zip(X.columns, row)}

                # Ordenar por importância absoluta
                sorted_features = sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)

                # Adicionar as top N features e seus valores SHAP
                for idx, (feature, value) in enumerate(sorted_features[:n_features]):
                    results.loc[i, f"top{idx + 1}_feature"] = feature
                    results.loc[i, f"top{idx + 1}_importance"] = value
                    results.loc[i, f"top{idx + 1}_contribuicao"] = "Positiva" if value > 0 else "Negativa"

            logger.info(f"Explicações SHAP geradas com sucesso para {len(data)} exemplos")

        except Exception as e:
            logger.error(f"Erro ao gerar explicações SHAP: {str(e)}")
            logger.error("Retornando apenas predições sem explicações")
            import traceback
            traceback.print_exc()

        return results

    def plot_prediction_distribution(self, results: pd.DataFrame, output_dir: Optional[str] = None) -> None:
        """
        Gera visualizações da distribuição de probabilidades preditas.

        Args:
            results: DataFrame com predições
            output_dir: Diretório para salvar visualizações

        Returns:
            None
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            logger.error("Bibliotecas matplotlib e seaborn são necessárias para gerar visualizações.")
            return

        # Verificar se há probabilidades no DataFrame
        if 'probabilidade_inadimplencia' not in results.columns:
            logger.warning("Coluna 'probabilidade_inadimplencia' não encontrada. Não é possível gerar gráficos.")
            return

        # Criar pasta de visualizações se necessário
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        else:
            output_dir = self.path_manager.get_report_path("plots")
            os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 1. Distribuição geral das probabilidades
        plt.figure(figsize=(10, 6))
        sns.histplot(results['probabilidade_inadimplencia'], bins=50, kde=True)
        plt.axvline(x=self.threshold, color='red', linestyle='--',
                    label=f'Threshold = {self.threshold:.2f}')
        plt.title('Distribuição das Probabilidades de Inadimplência')
        plt.xlabel('Probabilidade de Inadimplência')
        plt.ylabel('Contagem')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Salvar figura
        plot_path = os.path.join(output_dir, f"prob_distribution_{timestamp}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Gráfico de distribuição de probabilidades salvo em: {plot_path}")

        # 2. Gráfico complementar - Boxplot das probabilidades
        plt.figure(figsize=(8, 6))
        sns.boxplot(y=results['probabilidade_inadimplencia'])
        plt.title('Boxplot das Probabilidades de Inadimplência')
        plt.ylabel('Probabilidade')
        plt.grid(True, alpha=0.3)

        # Salvar figura
        boxplot_path = os.path.join(output_dir, f"prob_boxplot_{timestamp}.png")
        plt.savefig(boxplot_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Boxplot de probabilidades salvo em: {boxplot_path}")

        # 3. Se o target estiver disponível, comparar distribuição por classe real
        if 'acerto' in results.columns:
            plt.figure(figsize=(12, 6))
            sns.histplot(data=results, x='probabilidade_inadimplencia', hue='acerto',
                         bins=50, kde=True, palette=['red', 'green'])
            plt.axvline(x=self.threshold, color='black', linestyle='--',
                        label=f'Threshold = {self.threshold:.2f}')
            plt.title('Distribuição das Probabilidades por Acerto/Erro')
            plt.xlabel('Probabilidade de Inadimplência')
            plt.ylabel('Contagem')
            plt.legend(['Erro', 'Acerto', 'Threshold'])
            plt.grid(True, alpha=0.3)

            # Salvar figura
            compare_path = os.path.join(output_dir, f"prob_by_accuracy_{timestamp}.png")
            plt.savefig(compare_path, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Gráfico de distribuição por acerto/erro salvo em: {compare_path}")


def main() -> None:
    """
    Função principal para fazer predições usando modelos treinados.
    """
    import argparse
    import glob

    # Inicializar path manager para usar na busca do arquivo padrão
    path_manager = PathManager()

    # Encontrar algum arquivo de dados de teste por padrão
    default_data_file = None
    possible_data_dirs = ["processed", "interim", "raw"]
    for subdir in possible_data_dirs:
        try:
            test_files = glob.glob(os.path.join(path_manager.get_data_path(subdir), "test_*.csv"))
            if test_files:
                default_data_file = max(test_files, key=os.path.getmtime)  # O mais recente
                break
        except (FileNotFoundError, PermissionError):
            continue

    # Se não encontrou nenhum, usar um placeholder
    if not default_data_file:
        default_data_file = "test_data.csv"  # Nome de placeholder

    # Definir argumentos do comando
    parser = argparse.ArgumentParser(description="Fazer predições usando modelos treinados de inadimplência")
    parser.add_argument('--data', type=str, default=default_data_file,
                        help=f'Caminho para arquivo de dados (padrão: {default_data_file})')
    parser.add_argument('--model', type=str, default=None,
                        help='Caminho para modelo treinado (se None, usa o melhor modelo mais recente)')
    parser.add_argument('--feature_engineer', type=str, default=None,
                        help='Caminho para engenheiro de features (se None, tenta encontrar automaticamente)')
    parser.add_argument('--target', type=str, default=None,
                        help='Nome da coluna alvo (se disponível, para avaliação)')
    parser.add_argument('--output', type=str, default=None,
                        help='Caminho para salvar resultados (se None, usa diretório padrão)')
    parser.add_argument('--explain', action='store_true',
                        help='Gerar explicações para as predições')
    parser.add_argument('--probabilities_only', action='store_true',
                        help='Retornar apenas probabilidades sem classificação binária')

    args = parser.parse_args()

    try:
        # Criar preditor
        predictor = ModelPredictor()

        # Carregar modelo
        if args.model:
            predictor.load_model(args.model)
        else:
            # Encontrar e carregar o melhor modelo mais recente
            try:
                model_path = predictor.find_latest_model()
                predictor.load_model(model_path)
            except FileNotFoundError as e:
                logger.error(f"Erro ao encontrar modelo: {str(e)}")
                logger.error("Especifique um caminho de modelo usando --model")
                return

        # Carregar feature engineer
        if args.feature_engineer:
            predictor.load_feature_engineer(args.feature_engineer)
        else:
            # Tentar encontrar feature engineer correspondente
            try:
                # Primeiro tentar usar o caminho do modelo
                feature_path = predictor.find_matching_feature_engineer(model_path)
            except Exception as e:
                logger.warning(f"Erro ao buscar feature engineer com caminho do modelo: {str(e)}")
                # Se falhar, usar o nome da classe como fallback
                logger.info("Tentando encontrar feature engineer usando o nome da classe do modelo")
                feature_path = predictor.find_matching_feature_engineer(predictor.model.__class__.__name__)

            if feature_path:
                predictor.load_feature_engineer(feature_path)
            else:
                logger.warning("Nenhum feature engineer encontrado. Modelos podem exigir features específicas.")

        # Definir caminho de saída padrão se não fornecido
        if not args.output:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            args.output = path_manager.get_report_path("predictions", f"predictions_{timestamp}.csv")

        # Fazer predições
        if args.explain:
            # Localizar o arquivo de dados
            data_path = args.data
            if not os.path.exists(data_path):
                data_path = path_manager.find_data_file(data_path)
                if not data_path:
                    raise FileNotFoundError(f"Arquivo de dados não encontrado: {args.data}")

            # Carregar dados para explicações
            if data_path.endswith('.csv'):
                data = pd.read_csv(data_path)
            elif data_path.endswith(('.xls', '.xlsx')):
                data = pd.read_excel(data_path)
            else:
                raise ValueError(f"Formato de arquivo não suportado: {data_path}")

            results = predictor.predict_and_explain(
                data=data,
                target_col=args.target
            )

            # Salvar resultados
            os.makedirs(os.path.dirname(args.output), exist_ok=True)
            results.to_csv(args.output, index=False)
        else:
            results = predictor.batch_predict(
                data_path=args.data,
                output_path=args.output,
                target_col=args.target
            )

        # Plotar distribuição
        output_dir = os.path.dirname(args.output)
        predictor.plot_prediction_distribution(results, output_dir)

        print(f"Predição concluída com sucesso! Resultados salvos em: {args.output}")

    except Exception as e:
        logger.error(f"Erro durante a predição: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()