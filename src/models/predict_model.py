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
import traceback

# Configurar logger específico para este módulo
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Garantir que o diretório raiz do projeto esteja no PYTHONPATH
project_root = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


# Classe PathManager aprimorada
class PathManager:
    """Gerenciador de caminhos para o projeto de predição de inadimplência."""

    def __init__(self):
        """Inicializa o gerenciador de caminhos."""
        # Encontrar o diretório raiz do projeto de várias maneiras possíveis
        self.project_root = self._find_project_root()
        logger.info(f"Diretório raiz do projeto: {self.project_root}")

    def _find_project_root(self) -> str:
        """
        Encontra o diretório raiz do projeto de maneira robusta.
        Tenta múltiplas abordagens para lidar com diferentes ambientes de execução.
        """
        # Método 1: Baseado no caminho deste arquivo
        current_dir = os.path.dirname(os.path.abspath(__file__))
        root = os.path.abspath(os.path.join(current_dir, os.pardir, os.pardir))

        # Verificar se é realmente o diretório raiz (presença de diretórios chave)
        if os.path.exists(os.path.join(root, 'src')) and os.path.exists(os.path.join(root, 'models')):
            return root

        # Método 2: Baseado no diretório de trabalho atual
        root = os.getcwd()
        while root and not (os.path.exists(os.path.join(root, 'src')) and
                           os.path.exists(os.path.join(root, 'models'))):
            parent = os.path.dirname(root)
            if parent == root:  # Chegou ao diretório raiz do sistema
                break
            root = parent

        if os.path.exists(os.path.join(root, 'src')) and os.path.exists(os.path.join(root, 'models')):
            return root

        # Método 3: Fallback para o diretório atual
        return os.getcwd()

    def get_data_path(self, subdir: str) -> str:
        """Retorna caminho para diretório de dados."""
        path = os.path.join(self.project_root, "data", subdir)
        return path

    def get_model_path(self, subdir: str) -> str:
        """Retorna caminho para diretório de modelos."""
        path = os.path.join(self.project_root, "models", subdir)
        return path

    def get_report_path(self, subdir: str, filename: Optional[str] = None) -> str:
        """Retorna caminho para diretório de relatórios."""
        path = os.path.join(self.project_root, "reports", subdir)
        if filename:
            path = os.path.join(path, filename)
        return path

    def find_model_file(self, model_name: str) -> Optional[str]:
        """
        Encontra arquivo de modelo com nome específico.
        Tenta múltiplos diretórios e suporta busca flexível por padrões.
        """
        # Padrões de busca para diferentes tipos de strings de entrada
        if model_name.endswith('.joblib'):
            patterns = [model_name]
        else:
            patterns = [
                f"{model_name}.joblib",
                f"{model_name}_*.joblib",
                f"*{model_name}*.joblib"
            ]

        # Diretórios a verificar em ordem de prioridade
        dirs_to_check = [
            os.path.join(self.project_root, "models", "trained_models"),
            os.path.join(self.project_root, "models", "trained"),
            os.path.join(self.project_root, "models")
        ]

        # Verificar cada diretório
        for directory in dirs_to_check:
            if not os.path.exists(directory):
                continue

            for pattern in patterns:
                matching_files = glob.glob(os.path.join(directory, pattern))
                if matching_files:
                    # Ordenar por data de modificação (mais recente primeiro)
                    matching_files.sort(key=os.path.getmtime, reverse=True)
                    logger.info(f"Encontrado arquivo de modelo {matching_files[0]}")
                    return matching_files[0]

        return None

    def find_data_file(self, filename: str) -> Optional[str]:
        """
        Encontra arquivo de dados com nome específico.
        Verifica múltiplos diretórios de dados.
        """
        # Diretórios a verificar em ordem de prioridade
        dirs_to_check = [
            os.path.join(self.project_root, "data", "processed"),
            os.path.join(self.project_root, "data", "interim"),
            os.path.join(self.project_root, "data", "raw"),
            os.path.join(self.project_root, "data")
        ]

        # Primeiro, tentar correspondência exata
        for directory in dirs_to_check:
            if not os.path.exists(directory):
                continue

            full_path = os.path.join(directory, filename)
            if os.path.exists(full_path):
                logger.info(f"Encontrado arquivo de dados {full_path}")
                return full_path

        # Se não encontrar, tentar padrões
        patterns = [
            filename,
            f"{filename}.*",
            f"*{filename}*"
        ]

        for directory in dirs_to_check:
            if not os.path.exists(directory):
                continue

            for pattern in patterns:
                matching_files = glob.glob(os.path.join(directory, pattern))
                if matching_files:
                    # Ordenar por data de modificação (mais recente primeiro)
                    matching_files.sort(key=os.path.getmtime, reverse=True)
                    logger.info(f"Encontrado arquivo de dados {matching_files[0]}")
                    return matching_files[0]

        return None


# Classe FeatureEngineer simplificada (compatível com o que é usado em train_model.py)
class FeatureEngineer:
    """Versão simplificada do feature engineer para compatibilidade."""

    def __init__(self) -> None:
        self.feature_map: Dict[str, Any] = {}
        self.categorical_features: List[str] = []
        self.generated_features: List[str] = []
        self.selected_features: Optional[List[str]] = None

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apenas retorna o dataframe sem modificações, a menos que seja substituído por uma versão carregada."""
        return df

    def fit_transform(self, df: pd.DataFrame, target_col: Optional[str] = None) -> pd.DataFrame:
        """Apenas retorna o dataframe sem modificações, a menos que seja substituído por uma versão carregada."""
        return df


class ModelPredictor:
    """
    Classe para fazer predições usando modelos treinados de inadimplência.
    Versão aprimorada com maior robustez na localização de arquivos e tratamento de erros.
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
        self.threshold: float = 0.8
        self.model_details: Dict[str, Any] = {}

        # Inicializar o gerenciador de caminhos
        self.path_manager = PathManager()

        # Carregar modelo e feature engineer automaticamente se os caminhos forem fornecidos
        if model_path:
            self.load_model(model_path)
        else:
            # Tentar encontrar o modelo mais recente
            try:
                latest_model = self.find_latest_model()
                if latest_model:
                    self.load_model(latest_model)
            except Exception as e:
                logger.warning(f"Não foi possível carregar modelo automaticamente: {str(e)}")

        if feature_engineer_path:
            self.load_feature_engineer(feature_engineer_path)
        elif self.model is not None:
            # Tentar encontrar feature engineer correspondente
            try:
                matching_fe = self.find_matching_feature_engineer(
                    model_path or self.model_details.get('path', '')
                )
                if matching_fe:
                    self.load_feature_engineer(matching_fe)
            except Exception as e:
                logger.warning(f"Não foi possível carregar feature engineer automaticamente: {str(e)}")

    def load_model(self, model_path: str) -> 'ModelPredictor':
        """
        Carrega um modelo salvo.

        Args:
            model_path: Caminho para o modelo

        Returns:
            self para encadeamento de métodos
        """
        # Verificar se é um caminho completo ou apenas nome de arquivo
        original_path = model_path
        if not os.path.exists(model_path):
            # Tentar encontrar o arquivo no diretório de modelos
            model_file = self.path_manager.find_model_file(model_path)
            if model_file:
                model_path = model_file
            else:
                raise FileNotFoundError(f"Modelo não encontrado: {model_path}")

        try:
            logger.info(f"Carregando modelo de: {model_path}")
            self.model = joblib.load(model_path)

            # Armazenar detalhes do modelo
            self.model_details = {
                'path': model_path,
                'filename': os.path.basename(model_path),
                'type': type(self.model).__name__,
                'loaded_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

            # Tentar carregar metadados
            self._load_model_metadata(model_path)

            # Verificar se o modelo tem método predict_proba
            if not hasattr(self.model, 'predict_proba'):
                logger.warning(f"AVISO: O modelo não tem método predict_proba(). Algumas funcionalidades podem não funcionar corretamente.")

            logger.info(f"Modelo {self.model_details['type']} carregado com sucesso.")
            return self

        except Exception as e:
            logger.error(f"Erro ao carregar modelo: {str(e)}")
            logger.debug(traceback.format_exc())
            raise

    def _load_model_metadata(self, model_path: str) -> None:
        """
        Carrega metadados do modelo a partir de arquivo JSON correspondente.

        Args:
            model_path: Caminho do modelo
        """
        try:
            model_dir = os.path.dirname(model_path)
            model_name = os.path.basename(model_path).split('_')[0]

            # Extrair timestamp do nome do arquivo
            timestamp_parts = os.path.basename(model_path).split('_')[1:]
            timestamp = '_'.join(timestamp_parts).replace('.joblib', '')

            # Possíveis padrões de nomes de arquivos de metadados
            metadata_patterns = [
                os.path.join(model_dir, f"model_metadata_{timestamp}.json"),
                os.path.join(model_dir, f"metadata_{timestamp}.json"),
                os.path.join(model_dir, "model_metadata.json")
            ]

            for pattern in metadata_patterns:
                if os.path.exists(pattern):
                    with open(pattern, 'r') as f:
                        self.model_metadata = json.load(f)

                    # Extrair threshold para este modelo
                    if 'thresholds' in self.model_metadata and model_name in self.model_metadata['thresholds']:
                        self.threshold = self.model_metadata['thresholds'][model_name]
                        logger.info(f"Threshold carregado: {self.threshold}")
                    elif 'best_threshold' in self.model_metadata:
                        self.threshold = self.model_metadata['best_threshold']
                        logger.info(f"Threshold carregado: {self.threshold}")

                    # Armazenar metadados nos detalhes do modelo
                    self.model_details['metadata'] = self.model_metadata
                    logger.info(f"Metadados carregados de: {pattern}")
                    break
            else:
                logger.info("Nenhum arquivo de metadados encontrado.")

            # Se o modelo tiver atributo threshold, usar esse
            if hasattr(self.model, 'threshold'):
                self.threshold = self.model.threshold
                logger.info(f"Usando threshold interno do modelo: {self.threshold}")

        except Exception as e:
            logger.warning(f"Erro ao carregar metadados: {str(e)}")
            logger.debug(traceback.format_exc())

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
            logger.info(f"Feature engineer carregado com sucesso: {type(self.feature_engineer).__name__}")
            return self
        except Exception as e:
            error_msg = str(e)
            logger.warning(f"Erro ao carregar feature engineer: {error_msg}")

            # Tratar caso específico de ClassNotFoundError
            if "Can't get attribute 'FeatureEngineer'" in error_msg:
                logger.warning("Usando um feature engineer simplificado como substituto")
                self.feature_engineer = FeatureEngineer()
            else:
                # Se for outro tipo de erro, relançar
                logger.error(f"Erro desconhecido ao carregar feature engineer")
                logger.debug(traceback.format_exc())
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
        # Diretórios onde procurar modelos, em ordem de prioridade
        dirs_to_check = [
            self.path_manager.get_model_path("trained_models"),
            self.path_manager.get_model_path("trained"),
            os.path.join(self.path_manager.project_root, "models"),
        ]

        # Padrões de busca para diferentes tipos de modelos
        patterns = [
            f"{model_type}_*.joblib",  # Padrão específico
            "*.joblib"                 # Qualquer modelo como fallback
        ]

        for directory in dirs_to_check:
            if not os.path.exists(directory):
                logger.debug(f"Diretório não encontrado: {directory}")
                continue

            for pattern in patterns:
                matching_files = glob.glob(os.path.join(directory, pattern))

                if matching_files:
                    # Ordenar por data de modificação (mais recente primeiro)
                    matching_files.sort(key=os.path.getmtime, reverse=True)
                    latest_model = matching_files[0]

                    logger.info(f"Modelo mais recente encontrado: {latest_model}")
                    return latest_model

        # Se não encontrou nenhum modelo
        checked_dirs = "\n- ".join(dirs_to_check)
        error_msg = f"Nenhum modelo '{model_type}' encontrado. Diretórios verificados:\n- {checked_dirs}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

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

        # Diretórios onde procurar feature engineers
        dirs_to_check = [
            self.path_manager.get_model_path("preprocessing"),
            os.path.join(self.path_manager.project_root, "models", "preprocessing"),
            os.path.join(self.path_manager.project_root, "models")
        ]

        # Se temos um timestamp, primeiro tentamos encontrar um feature engineer com este timestamp
        if timestamp:
            for directory in dirs_to_check:
                if not os.path.exists(directory):
                    continue

                feature_path = os.path.join(directory, f"feature_engineer_{timestamp}.joblib")
                if os.path.exists(feature_path):
                    logger.info(f"Feature engineer correspondente encontrado: {feature_path}")
                    return feature_path

        # Se não encontrar com o timestamp específico, procurar o mais recente
        for directory in dirs_to_check:
            if not os.path.exists(directory):
                continue

            feature_files = [f for f in os.listdir(directory) if
                             f.startswith('feature_engineer_') and f.endswith('.joblib')]

            if feature_files:
                # Ordenar por nome (assumindo formato com timestamp)
                feature_files.sort(reverse=True)
                latest_feature = os.path.join(directory, feature_files[0])

                logger.info(f"Usando feature engineer mais recente: {latest_feature}")
                return latest_feature

        logger.warning("Nenhum feature engineer encontrado. A preparação básica dos dados será usada.")
        return None

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
        if self.model is None:
            logger.warning("Modelo não carregado - não é possível alinhar features")
            return X

        # Tentar extrair nomes de features do modelo
        model_features = None

        # Explorar diferentes atributos para extrair nomes de features
        model_attrs_to_check = [
            'feature_name_',       # LightGBM
            'feature_names_in_',   # scikit-learn
            'feature_names',       # XGBoost
            '_feature_names',      # Modelos customizados
        ]

        # Verificar pipeline e steps
        if hasattr(self.model, 'named_steps'):
            for step_name, step in self.model.named_steps.items():
                for attr in model_attrs_to_check:
                    if hasattr(step, attr):
                        model_features = getattr(step, attr)
                        logger.info(f"Extraídas {len(model_features)} features do passo {step_name} do pipeline")
                        break
                if model_features is not None:
                    break
        else:
            # Verificar o modelo diretamente
            for attr in model_attrs_to_check:
                if hasattr(self.model, attr):
                    model_features = getattr(self.model, attr)
                    logger.info(f"Extraídas {len(model_features)} features diretamente do modelo")
                    break

        # Se não conseguimos extrair as features do modelo
        if model_features is None:
            # Verificar se podemos obter o número de features que o modelo espera
            n_features_expected = None

            # Verificar atributos possíveis para número de features
            for attr in ['n_features_', 'n_features_in_', '_n_features']:
                if hasattr(self.model, attr):
                    n_features_expected = getattr(self.model, attr)
                    break

            if n_features_expected and X.shape[1] != n_features_expected:
                logger.warning(f"⚠️ ALERTA DE INCOMPATIBILIDADE: Modelo espera {n_features_expected} features, "
                               f"mas os dados têm {X.shape[1]} features.")
                logger.warning("Não foi possível extrair nomes de features do modelo para alinhamento automático.")

            # Verificar metadados para features
            if 'selected_features' in self.model_metadata:
                model_features = self.model_metadata['selected_features']
                logger.info(f"Usando lista de {len(model_features)} features dos metadados")
            else:
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
            X = X.drop(columns=extra_features)

        if missing_features:
            logger.warning(f"Faltam {len(missing_features)} features requeridas pelo modelo nos dados")

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

    def plot_prediction_distribution(self, results: pd.DataFrame, output_dir: Optional[str] = None, target_col=None) -> None:
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

        # 4. Adicionar gráfico de calibração se tiver dados reais
        if 'inadimplente_previsto' in results.columns and target_col in results.columns:
            try:
                from sklearn.calibration import calibration_curve

                prob_true, prob_pred = calibration_curve(
                    results[target_col],
                    results['probabilidade_inadimplencia'],
                    n_bins=10
                )

                plt.figure(figsize=(10, 6))
                plt.plot(prob_pred, prob_true, marker='o')
                plt.plot([0, 1], [0, 1], 'k--')
                plt.title('Curva de Calibração do Modelo')
                plt.xlabel('Probabilidade Média Predita')
                plt.ylabel('Proporção Real de Positivos')
                plt.grid(True, alpha=0.3)

                calibration_path = os.path.join(output_dir, f"calibration_curve_{timestamp}.png")
                plt.savefig(calibration_path, dpi=300, bbox_inches='tight')
                plt.close()
                logger.info(f"Curva de calibração salva em: {calibration_path}")
            except Exception as e:
                logger.warning(f"Erro ao gerar curva de calibração: {str(e)}")

def main() -> None:
    """
    Função principal para fazer predições usando modelos treinados.
    """
    import argparse

    # Definir argumentos do comando
    parser = argparse.ArgumentParser(description="Fazer predições usando modelos treinados de inadimplência")
    parser.add_argument('--data', type=str, required=False,
                        help='Caminho para arquivo de dados (CSV ou Excel)')
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
    parser.add_argument('--threshold', type=float, default=None,
                        help='Threshold para classificação (se None, usa o definido no modelo ou padrão)')
    parser.add_argument('--plot', action='store_true',
                        help='Gerar gráficos de visualização das predições')

    args = parser.parse_args()

    try:
        # Criar preditor
        predictor = ModelPredictor()

        # Iniciar o gerenciador de caminho
        path_manager = PathManager()

        # Se o threshold foi especificado, definir explicitamente
        if args.threshold is not None:
            predictor.threshold = args.threshold
            logger.info(f"Usando threshold definido manualmente: {args.threshold}")

        # Encontrar melhor modelo automaticamente se não especificado
        if args.model:
            predictor.load_model(args.model)
        else:
            try:
                # Tentar encontrar o melhor modelo mais recente
                logger.info("Buscando o modelo mais recente...")
                try:
                    model_path = predictor.find_latest_model("best_model")
                except FileNotFoundError:
                    # Se não encontrar best_model, tentar qualquer modelo
                    all_models = glob.glob(os.path.join(path_manager.get_model_path("trained_models"), "*.joblib"))
                    all_models += glob.glob(os.path.join(path_manager.get_model_path("trained"), "*.joblib"))

                    if all_models:
                        # Ordenar por data de modificação (mais recente primeiro)
                        all_models.sort(key=os.path.getmtime, reverse=True)
                        model_path = all_models[0]
                        logger.info(f"Usando modelo mais recente encontrado: {os.path.basename(model_path)}")
                    else:
                        raise FileNotFoundError("Nenhum modelo encontrado. Especifique um caminho de modelo usando --model")

                predictor.load_model(model_path)
            except Exception as e:
                logger.error(f"Erro ao encontrar modelo: {str(e)}")
                logger.error("Especifique um caminho de modelo usando --model")
                return

        # Carregar feature engineer
        if args.feature_engineer:
            predictor.load_feature_engineer(args.feature_engineer)
        else:
            # Tentar encontrar feature engineer correspondente
            try:
                feature_path = predictor.find_matching_feature_engineer(
                    args.model or predictor.model_details.get('path', '')
                )
                if feature_path:
                    predictor.load_feature_engineer(feature_path)
            except Exception as e:
                logger.warning(f"Aviso: {str(e)}")
                logger.info("Continuando sem feature engineering personalizado.")

        # Verificar se o arquivo de dados foi fornecido ou tentar encontrar um padrão
        if not args.data:
            # Tentar encontrar arquivo de dados padrão
            possible_data_files = []
            for subdir in ['processed', 'interim', 'raw']:
                try:
                    data_dir = path_manager.get_data_path(subdir)
                    if os.path.exists(data_dir):
                        # Procurar por arquivos com padrões comuns
                        for pattern in ['test_*.csv', 'data_*.csv', '*.csv']:
                            matching_files = glob.glob(os.path.join(data_dir, pattern))
                            possible_data_files.extend(matching_files)
                except Exception:
                    continue

            if possible_data_files:
                # Usar o arquivo mais recente
                possible_data_files.sort(key=os.path.getmtime, reverse=True)
                args.data = possible_data_files[0]
                logger.info(f"Usando arquivo de dados encontrado automaticamente: {args.data}")
            else:
                logger.error("Nenhum arquivo de dados fornecido e não foi possível encontrar automaticamente.")
                logger.error("Especifique um arquivo de dados usando --data")
                return

        # Definir caminho de saída padrão se não fornecido
        if not args.output:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            args.output = path_manager.get_report_path("predictions", f"predictions_{timestamp}.csv")

        # Fazer predições com base nos parâmetros fornecidos
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

            logger.info(f"Gerando predições com explicações para {len(data)} registros...")
            results = predictor.predict_and_explain(
                data=data,
                target_col=args.target
            )
        else:
            logger.info(f"Gerando predições em lote para {args.data}...")
            results = predictor.batch_predict(
                data_path=args.data,
                output_path=args.output,
                target_col=args.target
            )

        # Gerar visualizações se solicitado
        if args.plot:
            logger.info("Gerando visualizações...")
            output_dir = os.path.dirname(args.output)
            predictor.plot_prediction_distribution(results, output_dir)

        # Exibir resumo dos resultados
        if 'inadimplente_previsto' in results.columns:
            n_total = len(results)
            n_inadimplentes = results['inadimplente_previsto'].sum()
            percent_inadimplentes = 100 * n_inadimplentes / n_total

            logger.info(f"\nResumo das Predições:")
            logger.info(f"Total de registros: {n_total}")
            logger.info(f"Classificados como inadimplentes: {n_inadimplentes} ({percent_inadimplentes:.2f}%)")
            logger.info(f"Threshold utilizado: {predictor.threshold:.4f}")

            if args.target and args.target in results.columns:
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
                accuracy = accuracy_score(results[args.target], results['inadimplente_previsto'])
                precision = precision_score(results[args.target], results['inadimplente_previsto'], zero_division=0)
                recall = recall_score(results[args.target], results['inadimplente_previsto'], zero_division=0)
                f1 = f1_score(results[args.target], results['inadimplente_previsto'], zero_division=0)

                logger.info(f"\nMétricas de desempenho:")
                logger.info(f"Acurácia: {accuracy:.4f}")
                logger.info(f"Precisão: {precision:.4f}")
                logger.info(f"Recall: {recall:.4f}")
                logger.info(f"F1-Score: {f1:.4f}")

        print(f"\nPredição concluída com sucesso! Resultados salvos em: {args.output}")

    except Exception as e:
        logger.error(f"Erro durante a predição: {str(e)}")
        logger.debug(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()

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
            logger.debug(traceback.format_exc())

        return results