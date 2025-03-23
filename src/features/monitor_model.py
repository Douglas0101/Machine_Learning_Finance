"""
Módulo para monitoramento de modelos de predição de inadimplência em produção.
Implementa detecção de data drift, target drift e degradação de performance
para garantir a qualidade das previsões ao longo do tempo.
"""

import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import json
import logging
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
import scipy.stats as stats
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import KBinsDiscretizer

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


@dataclass
class DriftMetrics:
    """Armazena métricas de drift para uma variável."""
    feature_name: str
    psi: float = 0.0  # Population Stability Index
    ks_statistic: float = 0.0  # Kolmogorov-Smirnov statistic
    ks_pvalue: float = 1.0  # Kolmogorov-Smirnov p-value
    js_distance: float = 0.0  # Jensen-Shannon distance
    mean_diff: float = 0.0  # Diferença de médias
    std_diff: float = 0.0  # Diferença de desvios padrão
    drift_detected: bool = False  # Flag para drift detectado
    drift_severity: str = "Nenhum"  # Severidade do drift: Nenhum, Baixo, Médio, Alto, Crítico


@dataclass
class ModelHealthMetrics:
    """Armazena métricas de saúde do modelo."""
    timestamp: datetime = field(default_factory=datetime.now)
    auc: float = 0.0
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    average_precision: float = 0.0
    data_drift_features: List[str] = field(default_factory=list)
    data_drift_score: float = 0.0
    target_drift_score: float = 0.0
    prediction_drift_score: float = 0.0
    stability_score: float = 0.0
    health_status: str = "Ok"  # Ok, Atenção, Crítico


class ModelMonitor:
    """
    Monitor de modelo de inadimplência para detecção de drift e degradação.
    Implementa algoritmos para detectar mudanças nos dados, target e performance,
    gerando alertas e relatórios para ação preventiva.
    """

    def __init__(self, model_path: str,
                 reference_data_path: str,
                 feature_builder_path: Optional[str] = None,
                 model_metadata_path: Optional[str] = None,
                 threshold: Optional[float] = None,
                 drift_threshold: float = 0.2,
                 performance_decline_threshold: float = 0.1,
                 n_bins: int = 10):
        """
        Inicializa o monitor de modelo.

        Args:
            model_path: Caminho para o modelo serializado
            reference_data_path: Caminho para os dados de referência usados para treinar o modelo
            feature_builder_path: Caminho para o feature builder (opcional)
            model_metadata_path: Caminho para os metadados do modelo (opcional)
            threshold: Threshold de classificação (se None, usa metadata ou 0.5)
            drift_threshold: Threshold para considerar drift significativo (0.0-1.0)
            performance_decline_threshold: Threshold para considerar declínio significativo (0.0-1.0)
            n_bins: Número de bins para discretização de variáveis contínuas
        """
        self.model = self._load_model(model_path)
        self.feature_builder = self._load_feature_builder(feature_builder_path)
        self.metadata = self._load_metadata(model_metadata_path)
        self.threshold = threshold or self._get_threshold_from_metadata()
        self.drift_threshold = drift_threshold
        self.performance_decline_threshold = performance_decline_threshold
        self.n_bins = n_bins

        # Carregar dados de referência
        self.reference_data = self._load_reference_data(reference_data_path)

        # Calcular estatísticas de referência
        self.reference_stats = self._calculate_reference_stats()

        # Armazenar histórico de métricas
        self.metrics_history: List[ModelHealthMetrics] = []

        # Configurar diretório de saída
        project_root = get_project_root()
        self.output_dir = os.path.join(project_root, 'reports', 'model_monitoring')
        os.makedirs(self.output_dir, exist_ok=True)

    def _load_model(self, model_path: str) -> Any:
        """
        Carrega o modelo de classificação.

        Args:
            model_path: Caminho para o modelo serializado

        Returns:
            Modelo carregado
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Arquivo de modelo não encontrado: {model_path}")

        logger.info(f"Carregando modelo de: {model_path}")
        try:
            model = joblib.load(model_path)
            return model
        except Exception as e:
            raise RuntimeError(f"Erro ao carregar modelo: {str(e)}")

    def _load_feature_builder(self, feature_builder_path: Optional[str]) -> Optional[Any]:
        """
        Carrega o feature builder, se disponível.

        Args:
            feature_builder_path: Caminho para o feature builder

        Returns:
            Feature builder carregado ou None
        """
        if not feature_builder_path:
            return None

        if not os.path.exists(feature_builder_path):
            logger.warning(f"Feature builder não encontrado: {feature_builder_path}")
            return None

        logger.info(f"Carregando feature builder de: {feature_builder_path}")
        try:
            feature_builder = joblib.load(feature_builder_path)
            return feature_builder
        except Exception as e:
            logger.warning(f"Erro ao carregar feature builder: {str(e)}")
            return None

    def _load_metadata(self, metadata_path: Optional[str]) -> Dict:
        """
        Carrega metadados do modelo, se disponíveis.

        Args:
            metadata_path: Caminho para os metadados

        Returns:
            Dicionário com metadados
        """
        default_metadata = {
            'model_name': 'unknown',
            'model_type': 'unknown',
            'creation_date': datetime.now().strftime('%Y-%m-%d'),
            'features': [],
            'target': 'inadimplente',
            'metrics': {
                'auc': 0.0,
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0
            },
            'thresholds': {
                'default': 0.5
            }
        }

        if not metadata_path:
            return default_metadata

        if not os.path.exists(metadata_path):
            logger.warning(f"Arquivo de metadados não encontrado: {metadata_path}")
            return default_metadata

        logger.info(f"Carregando metadados de: {metadata_path}")
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            return metadata
        except Exception as e:
            logger.warning(f"Erro ao carregar metadados: {str(e)}")
            return default_metadata

    def _get_threshold_from_metadata(self) -> float:
        """
        Obtém o threshold de classificação dos metadados.

        Returns:
            Threshold de classificação
        """
        if not self.metadata:
            return 0.5

        thresholds = self.metadata.get('thresholds', {})

        # Tentar obter threshold específico para o tipo de modelo
        model_type = self.metadata.get('model_type', 'unknown')
        threshold = thresholds.get(model_type, None)

        # Caso não tenha, usar o padrão
        if threshold is None:
            threshold = thresholds.get('default', 0.5)

        return threshold

    def _load_reference_data(self, reference_data_path: str) -> pd.DataFrame:
        """
        Carrega os dados de referência.

        Args:
            reference_data_path: Caminho para os dados de referência

        Returns:
            DataFrame com dados de referência
        """
        if not os.path.exists(reference_data_path):
            raise FileNotFoundError(f"Arquivo de dados de referência não encontrado: {reference_data_path}")

        logger.info(f"Carregando dados de referência de: {reference_data_path}")

        try:
            # Determinar formato do arquivo
            file_ext = os.path.splitext(reference_data_path)[1].lower()

            if file_ext == '.csv':
                return pd.read_csv(reference_data_path)
            elif file_ext in ['.pkl', '.pickle']:
                return pd.read_pickle(reference_data_path)
            elif file_ext == '.parquet':
                return pd.read_parquet(reference_data_path)
            elif file_ext == '.feather':
                return pd.read_feather(reference_data_path)
            else:
                raise ValueError(f"Formato de arquivo não suportado: {file_ext}")

        except Exception as e:
            raise RuntimeError(f"Erro ao carregar dados de referência: {str(e)}")

    def _calculate_reference_stats(self) -> Dict:
        """
        Calcula estatísticas dos dados de referência para comparação futura.

        Returns:
            Dicionário com estatísticas
        """
        logger.info("Calculando estatísticas de referência...")

        stats = {}

        # Separar features e target
        target_col = self.metadata.get('target', 'inadimplente')

        if target_col in self.reference_data.columns:
            y_ref = self.reference_data[target_col]
            X_ref = self.reference_data.drop(columns=[target_col])
        else:
            logger.warning(f"Coluna target '{target_col}' não encontrada nos dados de referência.")
            y_ref = None
            X_ref = self.reference_data

        # Aplicar transformações se feature builder disponível
        if self.feature_builder:
            try:
                X_ref_transformed = self.feature_builder.transform(X_ref)
                # Verificar se a transformação foi bem-sucedida
                if X_ref_transformed.shape[0] == X_ref.shape[0]:
                    X_ref = X_ref_transformed
                else:
                    logger.warning("Transformação de features resultou em DataFrame com número diferente de linhas.")
            except Exception as e:
                logger.warning(f"Erro ao aplicar feature builder: {str(e)}")

        # Estatísticas para cada feature
        for column in X_ref.columns:
            col_data = X_ref[column]

            # Estatísticas diferentes baseadas no tipo de variável
            if pd.api.types.is_numeric_dtype(col_data):
                # Variável numérica
                stats[column] = {
                    'type': 'numeric',
                    'mean': col_data.mean(),
                    'median': col_data.median(),
                    'std': col_data.std(),
                    'min': col_data.min(),
                    'max': col_data.max(),
                    'q1': col_data.quantile(0.25),
                    'q3': col_data.quantile(0.75),
                    'missing': col_data.isna().mean(),
                    'histogram': np.histogram(col_data.dropna(), bins=self.n_bins)[0].tolist()
                }

                # Também armazenar os bins para comparação de distribuição
                stats[column]['bins'] = np.histogram_bin_edges(col_data.dropna(), bins=self.n_bins).tolist()

            else:
                # Variável categórica
                value_counts = col_data.value_counts(normalize=True, dropna=False)

                stats[column] = {
                    'type': 'categorical',
                    'categories': value_counts.index.tolist(),
                    'frequencies': value_counts.values.tolist(),
                    'missing': col_data.isna().mean(),
                    'n_unique': col_data.nunique()
                }

        # Estatísticas do target (se disponível)
        if y_ref is not None:
            if pd.api.types.is_numeric_dtype(y_ref) and y_ref.nunique() <= 2:
                # Target binário
                stats['target'] = {
                    'type': 'binary',
                    'mean': y_ref.mean(),
                    'positive_rate': y_ref.mean(),
                    'classes': sorted(y_ref.unique().tolist())
                }
            elif y_ref.nunique() <= 10:
                # Target categórico
                value_counts = y_ref.value_counts(normalize=True)
                stats['target'] = {
                    'type': 'categorical',
                    'classes': value_counts.index.tolist(),
                    'frequencies': value_counts.values.tolist()
                }
            else:
                # Target numérico contínuo
                stats['target'] = {
                    'type': 'continuous',
                    'mean': y_ref.mean(),
                    'median': y_ref.median(),
                    'std': y_ref.std(),
                    'min': y_ref.min(),
                    'max': y_ref.max()
                }

        # Calcular também as predições de referência
        if y_ref is not None:
            try:
                y_pred_proba = self.model.predict_proba(X_ref)[:, 1]
                stats['predictions'] = {
                    'mean': np.mean(y_pred_proba),
                    'median': np.median(y_pred_proba),
                    'std': np.std(y_pred_proba),
                    'min': np.min(y_pred_proba),
                    'max': np.max(y_pred_proba),
                    'histogram': np.histogram(y_pred_proba, bins=self.n_bins)[0].tolist(),
                    'bins': np.histogram_bin_edges(y_pred_proba, bins=self.n_bins).tolist()
                }

                # Calcular métricas de performance de referência
                y_pred = (y_pred_proba >= self.threshold).astype(int)

                stats['metrics'] = {
                    'auc': roc_auc_score(y_ref, y_pred_proba),
                    'accuracy': accuracy_score(y_ref, y_pred),
                    'precision': precision_score(y_ref, y_pred, zero_division=0),
                    'recall': recall_score(y_ref, y_pred, zero_division=0),
                    'f1': f1_score(y_ref, y_pred, zero_division=0),
                    'avg_precision': average_precision_score(y_ref, y_pred_proba)
                }

            except Exception as e:
                logger.warning(f"Erro ao calcular predições de referência: {str(e)}")

        return stats

    def check_data_drift(self, new_data: pd.DataFrame) -> Tuple[Dict[str, DriftMetrics], float]:
        """
        Detecta mudanças na distribuição das features entre os dados de referência e novos dados.

        Args:
            new_data: DataFrame com novos dados

        Returns:
            Tuple contendo dicionário com métricas de drift por feature e score geral de drift
        """
        logger.info("Verificando data drift...")

        # Separar features e target
        target_col = self.metadata.get('target', 'inadimplente')

        X_new = new_data.drop(columns=[target_col]) if target_col in new_data.columns else new_data

        # Aplicar transformações se feature builder disponível
        if self.feature_builder:
            try:
                X_new_transformed = self.feature_builder.transform(X_new)
                # Verificar se a transformação foi bem-sucedida
                if X_new_transformed.shape[0] == X_new.shape[0]:
                    X_new = X_new_transformed
                else:
                    logger.warning("Transformação de features resultou em DataFrame com número diferente de linhas.")
            except Exception as e:
                logger.warning(f"Erro ao aplicar feature builder: {str(e)}")

        # Inicializar resultados
        drift_metrics = {}
        drifted_features = []
        drift_scores = []

        # Verificar cada feature presente nos dados de referência e nos novos dados
        for column in X_new.columns:
            if column not in self.reference_stats:
                logger.warning(f"Feature '{column}' não encontrada nas estatísticas de referência.")
                continue

            ref_stats = self.reference_stats[column]
            col_type = ref_stats['type']

            # Inicializar métricas de drift
            metrics = DriftMetrics(feature_name=column)

            if col_type == 'numeric':
                # Variável numérica
                metrics = self._check_numeric_drift(X_new[column], ref_stats, metrics)

            else:
                # Variável categórica
                metrics = self._check_categorical_drift(X_new[column], ref_stats, metrics)

            # Avaliar severidade do drift
            if metrics.psi < 0.1:
                metrics.drift_severity = "Nenhum"
            elif metrics.psi < 0.2:
                metrics.drift_severity = "Baixo"
            elif metrics.psi < 0.3:
                metrics.drift_severity = "Médio"
            elif metrics.psi < 0.5:
                metrics.drift_severity = "Alto"
            else:
                metrics.drift_severity = "Crítico"

            # Estabelecer flag de drift
            metrics.drift_detected = metrics.psi >= self.drift_threshold

            # Armazenar métricas
            drift_metrics[column] = metrics

            # Acompanhar features com drift e scores
            if metrics.drift_detected:
                drifted_features.append(column)

            drift_scores.append(metrics.psi)

        # Calcular score geral de drift (média dos PSI)
        overall_drift_score = np.mean(drift_scores) if drift_scores else 0

        logger.info(f"Data drift detectado em {len(drifted_features)} features.")
        logger.info(f"Score geral de drift: {overall_drift_score:.4f}")

        if drifted_features:
            logger.info(f"Features com drift: {', '.join(drifted_features)}")

        return drift_metrics, overall_drift_score

    def _check_numeric_drift(self, new_data: pd.Series, ref_stats: Dict,
                             metrics: DriftMetrics) -> DriftMetrics:
        """
        Calcula métricas de drift para uma variável numérica.

        Args:
            new_data: Series com novos dados
            ref_stats: Estatísticas de referência
            metrics: Objeto para armazenar métricas

        Returns:
            Objeto atualizado com métricas de drift
        """
        # Calcular estatísticas básicas
        metrics.mean_diff = abs(new_data.mean() - ref_stats['mean']) / max(ref_stats['std'], 1e-6)
        metrics.std_diff = abs(new_data.std() - ref_stats['std']) / max(ref_stats['std'], 1e-6)

        # Calcular PSI (Population Stability Index)
        # Usar os mesmos bins que os dados de referência
        bins = ref_stats['bins']
        ref_hist = np.array(ref_stats['histogram']) / sum(ref_stats['histogram'])

        # Calcular histograma dos novos dados
        new_hist, _ = np.histogram(new_data.dropna(), bins=bins)
        new_hist = new_hist / sum(new_hist)

        # Evitar divisão por zero ou log de zero
        ref_hist = np.maximum(ref_hist, 1e-6)
        new_hist = np.maximum(new_hist, 1e-6)

        # Calcular PSI
        psi = np.sum((new_hist - ref_hist) * np.log(new_hist / ref_hist))
        metrics.psi = psi

        # Calcular Jensen-Shannon distance
        metrics.js_distance = jensenshannon(new_hist, ref_hist)

        # Calcular Kolmogorov-Smirnov test
        try:
            ks_stat, ks_pvalue = stats.ks_2samp(
                new_data.dropna().values,
                np.random.choice(
                    np.linspace(ref_stats['min'], ref_stats['max'], 1000),
                    size=min(1000, len(new_data)),
                    p=ref_hist / sum(ref_hist)
                )
            )
            metrics.ks_statistic = ks_stat
            metrics.ks_pvalue = ks_pvalue
        except Exception as e:
            logger.warning(f"Erro ao calcular teste KS para feature '{metrics.feature_name}': {str(e)}")

        return metrics

    def _check_categorical_drift(self, new_data: pd.Series, ref_stats: Dict,
                                 metrics: DriftMetrics) -> DriftMetrics:
        """
        Calcula métricas de drift para uma variável categórica.

        Args:
            new_data: Series com novos dados
            ref_stats: Estatísticas de referência
            metrics: Objeto para armazenar métricas

        Returns:
            Objeto atualizado com métricas de drift
        """
        # Obter categorias e frequências da referência
        ref_categories = ref_stats['categories']
        ref_freqs = np.array(ref_stats['frequencies'])

        # Calcular distribuição dos novos dados
        value_counts = new_data.value_counts(normalize=True, dropna=False)

        # Criar arrays alinhados para comparação
        new_freqs = np.zeros(len(ref_categories))

        for i, category in enumerate(ref_categories):
            new_freqs[i] = value_counts.get(category, 0)

        # Normalizar para garantir que soma = 1
        new_freqs = new_freqs / np.sum(new_freqs)

        # Evitar divisão por zero ou log de zero
        ref_freqs = np.maximum(ref_freqs, 1e-6)
        new_freqs = np.maximum(new_freqs, 1e-6)

        # Calcular PSI
        psi = np.sum((new_freqs - ref_freqs) * np.log(new_freqs / ref_freqs))
        metrics.psi = psi

        # Calcular Jensen-Shannon distance
        metrics.js_distance = jensenshannon(new_freqs, ref_freqs)

        # Calcular diferença de proporções para as top categorias
        if len(ref_categories) > 0:
            top_cat_idx = np.argmax(ref_freqs)
            metrics.mean_diff = abs(new_freqs[top_cat_idx] - ref_freqs[top_cat_idx])

        # Não é possível calcular KS teste para categóricas
        metrics.ks_statistic = 0
        metrics.ks_pvalue = 1.0

        return metrics

    def check_target_drift(self, new_data: pd.DataFrame) -> float:
        """
        Detecta mudanças na distribuição do target.

        Args:
            new_data: DataFrame com novos dados incluindo coluna target

        Returns:
            Score de drift do target (0-1)
        """
        logger.info("Verificando target drift...")

        target_col = self.metadata.get('target', 'inadimplente')

        if target_col not in new_data.columns:
            logger.warning(f"Coluna target '{target_col}' não encontrada nos novos dados.")
            return 0.0

        if 'target' not in self.reference_stats:
            logger.warning("Estatísticas de referência do target não disponíveis.")
            return 0.0

        target_stats = self.reference_stats['target']
        target_type = target_stats['type']

        y_new = new_data[target_col]

        if target_type == 'binary':
            # Target binário - calcular diferença na taxa positiva
            new_positive_rate = y_new.mean()
            ref_positive_rate = target_stats['positive_rate']

            # Calcular PSI para distribuição binária
            ref_dist = np.array([1 - ref_positive_rate, ref_positive_rate])
            new_dist = np.array([1 - new_positive_rate, new_positive_rate])

            # Evitar divisão por zero ou log de zero
            ref_dist = np.maximum(ref_dist, 1e-6)
            new_dist = np.maximum(new_dist, 1e-6)

            psi = np.sum((new_dist - ref_dist) * np.log(new_dist / ref_dist))

            # Calcular JS Distance
            js_distance = jensenshannon(new_dist, ref_dist)

            logger.info(f"Target drift - PSI: {psi:.4f}, JS Distance: {js_distance:.4f}")
            logger.info(f"Referência: taxa positiva = {ref_positive_rate:.4f}, Atual: {new_positive_rate:.4f}")

            return psi

        elif target_type == 'categorical':
            # Target categórico - comparar distribuições
            ref_categories = target_stats['classes']
            ref_freqs = np.array(target_stats['frequencies'])

            # Calcular distribuição dos novos dados
            value_counts = y_new.value_counts(normalize=True)

            # Criar arrays alinhados para comparação
            new_freqs = np.zeros(len(ref_categories))

            for i, category in enumerate(ref_categories):
                new_freqs[i] = value_counts.get(category, 0)

            # Normalizar para garantir que soma = 1
            new_freqs = new_freqs / np.sum(new_freqs)

            # Evitar divisão por zero ou log de zero
            ref_freqs = np.maximum(ref_freqs, 1e-6)
            new_freqs = np.maximum(new_freqs, 1e-6)

            # Calcular PSI
            psi = np.sum((new_freqs - ref_freqs) * np.log(new_freqs / ref_freqs))

            # Calcular JS Distance
            js_distance = jensenshannon(new_freqs, ref_freqs)

            logger.info(f"Target drift - PSI: {psi:.4f}, JS Distance: {js_distance:.4f}")

            return psi

        else:  # Continuous
            # Target contínuo - comparar distribuições
            # Discretizar para calcular PSI
            ref_mean = target_stats['mean']
            ref_std = target_stats.get('std', 1.0)

            try:
                # Usar KBinsDiscretizer para discretizar
                kbd = KBinsDiscretizer(n_bins=self.n_bins, encode='ordinal', strategy='quantile')
                y_new_disc = kbd.fit_transform(y_new.values.reshape(-1, 1)).flatten()

                # Calcular distribuição
                new_counts = np.bincount(y_new_disc.astype(int), minlength=self.n_bins)
                new_freqs = new_counts / np.sum(new_counts)

                # Distribuição uniforme como referência (assumindo quantis)
                ref_freqs = np.ones(self.n_bins) / self.n_bins

                # Calcular PSI entre as distribuições
                # Evitar divisão por zero ou log de zero
                ref_freqs = np.maximum(ref_freqs, 1e-6)
                new_freqs = np.maximum(new_freqs, 1e-6)

                psi = np.sum((new_freqs - ref_freqs) * np.log(new_freqs / ref_freqs))

                # Calcular JS Distance
                js_distance = jensenshannon(new_freqs, ref_freqs)

                # Calcular diferença normalizada de média
                mean_diff = abs(y_new.mean() - ref_mean) / max(ref_std, 1e-6)

                logger.info(f"Target drift - PSI: {psi:.4f}, JS Distance: {js_distance:.4f}")
                logger.info(f"Referência: média = {ref_mean:.4f}, Atual: {y_new.mean():.4f}")
                logger.info(f"Diferença normalizada: {mean_diff:.4f}")

                return psi

            except Exception as e:
                logger.warning(f"Erro ao calcular drift para target contínuo: {str(e)}")
                return 0.0

    def check_prediction_drift(self, new_data: pd.DataFrame) -> float:
        """
        Detecta mudanças na distribuição das predições do modelo.

        Args:
            new_data: DataFrame com novos dados

        Returns:
            Score de drift das predições (0-1)
        """
        logger.info("Verificando prediction drift...")

        # Separar features e target
        target_col = self.metadata.get('target', 'inadimplente')

        X_new = new_data.drop(columns=[target_col]) if target_col in new_data.columns else new_data

        # Aplicar transformações se feature builder disponível
        if self.feature_builder:
            try:
                X_new_transformed = self.feature_builder.transform(X_new)
                # Verificar se a transformação foi bem-sucedida
                if X_new_transformed.shape[0] == X_new.shape[0]:
                    X_new = X_new_transformed
                else:
                    logger.warning("Transformação de features resultou em DataFrame com número diferente de linhas.")
            except Exception as e:
                logger.warning(f"Erro ao aplicar feature builder: {str(e)}")

        # Verificar se temos estatísticas de predições de referência
        if 'predictions' not in self.reference_stats:
            logger.warning("Estatísticas de predições de referência não disponíveis.")
            return 0.0

        # Calcular predições para os novos dados
        try:
            y_pred_proba = self.model.predict_proba(X_new)[:, 1]
        except Exception as e:
            logger.warning(f"Erro ao calcular predições: {str(e)}")
            return 0.0

        # Estatísticas de referência
        ref_stats = self.reference_stats['predictions']
        ref_mean = ref_stats['mean']
        ref_std = ref_stats['std']
        ref_bins = ref_stats['bins']
        ref_hist = np.array(ref_stats['histogram']) / sum(ref_stats['histogram'])

        # Calcular histograma para novas predições
        new_hist, _ = np.histogram(y_pred_proba, bins=ref_bins)
        new_hist = new_hist / sum(new_hist)

        # Evitar divisão por zero ou log de zero
        ref_hist = np.maximum(ref_hist, 1e-6)
        new_hist = np.maximum(new_hist, 1e-6)

        # Calcular PSI
        psi = np.sum((new_hist - ref_hist) * np.log(new_hist / ref_hist))

        # Calcular JS Distance
        js_distance = jensenshannon(new_hist, ref_hist)

        # Calcular diferença normalizada da média
        mean_diff = abs(np.mean(y_pred_proba) - ref_mean) / max(ref_std, 1e-6)

        logger.info(f"Prediction drift - PSI: {psi:.4f}, JS Distance: {js_distance:.4f}")
        logger.info(f"Referência: média = {ref_mean:.4f}, Atual: {np.mean(y_pred_proba):.4f}")
        logger.info(f"Diferença normalizada: {mean_diff:.4f}")

        return psi

    def check_performance_drift(self, new_data: pd.DataFrame) -> ModelHealthMetrics:
        """
        Detecta degradação na performance do modelo.

        Args:
            new_data: DataFrame com novos dados incluindo coluna target

        Returns:
            Métricas de saúde do modelo
        """
        logger.info("Verificando performance drift...")

        # Separar features e target
        target_col = self.metadata.get('target', 'inadimplente')

        if target_col not in new_data.columns:
            logger.warning(
                f"Coluna target '{target_col}' não encontrada nos novos dados. Impossível verificar performance.")
            return ModelHealthMetrics()

        y_new = new_data[target_col]
        X_new = new_data.drop(columns=[target_col])

        # Aplicar transformações se feature builder disponível
        if self.feature_builder:
            try:
                X_new_transformed = self.feature_builder.transform(X_new)
                # Verificar se a transformação foi bem-sucedida
                if X_new_transformed.shape[0] == X_new.shape[0]:
                    X_new = X_new_transformed
                else:
                    logger.warning("Transformação de features resultou em DataFrame com número diferente de linhas.")
            except Exception as e:
                logger.warning(f"Erro ao aplicar feature builder: {str(e)}")

        # Verificar se temos métricas de referência
        if 'metrics' not in self.reference_stats:
            logger.warning("Métricas de performance de referência não disponíveis.")
            return ModelHealthMetrics()

        # Calcular predições para os novos dados
        try:
            y_pred_proba = self.model.predict_proba(X_new)[:, 1]
            y_pred = (y_pred_proba >= self.threshold).astype(int)
        except Exception as e:
            logger.warning(f"Erro ao calcular predições: {str(e)}")
            return ModelHealthMetrics()

        # Calcular métricas atuais
        metrics = ModelHealthMetrics()

        try:
            metrics.auc = roc_auc_score(y_new, y_pred_proba)
            metrics.accuracy = accuracy_score(y_new, y_pred)
            metrics.precision = precision_score(y_new, y_pred, zero_division=0)
            metrics.recall = recall_score(y_new, y_pred, zero_division=0)
            metrics.f1 = f1_score(y_new, y_pred, zero_division=0)
            metrics.average_precision = average_precision_score(y_new, y_pred_proba)
        except Exception as e:
            logger.warning(f"Erro ao calcular métricas: {str(e)}")

        # Obter métricas de referência
        ref_metrics = self.reference_stats['metrics']

        # Calcular diferenças de performance
        auc_diff = ref_metrics['auc'] - metrics.auc
        f1_diff = ref_metrics['f1'] - metrics.f1
        recall_diff = ref_metrics['recall'] - metrics.recall
        precision_diff = ref_metrics['precision'] - metrics.precision

        # Verificar se há degradação significativa
        degradation_scores = [
            auc_diff if auc_diff > 0 else 0,
            f1_diff if f1_diff > 0 else 0,
            recall_diff if recall_diff > 0 else 0,
            precision_diff if precision_diff > 0 else 0
        ]

        # Normalizar os scores
        degradation_scores = [min(score / self.performance_decline_threshold, 1.0) for score in degradation_scores]
        overall_degradation = np.mean(degradation_scores)

        # Log das métricas
        logger.info(f"Performance atual - AUC: {metrics.auc:.4f}, F1: {metrics.f1:.4f}")
        logger.info(f"Referência - AUC: {ref_metrics['auc']:.4f}, F1: {ref_metrics['f1']:.4f}")
        logger.info(f"Degradação - AUC diff: {auc_diff:.4f}, F1 diff: {f1_diff:.4f}")
        logger.info(f"Score de degradação: {overall_degradation:.4f}")

        # Definir status de saúde com base na degradação
        metrics.stability_score = 1.0 - overall_degradation

        if overall_degradation < 0.2:
            metrics.health_status = "Ok"
        elif overall_degradation < 0.5:
            metrics.health_status = "Atenção"
        else:
            metrics.health_status = "Crítico"

        return metrics

    def monitor_model_health(self, new_data: pd.DataFrame) -> ModelHealthMetrics:
        """
        Executa todas as verificações de saúde do modelo.

        Args:
            new_data: DataFrame com novos dados incluindo coluna target

        Returns:
            Métricas de saúde do modelo
        """
        logger.info("Iniciando monitoramento de saúde do modelo...")

        # Verificar data drift
        data_drift_metrics, data_drift_score = self.check_data_drift(new_data)

        # Identificar features com drift
        drifted_features = [
            feature for feature, metrics in data_drift_metrics.items()
            if metrics.drift_detected
        ]

        # Verificar target drift
        target_drift_score = self.check_target_drift(new_data)

        # Verificar prediction drift
        prediction_drift_score = self.check_prediction_drift(new_data)

        # Verificar performance drift
        performance_metrics = self.check_performance_drift(new_data)

        # Atualizar métricas consolidadas
        performance_metrics.data_drift_score = data_drift_score
        performance_metrics.target_drift_score = target_drift_score
        performance_metrics.prediction_drift_score = prediction_drift_score
        performance_metrics.data_drift_features = drifted_features
        performance_metrics.timestamp = datetime.now()

        # Determinar status geral de saúde
        combined_drift_score = np.mean([
            data_drift_score,
            target_drift_score,
            prediction_drift_score
        ])

        stability_score = 1.0 - max(combined_drift_score, 1.0 - performance_metrics.stability_score)
        performance_metrics.stability_score = stability_score

        if stability_score > 0.8:
            performance_metrics.health_status = "Ok"
        elif stability_score > 0.5:
            performance_metrics.health_status = "Atenção"
        else:
            performance_metrics.health_status = "Crítico"

        # Armazenar métricas no histórico
        self.metrics_history.append(performance_metrics)

        # Log de saúde geral
        logger.info(f"Status de saúde do modelo: {performance_metrics.health_status}")
        logger.info(f"Score de estabilidade: {stability_score:.4f}")

        return performance_metrics

    def generate_drift_report(self, output_format: str = 'html') -> str:
        """
        Gera relatório de drift e degradação.

        Args:
            output_format: Formato do relatório ('html', 'md', 'txt')

        Returns:
            Caminho para o relatório gerado
        """
        if not self.metrics_history:
            logger.warning("Nenhum histórico de métricas disponível para gerar relatório.")
            return ""

        logger.info(f"Gerando relatório de drift no formato {output_format}...")

        # Usar métricas mais recentes
        metrics = self.metrics_history[-1]

        # Criar diretório para relatórios
        reports_dir = os.path.join(self.output_dir, 'reports')
        os.makedirs(reports_dir, exist_ok=True)

        # Definir caminho do relatório
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if output_format == 'html':
            report_path = os.path.join(reports_dir, f"drift_report_{timestamp}.html")
            content = self._generate_html_report(metrics)
        elif output_format == 'md':
            report_path = os.path.join(reports_dir, f"drift_report_{timestamp}.md")
            content = self._generate_md_report(metrics)
        else:  # txt
            report_path = os.path.join(reports_dir, f"drift_report_{timestamp}.txt")
            content = self._generate_txt_report(metrics)

        # Salvar relatório
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(content)

        logger.info(f"Relatório de drift salvo em: {report_path}")

        # Gerar gráficos de monitoramento se tivermos histórico suficiente
        if len(self.metrics_history) >= 2:
            self._plot_metrics_history()

        return report_path

    def _generate_html_report(self, metrics: ModelHealthMetrics) -> str:
        """Gera relatório de drift em formato HTML."""
        model_name = self.metadata.get('model_name', 'Modelo')

        # Formatações baseadas no status
        health_status_color = {
            'Ok': '#27ae60',
            'Atenção': '#f39c12',
            'Crítico': '#e74c3c'
        }.get(metrics.health_status, '#7f8c8d')

        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Relatório de Monitoramento de Modelo - {model_name}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }}
        h1, h2, h3 {{ color: #2c3e50; }}
        .container {{ max-width: 1000px; margin: 0 auto; }}
        .header {{ background-color: #3498db; color: white; padding: 20px; margin-bottom: 20px; }}
        .section {{ margin-bottom: 30px; }}
        .metrics {{ display: flex; flex-wrap: wrap; gap: 20px; margin-bottom: 20px; }}
        .metric-card {{ background-color: #f8f9fa; border-radius: 5px; padding: 15px; min-width: 200px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
        .metric-title {{ font-weight: bold; margin-bottom: 5px; color: #7f8c8d; }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #2c3e50; }}
        .metric-context {{ font-size: 14px; color: #7f8c8d; }}
        .good {{ color: #27ae60; }}
        .medium {{ color: #f39c12; }}
        .bad {{ color: #e74c3c; }}
        table {{ width: 100%; border-collapse: collapse; margin-bottom: 20px; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #f2f2f2; }}
        tr:hover {{ background-color: #f5f5f5; }}
        .status-badge {{
            display: inline-block;
            padding: 5px 10px;
            border-radius: 15px;
            background-color: {health_status_color};
            color: white;
            font-weight: bold;
        }}
        .footer {{ text-align: center; margin-top: 50px; color: #7f8c8d; font-size: 14px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Relatório de Monitoramento de Modelo</h1>
            <p>Modelo: {model_name} | Data: {metrics.timestamp.strftime('%d/%m/%Y %H:%M')}</p>
        </div>

        <div class="section">
            <h2>Status do Modelo <span class="status-badge">{metrics.health_status}</span></h2>

            <div class="metrics">
                <div class="metric-card">
                    <div class="metric-title">Estabilidade</div>
                    <div class="metric-value {self._get_metric_color(metrics.stability_score)}">
                        {metrics.stability_score:.1%}
                    </div>
                    <div class="metric-context">Score de estabilidade geral</div>
                </div>

                <div class="metric-card">
                    <div class="metric-title">Data Drift</div>
                    <div class="metric-value {self._get_metric_color(1 - metrics.data_drift_score, inverse=True)}">
                        {metrics.data_drift_score:.2f}
                    </div>
                    <div class="metric-context">Score de mudança nas features</div>
                </div>

                <div class="metric-card">
                    <div class="metric-title">Target Drift</div>
                    <div class="metric-value {self._get_metric_color(1 - metrics.target_drift_score, inverse=True)}">
                        {metrics.target_drift_score:.2f}
                    </div>
                    <div class="metric-context">Score de mudança no target</div>
                </div>

                <div class="metric-card">
                    <div class="metric-title">Prediction Drift</div>
                    <div class="metric-value {self._get_metric_color(1 - metrics.prediction_drift_score, inverse=True)}">
                        {metrics.prediction_drift_score:.2f}
                    </div>
                    <div class="metric-context">Score de mudança nas predições</div>
                </div>
            </div>
        </div>

        <div class="section">
            <h2>Métricas de Performance</h2>

            <div class="metrics">
                <div class="metric-card">
                    <div class="metric-title">AUC</div>
                    <div class="metric-value {self._get_metric_color(metrics.auc)}">
                        {metrics.auc:.4f}
                    </div>
                    <div class="metric-context">Área sob a curva ROC</div>
                </div>

                <div class="metric-card">
                    <div class="metric-title">F1-Score</div>
                    <div class="metric-value {self._get_metric_color(metrics.f1)}">
                        {metrics.f1:.4f}
                    </div>
                    <div class="metric-context">Média harmônica entre precisão e recall</div>
                </div>

                <div class="metric-card">
                    <div class="metric-title">Precisão</div>
                    <div class="metric-value {self._get_metric_color(metrics.precision)}">
                        {metrics.precision:.4f}
                    </div>
                    <div class="metric-context">Positivos verdadeiros / positivos previstos</div>
                </div>

                <div class="metric-card">
                    <div class="metric-title">Recall</div>
                    <div class="metric-value {self._get_metric_color(metrics.recall)}">
                        {metrics.recall:.4f}
                    </div>
                    <div class="metric-context">Positivos verdadeiros / positivos reais</div>
                </div>
            </div>
        </div>
"""

        # Adicionar seção de features com drift se aplicável
        if metrics.data_drift_features:
            html += f"""
        <div class="section">
            <h2>Features com Drift Detectado</h2>
            <table>
                <tr>
                    <th>Feature</th>
                </tr>
"""
            for feature in metrics.data_drift_features:
                html += f"""
                <tr>
                    <td>{feature}</td>
                </tr>
"""
            html += """
            </table>
        </div>
"""

        html += f"""
        <div class="section">
            <h2>Recomendações</h2>
            <ul>
"""
        # Adicionar recomendações com base no status
        if metrics.health_status == "Crítico":
            html += """
                <li><strong>Reconsidere o uso deste modelo em produção.</strong> A performance e estabilidade estão significativamente comprometidas.</li>
                <li>Inicie um processo de retreinamento completo com dados atualizados.</li>
                <li>Investigue as features com drift para entender mudanças fundamentais no domínio.</li>
                <li>Verifique possíveis falhas na coleta ou processamento de dados.</li>
"""
        elif metrics.health_status == "Atenção":
            html += """
                <li>Monitore o modelo com maior frequência para acompanhar a evolução do drift.</li>
                <li>Considere retreinar o modelo se a degradação persistir.</li>
                <li>Valide se as features com drift representam mudanças reais no comportamento ou problemas de dados.</li>
                <li>Avalie o impacto financeiro da degradação atual na tomada de decisão.</li>
"""
        else:  # Ok
            html += """
                <li>Mantenha o monitoramento regular para detecção precoce de drift.</li>
                <li>Continue coletando dados para validação contínua.</li>
                <li>Documente a estabilidade atual como referência para comparações futuras.</li>
"""

        html += f"""
            </ul>
        </div>

        <div class="footer">
            <p>Relatório gerado automaticamente em {datetime.now().strftime('%d/%m/%Y %H:%M')}</p>
        </div>
    </div>
</body>
</html>
"""
        return html

    def _generate_md_report(self, metrics: ModelHealthMetrics) -> str:
        """Gera relatório de drift em formato Markdown."""
        model_name = self.metadata.get('model_name', 'Modelo')

        md = f"""# Relatório de Monitoramento de Modelo

**Modelo:** {model_name}  
**Data:** {metrics.timestamp.strftime('%d/%m/%Y %H:%M')}  
**Status:** {metrics.health_status}

## Status do Modelo

| Métrica | Valor | Interpretação |
|---------|-------|---------------|
| Estabilidade | {metrics.stability_score:.1%} | Score de estabilidade geral |
| Data Drift | {metrics.data_drift_score:.2f} | Score de mudança nas features |
| Target Drift | {metrics.target_drift_score:.2f} | Score de mudança no target |
| Prediction Drift | {metrics.prediction_drift_score:.2f} | Score de mudança nas predições |

## Métricas de Performance

| Métrica | Valor | Interpretação |
|---------|-------|---------------|
| AUC | {metrics.auc:.4f} | Área sob a curva ROC |
| F1-Score | {metrics.f1:.4f} | Média harmônica entre precisão e recall |
| Precisão | {metrics.precision:.4f} | Positivos verdadeiros / positivos previstos |
| Recall | {metrics.recall:.4f} | Positivos verdadeiros / positivos reais |
"""

        # Adicionar seção de features com drift se aplicável
        if metrics.data_drift_features:
            md += f"""
## Features com Drift Detectado

| Feature |
|---------|
"""
            for feature in metrics.data_drift_features:
                md += f"| {feature} |\n"

        md += """
## Recomendações

"""
        # Adicionar recomendações com base no status
        if metrics.health_status == "Crítico":
            md += """
* **Reconsidere o uso deste modelo em produção.** A performance e estabilidade estão significativamente comprometidas.
* Inicie um processo de retreinamento completo com dados atualizados.
* Investigue as features com drift para entender mudanças fundamentais no domínio.
* Verifique possíveis falhas na coleta ou processamento de dados.
"""
        elif metrics.health_status == "Atenção":
            md += """
* Monitore o modelo com maior frequência para acompanhar a evolução do drift.
* Considere retreinar o modelo se a degradação persistir.
* Valide se as features com drift representam mudanças reais no comportamento ou problemas de dados.
* Avalie o impacto financeiro da degradação atual na tomada de decisão.
"""
        else:  # Ok
            md += """
* Mantenha o monitoramento regular para detecção precoce de drift.
* Continue coletando dados para validação contínua.
* Documente a estabilidade atual como referência para comparações futuras.
"""

        md += f"""
---

Relatório gerado automaticamente em {datetime.now().strftime('%d/%m/%Y %H:%M')}
"""
        return md

    def _generate_txt_report(self, metrics: ModelHealthMetrics) -> str:
        """Gera relatório de drift em formato texto plano."""
        model_name = self.metadata.get('model_name', 'Modelo')

        txt = f"""RELATÓRIO DE MONITORAMENTO DE MODELO
======================================

Modelo: {model_name}
Data: {metrics.timestamp.strftime('%d/%m/%Y %H:%M')}
Status: {metrics.health_status}

STATUS DO MODELO
---------------
Estabilidade: {metrics.stability_score:.1%}
Data Drift: {metrics.data_drift_score:.2f}
Target Drift: {metrics.target_drift_score:.2f}
Prediction Drift: {metrics.prediction_drift_score:.2f}

MÉTRICAS DE PERFORMANCE
--------------------------
AUC: {metrics.auc:.4f}
F1-Score: {metrics.f1:.4f}
Precisão: {metrics.precision:.4f}
Recall: {metrics.recall:.4f}
"""

        # Adicionar seção de features com drift se aplicável
        if metrics.data_drift_features:
            txt += f"""
FEATURES COM DRIFT DETECTADO
----------------
"""
            for feature in metrics.data_drift_features:
                txt += f"- {feature}\n"

        txt += """
RECOMENDAÇÕES
------------
"""
        # Adicionar recomendações com base no status
        if metrics.health_status == "Crítico":
            txt += """
- RECONSIDERE O USO DESTE MODELO EM PRODUÇÃO. A performance e estabilidade estão significativamente comprometidas.
- Inicie um processo de retreinamento completo com dados atualizados.
- Investigue as features com drift para entender mudanças fundamentais no domínio.
- Verifique possíveis falhas na coleta ou processamento de dados.
"""
        elif metrics.health_status == "Atenção":
            txt += """
- Monitore o modelo com maior frequência para acompanhar a evolução do drift.
- Considere retreinar o modelo se a degradação persistir.
- Valide se as features com drift representam mudanças reais no comportamento ou problemas de dados.
- Avalie o impacto financeiro da degradação atual na tomada de decisão.
"""
        else:  # Ok
            txt += """
- Mantenha o monitoramento regular para detecção precoce de drift.
- Continue coletando dados para validação contínua.
- Documente a estabilidade atual como referência para comparações futuras.
"""

        txt += f"""
-------------------------------
Relatório gerado automaticamente em {datetime.now().strftime('%d/%m/%Y %H:%M')}
"""
        return txt

    def _get_metric_color(self, value: float, inverse: bool = False) -> str:
        """
        Determina a cor da métrica com base no valor.

        Args:
            value: Valor da métrica
            inverse: Se True, valores maiores são piores

        Returns:
            Classe CSS para cor
        """
        if inverse:
            if value < 0.6:
                return 'bad'
            elif value < 0.8:
                return 'medium'
            else:
                return 'good'
        else:
            if value > 0.8:
                return 'good'
            elif value > 0.6:
                return 'medium'
            else:
                return 'bad'

    def _plot_metrics_history(self) -> None:
        """
        Gera gráficos do histórico de métricas para acompanhamento ao longo do tempo.
        """
        if len(self.metrics_history) < 2:
            logger.warning("Histórico insuficiente para gerar gráficos de tendência.")
            return

        # Extrair dados do histórico
        timestamps = [m.timestamp for m in self.metrics_history]
        stability_scores = [m.stability_score for m in self.metrics_history]
        data_drift_scores = [m.data_drift_score for m in self.metrics_history]
        target_drift_scores = [m.target_drift_score for m in self.metrics_history]
        prediction_drift_scores = [m.prediction_drift_score for m in self.metrics_history]
        auc_scores = [m.auc for m in self.metrics_history]
        f1_scores = [m.f1 for m in self.metrics_history]

        # Criar diretório para gráficos
        plots_dir = os.path.join(self.output_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)

        # Formato para datas
        date_fmt = '%d/%m'

        # Plotar tendências de drift
        plt.figure(figsize=(12, 6))
        plt.plot(timestamps, stability_scores, 'o-', label='Estabilidade', color='#3498db')
        plt.plot(timestamps, [1 - s for s in data_drift_scores], 'o-', label='Data Estabilidade', color='#2ecc71')
        plt.plot(timestamps, [1 - s for s in target_drift_scores], 'o-', label='Target Estabilidade', color='#e74c3c')
        plt.plot(timestamps, [1 - s for s in prediction_drift_scores], 'o-', label='Prediction Estabilidade',
                 color='#f39c12')

        plt.axhline(y=0.8, color='green', linestyle='--', alpha=0.5, label='Ok')
        plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Crítico')

        plt.xlabel('Data')
        plt.ylabel('Score de Estabilidade')
        plt.title('Tendência de Estabilidade do Modelo')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1.05)

        # Formatar eixo x com datas
        plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter(date_fmt))
        plt.xticks(rotation=45)

        # Salvar gráfico
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'stability_trend.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # Plotar tendências de performance
        plt.figure(figsize=(12, 6))
        plt.plot(timestamps, auc_scores, 'o-', label='AUC', color='#3498db')
        plt.plot(timestamps, f1_scores, 'o-', label='F1-Score', color='#2ecc71')

        # Adicionar referências
        if 'metrics' in self.reference_stats:
            ref_auc = self.reference_stats['metrics']['auc']
            ref_f1 = self.reference_stats['metrics']['f1']

            plt.axhline(y=ref_auc, color='#3498db', linestyle='--', alpha=0.5, label='AUC de Referência')
            plt.axhline(y=ref_f1, color='#2ecc71', linestyle='--', alpha=0.5, label='F1 de Referência')

            # Adicionar limites de degradação
            plt.axhline(y=ref_auc * (1 - self.performance_decline_threshold), color='#3498db',
                        linestyle=':', alpha=0.5, label='Limite de Degradação AUC')
            plt.axhline(y=ref_f1 * (1 - self.performance_decline_threshold), color='#2ecc71',
                        linestyle=':', alpha=0.5, label='Limite de Degradação F1')

        plt.xlabel('Data')
        plt.ylabel('Score')
        plt.title('Tendência de Performance do Modelo')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Formatar eixo x com datas
        plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter(date_fmt))
        plt.xticks(rotation=45)

        # Salvar gráfico
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'performance_trend.png'), dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Gráficos de tendência gerados em: {plots_dir}")


def main():
    """
    Demonstração de uso do monitor de modelo.
    Esta função principal executa quando o script é chamado diretamente.
    """
    import argparse
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier

    # Configurar parser de argumentos
    parser = argparse.ArgumentParser(description='Demonstração de monitoramento de modelo')
    parser.add_argument('--model_path', type=str, help='Caminho para o modelo serializado')
    parser.add_argument('--reference_data_path', type=str, help='Caminho para os dados de referência')
    parser.add_argument('--new_data_path', type=str, help='Caminho para os novos dados')
    parser.add_argument('--feature_builder_path', type=str, help='Caminho para o feature builder (opcional)')
    parser.add_argument('--metadata_path', type=str, help='Caminho para os metadados do modelo (opcional)')
    parser.add_argument('--demo', action='store_true', default=True, help='Executar demonstração com dados sintéticos')
    parser.add_argument('--report_format', type=str, default='html', choices=['html', 'md', 'txt'],
                        help='Formato do relatório de drift')

    args = parser.parse_args()

    # Variáveis para caminhos de arquivos
    model_path = None
    reference_data_path = None
    new_data_path = None

    if args.model_path and args.reference_data_path and args.new_data_path:
        # Usar arquivos fornecidos
        model_path = args.model_path
        reference_data_path = args.reference_data_path
        new_data_path = args.new_data_path

    elif args.demo:
        logger.info("Executando demonstração com dados sintéticos...")

        # Criar diretório temporário para salvar arquivos
        project_root = get_project_root()
        temp_dir = os.path.join(project_root, 'temp')
        os.makedirs(temp_dir, exist_ok=True)

        # Gerar dados sintéticos
        X, y = make_classification(n_samples=1000, n_features=10, n_informative=5,
                                   n_redundant=2, n_classes=2, weights=[0.8, 0.2],
                                   random_state=42)

        # Dividir dados
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

        # Treinar modelo
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Salvar modelo
        model_path = os.path.join(temp_dir, 'model_demo.joblib')
        joblib.dump(model, model_path)

        # Criar e salvar dados de referência (dados de treino)
        train_df = pd.DataFrame(X_train, columns=[f'feature_{i}' for i in range(X_train.shape[1])])
        train_df['inadimplente'] = y_train

        reference_data_path = os.path.join(temp_dir, 'reference_data.csv')
        train_df.to_csv(reference_data_path, index=False)

        # Simular drift nos dados de teste
        X_test_drift = X_test.copy()

        # Introduzir drift em algumas features
        drift_features = [0, 2, 5]
        for feature in drift_features:
            # Adicionar drift na média e variância
            X_test_drift[:, feature] = X_test_drift[:, feature] * 1.5 + 0.5

        # Introduzir algumas mudanças na distribuição do target
        y_test_drift = y_test.copy()
        # Aumentar a taxa de inadimplência
        if sum(y_test) / len(y_test) < 0.3:
            # Mudar alguns negativos para positivos
            neg_indices = np.where(y_test == 0)[0]
            change_indices = np.random.choice(neg_indices, size=int(len(neg_indices) * 0.1), replace=False)
            y_test_drift[change_indices] = 1

        # Criar e salvar novos dados com drift
        test_df = pd.DataFrame(X_test_drift, columns=[f'feature_{i}' for i in range(X_test_drift.shape[1])])
        test_df['inadimplente'] = y_test_drift

        new_data_path = os.path.join(temp_dir, 'new_data_with_drift.csv')
        test_df.to_csv(new_data_path, index=False)

        # Criar metadados simplificados
        metadata = {
            'model_name': 'RandomForest_Demo',
            'model_type': 'RandomForestClassifier',
            'creation_date': datetime.now().strftime('%Y-%m-%d'),
            'features': [f'feature_{i}' for i in range(X_train.shape[1])],
            'target': 'inadimplente',
            'metrics': {
                'auc': 0.85,
                'accuracy': 0.8,
                'precision': 0.7,
                'recall': 0.6,
                'f1': 0.65
            },
            'thresholds': {
                'RandomForestClassifier': 0.5,
                'default': 0.5
            }
        }

        metadata_path = os.path.join(temp_dir, 'model_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)

        logger.info(f"Demonstração configurada. Modelo: {model_path}")
        logger.info(f"Dados de referência: {reference_data_path}")
        logger.info(f"Novos dados: {new_data_path}")
        logger.info(f"Metadados: {metadata_path}")

        # Usar o metadata_path gerado
        args.metadata_path = metadata_path

    else:
        logger.error("Nenhum dado fornecido e modo de demonstração desativado.")
        parser.print_help()
        return

    # Inicializar monitor de modelo
    try:
        monitor = ModelMonitor(
            model_path=model_path,
            reference_data_path=reference_data_path,
            feature_builder_path=args.feature_builder_path,
            model_metadata_path=args.metadata_path
        )

        # Carregar novos dados
        new_data = pd.read_csv(new_data_path)

        # Monitorar saúde do modelo
        health_metrics = monitor.monitor_model_health(new_data)

        # Gerar relatório de drift
        report_path = monitor.generate_drift_report(output_format=args.report_format)

        logger.info(f"\nMonitoramento concluído com sucesso!")
        logger.info(f"Status de saúde: {health_metrics.health_status}")
        logger.info(f"Relatório gerado em: {report_path}")

    except Exception as e:
        logger.error(f"Erro durante o monitoramento: {str(e)}")
        return


if __name__ == "__main__":
    main()