"""
Módulo para treinamento de modelos calibrados de inadimplência.
Implementa modelos com calibração de probabilidades para predições mais precisas.
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
import json
import logging
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Optional

# Modelos e calibração
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Métricas e validação
from sklearn.metrics import (
    precision_recall_curve, roc_auc_score,
    confusion_matrix, f1_score, average_precision_score,
    brier_score_loss, log_loss
)
from sklearn.calibration import calibration_curve

# Configurar logger
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

# Tentar importar PathManager
try:
    from src.utils.path_manager import PathManager
except ImportError:
    # Classe PathManager substituta
    class PathManager:
        """Versão simplificada do gerenciador de caminhos."""

        def __init__(self):
            """Inicializa o gerenciador de caminhos."""
            self.project_root = self._find_project_root()

        def _find_project_root(self) -> str:
            """Encontra o diretório raiz do projeto."""
            current_dir = os.path.dirname(os.path.abspath(__file__))
            root = os.path.abspath(os.path.join(current_dir, os.pardir, os.pardir))

            if os.path.exists(os.path.join(root, 'src')) and os.path.exists(os.path.join(root, 'models')):
                return root

            return os.getcwd()

        def get_data_path(self, subdir: str) -> str:
            """Retorna caminho para diretório de dados."""
            return os.path.join(self.project_root, "data", subdir)

        def get_model_path(self, subdir: str) -> str:
            """Retorna caminho para diretório de modelos."""
            return os.path.join(self.project_root, "models", subdir)

        def get_report_path(self, subdir: str, filename: Optional[str] = None) -> str:
            """Retorna caminho para diretório de relatórios."""
            path = os.path.join(self.project_root, "reports", subdir)
            if filename:
                path = os.path.join(path, filename)
            return path


class CalibratedModelTrainer:
    """
    Classe para treinar, avaliar e salvar modelos calibrados de predição de inadimplência.
    """

    def __init__(self, model_dir=None, eval_dir=None, default_threshold=0.5):
        """
        Inicializa o treinador de modelos calibrados.

        Args:
            model_dir: Diretório para salvar modelos treinados
            eval_dir: Diretório para salvar avaliações
            default_threshold: Limiar padrão para classificação
        """
        self.models = {}
        self.calibrators = {}
        self.evaluation_results = {}
        self.thresholds = {}
        self.default_threshold = default_threshold
        self.best_model_name = None

        # Configurar diretórios para salvar resultados
        self.path_manager = PathManager()
        self.model_dir = model_dir or os.path.join(self.path_manager.project_root, 'models', 'calibrated_models')
        self.eval_dir = eval_dir or os.path.join(self.path_manager.project_root, 'reports', 'model_evaluation')

        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.eval_dir, exist_ok=True)

        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def train_calibrated_logistic_regression(self, X_train, y_train, X_val=None, y_val=None,
                                             name="CalibratedLogisticRegression", **kwargs):
        """
        Treina uma regressão logística calibrada usando o método de Platt Scaling.

        Args:
            X_train: Features de treinamento
            y_train: Target de treinamento
            X_val: Features de validação (opcional)
            y_val: Target de validação (opcional)
            name: Nome do modelo
            **kwargs: Parâmetros adicionais para LogisticRegression

        Returns:
            self
        """
        logger.info(f"Treinando modelo calibrado: {name}")

        # Parâmetros padrão para LogisticRegression
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

        # Criar regressão logística como estimador
        estimator = LogisticRegression(**params)

        # Aplicar calibração usando Platt Scaling (sigmoid)
        calibrated_model = CalibratedClassifierCV(
            estimator=estimator,  # Changed from base_estimator to estimator
            method='sigmoid',  # Platt scaling
            cv=5,  # cross-validation para calibração
            n_jobs=-1  # usar todos os cores disponíveis
        )

        # Treinar o modelo
        calibrated_model.fit(X_train, y_train)

        # Armazenar modelo e calibrador
        self.models[name] = calibrated_model

        logger.info(f"Modelo {name} treinado com sucesso.")
        return self

    def train_calibrated_random_forest(self, X_train, y_train, X_val=None, y_val=None,
                                       name="CalibratedRandomForest", **kwargs):
        """
        Treina um Random Forest calibrado usando o método de isotonic regression.

        Args:
            X_train: Features de treinamento
            y_train: Target de treinamento
            X_val: Features de validação (opcional)
            y_val: Target de validação (opcional)
            name: Nome do modelo
            **kwargs: Parâmetros adicionais para RandomForestClassifier

        Returns:
            self
        """
        logger.info(f"Treinando modelo calibrado: {name}")

        # Parâmetros padrão para RandomForestClassifier
        params = {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'class_weight': 'balanced',
            'random_state': 42,
            'n_jobs': -1
        }

        # Atualizar com parâmetros fornecidos
        params.update(kwargs)

        # Criar random forest como estimador
        estimator = RandomForestClassifier(**params)

        # Aplicar calibração usando isotonic regression
        calibrated_model = CalibratedClassifierCV(
            estimator=estimator,  # Changed from base_estimator to estimator
            method='isotonic',  # isotonic regression
            cv=5,  # cross-validation para calibração
            n_jobs=-1  # usar todos os cores disponíveis
        )

        # Treinar o modelo
        calibrated_model.fit(X_train, y_train)

        # Armazenar modelo e calibrador
        self.models[name] = calibrated_model

        logger.info(f"Modelo {name} treinado com sucesso.")
        return self

    def train_calibrated_gradient_boosting(self, X_train, y_train, X_val=None, y_val=None,
                                           name="CalibratedGradientBoosting", **kwargs):
        """
        Treina um Gradient Boosting calibrado, utilizando calibração após o treinamento.

        Args:
            X_train: Features de treinamento
            y_train: Target de treinamento
            X_val: Features de validação (opcional)
            y_val: Target de validação (opcional)
            name: Nome do modelo
            **kwargs: Parâmetros adicionais para GradientBoostingClassifier

        Returns:
            self
        """
        logger.info(f"Treinando modelo calibrado: {name}")

        # Parâmetros padrão para GradientBoostingClassifier
        params = {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 3,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'subsample': 0.8,
            'random_state': 42
        }

        # Atualizar com parâmetros fornecidos
        params.update(kwargs)

        # Criar gradient boosting como estimador
        estimator = GradientBoostingClassifier(**params)

        # Para Gradient Boosting, testar ambos os métodos de calibração
        calibrated_model = CalibratedClassifierCV(
            estimator=estimator,  # Changed from base_estimator to estimator
            method='sigmoid',  # também pode usar 'isotonic'
            cv=5,
            n_jobs=-1
        )

        # Treinar o modelo
        calibrated_model.fit(X_train, y_train)

        # Armazenar modelo e calibrador
        self.models[name] = calibrated_model

        logger.info(f"Modelo {name} treinado com sucesso.")
        return self

    def train_svm_with_calibration(self, X_train, y_train, X_val=None, y_val=None,
                                   name="CalibratedSVM", **kwargs):
        """
        Treina um SVM com calibração de Platt para obter probabilidades bem calibradas.

        Args:
            X_train: Features de treinamento
            y_train: Target de treinamento
            X_val: Features de validação (opcional)
            y_val: Target de validação (opcional)
            name: Nome do modelo
            **kwargs: Parâmetros adicionais para LinearSVC

        Returns:
            self
        """
        logger.info(f"Treinando modelo calibrado: {name}")

        # Parâmetros padrão para LinearSVC
        params = {
            'C': 1.0,
            'class_weight': 'balanced',
            'random_state': 42,
            'max_iter': 1000,
            'dual': False
        }

        # Atualizar com parâmetros fornecidos
        params.update(kwargs)

        # Criar pipeline com normalização e SVM
        estimator = Pipeline([
            ('scaler', StandardScaler()),
            ('svm', LinearSVC(**params))
        ])

        # SVM não gera probabilidades naturalmente, então a calibração é essencial
        calibrated_model = CalibratedClassifierCV(
            estimator=estimator,  # Changed from base_estimator to estimator
            method='sigmoid',  # Platt scaling é mais adequado para SVM
            cv=5,
            n_jobs=-1
        )

        # Treinar o modelo
        calibrated_model.fit(X_train, y_train)

        # Armazenar modelo
        self.models[name] = calibrated_model

        logger.info(f"Modelo {name} treinado com sucesso.")
        return self

    def train_bayesian_calibration(self, X_train, y_train, X_val=None, y_val=None,
                                   name="CalibratedNaiveBayes", **kwargs):
        """
        Treina um modelo Naive Bayes com calibração para corrigir o viés.

        Args:
            X_train: Features de treinamento
            y_train: Target de treinamento
            X_val: Features de validação (opcional)
            y_val: Target de validação (opcional)
            name: Nome do modelo
            **kwargs: Parâmetros adicionais para GaussianNB

        Returns:
            self
        """
        logger.info(f"Treinando modelo calibrado: {name}")

        # Parâmetros padrão para GaussianNB
        params = {
            'var_smoothing': 1e-9  # parâmetro de suavização
        }

        # Atualizar com parâmetros fornecidos
        params.update(kwargs)

        # Criar modelo Naive Bayes como estimador
        estimator = GaussianNB(**params)

        # Naive Bayes tende a produzir probabilidades extremas, calibração é importante
        calibrated_model = CalibratedClassifierCV(
            estimator=estimator,  # Changed from base_estimator to estimator
            method='isotonic',  # isotonic pode trabalhar bem para corrigir as probabilidades extremas
            cv=5,
            n_jobs=-1
        )

        # Treinar o modelo
        calibrated_model.fit(X_train, y_train)

        # Armazenar modelo
        self.models[name] = calibrated_model

        logger.info(f"Modelo {name} treinado com sucesso.")
        return self

    def train_knn_calibrated(self, X_train, y_train, X_val=None, y_val=None,
                             name="CalibratedKNN", **kwargs):
        """
        Treina um modelo KNN com calibração.

        Args:
            X_train: Features de treinamento
            y_train: Target de treinamento
            X_val: Features de validação (opcional)
            y_val: Target de validação (opcional)
            name: Nome do modelo
            **kwargs: Parâmetros adicionais para KNeighborsClassifier

        Returns:
            self
        """
        logger.info(f"Treinando modelo calibrado: {name}")

        # Parâmetros padrão para KNeighborsClassifier
        params = {
            'n_neighbors': 5,
            'weights': 'distance',
            'algorithm': 'auto',
            'n_jobs': -1
        }

        # Atualizar com parâmetros fornecidos
        params.update(kwargs)

        # Criar modelo KNN como estimador
        estimator = Pipeline([
            ('scaler', StandardScaler()),
            ('knn', KNeighborsClassifier(**params))
        ])

        # Calibrar o modelo KNN
        calibrated_model = CalibratedClassifierCV(
            estimator=estimator,  # Changed from base_estimator to estimator
            method='sigmoid',
            cv=5,
            n_jobs=-1
        )

        # Treinar o modelo
        calibrated_model.fit(X_train, y_train)

        # Armazenar modelo
        self.models[name] = calibrated_model

        logger.info(f"Modelo {name} treinado com sucesso.")
        return self

    def train_ensemble_calibrated(self, base_models=None, method='isotonic'):
        """
        Cria um ensemble calibrado a partir dos modelos já treinados.

        Args:
            base_models: Lista de nomes dos modelos a incluir no ensemble
            method: Método de calibração ('sigmoid' ou 'isotonic')

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

        logger.info(f"Criando modelo ensemble calibrado com {len(base_models)} modelos base: {base_models}")

        # Classe do modelo ensemble
        class CalibratedEnsembleModel:
            """Modelo ensemble que combina múltiplos modelos calibrados."""

            def __init__(self, models, model_weights=None):
                self.models = models
                # Se não houver pesos, usar pesos iguais
                if model_weights is None:
                    self.weights = [1 / len(models)] * len(models)
                else:
                    # Normalizar pesos
                    total_weight = sum(model_weights)
                    self.weights = [w / total_weight for w in model_weights]

                self.threshold = 0.5

            def predict_proba(self, X):
                """Combina probabilidades de todos os modelos base com pesos."""
                probas = np.zeros((X.shape[0], 2))

                for i, (name, model) in enumerate(self.models.items()):
                    model_proba = model.predict_proba(X)
                    probas += self.weights[i] * model_proba

                # Normalizar (garantir que soma = 1)
                row_sums = probas.sum(axis=1)
                probas = probas / row_sums[:, np.newaxis]

                return probas

            def predict(self, X):
                """Faz predições com base no threshold definido."""
                probas = self.predict_proba(X)
                return (probas[:, 1] >= self.threshold).astype(int)

        # Coletar modelos base
        base_model_dict = {name: self.models[name] for name in base_models}

        # Definir pesos com base nas performances (se já avaliados)
        weights = None
        if self.evaluation_results:
            # Usar métrica de calibração como peso
            weights = []
            for name in base_models:
                if name in self.evaluation_results and 'brier_score' in self.evaluation_results[name]:
                    # Menor Brier score é melhor, então invertemos
                    brier = self.evaluation_results[name]['brier_score']
                    weights.append(1.0 / (brier + 0.01))  # Adicionar pequeno valor para evitar divisão por zero
                else:
                    weights.append(1.0)  # Peso padrão

        # Criar e armazenar o modelo ensemble
        ensemble = CalibratedEnsembleModel(base_model_dict, weights)
        self.models['CalibratedEnsemble'] = ensemble

        logger.info("Modelo ensemble calibrado criado com sucesso.")
        return self

    def evaluate_calibration(self, model_name, X_val, y_val, n_bins=10):
        """
        Avalia a calibração de um modelo específico.

        Args:
            model_name: Nome do modelo
            X_val: Features de validação
            y_val: Target de validação
            n_bins: Número de bins para a curva de calibração

        Returns:
            Dicionário com métricas de calibração
        """
        if model_name not in self.models:
            raise ValueError(f"Modelo '{model_name}' não encontrado.")

        model = self.models[model_name]
        logger.info(f"Avaliando calibração do modelo '{model_name}'...")

        # Obter probabilidades
        y_proba = model.predict_proba(X_val)[:, 1]

        # Calcular curva de calibração
        prob_true, prob_pred = calibration_curve(y_val, y_proba, n_bins=n_bins)

        # Calcular Brier score (erro quadrático médio)
        brier_score = brier_score_loss(y_val, y_proba)

        # Calcular log-loss (entropia cruzada)
        log_loss_score = log_loss(y_val, y_proba)

        # Calcular outras métricas
        auc = roc_auc_score(y_val, y_proba)

        # Verificar confiabilidade da calibração (ECE - Expected Calibration Error)
        # Uma implementação simplificada do ECE
        ece = np.mean(np.abs(prob_true - prob_pred))

        # Armazenar resultados
        calibration_results = {
            'model_name': model_name,
            'brier_score': brier_score,  # menor é melhor
            'log_loss': log_loss_score,  # menor é melhor
            'ece': ece,  # menor é melhor
            'auc': auc,  # maior é melhor
            'prob_true': prob_true.tolist(),
            'prob_pred': prob_pred.tolist(),
        }

        # Adicionar aos resultados de avaliação
        if model_name in self.evaluation_results:
            self.evaluation_results[model_name].update(calibration_results)
        else:
            self.evaluation_results[model_name] = calibration_results

        logger.info(f"Calibração do modelo '{model_name}':")
        logger.info(f"  Brier Score: {brier_score:.4f} (menor é melhor)")
        logger.info(f"  Log Loss: {log_loss_score:.4f} (menor é melhor)")
        logger.info(f"  ECE: {ece:.4f} (menor é melhor)")
        logger.info(f"  AUC: {auc:.4f}")

        return calibration_results

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

        # Métricas de calibração
        brier_score = brier_score_loss(y_test, y_proba)
        log_loss_score = log_loss(y_test, y_proba)

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
            'brier_score': brier_score,
            'log_loss': log_loss_score,
            'tn': tn,
            'fp': fp,
            'fn': fn,
            'tp': tp,
            'aprovacao_rate': aprovacao_rate,
            'inadimplencia_portfolio': inadimplencia_portfolio,
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
        logger.info(f"Brier Score: {brier_score:.4f}")
        logger.info(f"Log Loss: {log_loss_score:.4f}")
        logger.info("\nMatriz de Confusão:")
        logger.info(f"TN: {tn}, FP: {fp}")
        logger.info(f"FN: {fn}, TP: {tp}")
        logger.info("\nMétricas de Negócio:")
        logger.info(f"Taxa de Aprovação: {aprovacao_rate:.2%}")
        logger.info(f"Taxa de Inadimplência no Portfolio: {inadimplencia_portfolio:.2%}")

        # Gerar visualizações
        self._plot_calibration_curve(model_name, X_test, y_test)

        return results

    def optimize_threshold(self, model_name, X_val, y_val, optimization_metric='cost',
                           cost_fn_ratio=5.0, approval_target=None):
        """
        Otimiza o threshold de classificação para um modelo específico.

        Args:
            model_name: Nome do modelo
            X_val: Features de validação
            y_val: Target de validação
            optimization_metric: Métrica para otimização ('cost', 'f1', 'calibration')
            cost_fn_ratio: Custo relativo de falsos negativos vs. falsos positivos
            approval_target: Taxa alvo de aprovação (0-1)

        Returns:
            Threshold otimizado
        """
        if model_name not in self.models:
            raise ValueError(f"Modelo '{model_name}' não encontrado.")

        model = self.models[model_name]
        logger.info(f"Otimizando threshold para modelo '{model_name}'...")

        # Obter probabilidades
        y_proba = model.predict_proba(X_val)[:, 1]

        # Definir thresholds para testar
        thresholds = np.linspace(0.01, 0.99, 99)

        if optimization_metric == 'cost':
            # Otimizar para custo de negócio
            best_threshold = 0.5
            min_cost = float('inf')

            for thresh in thresholds:
                y_pred = (y_proba >= thresh).astype(int)

                # Calcular matriz de confusão
                tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()

                # Calcular custo (FP + cost_ratio * FN)
                cost = fp + cost_fn_ratio * fn

                # Se há target de aprovação, penalizar desvios
                if approval_target:
                    approval_rate = (tp + fp) / (tp + tn + fp + fn)
                    approval_deviation = abs(approval_rate - approval_target)
                    # Adicionar penalidade proporcional ao desvio
                    cost += cost * approval_deviation * 2

                if cost < min_cost:
                    min_cost = cost
                    best_threshold = thresh

            logger.info(f"Threshold otimizado para custo de negócio: {best_threshold:.4f}")

        elif optimization_metric == 'f1':
            # Otimizar para F1-score
            best_threshold = 0.5
            max_f1 = 0

            for thresh in thresholds:
                y_pred = (y_proba >= thresh).astype(int)

                # Calcular F1-score
                f1 = f1_score(y_val, y_pred)

                if f1 > max_f1:
                    max_f1 = f1
                    best_threshold = thresh

            logger.info(f"Threshold otimizado para F1-score: {best_threshold:.4f} (F1: {max_f1:.4f})")

        elif optimization_metric == 'calibration':
            # Otimizar para calibração (minimizar Brier score)
            best_threshold = 0.5
            min_brier = float('inf')

            for thresh in thresholds:
                y_pred = (y_proba >= thresh).astype(int)

                # Calcular Brier score ajustado pelo threshold
                # Note: o threshold não afeta diretamente o Brier score,
                # mas podemos usar uma métrica composta
                brier = brier_score_loss(y_val, y_proba)

                # Combinar com precisão da classificação no threshold
                accuracy = (y_pred == y_val).mean()
                combined_score = brier * (1.0 - accuracy)

                if combined_score < min_brier:
                    min_brier = combined_score
                    best_threshold = thresh

            logger.info(f"Threshold otimizado para calibração: {best_threshold:.4f}")

        else:
            # Default: equilibrar precision e recall
            precision, recall, thresholds_pr = precision_recall_curve(y_val, y_proba)

            # Adicionar threshold = 1.0 (ausente no retorno da função)
            thresholds_pr = np.append(thresholds_pr, 1.0)

            # Encontrar o ponto onde precision e recall estão mais próximos
            closest_idx = np.argmin(np.abs(precision - recall))
            best_threshold = thresholds_pr[closest_idx]

            logger.info(f"Threshold otimizado para equilíbrio precision-recall: {best_threshold:.4f}")

        # Armazenar threshold otimizado
        self.thresholds[model_name] = best_threshold

        # Se o modelo tiver atributo threshold, atualizar
        if hasattr(model, 'threshold'):
            model.threshold = best_threshold

        # Plotar impacto do threshold
        self._plot_threshold_impact(model_name, X_val, y_val, thresholds)

        return best_threshold

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
            self.evaluate_calibration(name, X_test, y_test)

            # Extrair métricas relevantes
            metrics = {
                'Modelo': name,
                'AUC': self.evaluation_results[name]['auc'],
                'F1-Score': self.evaluation_results[name]['f1_score'],
                'Precision': self.evaluation_results[name]['precision'],
                'Recall': self.evaluation_results[name]['recall'],
                'Brier Score': self.evaluation_results[name]['brier_score'],
                'Log Loss': self.evaluation_results[name]['log_loss'],
                'Taxa de Aprovação': self.evaluation_results[name]['aprovacao_rate'],
                'Taxa de Inadimplência': self.evaluation_results[name]['inadimplencia_portfolio']
            }

            results.append(metrics)

        # Criar DataFrame com resultados
        results_df = pd.DataFrame(results)

        # Encontrar melhor modelo (menor Brier score - melhor calibração)
        best_model_idx = results_df['Brier Score'].idxmin()
        self.best_model_name = results_df.loc[best_model_idx, 'Modelo']

        logger.info(f"\n>>> Melhor modelo por calibração: {self.best_model_name}")
        logger.info(f"Brier Score: {results_df.loc[best_model_idx, 'Brier Score']:.4f}")

        # Salvar resultados
        results_file = os.path.join(self.eval_dir, f"calibrated_model_comparison_{self.timestamp}.csv")
        results_df.to_csv(results_file, index=False)
        logger.info(f"Resultados comparativos salvos em: {results_file}")

        # Criar visualização comparativa
        self._plot_model_comparison(results_df)
        self._plot_calibration_comparison()

        return results_df

    def _plot_calibration_curve(self, model_name, X_test, y_test, n_bins=10, save_path=None):
        """
        Gera e salva um gráfico da curva de calibração para um modelo específico.

        Args:
            model_name: Nome do modelo
            X_test: Features de teste
            y_test: Target de teste
            n_bins: Número de bins para a curva de calibração
            save_path: Caminho para salvar o gráfico (opcional)
        """
        model = self.models[model_name]

        # Predições de probabilidade
        y_proba = model.predict_proba(X_test)[:, 1]

        # Calcular curva de calibração
        prob_true, prob_pred = calibration_curve(y_test, y_proba, n_bins=n_bins)

        # Calcular Brier Score
        brier = brier_score_loss(y_test, y_proba)

        # Criar gráfico
        plt.figure(figsize=(10, 8))

        # Plotar curva de calibração
        plt.plot(prob_pred, prob_true, marker='o', linewidth=2, label=f'{model_name}')

        # Linha de calibração perfeita
        plt.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated')

        # Configurações do gráfico
        plt.xlabel('Probabilidade Média Predita')
        plt.ylabel('Fração de Positivos')
        plt.title(f'Curva de Calibração - {model_name}\nBrier Score: {brier:.4f} (menor é melhor)')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)

        # Determinar caminho para salvar
        if save_path is None:
            plots_dir = os.path.join(self.eval_dir, 'plots', model_name)
            os.makedirs(plots_dir, exist_ok=True)
            save_path = os.path.join(plots_dir, f'calibration_curve_{self.timestamp}.png')

        # Salvar gráfico
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Curva de calibração salva em: {save_path}")

    def _plot_threshold_impact(self, model_name, X_val, y_val, thresholds, save_path=None):
        """
        Gera e salva gráficos mostrando o impacto do threshold nas métricas.

        Args:
            model_name: Nome do modelo
            X_val: Features de validação
            y_val: Target de validação
            thresholds: Array de thresholds testados
            save_path: Caminho para salvar o gráfico (opcional)
        """
        # Predições de probabilidade
        y_proba = self.models[model_name].predict_proba(X_val)[:, 1]

        # Métricas para cada threshold
        precision_values = []
        recall_values = []
        f1_values = []
        accuracy_values = []
        approval_rates = []
        default_rates = []
        brier_scores = []  # O Brier score não muda com o threshold, mas plotamos para referência

        # Calcular métricas para cada threshold
        brier_score = brier_score_loss(y_val, y_proba)

        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)

            # Calcular matriz de confusão
            tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()

            # Métricas de classificação
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            accuracy = (tp + tn) / (tp + tn + fp + fn)

            # Métricas de negócio
            approval_rate = (tp + fp) / (tp + tn + fp + fn)
            default_rate = fn / (tp + fn) if (tp + fn) > 0 else 0

            # Armazenar métricas
            precision_values.append(precision)
            recall_values.append(recall)
            f1_values.append(f1)
            accuracy_values.append(accuracy)
            approval_rates.append(approval_rate)
            default_rates.append(default_rate)
            brier_scores.append(brier_score)  # constante para todos os thresholds

        # Threshold otimizado
        optimal_threshold = self.thresholds.get(model_name, 0.5)

        # Criar gráfico
        plt.figure(figsize=(12, 10))

        # 1. Métricas de classificação
        plt.subplot(2, 2, 1)
        plt.plot(thresholds, precision_values, label='Precision', color='blue')
        plt.plot(thresholds, recall_values, label='Recall', color='red')
        plt.plot(thresholds, f1_values, label='F1', color='green')
        plt.axvline(x=optimal_threshold, color='black', linestyle='--')
        plt.xlabel('Threshold')
        plt.ylabel('Score')
        plt.title('Precision, Recall e F1')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 2. Taxas de aprovação e inadimplência
        plt.subplot(2, 2, 2)
        plt.plot(thresholds, approval_rates, label='Taxa de Aprovação', color='blue')
        plt.plot(thresholds, default_rates, label='Taxa de Inadimplência', color='red')
        plt.axvline(x=optimal_threshold, color='black', linestyle='--')
        plt.xlabel('Threshold')
        plt.ylabel('Taxa')
        plt.title('Taxas de Aprovação e Inadimplência')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 3. Acurácia
        plt.subplot(2, 2, 3)
        plt.plot(thresholds, accuracy_values, color='purple')
        plt.axvline(x=optimal_threshold, color='black', linestyle='--')
        plt.xlabel('Threshold')
        plt.ylabel('Acurácia')
        plt.title('Acurácia vs Threshold')
        plt.grid(True, alpha=0.3)

        # 4. Brier Score (constante para referência)
        plt.subplot(2, 2, 4)
        plt.plot(thresholds, brier_scores, color='orange')
        plt.axvline(x=optimal_threshold, color='black', linestyle='--')
        plt.xlabel('Threshold')
        plt.ylabel('Brier Score')
        plt.title(f'Brier Score (Calibração): {brier_score:.4f}')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.suptitle(f'Impacto do Threshold - {model_name}', fontsize=16, y=1.05)

        # Determinar caminho para salvar
        if save_path is None:
            plots_dir = os.path.join(self.eval_dir, 'plots', model_name)
            os.makedirs(plots_dir, exist_ok=True)
            save_path = os.path.join(plots_dir, f'threshold_impact_{self.timestamp}.png')

        # Salvar gráfico
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Gráfico de impacto do threshold salvo em: {save_path}")

    def _plot_model_comparison(self, results_df):
        """
        Gera visualizações comparativas entre modelos.

        Args:
            results_df: DataFrame com resultados dos modelos
        """
        # 1. Comparação de métricas de calibração
        plt.figure(figsize=(12, 8))
        metrics = ['Brier Score', 'Log Loss']
        results_df.set_index('Modelo')[metrics].plot(kind='barh')
        plt.title('Comparação de Métricas de Calibração (menor é melhor)')
        plt.xlabel('Valor')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # Salvar gráfico
        plt.savefig(os.path.join(self.eval_dir, f'calibration_metrics_comparison_{self.timestamp}.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

        # 2. Comparação de métricas de classificação
        plt.figure(figsize=(12, 8))
        metrics = ['AUC', 'F1-Score', 'Precision', 'Recall']
        results_df.set_index('Modelo')[metrics].plot(kind='barh')
        plt.title('Comparação de Métricas de Classificação (maior é melhor)')
        plt.xlabel('Valor')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # Salvar gráfico
        plt.savefig(os.path.join(self.eval_dir, f'classification_metrics_comparison_{self.timestamp}.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

        # 3. Comparação de taxas de aprovação e inadimplência
        plt.figure(figsize=(12, 8))
        results_df.set_index('Modelo')[['Taxa de Aprovação', 'Taxa de Inadimplência']].plot(kind='barh')
        plt.title('Taxas de Aprovação vs. Inadimplência')
        plt.xlabel('Taxa')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # Salvar gráfico
        plt.savefig(os.path.join(self.eval_dir, f'approval_vs_default_{self.timestamp}.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_calibration_comparison(self):
        """
        Gera um gráfico comparativo das curvas de calibração de todos os modelos.
        """
        plt.figure(figsize=(12, 10))

        # Verificar quais modelos têm dados de calibração
        models_with_calibration = []
        for name, results in self.evaluation_results.items():
            if 'prob_true' in results and 'prob_pred' in results:
                models_with_calibration.append(name)

        if not models_with_calibration:
            logger.warning("Nenhum modelo com dados de calibração disponíveis para comparação.")
            return

        # Plotar curva de calibração para cada modelo
        for name in models_with_calibration:
            results = self.evaluation_results[name]
            prob_true = results['prob_true']
            prob_pred = results['prob_pred']
            brier = results.get('brier_score', 0)

            plt.plot(prob_pred, prob_true, marker='o', linewidth=2,
                     label=f'{name} (Brier: {brier:.4f})')

        # Linha de calibração perfeita
        plt.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated')

        # Configurações do gráfico
        plt.xlabel('Probabilidade Média Predita')
        plt.ylabel('Fração de Positivos')
        plt.title('Comparação das Curvas de Calibração')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)

        # Salvar gráfico
        plt.savefig(os.path.join(self.eval_dir, f'calibration_curves_comparison_{self.timestamp}.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Comparação de curvas de calibração salva em: {self.eval_dir}")

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

            try:
                # Salvar modelo
                joblib.dump(model, file_path)
                saved_paths[name] = file_path

                logger.info(f"Modelo '{name}' salvo em: {file_path}")
            except Exception as e:
                logger.error(f"Erro ao salvar modelo '{name}': {str(e)}")

        # Salvar métricas e thresholds para referência
        metadata = {
            'timestamp': self.timestamp,
            'models': list(self.models.keys()),
            'thresholds': self.thresholds,
            'best_model': self.best_model_name,
            'evaluation_summary': {
                name: {
                    'auc': results.get('auc', None),
                    'f1_score': results.get('f1_score', None),
                    'brier_score': results.get('brier_score', None),
                    'log_loss': results.get('log_loss', None),
                    'threshold': self.thresholds.get(name, self.default_threshold)
                }
                for name, results in self.evaluation_results.items()
            }
        }

        metadata_file = os.path.join(self.model_dir, f"calibrated_model_metadata_{self.timestamp}.json")
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=4, default=str)

        logger.info(f"Metadados dos modelos salvos em: {metadata_file}")

        # Salvar o melhor modelo separadamente
        if self.best_model_name:
            best_model_path = os.path.join(self.model_dir, f"best_calibrated_model_{self.timestamp}.joblib")
            joblib.dump(self.models[self.best_model_name], best_model_path)
            logger.info(f"Melhor modelo ({self.best_model_name}) salvo separadamente em: {best_model_path}")

        return saved_paths


def load_and_prepare_data(data_dir, timestamp=None, target_col='Inadimplente'):
    """
    Carrega e prepara dados para treinamento de modelos calibrados.

    Args:
        data_dir: Diretório com dados processados
        timestamp: Timestamp específico dos dados a usar
        target_col: Nome da coluna alvo

    Returns:
        Dicionário com dados preparados
    """
    logger.info("Carregando e preparando dados para modelagem...")

    path_manager = PathManager()

    # Obter caminho absoluto do diretório
    if not os.path.isabs(data_dir):
        data_dir = os.path.join(path_manager.project_root, data_dir)

    # Se timestamp não fornecido, encontrar dados mais recentes
    if timestamp is None:
        # Procurar arquivos de metadados
        meta_files = [f for f in os.listdir(data_dir) if f.startswith('metadata_') and f.endswith('.json')]

        if meta_files:
            # Ordenar por timestamp (mais recente primeiro)
            meta_files.sort(reverse=True)
            timestamp = meta_files[0].replace('metadata_', '').split('.')[0]
            logger.info(f"Usando timestamp de metadados: {timestamp}")
        else:
            # Procurar diretamente por arquivos de dados
            train_files = [f for f in os.listdir(data_dir) if f.startswith('train_') and f.endswith('.csv')]

            if train_files:
                train_files.sort(reverse=True)
                timestamp = train_files[0].replace('train_', '').replace('.csv', '')
                logger.info(f"Usando timestamp do arquivo de treino: {timestamp}")
            else:
                raise FileNotFoundError(f"Não foi possível encontrar arquivos de dados em {data_dir}")

    logger.info(f"Usando timestamp: {timestamp}")

    # Carregar arquivos de dados
    train_file = os.path.join(data_dir, f"train_{timestamp}.csv")
    val_file = os.path.join(data_dir, f"val_{timestamp}.csv")
    test_file = os.path.join(data_dir, f"test_{timestamp}.csv")

    # Verificar se os arquivos existem
    for file_path in [train_file, val_file, test_file]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Arquivo não encontrado: {file_path}")

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
        possible_targets = ['inadimplente', 'target', 'default', 'risco_inadimplencia']
        for col in df_train.columns:
            if col.lower() in possible_targets:
                target_col = col
                logger.info(f"Coluna alvo identificada: {target_col}")
                break
        else:
            raise ValueError(
                f"Coluna alvo '{target_col}' não encontrada e não foi possível identificar automaticamente.")

    # Separar features e target
    X_train = df_train.drop(columns=[target_col])
    y_train = df_train[target_col]

    X_val = df_val.drop(columns=[target_col])
    y_val = df_val[target_col]

    X_test = df_test.drop(columns=[target_col])
    y_test = df_test[target_col]

    # Remover colunas que não são úteis para modelagem
    columns_to_exclude = [
        'ID_Cliente', 'Nome', 'CPF', 'Email', 'Telefone', 'Data_Referencia',
        'Nome_Completo', 'RG', 'CEP', 'Endereco'
    ]

    for col in columns_to_exclude:
        if col in X_train.columns:
            logger.info(f"Removendo coluna: {col}")
            X_train = X_train.drop(columns=[col])
            X_val = X_val.drop(columns=[col])
            X_test = X_test.drop(columns=[col])

    # Identificar e tratar colunas não numéricas
    categorical_cols = []
    for col in X_train.columns:
        if X_train[col].dtype == 'object':
            categorical_cols.append(col)

    if categorical_cols:
        logger.info(f"Tratando {len(categorical_cols)} colunas categóricas...")

        # Usar encoding simples (label encoding)
        for col in categorical_cols:
            # Criar mapeamento de categorias para inteiros
            categories = X_train[col].dropna().unique()
            cat_mapping = {cat: i for i, cat in enumerate(categories)}

            # Aplicar mapeamento
            X_train[col] = X_train[col].map(cat_mapping).fillna(-1)
            X_val[col] = X_val[col].map(cat_mapping).fillna(-1)
            X_test[col] = X_test[col].map(cat_mapping).fillna(-1)

    # Verificar e tratar valores ausentes
    if X_train.isnull().any().any():
        logger.info("Tratando valores ausentes...")

        # Usar imputação simples para valores ausentes
        for col in X_train.columns:
            if X_train[col].isnull().any():
                # Para features numéricas, usar mediana
                if X_train[col].dtype in ['int64', 'float64']:
                    fill_value = X_train[col].median()
                else:
                    # Para categóricas já codificadas, usar moda ou -1
                    fill_value = X_train[col].mode()[0] if not X_train[col].mode().empty else -1

                X_train[col] = X_train[col].fillna(fill_value)
                X_val[col] = X_val[col].fillna(fill_value)
                X_test[col] = X_test[col].fillna(fill_value)

    # Resumo dos dados
    logger.info("\nResumo dos dados:")
    logger.info(f"Conjunto de Treino: {X_train.shape[0]} exemplos, {X_train.shape[1]} features")
    logger.info(f"Conjunto de Validação: {X_val.shape[0]} exemplos, {X_val.shape[1]} features")
    logger.info(f"Conjunto de Teste: {X_test.shape[0]} exemplos, {X_test.shape[1]} features")

    # Verificar distribuição da variável alvo
    train_pos_rate = y_train.mean()
    val_pos_rate = y_val.mean()
    test_pos_rate = y_test.mean()

    logger.info("\nDistribuição da variável alvo:")
    logger.info(f"- Treino: {train_pos_rate:.2%} positivos")
    logger.info(f"- Validação: {val_pos_rate:.2%} positivos")
    logger.info(f"- Teste: {test_pos_rate:.2%} positivos")

    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'X_test': X_test,
        'y_test': y_test,
        'target_col': target_col,
        'timestamp': timestamp
    }


def train_calibrated_models(data_dict):
    """
    Treina e avalia os modelos calibrados.

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

    # Criar trainer
    trainer = CalibratedModelTrainer()

    # 1. Treinar Regressão Logística Calibrada
    logger.info("\n" + "=" * 80)
    logger.info("Treinando Regressão Logística Calibrada...")
    logger.info("=" * 80)

    trainer.train_calibrated_logistic_regression(
        X_train, y_train,
        X_val, y_val,
        name="CalibratedLogisticRegression",
        C=1.0,
        class_weight='balanced'
    )

    # 2. Treinar Random Forest Calibrado
    logger.info("\n" + "=" * 80)
    logger.info("Treinando Random Forest Calibrado...")
    logger.info("=" * 80)

    trainer.train_calibrated_random_forest(
        X_train, y_train,
        X_val, y_val,
        name="CalibratedRandomForest",
        n_estimators=100,
        max_depth=10
    )

    # 3. Treinar Gradient Boosting Calibrado
    logger.info("\n" + "=" * 80)
    logger.info("Treinando Gradient Boosting Calibrado...")
    logger.info("=" * 80)

    trainer.train_calibrated_gradient_boosting(
        X_train, y_train,
        X_val, y_val,
        name="CalibratedGradientBoosting",
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3
    )

    # 4. Treinar SVM Calibrado
    logger.info("\n" + "=" * 80)
    logger.info("Treinando SVM Calibrado...")
    logger.info("=" * 80)

    trainer.train_svm_with_calibration(
        X_train, y_train,
        X_val, y_val,
        name="CalibratedSVM"
    )

    # 5. Treinar Naive Bayes Calibrado
    logger.info("\n" + "=" * 80)
    logger.info("Treinando Naive Bayes Calibrado...")
    logger.info("=" * 80)

    trainer.train_bayesian_calibration(
        X_train, y_train,
        X_val, y_val,
        name="CalibratedNaiveBayes"
    )

    # 6. Treinar KNN Calibrado
    logger.info("\n" + "=" * 80)
    logger.info("Treinando KNN Calibrado...")
    logger.info("=" * 80)

    trainer.train_knn_calibrated(
        X_train, y_train,
        X_val, y_val,
        name="CalibratedKNN",
        n_neighbors=7
    )

    # 7. Criar Ensemble Calibrado
    logger.info("\n" + "=" * 80)
    logger.info("Criando Ensemble Calibrado...")
    logger.info("=" * 80)

    trainer.train_ensemble_calibrated()

    # 8. Otimizar thresholds
    logger.info("\n" + "=" * 80)
    logger.info("Otimizando thresholds...")
    logger.info("=" * 80)

    for name in trainer.models:
        trainer.optimize_threshold(
            name,
            X_val, y_val,
            optimization_metric='calibration'
        )

    # 9. Avaliar modelos
    logger.info("\n" + "=" * 80)
    logger.info("Avaliando modelos...")
    logger.info("=" * 80)

    results_df = trainer.evaluate_all_models(X_test, y_test)

    # 10. Salvar modelos
    logger.info("\n" + "=" * 80)
    logger.info("Salvando modelos...")
    logger.info("=" * 80)

    trainer.save_models()

    return trainer, results_df


def main():
    """
    Função principal para treinamento e avaliação de modelos calibrados.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Treinamento de modelos calibrados para predição de inadimplência")
    parser.add_argument('--data_dir', type=str, default='data/processed',
                        help='Diretório com dados processados (padrão: data/processed)')
    parser.add_argument('--timestamp', type=str, default=None,
                        help='Timestamp dos dados a serem usados')

    args = parser.parse_args()

    try:
        # 1. Carregar e preparar dados
        logger.info("\n" + "=" * 80)
        logger.info("CARREGANDO E PREPARANDO DADOS")
        logger.info("=" * 80)

        data_dict = load_and_prepare_data(args.data_dir, args.timestamp)

        # 2. Treinar e avaliar modelos
        logger.info("\n" + "=" * 80)
        logger.info("TREINANDO E AVALIANDO MODELOS CALIBRADOS")
        logger.info("=" * 80)

        trainer, results = train_calibrated_models(data_dict)

        # 3. Exibir resumo final
        logger.info("\n" + "=" * 80)
        logger.info("RESUMO FINAL")
        logger.info("=" * 80)

        logger.info("\nResultados dos modelos calibrados:")
        logger.info(results.to_string())

        # Melhor modelo
        best_model_name = trainer.best_model_name
        if best_model_name:
            logger.info(f"\nMelhor modelo calibrado: {best_model_name}")
            logger.info(f"Métricas do melhor modelo:")
            metrics = trainer.evaluation_results[best_model_name]
            logger.info(f"- AUC: {metrics['auc']:.4f}")
            logger.info(f"- Brier Score: {metrics['brier_score']:.4f}")
            logger.info(f"- Log Loss: {metrics['log_loss']:.4f}")
            logger.info(f"- F1-Score: {metrics['f1_score']:.4f}")
            logger.info(f"- Taxa de Aprovação: {metrics['aprovacao_rate']:.2%}")
            logger.info(f"- Taxa de Inadimplência: {metrics['inadimplencia_portfolio']:.2%}")

        logger.info("\nTreinamento e avaliação de modelos calibrados concluídos com sucesso!")
        logger.info(f"Resultados salvos em: {trainer.eval_dir}")
        logger.info(f"Modelos salvos em: {trainer.model_dir}")

    except Exception as e:
        logger.error(f"Erro durante a execução: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()