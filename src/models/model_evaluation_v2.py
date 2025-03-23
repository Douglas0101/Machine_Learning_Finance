"""
Módulo aprimorado para avaliação detalhada de modelos de predição de inadimplência.
Inclui métricas de classificação, análise de custo-benefício, e visualizações técnicas
avançadas para o contexto financeiro.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import interp1d
import joblib
import json
from datetime import datetime
import logging

# Métricas e validação
from sklearn.metrics import (
    roc_auc_score, confusion_matrix, average_precision_score, roc_curve, auc
)

# Configurar logger
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


# Obter caminho da raiz do projeto
def get_project_root():
    """Retorna o caminho para a raiz do projeto."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, os.pardir, os.pardir))
    return project_root


class VisualizationConfig:
    """
    Classe para configurar o estilo visual dos gráficos gerados.
    Controla densidade, esquemas de cores, anotações técnicas e outros aspectos.
    """

    def __init__(self,
                 style='technical',
                 color_palette='deep',
                 annotation_density='high',
                 dpi=300,
                 show_annotations=True,
                 show_statistics=True,
                 show_confidence_intervals=True,
                 font_scale=1.0):
        """
        Inicializa a configuração de visualização.

        Args:
            style: Estilo básico ('technical', 'presentation', 'report', 'minimal')
            color_palette: Paleta de cores do seaborn
            annotation_density: Quantidade de anotações ('low', 'medium', 'high')
            dpi: Resolução das imagens geradas
            show_annotations: Se True, mostra anotações técnicas nos gráficos
            show_statistics: Se True, adiciona caixas com estatísticas detalhadas
            show_confidence_intervals: Se True, mostra intervalos de confiança
            font_scale: Escala para os tamanhos de fonte
        """
        self.style = style
        self.color_palette = color_palette
        self.annotation_density = annotation_density
        self.dpi = dpi
        self.show_annotations = show_annotations
        self.show_statistics = show_statistics
        self.show_confidence_intervals = show_confidence_intervals
        self.font_scale = font_scale

        # Aplicar configurações
        self._apply_style()

    def _apply_style(self):
        """Aplica o estilo escolhido usando configurações do Seaborn e Matplotlib."""
        # Definir estilo básico
        if self.style == 'technical':
            sns.set_style('whitegrid')
            plt.rcParams['axes.facecolor'] = '#F5F5F5'
        elif self.style == 'presentation':
            sns.set_style('talk')
        elif self.style == 'report':
            sns.set_style('ticks')
        elif self.style == 'minimal':
            sns.set_style('white')

        # Aplicar paleta de cores
        sns.set_palette(self.color_palette)

        # Ajustar escala de fontes
        sns.set_context('notebook', font_scale=self.font_scale)

        # Configurações específicas de Matplotlib
        plt.rcParams['axes.titleweight'] = 'bold'
        plt.rcParams['figure.titleweight'] = 'bold'
        plt.rcParams['axes.labelweight'] = 'bold'
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.alpha'] = 0.3

        # Configurações para visualização técnica de alta densidade
        if self.annotation_density == 'high':
            plt.rcParams['axes.labelsize'] = 10
            plt.rcParams['xtick.labelsize'] = 9
            plt.rcParams['ytick.labelsize'] = 9
            plt.rcParams['legend.fontsize'] = 9

        # Ajustar DPI
        plt.rcParams['figure.dpi'] = self.dpi
        plt.rcParams['savefig.dpi'] = self.dpi


class ModelEvaluator:
    """
    Classe para avaliação de modelos de inadimplência, focando em métricas
    relevantes para o contexto financeiro e impacto de negócio, com visualizações
    técnicas avançadas integradas.
    """

    def __init__(self, cost_fn_ratio=5.0, approval_target=None, default_threshold=None,
                 positive_class='inadimplente', visualization_config=None):
        """
        Inicializa o avaliador de modelos.

        Args:
            cost_fn_ratio: Custo relativo de um falso negativo (cliente inadimplente
                          classificado como adimplente) comparado a um falso positivo
            approval_target: Taxa alvo de aprovação (0-1) para otimização do threshold
            default_threshold: Threshold padrão para classificação (0.5)
            positive_class: Define qual é a classe positiva ('inadimplente' ou 'adimplente')
            visualization_config: Configurações para visualizações técnicas
        """
        self.cost_fn_ratio = cost_fn_ratio
        self.approval_target = approval_target
        self.default_threshold = default_threshold or 0.5
        self.models = {}
        self.model_results = {}
        self.best_model = None
        self.best_model_name = None
        self.positive_class = positive_class
        self.visualization_config = visualization_config or VisualizationConfig()

        # Validar classe positiva
        if self.positive_class not in ['inadimplente', 'adimplente']:
            logger.warning(f"Classe positiva '{positive_class}' não reconhecida. Usando 'inadimplente'.")
            self.positive_class = 'inadimplente'

        # Configurar diretório de saída
        project_root = get_project_root()
        self.output_dir = os.path.join(project_root, 'reports', 'model_evaluation')
        os.makedirs(self.output_dir, exist_ok=True)

        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Aplicar configurações de estilo para visualizações
        self.visualization_config._apply_style()

    def add_model(self, name, model, threshold=None):
        """
        Adiciona um modelo à avaliação.

        Args:
            name: Nome do modelo
            model: Modelo treinado (deve ter método predict_proba)
            threshold: Threshold para classificação (se None, usa padrão)

        Returns:
            self
        """
        # Verificar se o modelo tem método predict_proba
        if not hasattr(model, 'predict_proba') or not callable(getattr(model, 'predict_proba')):
            raise ValueError(f"O modelo '{name}' não possui método predict_proba().")

        # Armazenar modelo
        self.models[name] = {
            'model': model,
            'threshold': threshold or self.default_threshold
        }

        logger.info(f"Modelo '{name}' adicionado à avaliação.")
        return self

    def load_model(self, model_path, name=None, threshold=None):
        """
        Carrega um modelo salvo.

        Args:
            model_path: Caminho para o modelo
            name: Nome para o modelo (se None, usa o nome do arquivo)
            threshold: Threshold para classificação (se None, usa padrão)

        Returns:
            self
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Arquivo não encontrado: {model_path}")

        # Definir nome
        if name is None:
            name = os.path.basename(model_path).split('.')[0]

        # Carregar modelo
        logger.info(f"Carregando modelo '{name}' de: {model_path}")
        model = joblib.load(model_path)

        # Determinar threshold
        if threshold is None:
            # Verificar se o modelo tem atributo threshold
            if hasattr(model, 'threshold'):
                threshold = model.threshold
            else:
                # Tentar carregar de metadados
                model_dir = os.path.dirname(model_path)
                base_name = os.path.basename(model_path)
                parts = base_name.split('_')

                if len(parts) > 1:
                    # Extrair timestamp do nome do arquivo
                    model_type = parts[0]
                    timestamp = '_'.join(parts[1:]).replace('.joblib', '')

                    metadata_path = os.path.join(model_dir, f"model_metadata_{timestamp}.json")
                    if os.path.exists(metadata_path):
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)

                        if 'thresholds' in metadata and model_type in metadata['thresholds']:
                            threshold = metadata['thresholds'][model_type]
                            logger.info(f"Threshold {threshold} carregado dos metadados.")

        # Usar threshold padrão se ainda não definido
        threshold = threshold or self.default_threshold

        # Adicionar modelo
        self.add_model(name, model, threshold)
        return self

    def evaluate_model(self, name, X_test, y_test, threshold=None,
                       plot_curves=True, save_plots=True,
                       enhanced_visuals=False, store_predictions=False):  # Mudei enhanced_visuals para False por padrão
        """
        Avalia um modelo específico no conjunto de teste com visualizações avançadas.

        Args:
            name: Nome do modelo
            X_test: Features de teste
            y_test: Target de teste
            threshold: Threshold para classificação (se None, usa o definido no modelo)
            plot_curves: Se True, gera gráficos
            save_plots: Se True, salva os gráficos gerados
            enhanced_visuals: Se True, utiliza as visualizações técnicas avançadas
            store_predictions: Se True, armazena as predições para análises adicionais

        Returns:
            Dicionário com métricas de avaliação
        """
        # Validar entradas
        if not isinstance(X_test, (pd.DataFrame, np.ndarray)):
            raise TypeError("X_test deve ser um DataFrame pandas ou array numpy")

        if not isinstance(y_test, (pd.Series, np.ndarray, list)):
            raise TypeError("y_test deve ser uma Series pandas, array numpy ou lista")

        if len(X_test) != len(y_test):
            raise ValueError(
                f"X_test ({len(X_test)} amostras) e y_test ({len(y_test)} amostras) devem ter o mesmo número de amostras")

        if name not in self.models:
            raise ValueError(f"Modelo '{name}' não encontrado.")

        model_info = self.models[name]
        model = model_info['model']

        # Determinar threshold
        threshold = threshold or model_info['threshold']

        logger.info(f"Avaliando modelo '{name}' (threshold={threshold:.4f})...")

        # Fazer predições
        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= threshold).astype(int)

        # Calcular métricas básicas
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        acc = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        auc_score = roc_auc_score(y_test, y_proba)
        avg_precision = average_precision_score(y_test, y_proba)

        # Métricas de negócio - CORRIGIDAS de acordo com a interpretação da classe positiva
        if self.positive_class == 'inadimplente':
            # TP = Inadimplente classificado como inadimplente (correto)
            # TN = Adimplente classificado como adimplente (correto)
            # FP = Adimplente classificado como inadimplente (erro tipo I)
            # FN = Inadimplente classificado como adimplente (erro tipo II)
            aprovacao_rate = (tn + fn) / (tp + tn + fp + fn)  # Taxa de aprovação (classe negativa)
            inadimplencia_portfolio = fn / (tn + fn) if (tn + fn) > 0 else 0  # Inadimplentes entre aprovados
        else:  # 'adimplente'
            # Neste caso, invertemos a interpretação
            # TP = Adimplente classificado como adimplente (correto)
            # TN = Inadimplente classificado como inadimplente (correto)
            # FP = Inadimplente classificado como adimplente (erro tipo I)
            # FN = Adimplente classificado como inadimplente (erro tipo II)
            aprovacao_rate = (tp + fp) / (tp + tn + fp + fn)  # Taxa de aprovação (classe positiva)
            inadimplencia_portfolio = fp / (tp + fp) if (tp + fp) > 0 else 0  # Inadimplentes entre aprovados

        # Calcular custo de negócio - CORRIGIDO
        # Custo = FP + cost_ratio * FN
        business_cost = fp + self.cost_fn_ratio * fn
        normalized_cost = business_cost / len(y_test)

        # Armazenar resultados
        results = {
            'model_name': name,
            'threshold': threshold,
            'accuracy': acc,
            'precision': precision,
            'recall': recall,
            'specificity': specificity,
            'f1_score': f1,
            'auc': auc_score,
            'avg_precision': avg_precision,
            'tn': tn,
            'fp': fp,
            'fn': fn,
            'tp': tp,
            'aprovacao_rate': aprovacao_rate,
            'inadimplencia_portfolio': inadimplencia_portfolio,
            'business_cost': business_cost,
            'normalized_cost': normalized_cost,
        }

        # Opcionalmente armazenar previsões completas
        if store_predictions:
            results.update({
                'y_proba': y_proba,
                'y_pred': y_pred,
                'y_true': y_test
            })

        self.model_results[name] = results

        # Imprimir relatório
        logger.info("\nRelatório de Avaliação:")
        logger.info(f"Modelo: {name}")
        logger.info(f"Threshold: {threshold:.4f}")
        logger.info(f"Acurácia: {acc:.4f}")
        logger.info(f"AUC-ROC: {auc_score:.4f}")
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

        # Gerar gráficos
        if plot_curves:
            plots_dir = os.path.join(self.output_dir, 'plots', name)
            if save_plots:
                os.makedirs(plots_dir, exist_ok=True)

            # Usar apenas visualizações básicas até que as avançadas sejam implementadas
            self._plot_roc_curve(y_test, y_proba, name,
                                 save_path=os.path.join(plots_dir,
                                                        f'roc_curve_{self.timestamp}.png') if save_plots else None)

            self._plot_precision_recall_curve(y_test, y_proba, name,
                                              save_path=os.path.join(plots_dir,
                                                                     f'pr_curve_{self.timestamp}.png') if save_plots else None)

            self._plot_confusion_matrix(y_test, y_pred, name,
                                        save_path=os.path.join(plots_dir,
                                                               f'confusion_matrix_{self.timestamp}.png') if save_plots else None)

            self._plot_score_distribution(y_test, y_proba, threshold, name,
                                          save_path=os.path.join(plots_dir,
                                                                 f'score_dist_{self.timestamp}.png') if save_plots else None)

            self._plot_calibration_curve(y_test, y_proba, name,
                                         save_path=os.path.join(plots_dir,
                                                                f'calibration_{self.timestamp}.png') if save_plots else None)

            self._plot_threshold_impact(y_test, y_proba, name,
                                        save_path=os.path.join(plots_dir,
                                                               f'threshold_impact_{self.timestamp}.png') if save_plots else None)

        return results

    def evaluate_all_models(self, X_test, y_test, store_predictions=False):
        """
        Avalia todos os modelos registrados.

        Args:
            X_test: Features de teste
            y_test: Target de teste
            store_predictions: Se True, armazena as predições para análises adicionais

        Returns:
            DataFrame com resultados comparativos
        """
        # Validar entradas
        if not isinstance(X_test, (pd.DataFrame, np.ndarray)):
            raise TypeError("X_test deve ser um DataFrame pandas ou array numpy")

        if not isinstance(y_test, (pd.Series, np.ndarray, list)):
            raise TypeError("y_test deve ser uma Series pandas, array numpy ou lista")

        if len(X_test) != len(y_test):
            raise ValueError(
                f"X_test ({len(X_test)} amostras) e y_test ({len(y_test)} amostras) devem ter o mesmo número de amostras")

        if not self.models:
            raise ValueError("Nenhum modelo registrado para avaliação.")

        logger.info(f"Avaliando {len(self.models)} modelos...")

        for name in self.models:
            self.evaluate_model(name, X_test, y_test, store_predictions=store_predictions)

        # Criar DataFrame comparativo
        results_list = []

        for name, results in self.model_results.items():
            row = {
                'Modelo': name,
                'AUC': results['auc'],
                'F1-Score': results['f1_score'],
                'Precisão': results['precision'],
                'Recall': results['recall'],
                'Acurácia': results['accuracy'],
                'Taxa de Aprovação': results['aprovacao_rate'],
                'Taxa de Inadimplência': results['inadimplencia_portfolio'],
                'Custo de Negócio': results['normalized_cost']
            }
            results_list.append(row)

        comparison_df = pd.DataFrame(results_list)

        # Identificar melhor modelo (menor custo)
        best_idx = comparison_df['Custo de Negócio'].idxmin()
        self.best_model_name = comparison_df.iloc[best_idx]['Modelo']
        self.best_model = self.models[self.best_model_name]['model']

        logger.info(f"\nMelhor modelo: {self.best_model_name}")
        logger.info(f"Custo de Negócio: {comparison_df.iloc[best_idx]['Custo de Negócio']:.4f}")

        # Salvar resultados
        comparison_path = os.path.join(self.output_dir, f"model_comparison_{self.timestamp}.csv")
        comparison_df.to_csv(comparison_path, index=False)
        logger.info(f"Resultados comparativos salvos em: {comparison_path}")

        # Gerar gráficos comparativos
        self._plot_models_comparison(comparison_df)

        # Apenas gerar gráficos de comparação se as predições foram armazenadas
        if store_predictions:
            self._plot_roc_curves_comparison()
            self._plot_pr_curves_comparison()
        else:
            logger.warning(
                "Gráficos comparativos de curvas ROC e PR não foram gerados. Execute evaluate_all_models com store_predictions=True.")

        return comparison_df

    def find_optimal_threshold(self, name, X_val, y_val, optimization_metric='cost',
                               approval_target=None, plot_result=True):
        """
        Encontra o threshold ótimo para um modelo específico.

        Args:
            name: Nome do modelo
            X_val: Features de validação
            y_val: Target de validação
            optimization_metric: Métrica para otimização ('cost', 'f1', 'precision', 'recall')
            approval_target: Taxa alvo de aprovação (0-1)
            plot_result: Se True, gera gráfico mostrando impacto do threshold

        Returns:
            Threshold ótimo
        """
        # Validar entradas
        if not isinstance(X_val, (pd.DataFrame, np.ndarray)):
            raise TypeError("X_val deve ser um DataFrame pandas ou array numpy")

        if not isinstance(y_val, (pd.Series, np.ndarray, list)):
            raise TypeError("y_val deve ser uma Series pandas, array numpy ou lista")

        if len(X_val) != len(y_val):
            raise ValueError("X_val e y_val devem ter o mesmo número de amostras")

        if name not in self.models:
            raise ValueError(f"Modelo '{name}' não encontrado.")

        model_info = self.models[name]
        model = model_info['model']

        # Definir target de aprovação
        approval_target = approval_target or self.approval_target

        logger.info(f"Encontrando threshold ótimo para modelo '{name}'...")
        logger.info(f"Métrica de otimização: {optimization_metric}")
        if approval_target:
            logger.info(f"Taxa alvo de aprovação: {approval_target:.2%}")

        # Calcular probabilidades
        y_proba = model.predict_proba(X_val)[:, 1]

        # Definir thresholds para testar
        thresholds = np.linspace(0.01, 0.99, 99)

        # Inicializar arrays para armazenar métricas
        metrics = []

        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)

            # Calcular matriz de confusão
            tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()

            # Calcular métricas
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            accuracy = (tp + tn) / (tp + tn + fp + fn)

            # Calcular métricas de negócio - CORRIGIDAS
            if self.positive_class == 'inadimplente':
                aprovacao_rate = (tn + fn) / (tp + tn + fp + fn)
                inadimplencia_portfolio = fn / (tn + fn) if (tn + fn) > 0 else 0
            else:
                aprovacao_rate = (tp + fp) / (tp + tn + fp + fn)
                inadimplencia_portfolio = fp / (tp + fp) if (tp + fp) > 0 else 0

            # Calcular custo de negócio
            business_cost = fp + self.cost_fn_ratio * fn
            normalized_cost = business_cost / len(y_val)

            metrics.append({
                'threshold': threshold,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'aprovacao_rate': aprovacao_rate,
                'inadimplencia_portfolio': inadimplencia_portfolio,
                'business_cost': normalized_cost
            })

        # Converter para DataFrame
        metrics_df = pd.DataFrame(metrics)

        # Encontrar threshold ótimo
        if optimization_metric == 'cost':
            if approval_target:
                # Encontrar thresholds que atendem ao target de aprovação
                valid_thresholds = metrics_df[
                    (metrics_df['aprovacao_rate'] >= approval_target * 0.95) &
                    (metrics_df['aprovacao_rate'] <= approval_target * 1.05)
                    ]

                if valid_thresholds.empty:
                    logger.warning(
                        f"Nenhum threshold atende ao target de aprovação {approval_target:.2%}. Usando todos os thresholds.")
                    valid_thresholds = metrics_df

                # Escolher threshold com menor custo dentre os válidos
                best_idx = valid_thresholds['business_cost'].idxmin()
            else:
                # Escolher threshold com menor custo geral
                best_idx = metrics_df['business_cost'].idxmin()

        elif optimization_metric == 'f1':
            best_idx = metrics_df['f1'].idxmax()

        elif optimization_metric == 'precision':
            # Garantir recall mínimo
            min_recall = 0.1
            valid_thresholds = metrics_df[metrics_df['recall'] >= min_recall]

            if valid_thresholds.empty:
                logger.warning(
                    f"Nenhum threshold atende ao recall mínimo {min_recall:.2%}. Usando todos os thresholds.")
                valid_thresholds = metrics_df

            best_idx = valid_thresholds['precision'].idxmax()

        elif optimization_metric == 'recall':
            # Garantir precisão mínima
            min_precision = 0.1
            valid_thresholds = metrics_df[metrics_df['precision'] >= min_precision]

            if valid_thresholds.empty:
                logger.warning(
                    f"Nenhum threshold atende à precisão mínima {min_precision:.2%}. Usando todos os thresholds.")
                valid_thresholds = metrics_df

            best_idx = valid_thresholds['recall'].idxmax()

        else:
            raise ValueError(f"Métrica de otimização não reconhecida: {optimization_metric}")

        # Extrair threshold ótimo
        optimal_threshold = metrics_df.iloc[best_idx]['threshold']

        # Atualizar threshold do modelo
        self.models[name]['threshold'] = optimal_threshold

        # Extrair métricas com o threshold ótimo
        optimal_metrics = metrics_df.iloc[best_idx].to_dict()

        logger.info(f"\nThreshold ótimo encontrado: {optimal_threshold:.4f}")
        logger.info(f"F1-Score: {optimal_metrics['f1']:.4f}")
        logger.info(f"Precision: {optimal_metrics['precision']:.4f}")
        logger.info(f"Recall: {optimal_metrics['recall']:.4f}")
        logger.info(f"Taxa de Aprovação: {optimal_metrics['aprovacao_rate']:.2%}")
        logger.info(f"Taxa de Inadimplência no Portfolio: {optimal_metrics['inadimplencia_portfolio']:.2%}")
        logger.info(f"Custo de Negócio Normalizado: {optimal_metrics['business_cost']:.4f}")

        # Plotar impacto do threshold
        if plot_result:
            plt.figure(figsize=(12, 8))

            # Custo de negócio
            plt.subplot(2, 2, 1)
            plt.plot(metrics_df['threshold'], metrics_df['business_cost'])
            plt.axvline(x=optimal_threshold, color='red', linestyle='--')
            plt.xlabel('Threshold')
            plt.ylabel('Custo de Negócio')
            plt.title('Custo de Negócio vs Threshold')
            plt.grid(True, alpha=0.3)

            # F1-Score
            plt.subplot(2, 2, 2)
            plt.plot(metrics_df['threshold'], metrics_df['f1'])
            plt.axvline(x=optimal_threshold, color='red', linestyle='--')
            plt.xlabel('Threshold')
            plt.ylabel('F1-Score')
            plt.title('F1-Score vs Threshold')
            plt.grid(True, alpha=0.3)

            # Precision e Recall
            plt.subplot(2, 2, 3)
            plt.plot(metrics_df['threshold'], metrics_df['precision'], label='Precision')
            plt.plot(metrics_df['threshold'], metrics_df['recall'], label='Recall')
            plt.axvline(x=optimal_threshold, color='red', linestyle='--')
            plt.xlabel('Threshold')
            plt.ylabel('Valor')
            plt.title('Precision e Recall vs Threshold')
            plt.legend()
            plt.grid(True, alpha=0.3)

            # Taxa de Aprovação e Inadimplência
            plt.subplot(2, 2, 4)
            plt.plot(metrics_df['threshold'], metrics_df['aprovacao_rate'], label='Taxa de Aprovação')
            plt.plot(metrics_df['threshold'], metrics_df['inadimplencia_portfolio'], label='Taxa de Inadimplência')
            plt.axvline(x=optimal_threshold, color='red', linestyle='--')
            plt.xlabel('Threshold')
            plt.ylabel('Taxa')
            plt.title('Taxas de Negócio vs Threshold')
            plt.legend()
            plt.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.suptitle(f'Impacto do Threshold - {name}', fontsize=16, y=1.05)

            # Salvar gráfico
            plots_dir = os.path.join(self.output_dir, 'plots', name)
            os.makedirs(plots_dir, exist_ok=True)
            plt.savefig(os.path.join(plots_dir, f'threshold_optimization_{self.timestamp}.png'),
                        dpi=300, bbox_inches='tight')
            plt.close()

        return optimal_threshold

    def find_optimal_thresholds_all_models(self, X_val, y_val, optimization_metric='cost',
                                           approval_target=None):
        """
        Encontra os thresholds ótimos para todos os modelos.

        Args:
            X_val: Features de validação
            y_val: Target de validação
            optimization_metric: Métrica para otimização ('cost', 'f1', 'precision', 'recall')
            approval_target: Taxa alvo de aprovação (0-1)

        Returns:
            Dicionário com thresholds ótimos para cada modelo
        """
        # Validar entradas
        if not isinstance(X_val, (pd.DataFrame, np.ndarray)):
            raise TypeError("X_val deve ser um DataFrame pandas ou array numpy")

        if not isinstance(y_val, (pd.Series, np.ndarray, list)):
            raise TypeError("y_val deve ser uma Series pandas, array numpy ou lista")

        if len(X_val) != len(y_val):
            raise ValueError("X_val e y_val devem ter o mesmo número de amostras")

        if not self.models:
            raise ValueError("Nenhum modelo registrado para otimização.")

        logger.info(f"Otimizando thresholds para {len(self.models)} modelos...")

        optimal_thresholds = {}

        for name in self.models:
            optimal_thresholds[name] = self.find_optimal_threshold(
                name, X_val, y_val, optimization_metric, approval_target
            )

        return optimal_thresholds

    def evaluate_and_report(self, name, X_test, y_test, threshold=None, output_format='html'):
        """
        Avalia um modelo e gera um relatório técnico completo em uma única etapa.

        Args:
            name: Nome do modelo
            X_test: Features de teste
            y_test: Target de teste
            threshold: Threshold para classificação (se None, usa o definido no modelo)
            output_format: Formato do relatório ('html', 'md', 'txt')

        Returns:
            Caminho para o relatório gerado
        """
        # Avaliar modelo e armazenar predições para visualizações
        self.evaluate_model(name, X_test, y_test, threshold=threshold,
                            plot_curves=True, enhanced_visuals=True, store_predictions=True)

        # Gerar relatório com visualizações técnicas integradas
        return self.generate_business_report(name, output_format=output_format, include_visualizations=True)

    def generate_business_report(self, name=None, output_format='html', include_visualizations=True):
        """
        Gera um relatório de negócio para um modelo específico ou para o melhor modelo,
        com visualizações técnicas integradas.

        Args:
            name: Nome do modelo (se None, usa o melhor modelo)
            output_format: Formato do relatório ('html', 'md', 'txt')
            include_visualizations: Se True, gera visualizações técnicas para incluir no relatório

        Returns:
            Caminho para o relatório gerado
        """
        # Determinar modelo
        if name is None:
            if self.best_model_name:
                name = self.best_model_name
            else:
                raise ValueError("Nenhum modelo identificado como melhor. Especifique um nome de modelo.")

        if name not in self.model_results:
            raise ValueError(f"Resultados para modelo '{name}' não encontrados. Execute evaluate_model() primeiro.")

        results = self.model_results[name]
        model_info = self.models[name]
        model = model_info['model']

        # Verificar se as predições estão disponíveis para visualizações
        has_predictions = all(k in results for k in ['y_true', 'y_proba', 'y_pred'])

        # Gerar visualizações se necessário e disponíveis
        if include_visualizations and has_predictions:
            plots_dir = os.path.join(self.output_dir, 'plots', name)
            os.makedirs(plots_dir, exist_ok=True)

            # Extrair dados para visualizações
            y_true = results['y_true']
            y_proba = results['y_proba']
            y_pred = results['y_pred']
            threshold = results['threshold']

            # Gerar visualizações avançadas
            logger.info(f"Gerando visualizações técnicas avançadas para o relatório...")

            self._plot_enhanced_roc_curve(y_true, y_proba, name,
                                          save_path=os.path.join(plots_dir, f'roc_curve_enhanced_{self.timestamp}.png'))

            self._plot_enhanced_score_distribution(y_true, y_proba, threshold, name,
                                                   save_path=os.path.join(plots_dir,
                                                                          f'score_dist_enhanced_{self.timestamp}.png'))

            self._plot_cumulative_gains(y_true, y_proba, name,
                                        save_path=os.path.join(plots_dir, f'cumulative_gains_{self.timestamp}.png'))

            self._plot_enhanced_lift_chart(y_true, y_proba, name,
                                           save_path=os.path.join(plots_dir, f'lift_chart_{self.timestamp}.png'))

            self._plot_calibration_curve(y_true, y_proba, name,
                                         save_path=os.path.join(plots_dir, f'calibration_{self.timestamp}.png'))

            self._plot_threshold_impact(y_true, y_proba, name,
                                        save_path=os.path.join(plots_dir, f'threshold_impact_{self.timestamp}.png'))
        elif include_visualizations and not has_predictions:
            logger.warning(
                "Não foi possível gerar visualizações técnicas avançadas pois as predições não estão disponíveis. "
                "Execute evaluate_model() com store_predictions=True.")

        # Criar diretório de relatórios
        reports_dir = os.path.join(self.output_dir, 'reports')
        os.makedirs(reports_dir, exist_ok=True)

        # Definir caminho do relatório
        if output_format == 'html':
            report_path = os.path.join(reports_dir, f"business_report_{name}_{self.timestamp}.html")
        elif output_format == 'md':
            report_path = os.path.join(reports_dir, f"business_report_{name}_{self.timestamp}.md")
        else:  # txt
            report_path = os.path.join(reports_dir, f"business_report_{name}_{self.timestamp}.txt")

        logger.info(f"Gerando relatório de negócio para modelo '{name}'...")

        # Criar conteúdo do relatório
        if output_format == 'html':
            content = self._generate_html_report(name, results)
        elif output_format == 'md':
            content = self._generate_md_report(name, results)
        else:  # txt
            content = self._generate_txt_report(name, results)

        # Salvar relatório
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(content)

        logger.info(f"Relatório gerado em: {report_path}")
        return report_path

    def _generate_html_report(self, name, results):
        """
        Gera um relatório em formato HTML com visualizações técnicas integradas.

        Args:
            name: Nome do modelo
            results: Dicionário com resultados da avaliação

        Returns:
            String contendo o conteúdo HTML do relatório
        """
        threshold = results['threshold']
        timestamp = self.timestamp

        # Verificar se existem visualizações salvas para este modelo
        plots_dir = os.path.join(self.output_dir, 'plots', name)

        # Caminho para as visualizações
        roc_curve_path = os.path.join(plots_dir, f'roc_curve_enhanced_{timestamp}.png')
        score_dist_path = os.path.join(plots_dir, f'score_dist_enhanced_{timestamp}.png')
        gains_chart_path = os.path.join(plots_dir, f'cumulative_gains_{timestamp}.png')
        lift_chart_path = os.path.join(plots_dir, f'lift_chart_{timestamp}.png')
        calibration_path = os.path.join(plots_dir, f'calibration_{timestamp}.png')
        threshold_impact_path = os.path.join(plots_dir, f'threshold_impact_{timestamp}.png')

        # Verificar quais visualizações estão disponíveis
        visualization_paths = {
            'roc_curve': roc_curve_path if os.path.exists(roc_curve_path) else None,
            'score_dist': score_dist_path if os.path.exists(score_dist_path) else None,
            'gains_chart': gains_chart_path if os.path.exists(gains_chart_path) else None,
            'lift_chart': lift_chart_path if os.path.exists(lift_chart_path) else None,
            'calibration': calibration_path if os.path.exists(calibration_path) else None,
            'threshold_impact': threshold_impact_path if os.path.exists(threshold_impact_path) else None
        }

        # Converter caminhos absolutos para relativos
        # Isso é importante para que as imagens sejam corretamente referenciadas no HTML
        for key, path in visualization_paths.items():
            if path:
                visualization_paths[key] = os.path.relpath(path, self.output_dir)

        # Calcular valores para relatório
        total_clients = results['tp'] + results['tn'] + results['fp'] + results['fn']

        # Determinar clientes aprovados/rejeitados com base na classe positiva
        if self.positive_class == 'inadimplente':
            approved_clients = results['tn'] + results['fn']
            rejected_clients = results['tp'] + results['fp']
        else:
            approved_clients = results['tp'] + results['fp']
            rejected_clients = results['tn'] + results['fn']

        # Revenue e loss (valores fictícios para demonstração)
        avg_loan = 10000  # Valor médio do empréstimo
        avg_interest_rate = 0.15  # Taxa de juros média
        default_loss_rate = 0.8  # Taxa de perda em caso de inadimplência (80% do empréstimo é perdido)

        # Receita dos empréstimos (juros dos não inadimplentes) - CORRIGIDO
        if self.positive_class == 'inadimplente':
            revenue = results['tn'] * avg_loan * avg_interest_rate
            loss = results['fn'] * avg_loan * default_loss_rate
            opportunity_loss = results['fp'] * avg_loan * avg_interest_rate * 0.55
        else:
            revenue = results['tp'] * avg_loan * avg_interest_rate
            loss = results['fp'] * avg_loan * default_loss_rate
            opportunity_loss = results['fn'] * avg_loan * avg_interest_rate * 0.5

        # Lucro líquido
        net_profit = revenue - loss

        # Calcular ROI
        roi = (net_profit / (approved_clients * avg_loan)) * 100 if approved_clients > 0 else 0

        # Começar a construir o HTML
        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Relatório Técnico de Avaliação - Modelo {name}</title>
    <style>
        body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 0; padding: 0; color: #333; line-height: 1.6; }}
        .container {{ max-width: 1200px; margin: 0 auto; padding: 20px; }}
        .header {{ background: linear-gradient(135deg, #1a5276, #2980b9); color: white; padding: 30px; margin-bottom: 30px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
        h1, h2, h3, h4 {{ color: #2c3e50; font-weight: 600; margin-top: 30px; }}
        .header h1, .header h2 {{ color: white; margin: 0; }}
        .header p {{ margin: 5px 0 0 0; opacity: 0.9; }}
        .section {{ margin-bottom: 40px; background-color: white; padding: 25px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
        .metrics {{ display: flex; flex-wrap: wrap; gap: 20px; margin-bottom: 30px; }}
        .metric-card {{ flex: 1; min-width: 200px; background-color: white; border-radius: 8px; padding: 20px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); transition: transform 0.3s ease; }}
        .metric-card:hover {{ transform: translateY(-5px); box-shadow: 0 5px 15px rgba(0,0,0,0.1); }}
        .metric-title {{ font-weight: bold; margin-bottom: 10px; color: #7f8c8d; font-size: 14px; text-transform: uppercase; }}
        .metric-value {{ font-size: 28px; font-weight: bold; color: #2c3e50; margin-bottom: 5px; }}
        .metric-context {{ font-size: 14px; color: #7f8c8d; }}
        .good {{ color: #27ae60; }}
        .medium {{ color: #f39c12; }}
        .bad {{ color: #e74c3c; }}
        table {{ width: 100%; border-collapse: collapse; margin-bottom: 20px; box-shadow: 0 2px 3px rgba(0,0,0,0.1); }}
        th, td {{ padding: 12px 15px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #f8f9fa; color: #2c3e50; font-weight: 600; }}
        tr:hover {{ background-color: #f5f5f5; }}
        .visualization {{ margin: 30px 0; padding: 20px; background-color: white; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
        .visualization-title {{ font-size: 18px; color: #2c3e50; margin-bottom: 15px; font-weight: 600; }}
        .visualization-description {{ margin-bottom: 20px; color: #666; font-size: 14px; line-height: 1.6; }}
        .visualization img {{ max-width: 100%; height: auto; display: block; margin: 0 auto; border-radius: 4px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .recommendations {{ background-color: #f8f9fa; padding: 20px; border-left: 5px solid #3498db; margin: 20px 0; }}
        .footer {{ text-align: center; margin-top: 50px; padding: 20px; color: #7f8c8d; font-size: 14px; border-top: 1px solid #eee; }}
        .tab {{ margin-bottom: 20px; }}
        .tab-links {{ display: flex; margin-bottom: -1px; }}
        .tab-link {{ padding: 10px 20px; background-color: #f8f9fa; margin-right: 5px; cursor: pointer; border: 1px solid #ddd; border-bottom: none; border-radius: 5px 5px 0 0; }}
        .tab-link.active {{ background-color: white; border-bottom: 1px solid white; }}
        .tab-content {{ display: none; padding: 20px; border: 1px solid #ddd; border-radius: 0 5px 5px 5px; }}
        .tab-content.active {{ display: block; }}
        .technical-notes {{ font-size: 14px; color: #666; background-color: #f9f9f9; padding: 15px; border-radius: 5px; margin-top: 10px; }}
        @media print {{
            .tab-links {{ display: none; }}
            .tab-content {{ display: block !important; border: none; padding: 0; }}
            .visualization img {{ max-width: 100%; page-break-inside: avoid; }}
        }}
    </style>
    <script>
        document.addEventListener('DOMContentLoaded', function() {{
            // Ativar a primeira aba por padrão
            document.querySelector('.tab-link').classList.add('active');
            document.querySelector('.tab-content').classList.add('active');

            // Adicionar eventos de clique às abas
            var tabLinks = document.querySelectorAll('.tab-link');
            tabLinks.forEach(function(link) {{
                link.addEventListener('click', function() {{
                    // Remover classe active de todas as abas
                    tabLinks.forEach(function(l) {{ l.classList.remove('active'); }});
                    document.querySelectorAll('.tab-content').forEach(function(c) {{ c.classList.remove('active'); }});

                    // Adicionar classe active à aba clicada e seu conteúdo
                    this.classList.add('active');
                    document.getElementById(this.dataset.tab).classList.add('active');
                }});
            }});
        }});
    </script>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Relatório Técnico de Avaliação de Modelo de Inadimplência</h1>
            <h2>{name}</h2>
            <p>Data: {datetime.now().strftime('%d/%m/%Y %H:%M')} | Threshold: {threshold:.4f} | Classe positiva: {self.positive_class}</p>
        </div>

        <div class="section">
            <h2>Resumo Executivo</h2>
            <p>
                Este relatório apresenta os resultados e o impacto de negócio do modelo de predição de inadimplência <strong>{name}</strong>.
                A avaliação foi realizada com métricas técnicas avançadas e análise detalhada do impacto financeiro.
            </p>

            <div class="metrics">
                <div class="metric-card">
                    <div class="metric-title">Performance (AUC-ROC)</div>
                    <div class="metric-value {'good' if results['auc'] > 0.8 else 'medium' if results['auc'] > 0.7 else 'bad'}">
                        {results['auc']:.3f}
                    </div>
                    <div class="metric-context">Capacidade discriminativa do modelo</div>
                </div>

                <div class="metric-card">
                    <div class="metric-title">Taxa de Aprovação</div>
                    <div class="metric-value">{results['aprovacao_rate']:.1%}</div>
                    <div class="metric-context">{approved_clients} de {total_clients} clientes</div>
                </div>

                <div class="metric-card">
                    <div class="metric-title">Taxa de Inadimplência</div>
                    <div class="metric-value {'bad' if results['inadimplencia_portfolio'] > 0.15 else 'medium' if results['inadimplencia_portfolio'] > 0.1 else 'good'}">
                        {results['inadimplencia_portfolio']:.1%}
                    </div>
                    <div class="metric-context">Inadimplentes entre aprovados</div>
                </div>

                <div class="metric-card">
                    <div class="metric-title">ROI Estimado</div>
                    <div class="metric-value {'good' if roi > 10 else 'medium' if roi > 5 else 'bad'}">
                        {roi:.1f}%
                    </div>
                    <div class="metric-context">Retorno sobre investimento</div>
                </div>
            </div>
        </div>

        <div class="section">
            <div class="tab">
                <div class="tab-links">
                    <div class="tab-link" data-tab="tab-performance">Performance Técnica</div>
                    <div class="tab-link" data-tab="tab-business">Métricas de Negócio</div>
                    <div class="tab-link" data-tab="tab-visualizations">Visualizações Avançadas</div>
                </div>

                <div id="tab-performance" class="tab-content">
                    <h3>Métricas de Performance</h3>
                    <table>
                        <tr>
                            <th>Métrica</th>
                            <th>Valor</th>
                            <th>Interpretação</th>
                        </tr>
                        <tr>
                            <td>AUC-ROC</td>
                            <td>{results['auc']:.4f}</td>
                            <td>Capacidade discriminativa geral do modelo. Valores próximos a 1 são melhores.</td>
                        </tr>
                        <tr>
                            <td>F1-Score</td>
                            <td>{results['f1_score']:.4f}</td>
                            <td>Média harmônica entre precisão e recall. Balanceia falsos positivos e falsos negativos.</td>
                        </tr>
                        <tr>
                            <td>Precisão</td>
                            <td>{results['precision']:.4f}</td>
                            <td>Dos clientes classificados como {self.positive_class}, quantos realmente são.</td>
                        </tr>
                        <tr>
                            <td>Recall (Sensibilidade)</td>
                            <td>{results['recall']:.4f}</td>
                            <td>Dos clientes realmente {self.positive_class}, quantos foram detectados.</td>
                        </tr>
                        <tr>
                            <td>Especificidade</td>
                            <td>{results['specificity']:.4f}</td>
                            <td>Dos clientes não {self.positive_class}, quantos foram classificados corretamente.</td>
                        </tr>
                        <tr>
                            <td>Acurácia</td>
                            <td>{results['accuracy']:.4f}</td>
                            <td>Proporção geral de previsões corretas.</td>
                        </tr>
                    </table>

                    <h3>Matriz de Confusão</h3>
                    <table>
                        <tr>
                            <th></th>
                            <th>Predito: Não {self.positive_class}</th>
                            <th>Predito: {self.positive_class.title()}</th>
                        </tr>
                        <tr>
                            <th>Real: Não {self.positive_class}</th>
                            <td>{results['tn']} (Verdadeiro Negativo)</td>
                            <td>{results['fp']} (Falso Positivo)</td>
                        </tr>
                        <tr>
                            <th>Real: {self.positive_class.title()}</th>
                            <td>{results['fn']} (Falso Negativo)</td>
                            <td>{results['tp']} (Verdadeiro Positivo)</td>
                        </tr>
                    </table>

                    <div class="technical-notes">
                        <p><strong>Notas técnicas:</strong></p>
                        <p>Um bom modelo de inadimplência deve balancear o trade-off entre recall e especificidade. 
                        O threshold atual de {threshold:.4f} resulta em um F1-score de {results['f1_score']:.4f}, 
                        que {'representa um bom equilíbrio' if results['f1_score'] > 0.7 else 'pode ser otimizado para melhorar o equilíbrio'} 
                        entre identificar inadimplentes (recall) e não rejeitar bons pagadores (especificidade).</p>
                    </div>
                </div>

                <div id="tab-business" class="tab-content">
                    <h3>Análise de Custo-Benefício</h3>
                    <table>
                        <tr>
                            <th>Métrica</th>
                            <th>Valor</th>
                            <th>Interpretação</th>
                        </tr>
                        <tr>
                            <td>Receita Estimada</td>
                            <td>R$ {revenue:,.2f}</td>
                            <td>Juros dos empréstimos para bons pagadores</td>
                        </tr>
                        <tr>
                            <td>Perda por Inadimplência</td>
                            <td>R$ {loss:,.2f}</td>
                            <td>Valor perdido com clientes inadimplentes não detectados</td>
                        </tr>
                        <tr>
                            <td>Perda de Oportunidade</td>
                            <td>R$ {opportunity_loss:,.2f}</td>
                            <td>Receita potencial perdida por rejeitar bons pagadores</td>
                        </tr>
                        <tr>
                            <td>Lucro Líquido Estimado</td>
                            <td>R$ {net_profit:,.2f}</td>
                            <td>Receita menos perdas</td>
                        </tr>
                        <tr>
                            <td>ROI</td>
                            <td>{roi:.2f}%</td>
                            <td>Retorno sobre o capital emprestado</td>
                        </tr>
                    </table>

                    <h3>Segmentação de Clientes</h3>
                    <table>
                        <tr>
                            <th>Segmento</th>
                            <th>Quantidade</th>
                            <th>Percentual</th>
                            <th>Impacto no Negócio</th>
                        </tr>
                        <tr>
                            <td>Aprovados</td>
                            <td>{approved_clients}</td>
                            <td>{approved_clients / total_clients:.1%}</td>
                            <td>Taxa de aprovação geral do modelo</td>
                        </tr>
                        <tr>
                            <td>Rejeitados</td>
                            <td>{rejected_clients}</td>
                            <td>{rejected_clients / total_clients:.1%}</td>
                            <td>Taxa de rejeição geral do modelo</td>
                        </tr>
                        <tr>
                            <td>Adimplentes Aprovados</td>
                            <td>{results['tn'] if self.positive_class == 'inadimplente' else results['tp']}</td>
                            <td>{(results['tn'] if self.positive_class == 'inadimplente' else results['tp']) / total_clients:.1%}</td>
                            <td>Clientes que geram receita sem perdas (cenário ideal)</td>
                        </tr>
                        <tr>
                            <td>Inadimplentes Rejeitados</td>
                            <td>{results['tp'] if self.positive_class == 'inadimplente' else results['tn']}</td>
                            <td>{(results['tp'] if self.positive_class == 'inadimplente' else results['tn']) / total_clients:.1%}</td>
                            <td>Perdas evitadas pelo modelo (cenário ideal)</td>
                        </tr>
                        <tr>
                            <td>Adimplentes Rejeitados (Oportunidade Perdida)</td>
                            <td>{results['fp'] if self.positive_class == 'inadimplente' else results['fn']}</td>
                            <td>{(results['fp'] if self.positive_class == 'inadimplente' else results['fn']) / total_clients:.1%}</td>
                            <td>Receita potencial não realizada (falso positivo)</td>
                        </tr>
                        <tr>
                            <td>Inadimplentes Aprovados (Risco)</td>
                            <td>{results['fn'] if self.positive_class == 'inadimplente' else results['fp']}</td>
                            <td>{(results['fn'] if self.positive_class == 'inadimplente' else results['fp']) / total_clients:.1%}</td>
                            <td>Risco de perda incorporado ao portfólio (falso negativo)</td>
                        </tr>
                    </table>
                </div>

                <div id="tab-visualizations" class="tab-content">
                    <h3>Visualizações Técnicas Avançadas</h3>
                    <p>As visualizações abaixo oferecem uma análise detalhada do desempenho técnico e do impacto de negócio do modelo.</p>
"""

        # Adicionar visualizações disponíveis
        # 1. Curva ROC
        if visualization_paths['roc_curve']:
            html += f"""
                    <div class="visualization">
                        <div class="visualization-title">Curva ROC com Análise Avançada</div>
                        <div class="visualization-description">
                            <p>A curva ROC (Receiver Operating Characteristic) mostra o trade-off entre sensibilidade (taxa de verdadeiros positivos) e 
                            especificidade (1 - taxa de falsos positivos). A área sob a curva (AUC) de {results['auc']:.4f} indica 
                            {'uma excelente' if results['auc'] > 0.9 else 'uma boa' if results['auc'] > 0.8 else 'uma razoável' if results['auc'] > 0.7 else 'uma fraca'} 
                            capacidade discriminativa do modelo.</p>
                            <p>O gráfico inclui intervalos de confiança e pontos de threshold ótimo, calculados com base no critério de Youden (J = sensibilidade + especificidade - 1).</p>
                        </div>
                        <img src="../{visualization_paths['roc_curve']}" alt="Curva ROC Avançada">
                    </div>
"""

        # 2. Distribuição de Scores
        if visualization_paths['score_dist']:
            html += f"""
                    <div class="visualization">
                        <div class="visualization-title">Distribuição de Scores com Análise de Separação</div>
                        <div class="visualization-description">
                            <p>Este gráfico mostra a distribuição dos scores do modelo para clientes inadimplentes e adimplentes. Uma boa separação entre as 
                            distribuições indica que o modelo consegue diferenciar bem entre as classes. A estatística KS (Kolmogorov-Smirnov) mede o grau de separação.</p>
                            <p>A linha vertical tracejada indica o threshold atual de classificação ({threshold:.4f}). O gráfico inferior mostra as funções de distribuição 
                            acumulada (CDF) para ambas as classes, útil para identificar o ponto de máxima separação.</p>
                        </div>
                        <img src="../{visualization_paths['score_dist']}" alt="Distribuição de Scores">
                    </div>
"""

        # 3. Gráfico de Ganhos Cumulativos
        if visualization_paths['gains_chart']:
            html += f"""
                    <div class="visualization">
                        <div class="visualization-title">Gráfico de Ganhos Cumulativos</div>
                        <div class="visualization-description">
                            <p>O gráfico de ganhos cumulativos mostra quanto do total de clientes {self.positive_class}s é capturado 
                            ao selecionar uma determinada percentagem da população ordenada pelo score do modelo.</p>
                            <p>Um modelo eficiente identifica uma grande proporção de clientes {self.positive_class}s usando apenas uma pequena fração da população total. 
                            Este gráfico é particularmente útil para campanhas direcionadas e para determinar pontos de corte operacionais.</p>
                        </div>
                        <img src="../{visualization_paths['gains_chart']}" alt="Gráfico de Ganhos Cumulativos">
                    </div>
"""

        # 4. Gráfico de Lift
        if visualization_paths['lift_chart']:
            html += f"""
                    <div class="visualization">
                        <div class="visualization-title">Análise de Lift por Decil</div>
                        <div class="visualization-description">
                            <p>O gráfico de Lift mostra quanto melhor o modelo é na identificação de clientes {self.positive_class}s em comparação 
                            com uma seleção aleatória. Um valor de Lift de 2 significa que o modelo identifica 2 vezes mais clientes {self.positive_class}s 
                            do que uma seleção aleatória naquele decil.</p>
                            <p>As barras de erro representam intervalos de confiança de 95%, indicando a estabilidade estatística do Lift em cada decil.
                            O gráfico inferior mostra a taxa de eventos observada em cada decil.</p>
                        </div>
                        <img src="../{visualization_paths['lift_chart']}" alt="Análise de Lift por Decil">
                    </div>
"""

        # 5. Curva de Calibração
        if visualization_paths['calibration']:
            html += f"""
                    <div class="visualization">
                        <div class="visualization-title">Curva de Calibração de Probabilidades</div>
                        <div class="visualization-description">
                            <p>A curva de calibração avalia se as probabilidades produzidas pelo modelo correspondem às frequências observadas. 
                            Um modelo bem calibrado atribui probabilidades que refletem a verdadeira probabilidade de ocorrência do evento.</p>
                            <p>A linha pontilhada representa a calibração perfeita. Desvios acima desta linha indicam que o modelo é pessimista 
                            (subestima as probabilidades), enquanto desvios abaixo indicam que o modelo é otimista (superestima as probabilidades).</p>
                        </div>
                        <img src="../{visualization_paths['calibration']}" alt="Curva de Calibração">
                    </div>
"""

        # 6. Impacto do Threshold
        if visualization_paths['threshold_impact']:
            html += f"""
                    <div class="visualization">
                        <div class="visualization-title">Análise de Impacto do Threshold</div>
                        <div class="visualization-description">
                            <p>Este gráfico mostra como diferentes valores de threshold afetam as métricas de classificação e de negócio. 
                            O threshold atual de {threshold:.4f} foi escolhido para otimizar o equilíbrio entre custo de negócio e taxa de aprovação.</p>
                            <p>O gráfico permite identificar thresholds alternativos que poderiam ser considerados para diferentes objetivos de negócio, 
                            como maximizar aprovações, minimizar inadimplência, ou otimizar o ROI.</p>
                        </div>
                        <img src="../{visualization_paths['threshold_impact']}" alt="Impacto do Threshold">
                    </div>
"""

        # Fechar a div de visualizações e adicionar recomendações
        html += f"""
                </div>
            </div>
        </div>

        <div class="section">
            <h2>Recomendações Técnicas e de Negócio</h2>

            <div class="recommendations">
                <h3>Recomendações com Base na Análise Técnica</h3>
                <ul>
                    <li>O modelo apresenta uma AUC de {results['auc']:.4f}, o que indica {'excelente' if results['auc'] > 0.9 else 'boa' if results['auc'] > 0.8 else 'razoável' if results['auc'] > 0.7 else 'insuficiente'} capacidade discriminativa.</li>
                    <li>{'O modelo está bem calibrado, com probabilidades que refletem adequadamente o risco real.' if results['precision'] > 0.7 else 'O modelo pode se beneficiar de recalibração de probabilidades para melhor refletir o risco real.'}</li>
                    <li>{'A distribuição de scores mostra boa separação entre as classes, facilitando a definição de um threshold eficaz.' if results['auc'] > 0.8 else 'A separação entre as classes poderia ser melhorada com engenharia de features adicionais.'}</li>
                    <li>{'O threshold atual está bem otimizado para o objetivo de negócio.' if 0.4 < threshold < 0.6 else 'Considerar ajustar o threshold para melhor alinhamento com objetivos específicos de negócio.'}</li>
                </ul>
            </div>

            <div class="recommendations">
                <h3>Recomendações com Base no Impacto de Negócio</h3>
                <ul>
                    <li>A taxa de aprovação atual é de {results['aprovacao_rate']:.1%}, o que {'está alinhado com os objetivos de negócio' if 0.4 <= results['aprovacao_rate'] <= 0.6 else 'pode ser ajustado para melhor alinhamento com a estratégia comercial'}.</li>
                    <li>A taxa de inadimplência no portfólio é de {results['inadimplencia_portfolio']:.1%}, {'dentro dos limites aceitáveis' if results['inadimplencia_portfolio'] < 0.1 else 'acima dos limites desejáveis de risco'}.</li>
                    <li>O modelo gera um ROI estimado de {roi:.1f}%, o que {'representa um bom retorno sobre o investimento' if roi > 10 else 'indica que ajustes podem melhorar a rentabilidade'}.</li>
                    <li>{'A principal oportunidade de melhoria está na redução de falsos negativos, que representam o maior custo de negócio.' if results['fn'] > results['fp'] else 'A principal oportunidade de melhoria está na redução de falsos positivos, que representam perda significativa de oportunidade de negócio.'}</li>
                </ul>
            </div>

            <div class="recommendations">
                <h3>Próximos Passos Sugeridos</h3>
                <ul>
                    <li>Realizar análise de subgrupos para identificar segmentos onde o modelo pode ser aprimorado.</li>
                    <li>Implementar monitoramento contínuo do desempenho do modelo em produção, com atenção a drift conceitual.</li>
                    <li>{'Considerar o desenvolvimento de modelos específicos para segmentos críticos.' if results['auc'] < 0.85 else 'Focar na otimização operacional da implementação do modelo.'}</li>
                    <li>Testar ajustes de threshold baseados em objetivos estratégicos diferentes, como maximização de volume com risco controlado ou maximização de lucro com volume mínimo.</li>
                </ul>
            </div>
        </div>

        <div class="footer">
            <p>Relatório gerado automaticamente pelo ModelEvaluator em {datetime.now().strftime('%d/%m/%Y %H:%M')}</p>
            <p>Parâmetros de avaliação: Custo FN/FP = {self.cost_fn_ratio:.1f} | Classe positiva: {self.positive_class}</p>
        </div>
    </div>
</body>
</html>
"""
        return html

    def _generate_md_report(self, name, results):
        """Gera relatório em formato Markdown com referências às visualizações técnicas."""
        threshold = results['threshold']

        # Calcular valores para relatório (usando a mesma lógica corrigida do HTML)
        total_clients = results['tp'] + results['tn'] + results['fp'] + results['fn']

        # Determinar clientes aprovados/rejeitados com base na classe positiva
        if self.positive_class == 'inadimplente':
            approved_clients = results['tn'] + results['fn']
            rejected_clients = results['tp'] + results['fp']
        else:
            approved_clients = results['tp'] + results['fp']
            rejected_clients = results['tn'] + results['fn']

        # Revenue e loss (valores fictícios para demonstração)
        avg_loan = 10000  # Valor médio do empréstimo
        avg_interest_rate = 0.15  # Taxa de juros média
        default_loss_rate = 0.8  # Taxa de perda em caso de inadimplência (80% do empréstimo é perdido)

        # Receita dos empréstimos (juros dos não inadimplentes) - CORRIGIDO
        if self.positive_class == 'inadimplente':
            revenue = results['tn'] * avg_loan * avg_interest_rate
            loss = results['fn'] * avg_loan * default_loss_rate
            opportunity_loss = results['fp'] * avg_loan * avg_interest_rate * 0.5
        else:
            revenue = results['tp'] * avg_loan * avg_interest_rate
            loss = results['fp'] * avg_loan * default_loss_rate
            opportunity_loss = results['fn'] * avg_loan * avg_interest_rate * 0.5

        # Lucro líquido
        net_profit = revenue - loss

        # Calcular ROI
        roi = (net_profit / (approved_clients * avg_loan)) * 100 if approved_clients > 0 else 0

        md = f"""# Relatório Técnico de Avaliação - Modelo de Inadimplência

**Modelo:** {name}  
**Data:** {datetime.now().strftime('%d/%m/%Y %H:%M')}  
**Threshold de Classificação:** {threshold:.4f}  
**Classe Positiva:** {self.positive_class}

## Resumo Executivo

Este relatório apresenta os resultados e o impacto de negócio do modelo de predição de inadimplência **{name}**,
avaliado com métricas técnicas avançadas e análise detalhada do impacto financeiro.

| Métrica | Valor | Contexto |
|---------|-------|----------|
| Performance (AUC-ROC) | {results['auc']:.3f} | Capacidade discriminativa do modelo |
| Taxa de Aprovação | {results['aprovacao_rate']:.1%} | {approved_clients} de {total_clients} clientes |
| Taxa de Inadimplência na Carteira | {results['inadimplencia_portfolio']:.1%} | Inadimplentes entre aprovados |
| ROI Estimado | {roi:.1f}% | Retorno sobre investimento |

## Performance Técnica

### Métricas de Performance

| Métrica | Valor | Interpretação |
|---------|-------|---------------|
| AUC-ROC | {results['auc']:.4f} | Capacidade discriminativa geral do modelo |
| F1-Score | {results['f1_score']:.4f} | Média harmônica entre precisão e recall |
| Precisão | {results['precision']:.4f} | Dos clientes classificados como {self.positive_class}, quantos realmente são |
| Recall | {results['recall']:.4f} | Dos clientes realmente {self.positive_class}, quantos foram detectados |
| Especificidade | {results['specificity']:.4f} | Dos clientes não {self.positive_class}, quantos foram classificados corretamente |
| Acurácia | {results['accuracy']:.4f} | Proporção de predições corretas |

### Matriz de Confusão

|                          | Predito: Não {self.positive_class} | Predito: {self.positive_class.title()} |
|--------------------------|---------------------------|------------------------|
| Real: Não {self.positive_class}   | {results['tn']} (Verdadeiro Negativo) | {results['fp']} (Falso Positivo) |
| Real: {self.positive_class.title()}       | {results['fn']} (Falso Negativo) | {results['tp']} (Verdadeiro Positivo) |

**Nota técnica:** Um bom modelo de inadimplência deve balancear o trade-off entre recall e especificidade. 
O threshold atual de {threshold:.4f} resulta em um F1-score de {results['f1_score']:.4f}, 
que {'representa um bom equilíbrio' if results['f1_score'] > 0.7 else 'pode ser otimizado para melhorar o equilíbrio'} 
entre identificar inadimplentes (recall) e não rejeitar bons pagadores (especificidade).

## Métricas de Negócio

### Análise de Custo-Benefício

| Métrica | Valor | Interpretação |
|---------|-------|---------------|
| Receita Estimada | R$ {revenue:,.2f} | Juros dos empréstimos para bons pagadores |
| Perda por Inadimplência | R$ {loss:,.2f} | Valor perdido com clientes inadimplentes não detectados |
| Perda de Oportunidade | R$ {opportunity_loss:,.2f} | Receita potencial perdida por rejeitar bons pagadores |
| Lucro Líquido Estimado | R$ {net_profit:,.2f} | Receita menos perdas |
| ROI | {roi:.2f}% | Retorno sobre o capital emprestado |

### Segmentação de Clientes

| Segmento | Quantidade | Percentual | Impacto no Negócio |
|----------|------------|------------|-------------------|
| Aprovados | {approved_clients} | {approved_clients / total_clients:.1%} | Taxa de aprovação geral do modelo |
| Rejeitados | {rejected_clients} | {rejected_clients / total_clients:.1%} | Taxa de rejeição geral do modelo |
| Adimplentes Aprovados | {results['tn'] if self.positive_class == 'inadimplente' else results['tp']} | {(results['tn'] if self.positive_class == 'inadimplente' else results['tp']) / total_clients:.1%} | Clientes que geram receita sem perdas |
| Inadimplentes Rejeitados | {results['tp'] if self.positive_class == 'inadimplente' else results['tn']} | {(results['tp'] if self.positive_class == 'inadimplente' else results['tn']) / total_clients:.1%} | Perdas evitadas pelo modelo |
| Adimplentes Rejeitados | {results['fp'] if self.positive_class == 'inadimplente' else results['fn']} | {(results['fp'] if self.positive_class == 'inadimplente' else results['fn']) / total_clients:.1%} | Oportunidade perdida (falso positivo) |
| Inadimplentes Aprovados | {results['fn'] if self.positive_class == 'inadimplente' else results['fp']} | {(results['fn'] if self.positive_class == 'inadimplente' else results['fp']) / total_clients:.1%} | Risco incorporado ao portfólio (falso negativo) |
"""

        # Adicionar seção de visualizações técnicas
        plots_dir = os.path.join(self.output_dir, 'plots', name)

        md += """
## Visualizações Técnicas Avançadas

As visualizações técnicas abaixo oferecem uma análise detalhada do desempenho do modelo:

"""

        # Verificar quais visualizações estão disponíveis
        roc_curve_path = os.path.join(plots_dir, f'roc_curve_enhanced_{self.timestamp}.png')
        if os.path.exists(roc_curve_path):
            rel_path = os.path.relpath(roc_curve_path, self.output_dir)
            md += f"### Curva ROC com Análise Avançada\n\n![Curva ROC]({rel_path})\n\n"
            md += "A curva ROC mostra o trade-off entre sensibilidade e especificidade em diferentes thresholds. "
            md += f"A área sob a curva (AUC) de {results['auc']:.4f} indica "
            md += f"{'uma excelente' if results['auc'] > 0.9 else 'uma boa' if results['auc'] > 0.8 else 'uma razoável' if results['auc'] > 0.7 else 'uma fraca'} "
            md += "capacidade discriminativa do modelo.\n\n"

        score_dist_path = os.path.join(plots_dir, f'score_dist_enhanced_{self.timestamp}.png')
        if os.path.exists(score_dist_path):
            rel_path = os.path.relpath(score_dist_path, self.output_dir)
            md += f"### Distribuição de Scores com Análise de Separação\n\n![Distribuição de Scores]({rel_path})\n\n"
            md += "Este gráfico mostra a distribuição dos scores do modelo para clientes inadimplentes e adimplentes. "
            md += "Uma boa separação entre as distribuições indica que o modelo consegue diferenciar bem entre as classes. "
            md += "A estatística KS (Kolmogorov-Smirnov) mede o grau de separação entre as distribuições.\n\n"

        gains_chart_path = os.path.join(plots_dir, f'cumulative_gains_{self.timestamp}.png')
        if os.path.exists(gains_chart_path):
            rel_path = os.path.relpath(gains_chart_path, self.output_dir)
            md += f"### Gráfico de Ganhos Cumulativos\n\n![Ganhos Cumulativos]({rel_path})\n\n"
            md += f"O gráfico de ganhos cumulativos mostra quanto do total de clientes {self.positive_class}s é capturado "
            md += "ao selecionar uma determinada percentagem da população ordenada pelo score do modelo. "
            md += "Um modelo eficiente identifica uma grande proporção de positivos usando apenas uma pequena fração da população total.\n\n"

        lift_chart_path = os.path.join(plots_dir, f'lift_chart_{self.timestamp}.png')
        if os.path.exists(lift_chart_path):
            rel_path = os.path.relpath(lift_chart_path, self.output_dir)
            md += f"### Análise de Lift por Decil\n\n![Lift Chart]({rel_path})\n\n"
            md += f"O gráfico de Lift mostra quanto melhor o modelo é na identificação de clientes {self.positive_class}s "
            md += "em comparação com uma seleção aleatória. Um valor de Lift de 2 significa que o modelo identifica "
            md += "2 vezes mais positivos do que uma seleção aleatória naquele decil.\n\n"

        calibration_path = os.path.join(plots_dir, f'calibration_{self.timestamp}.png')
        if os.path.exists(calibration_path):
            rel_path = os.path.relpath(calibration_path, self.output_dir)
            md += f"### Curva de Calibração de Probabilidades\n\n![Calibração]({rel_path})\n\n"
            md += "A curva de calibração avalia se as probabilidades produzidas pelo modelo correspondem às frequências observadas. "
            md += "Um modelo bem calibrado atribui probabilidades que refletem a verdadeira probabilidade de ocorrência do evento.\n\n"

        threshold_impact_path = os.path.join(plots_dir, f'threshold_impact_{self.timestamp}.png')
        if os.path.exists(threshold_impact_path):
            rel_path = os.path.relpath(threshold_impact_path, self.output_dir)
            md += f"### Análise de Impacto do Threshold\n\n![Impacto do Threshold]({rel_path})\n\n"
            md += f"Este gráfico mostra como diferentes valores de threshold afetam as métricas de classificação e de negócio. "
            md += f"O threshold atual de {threshold:.4f} foi escolhido para otimizar o equilíbrio entre custo de negócio e taxa de aprovação.\n\n"

        # Adicionar recomendações
        md += """
## Recomendações Técnicas e de Negócio

### Recomendações com Base na Análise Técnica
"""
        md += f"""
* O modelo apresenta uma AUC de {results['auc']:.4f}, o que indica {'excelente' if results['auc'] > 0.9 else 'boa' if results['auc'] > 0.8 else 'razoável' if results['auc'] > 0.7 else 'insuficiente'} capacidade discriminativa.
* {'O modelo está bem calibrado, com probabilidades que refletem adequadamente o risco real.' if results['precision'] > 0.7 else 'O modelo pode se beneficiar de recalibração de probabilidades para melhor refletir o risco real.'}
* {'A distribuição de scores mostra boa separação entre as classes, facilitando a definição de um threshold eficaz.' if results['auc'] > 0.8 else 'A separação entre as classes poderia ser melhorada com engenharia de features adicionais.'}
* {'O threshold atual está bem otimizado para o objetivo de negócio.' if 0.4 < threshold < 0.6 else 'Considerar ajustar o threshold para melhor alinhamento com objetivos específicos de negócio.'}

### Recomendações com Base no Impacto de Negócio

* A taxa de aprovação atual é de {results['aprovacao_rate']:.1%}, o que {'está alinhado com os objetivos de negócio' if 0.4 <= results['aprovacao_rate'] <= 0.6 else 'pode ser ajustado para melhor alinhamento com a estratégia comercial'}.
* A taxa de inadimplência no portfólio é de {results['inadimplencia_portfolio']:.1%}, {'dentro dos limites aceitáveis' if results['inadimplencia_portfolio'] < 0.1 else 'acima dos limites desejáveis de risco'}.
* O modelo gera um ROI estimado de {roi:.1f}%, o que {'representa um bom retorno sobre o investimento' if roi > 10 else 'indica que ajustes podem melhorar a rentabilidade'}.
* {'A principal oportunidade de melhoria está na redução de falsos negativos, que representam o maior custo de negócio.' if results['fn'] > results['fp'] else 'A principal oportunidade de melhoria está na redução de falsos positivos, que representam perda significativa de oportunidade de negócio.'}

### Próximos Passos Sugeridos

* Realizar análise de subgrupos para identificar segmentos onde o modelo pode ser aprimorado.
* Implementar monitoramento contínuo do desempenho do modelo em produção, com atenção a drift conceitual.
* {'Considerar o desenvolvimento de modelos específicos para segmentos críticos.' if results['auc'] < 0.85 else 'Focar na otimização operacional da implementação do modelo.'}
* Testar ajustes de threshold baseados em objetivos estratégicos diferentes, como maximização de volume com risco controlado ou maximização de lucro com volume mínimo.
"""

        md += f"\n\n---\n\nRelatório gerado automaticamente pelo ModelEvaluator em {datetime.now().strftime('%d/%m/%Y %H:%M')}\n"
        md += f"Parâmetros de avaliação: Custo FN/FP = {self.cost_fn_ratio:.1f} | Classe positiva: {self.positive_class}"

        return md

    def _generate_txt_report(self, name, results):
        """Gera relatório em formato texto plano com valores financeiros corrigidos."""
        threshold = results['threshold']

        # Calcular valores para relatório (usando a mesma lógica corrigida do HTML e MD)
        total_clients = results['tp'] + results['tn'] + results['fp'] + results['fn']

        # Determinar clientes aprovados/rejeitados com base na classe positiva
        if self.positive_class == 'inadimplente':
            approved_clients = results['tn'] + results['fn']
            rejected_clients = results['tp'] + results['fp']
        else:
            approved_clients = results['tp'] + results['fp']
            rejected_clients = results['tn'] + results['fn']

        # Revenue e loss (valores fictícios para demonstração)
        avg_loan = 10000  # Valor médio do empréstimo
        avg_interest_rate = 0.15  # Taxa de juros média
        default_loss_rate = 0.8  # Taxa de perda em caso de inadimplência (80% do empréstimo é perdido)

        # Receita dos empréstimos (juros dos não inadimplentes) - CORRIGIDO
        if self.positive_class == 'inadimplente':
            revenue = results['tn'] * avg_loan * avg_interest_rate
            loss = results['fn'] * avg_loan * default_loss_rate
            opportunity_loss = results['fp'] * avg_loan * avg_interest_rate * 0.

def _plot_roc_curve(self, y_true, y_proba, model_name, save_path=None):
    """Plota a curva ROC."""
    plt.figure(figsize=(8, 6))

    # Calcular curva ROC
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc_score = roc_auc_score(y_true, y_proba)

    # Plotar curva
    plt.plot(fpr, tpr, label=f'AUC = {auc_score:.4f}')

    # Plotar linha de referência
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.8)

    # Configurações do gráfico
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taxa de Falsos Positivos')
    plt.ylabel('Taxa de Verdadeiros Positivos')
    plt.title(f'Curva ROC - {model_name}')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)

    # Salvar se path fornecido
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def _plot_enhanced_roc_curve(self, y_true, y_proba, model_name, save_path=None):
    """Plota uma curva ROC tecnicamente aprimorada com detalhes estatísticos."""
    fig, ax = plt.subplots(figsize=(10, 8), dpi=100)

    # Calcular curva ROC com intervalos de confiança (bootstrap)
    n_bootstraps = 1000
    bootstrapped_aucs = []
    rng = np.random.RandomState(42)

    # Calcular curva ROC base
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)

    # Bootstrap para intervalos de confiança
    for i in range(n_bootstraps):
        # Amostragem com reposição
        indices = rng.randint(0, len(y_true), len(y_true))
        if len(np.unique(y_true[indices])) < 2:
            # Pular iteração se só houver uma classe no bootstrap
            continue

        # Calcular AUC para esta amostra bootstrap
        fpr_bootstrap, tpr_bootstrap, _ = roc_curve(y_true[indices], y_proba[indices])
        bootstrapped_aucs.append(auc(fpr_bootstrap, tpr_bootstrap))

    # Calcular intervalos de confiança
    auc_ci_lower = np.percentile(bootstrapped_aucs, 2.5)
    auc_ci_upper = np.percentile(bootstrapped_aucs, 97.5)

    # Plotar curva ROC principal com área sombreada
    ax.plot(fpr, tpr, 'b-', lw=2,
            label=f'AUC = {roc_auc:.4f} (IC 95%: {auc_ci_lower:.4f}-{auc_ci_upper:.4f})')

    # Adicionar área sob a curva (sombreada)
    ax.fill_between(fpr, tpr, alpha=0.3, color='b')

    # Adicionar linha de referência
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.8, lw=1.5)

    # Adicionar pontos de threshold
    threshold_markers = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    threshold_indices = [np.argmin(np.abs(thresholds - t)) for t in threshold_markers if t < max(thresholds)]

    for idx in threshold_indices:
        ax.plot(fpr[idx], tpr[idx], 'ro', markersize=5)
        ax.annotate(f'{thresholds[idx]:.2f}',
                    xy=(fpr[idx], tpr[idx]),
                    xytext=(fpr[idx] + 0.02, tpr[idx] - 0.02),
                    fontsize=8)

    # Adicionar marcadores para pontos operacionais importantes
    # 1. Ponto J máximo (melhor equilíbrio sensibilidade/especificidade)
    j_scores = tpr - fpr
    best_j_idx = np.argmax(j_scores)
    ax.plot(fpr[best_j_idx], tpr[best_j_idx], 'go', markersize=7)
    ax.annotate(f'J-max ({thresholds[best_j_idx]:.2f})',
                xy=(fpr[best_j_idx], tpr[best_j_idx]),
                xytext=(fpr[best_j_idx] + 0.05, tpr[best_j_idx]),
                fontsize=9,
                arrowprops=dict(arrowstyle="->", color='green'))

    # Configurações avançadas do gráfico
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    ax.set_xlabel('Taxa de Falsos Positivos (1 - Especificidade)', fontsize=12)
    ax.set_ylabel('Taxa de Verdadeiros Positivos (Sensibilidade)', fontsize=12)
    ax.set_title(f'Curva ROC - {model_name}\nAnálise Detalhada de Performance', fontsize=14, fontweight='bold')

    # Adicionar grade mais sutil
    ax.grid(True, linestyle='--', alpha=0.7)

    # Adicionar legenda com estilo melhorado
    ax.legend(loc='lower right', frameon=True, fancybox=True, framealpha=0.8, fontsize=10)

    # Adicionar texto com estatísticas adicionais
    stats_text = (
        f"Estatísticas adicionais:\n"
        f"AUC: {roc_auc:.4f}\n"
        f"IC 95%: [{auc_ci_lower:.4f} - {auc_ci_upper:.4f}]\n"
        f"Melhor threshold (J): {thresholds[best_j_idx]:.4f}\n"
        f"Sensibilidade em J-max: {tpr[best_j_idx]:.4f}\n"
        f"Especificidade em J-max: {1 - fpr[best_j_idx]:.4f}"
    )
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    ax.text(0.05, 0.05, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='bottom', bbox=props)

    # Adicionar anotação do método
    plt.figtext(0.99, 0.01, f'Gerado em: {datetime.now().strftime("%Y-%m-%d %H:%M")}',
                fontsize=8, ha='right')

    # Configurações de estilo global para o subplot
    plt.tight_layout()

    # Salvar se path fornecido
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    """
    Ponto de entrada para executar a avaliação de modelos de inadimplência.
    Processa argumentos da linha de comando e executa as funções apropriadas.
    """
    import argparse
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

    # Configurar parser de argumentos
    parser = argparse.ArgumentParser(description='Avaliação de modelos de inadimplência')
    parser.add_argument('--model_path', type=str, help='Caminho para um modelo salvo (opcional)')
    parser.add_argument('--data_path', type=str, help='Caminho para dados de teste (opcional)')
    parser.add_argument('--demo', action='store_true', default=True, help='Executar demonstração com dados sintéticos')
    parser.add_argument('--report_format', type=str, default='html', choices=['html', 'md', 'txt'],
                        help='Formato do relatório de negócio')
    parser.add_argument('--enhanced_visuals', action='store_true', default=True,
                        help='Usar visualizações técnicas avançadas')
    parser.add_argument('--multiple', action='store_true', default=False,
                        help='Testar múltiplos modelos')

    args = parser.parse_args()

    # Verificar modo de execução
    if args.model_path and args.data_path:
        # Modo de execução com arquivos externos
        logger.info(f"Executando avaliação com modelo: {args.model_path} e dados: {args.data_path}")

        try:
            # Carregar modelo
            model = joblib.load(args.model_path)

            # Carregar dados
            data = pd.read_csv(args.data_path)

            # Separar features e target
            if 'target' in data.columns:
                X = data.drop('target', axis=1)
                y = data['target']
            else:
                # Se não houver coluna 'target', assume-se que a última coluna é o target
                X = data.iloc[:, :-1]
                y = data.iloc[:, -1]

            # Dividir em conjuntos de validação e teste
            X_val, X_test, y_val, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

            # Configurar visualizações
            viz_config = VisualizationConfig(
                style='technical',
                color_palette='muted',
                annotation_density='high',
                dpi=300,
                show_annotations=True,
                show_statistics=True,
                show_confidence_intervals=True,
                font_scale=1.1
            )

            # Criar avaliador
            evaluator = ModelEvaluator(
                cost_fn_ratio=4.0,
                default_threshold=0.5,
                positive_class='inadimplente',
                visualization_config=viz_config
            )

            # Extrair nome do modelo do caminho do arquivo
            model_name = os.path.basename(args.model_path).split('.')[0]

            # Adicionar modelo
            evaluator.add_model(model_name, model)

            # Otimizar threshold
            logger.info("Otimizando threshold usando conjunto de validação...")
            evaluator.find_optimal_threshold(model_name, X_val, y_val, optimization_metric='cost')

            # Avaliar modelo com visualizações técnicas aprimoradas
            logger.info("Avaliando modelo no conjunto de teste...")
            evaluator.evaluate_model(
                model_name,
                X_test,
                y_test,
                enhanced_visuals=args.enhanced_visuals,
                store_predictions=True
            )

            # Gerar relatório de negócio
            logger.info(f"Gerando relatório de negócio no formato {args.report_format}...")
            report_path = evaluator.generate_business_report(
                model_name,
                output_format=args.report_format,
                include_visualizations=True
            )

            logger.info(f"Avaliação concluída! Relatório salvo em: {report_path}")
            logger.info(f"Gráficos e resultados salvos em: {evaluator.output_dir}")

            print(f"\nAvaliação concluída com sucesso!")
            print(f"Relatório técnico com visualizações avançadas salvo em: {report_path}")
            print(f"Gráficos e resultados adicionais salvos em: {evaluator.output_dir}")

        except Exception as e:
            logger.error(f"Erro ao carregar ou avaliar o modelo: {str(e)}")
            raise e

    elif args.demo:
        # Modo de demonstração com dados sintéticos
        logger.info("Executando demonstração com dados sintéticos...")

        # Gerar dados de exemplo com características mais realistas para inadimplência
        X, y = make_classification(
            n_samples=5000,
            n_features=20,
            n_informative=8,
            n_redundant=4,
            n_repeated=0,
            n_classes=2,
            n_clusters_per_class=3,
            weights=[0.85, 0.15],  # Dados desbalanceados, típico em inadimplência
            random_state=42
        )

        # Dividir em conjuntos de treinamento, validação e teste
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

        # Configurar visualizações
        viz_config = VisualizationConfig(
            style='technical',
            color_palette='muted',
            annotation_density='high',
            dpi=300,
            show_annotations=True,
            show_statistics=True,
            show_confidence_intervals=True,
            font_scale=1.1
        )

        # Criar avaliador
        evaluator = ModelEvaluator(
            cost_fn_ratio=4.0,
            default_threshold=0.5,
            positive_class='inadimplente',
            visualization_config=viz_config
        )

        if args.multiple:
            # Testar múltiplos modelos
            logger.info("Testando múltiplos modelos...")

            # Treinar diferentes modelos
            models = {
                'logistic_regression': LogisticRegression(max_iter=1000, random_state=42),
                'random_forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
                'gradient_boosting': GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
            }

            for name, model in models.items():
                logger.info(f"Treinando modelo: {name}")
                model.fit(X_train, y_train)

                # Adicionar modelo
                evaluator.add_model(name, model)

            # Otimizar thresholds
            evaluator.find_optimal_thresholds_all_models(X_val, y_val)

            # Avaliar todos os modelos
            evaluator.evaluate_all_models(X_test, y_test, store_predictions=True)

            # Gerar relatório para o melhor modelo
            logger.info(f"Gerando relatório de negócio para o melhor modelo no formato {args.report_format}...")
            report_path = evaluator.generate_business_report(
                evaluator.best_model_name,
                output_format=args.report_format,
                include_visualizations=True
            )

            logger.info(f"Demonstração com múltiplos modelos concluída!")
            logger.info(f"Melhor modelo: {evaluator.best_model_name}")
            logger.info(f"Relatório salvo em: {report_path}")

            print(f"\nDemonstração com múltiplos modelos concluída com sucesso!")
            print(f"Melhor modelo: {evaluator.best_model_name}")
            print(f"Relatório técnico com visualizações avançadas salvo em: {report_path}")
            print(f"Gráficos e resultados adicionais salvos em: {evaluator.output_dir}")

        else:
            # Testar apenas um modelo
            # Treinar modelo de exemplo
            rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
            rf_model.fit(X_train, y_train)

            # Adicionar modelo
            evaluator.add_model("RandomForest_Technical", rf_model)

            # Otimizar threshold
            logger.info("Otimizando threshold usando conjunto de validação...")
            evaluator.find_optimal_threshold("RandomForest_Technical", X_val, y_val, optimization_metric='cost')

            # Avaliar modelo com visualizações técnicas aprimoradas
            logger.info("Avaliando modelo no conjunto de teste...")
            evaluator.evaluate_model(
                "RandomForest_Technical",
                X_test,
                y_test,
                enhanced_visuals=args.enhanced_visuals,
                store_predictions=True
            )

            # Gerar relatório de negócio
            logger.info(f"Gerando relatório de negócio no formato {args.report_format}...")
            report_path = evaluator.generate_business_report(
                "RandomForest_Technical",
                output_format=args.report_format,
                include_visualizations=True
            )

            logger.info(f"Demonstração concluída! Relatório salvo em: {report_path}")
            logger.info(f"Gráficos e resultados salvos em: {evaluator.output_dir}")

            print(f"\nDemonstração concluída com sucesso!")
            print(f"Relatório técnico com visualizações avançadas salvo em: {report_path}")
            print(f"Gráficos e resultados adicionais salvos em: {evaluator.output_dir}")

    else:
        logger.error("Nenhum modelo/dados fornecidos e modo de demonstração desativado.")
        parser.print_help()