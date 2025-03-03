"""
Módulo para avaliação detalhada de modelos de predição de inadimplência.
Inclui métricas de classificação, análise de custo-benefício, e visualizações específicas
para o contexto financeiro.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
from datetime import datetime
import logging

# Métricas e validação
from sklearn.metrics import (
    roc_curve, precision_recall_curve, roc_auc_score,
    confusion_matrix, average_precision_score
)
from sklearn.calibration import calibration_curve  # Importação para a curva de calibração otimizada

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


class ModelEvaluator:
    """
    Classe para avaliação de modelos de inadimplência, focando em métricas
    relevantes para o contexto financeiro e impacto de negócio.
    """

    def __init__(self, cost_fn_ratio=5.0, approval_target=None, default_threshold=None, positive_class='inadimplente'):
        """
        Inicializa o avaliador de modelos.

        Args:
            cost_fn_ratio: Custo relativo de um falso negativo (cliente inadimplente
                          classificado como adimplente) comparado a um falso positivo
            approval_target: Taxa alvo de aprovação (0-1) para otimização do threshold
            default_threshold: Threshold padrão para classificação (0.5)
            positive_class: Define qual é a classe positiva ('inadimplente' ou 'adimplente')
        """
        self.cost_fn_ratio = cost_fn_ratio
        self.approval_target = approval_target
        self.default_threshold = default_threshold or 0.5
        self.models = {}
        self.model_results = {}
        self.best_model = None
        self.best_model_name = None
        self.positive_class = positive_class

        # Validar classe positiva
        if self.positive_class not in ['inadimplente', 'adimplente']:
            logger.warning(f"Classe positiva '{positive_class}' não reconhecida. Usando 'inadimplente'.")
            self.positive_class = 'inadimplente'

        # Configurar diretório de saída
        project_root = get_project_root()
        self.output_dir = os.path.join(project_root, 'reports', 'model_evaluation')
        os.makedirs(self.output_dir, exist_ok=True)

        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

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
                       plot_curves=True, save_plots=True, store_predictions=False):
        """
        Avalia um modelo específico no conjunto de teste.

        Args:
            name: Nome do modelo
            X_test: Features de teste
            y_test: Target de teste
            threshold: Threshold para classificação (se None, usa o definido no modelo)
            plot_curves: Se True, gera gráficos
            save_plots: Se True, salva os gráficos gerados
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
            raise ValueError(f"X_test ({len(X_test)} amostras) e y_test ({len(y_test)} amostras) devem ter o mesmo número de amostras")

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
        auc = roc_auc_score(y_test, y_proba)
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

        # Gerar gráficos
        if plot_curves:
            plots_dir = os.path.join(self.output_dir, 'plots', name)
            if save_plots:
                os.makedirs(plots_dir, exist_ok=True)

            self._plot_roc_curve(y_test, y_proba, name, save_path=os.path.join(plots_dir,
                                                                               f'roc_curve_{self.timestamp}.png') if save_plots else None)
            self._plot_precision_recall_curve(y_test, y_proba, name, save_path=os.path.join(plots_dir,
                                                                                            f'pr_curve_{self.timestamp}.png') if save_plots else None)
            self._plot_confusion_matrix(y_test, y_pred, name, save_path=os.path.join(plots_dir,
                                                                                     f'confusion_matrix_{self.timestamp}.png') if save_plots else None)
            self._plot_score_distribution(y_test, y_proba, threshold, name, save_path=os.path.join(plots_dir,
                                                                                                   f'score_dist_{self.timestamp}.png') if save_plots else None)
            self._plot_calibration_curve(y_test, y_proba, name, save_path=os.path.join(plots_dir,
                                                                                       f'calibration_{self.timestamp}.png') if save_plots else None)
            self._plot_threshold_impact(y_test, y_proba, name, save_path=os.path.join(plots_dir,
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
            raise ValueError(f"X_test ({len(X_test)} amostras) e y_test ({len(y_test)} amostras) devem ter o mesmo número de amostras")

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
            logger.warning("Gráficos comparativos de curvas ROC e PR não foram gerados. Execute evaluate_all_models com store_predictions=True.")

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

        # Coletar métricas para cada threshold
        metrics = []

        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)

            # Calcular matriz de confusão
            tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()

            # Calcular métricas
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

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

    def generate_business_report(self, name=None, output_format='html'):
        """
        Gera um relatório de negócio para um modelo específico ou para o melhor modelo.

        Args:
            name: Nome do modelo (se None, usa o melhor modelo)
            output_format: Formato do relatório ('html', 'md', 'txt')

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
        """Gera relatório em formato HTML com valores financeiros corrigidos."""
        threshold = results['threshold']

        # Calcular valores para relatório
        total_clients = results['tp'] + results['tn'] + results['fp'] + results['fn']

        # Determinar clientes aprovados/rejeitados com base na classe positiva
        if self.positive_class == 'inadimplente':
            # Aprovados são os que NÃO foram classificados como inadimplentes
            approved_clients = results['tn'] + results['fn']
            rejected_clients = results['tp'] + results['fp']
        else:
            # Aprovados são os que foram classificados como adimplentes
            approved_clients = results['tp'] + results['fp']
            rejected_clients = results['tn'] + results['fn']

        # Revenue e loss (valores fictícios para demonstração)
        avg_loan = 10000  # Valor médio do empréstimo
        avg_interest_rate = 0.15  # Taxa de juros média
        default_loss_rate = 0.8  # Taxa de perda em caso de inadimplência (80% do empréstimo é perdido)

        # Receita dos empréstimos (juros dos não inadimplentes) - CORRIGIDO
        if self.positive_class == 'inadimplente':
            # Receita vem dos verdadeiros negativos (adimplentes aprovados)
            revenue = results['tn'] * avg_loan * avg_interest_rate
            # Perda vem dos falsos negativos (inadimplentes aprovados como adimplentes)
            loss = results['fn'] * avg_loan * default_loss_rate
            # Perda de oportunidade de bons pagadores rejeitados como inadimplentes
            opportunity_loss = results['fp'] * avg_loan * avg_interest_rate * 0.5
        else:
            # Receita vem dos verdadeiros positivos (adimplentes classificados como tal)
            revenue = results['tp'] * avg_loan * avg_interest_rate
            # Perda vem dos falsos positivos (inadimplentes aprovados como adimplentes)
            loss = results['fp'] * avg_loan * default_loss_rate
            # Perda de oportunidade de bons pagadores rejeitados como inadimplentes
            opportunity_loss = results['fn'] * avg_loan * avg_interest_rate * 0.5

        # Lucro líquido
        net_profit = revenue - loss

        # Calcular ROI
        roi = (net_profit / (approved_clients * avg_loan)) * 100 if approved_clients > 0 else 0

        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Relatório de Negócio - Modelo de Inadimplência</title>
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
        .footer {{ text-align: center; margin-top: 50px; color: #7f8c8d; font-size: 14px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Relatório de Negócio - Modelo de Inadimplência</h1>
            <p>Modelo: {name} | Data: {datetime.now().strftime('%d/%m/%Y %H:%M')}</p>
        </div>

        <div class="section">
            <h2>Resumo Executivo</h2>
            <p>
                Este relatório apresenta os resultados e o impacto de negócio do modelo de predição de inadimplência <strong>{name}</strong>.
                O modelo foi avaliado com um threshold de <strong>{threshold:.4f}</strong> para classificação.
                Classe positiva definida como: <strong>{self.positive_class}</strong>.
            </p>

            <div class="metrics">
                <div class="metric-card">
                    <div class="metric-title">Taxa de Aprovação</div>
                    <div class="metric-value">{results['aprovacao_rate']:.1%}</div>
                    <div class="metric-context">{approved_clients} de {total_clients} clientes</div>
                </div>

                <div class="metric-card">
                    <div class="metric-title">Taxa de Inadimplência na Carteira</div>
                    <div class="metric-value {'bad' if results['inadimplencia_portfolio'] > 0.15 else 'medium' if results['inadimplencia_portfolio'] > 0.1 else 'good'}">
                        {results['inadimplencia_portfolio']:.1%}
                    </div>
                    <div class="metric-context">Inadimplentes entre aprovados</div>
                </div>

                <div class="metric-card">
                    <div class="metric-title">AUC-ROC</div>
                    <div class="metric-value {'good' if results['auc'] > 0.8 else 'medium' if results['auc'] > 0.7 else 'bad'}">
                        {results['auc']:.3f}
                    </div>
                    <div class="metric-context">Capacidade de discriminação do modelo</div>
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
            <h2>Performance de Classificação</h2>

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

            <h3>Métricas de Performance</h3>
            <table>
                <tr>
                    <th>Métrica</th>
                    <th>Valor</th>
                    <th>Interpretação</th>
                </tr>
                <tr>
                    <td>Acurácia</td>
                    <td>{results['accuracy']:.4f}</td>
                    <td>Proporção de predições corretas</td>
                </tr>
                <tr>
                    <td>Precisão</td>
                    <td>{results['precision']:.4f}</td>
                    <td>Dos clientes classificados como {self.positive_class}, quantos realmente são</td>
                </tr>
                <tr>
                    <td>Recall</td>
                    <td>{results['recall']:.4f}</td>
                    <td>Dos clientes realmente {self.positive_class}, quantos foram detectados</td>
                </tr>
                <tr>
                    <td>Especificidade</td>
                    <td>{results['specificity']:.4f}</td>
                    <td>Dos clientes não {self.positive_class}, quantos foram classificados corretamente</td>
                </tr>
                <tr>
                    <td>F1-Score</td>
                    <td>{results['f1_score']:.4f}</td>
                    <td>Média harmônica entre precisão e recall</td>
                </tr>
            </table>
        </div>

        <div class="section">
            <h2>Impacto de Negócio</h2>

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
            </table>

            <h3>Segmentação de Clientes</h3>
            <table>
                <tr>
                    <th>Segmento</th>
                    <th>Quantidade</th>
                    <th>Percentual</th>
                </tr>
                <tr>
                    <td>Aprovados</td>
                    <td>{approved_clients}</td>
                    <td>{approved_clients / total_clients:.1%}</td>
                </tr>
                <tr>
                    <td>Rejeitados</td>
                    <td>{rejected_clients}</td>
                    <td>{rejected_clients / total_clients:.1%}</td>
                </tr>
                <tr>
                    <td>Adimplentes Aprovados</td>
                    <td>{results['tn'] if self.positive_class == 'inadimplente' else results['tp']}</td>
                    <td>{(results['tn'] if self.positive_class == 'inadimplente' else results['tp']) / total_clients:.1%}</td>
                </tr>
                <tr>
                    <td>Inadimplentes Rejeitados</td>
                    <td>{results['tp'] if self.positive_class == 'inadimplente' else results['tn']}</td>
                    <td>{(results['tp'] if self.positive_class == 'inadimplente' else results['tn']) / total_clients:.1%}</td>
                </tr>
                <tr>
                    <td>Adimplentes Rejeitados (Oportunidade Perdida)</td>
                    <td>{results['fp'] if self.positive_class == 'inadimplente' else results['fn']}</td>
                    <td>{(results['fp'] if self.positive_class == 'inadimplente' else results['fn']) / total_clients:.1%}</td>
                </tr>
                <tr>
                    <td>Inadimplentes Aprovados (Risco)</td>
                    <td>{results['fn'] if self.positive_class == 'inadimplente' else results['fp']}</td>
                    <td>{(results['fn'] if self.positive_class == 'inadimplente' else results['fp']) / total_clients:.1%}</td>
                </tr>
            </table>
        </div>

        <div class="section">
            <h2>Recomendações</h2>
            <ul>
                <li>O modelo apresenta uma taxa de aprovação de {results['aprovacao_rate']:.1%}, o que {'está dentro da faixa desejada' if 0.4 <= results['aprovacao_rate'] <= 0.6 else 'pode ser ajustado para atingir a faixa desejada de 40-60%'}.</li>
                <li>A taxa de inadimplência na carteira é de {results['inadimplencia_portfolio']:.1%}, o que {'está dentro do limite aceitável (<10%)' if results['inadimplencia_portfolio'] < 0.1 else 'está acima do limite aceitável e deve ser reduzida'}.</li>
                <li>{'O threshold atual de classificação está bem calibrado.' if 0.4 <= results['accuracy'] <= 0.6 else 'O threshold pode ser ajustado para equilibrar melhor aprovação e risco.'}</li>
                <li>{'O modelo tem boa capacidade de discriminação (AUC > 0.8).' if results['auc'] > 0.8 else 'A capacidade discriminativa do modelo pode ser melhorada.'}</li>
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

    def _generate_md_report(self, name, results):
        """Gera relatório em formato Markdown com valores financeiros corrigidos."""
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

        md = f"""# Relatório de Negócio - Modelo de Inadimplência

**Modelo:** {name}  
**Data:** {datetime.now().strftime('%d/%m/%Y %H:%M')}  
**Threshold de Classificação:** {threshold:.4f}  
**Classe Positiva:** {self.positive_class}

## Resumo Executivo

Este relatório apresenta os resultados e o impacto de negócio do modelo de predição de inadimplência **{name}**.
O modelo foi avaliado com um threshold de **{threshold:.4f}** para classificação.

| Métrica | Valor | Contexto |
|---------|-------|----------|
| Taxa de Aprovação | {results['aprovacao_rate']:.1%} | {approved_clients} de {total_clients} clientes |
| Taxa de Inadimplência na Carteira | {results['inadimplencia_portfolio']:.1%} | Inadimplentes entre aprovados |
| AUC-ROC | {results['auc']:.3f} | Capacidade de discriminação do modelo |
| ROI Estimado | {roi:.1f}% | Retorno sobre investimento |

## Performance de Classificação

### Matriz de Confusão

|                          | Predito: Não {self.positive_class} | Predito: {self.positive_class.title()} |
|--------------------------|---------------------------|------------------------|
| Real: Não {self.positive_class}   | {results['tn']} (Verdadeiro Negativo) | {results['fp']} (Falso Positivo) |
| Real: {self.positive_class.title()}       | {results['fn']} (Falso Negativo) | {results['tp']} (Verdadeiro Positivo) |

### Métricas de Performance

| Métrica | Valor | Interpretação |
|---------|-------|---------------|
| Acurácia | {results['accuracy']:.4f} | Proporção de predições corretas |
| Precisão | {results['precision']:.4f} | Dos clientes classificados como {self.positive_class}, quantos realmente são |
| Recall | {results['recall']:.4f} | Dos clientes realmente {self.positive_class}, quantos foram detectados |
| Especificidade | {results['specificity']:.4f} | Dos clientes não {self.positive_class}, quantos foram classificados corretamente |
| F1-Score | {results['f1_score']:.4f} | Média harmônica entre precisão e recall |

## Impacto de Negócio

### Análise de Custo-Benefício

| Métrica | Valor | Interpretação |
|---------|-------|---------------|
| Receita Estimada | R$ {revenue:,.2f} | Juros dos empréstimos para bons pagadores |
| Perda por Inadimplência | R$ {loss:,.2f} | Valor perdido com clientes inadimplentes não detectados |
| Perda de Oportunidade | R$ {opportunity_loss:,.2f} | Receita potencial perdida por rejeitar bons pagadores |
| Lucro Líquido Estimado | R$ {net_profit:,.2f} | Receita menos perdas |

### Segmentação de Clientes

| Segmento | Quantidade | Percentual |
|----------|------------|------------|
| Aprovados | {approved_clients} | {approved_clients / total_clients:.1%} |
| Rejeitados | {rejected_clients} | {rejected_clients / total_clients:.1%} |
| Adimplentes Aprovados | {results['tn'] if self.positive_class == 'inadimplente' else results['tp']} | {(results['tn'] if self.positive_class == 'inadimplente' else results['tp']) / total_clients:.1%} |
| Inadimplentes Rejeitados | {results['tp'] if self.positive_class == 'inadimplente' else results['tn']} | {(results['tp'] if self.positive_class == 'inadimplente' else results['tn']) / total_clients:.1%} |
| Adimplentes Rejeitados (Oportunidade Perdida) | {results['fp'] if self.positive_class == 'inadimplente' else results['fn']} | {(results['fp'] if self.positive_class == 'inadimplente' else results['fn']) / total_clients:.1%} |
| Inadimplentes Aprovados (Risco) | {results['fn'] if self.positive_class == 'inadimplente' else results['fp']} | {(results['fn'] if self.positive_class == 'inadimplente' else results['fp']) / total_clients:.1%} |

## Recomendações

* O modelo apresenta uma taxa de aprovação de {results['aprovacao_rate']:.1%}, o que {'está dentro da faixa desejada' if 0.4 <= results['aprovacao_rate'] <= 0.6 else 'pode ser ajustado para atingir a faixa desejada de 40-60%'}.
* A taxa de inadimplência na carteira é de {results['inadimplencia_portfolio']:.1%}, o que {'está dentro do limite aceitável (<10%)' if results['inadimplencia_portfolio'] < 0.1 else 'está acima do limite aceitável e deve ser reduzida'}.
* {'O threshold atual de classificação está bem calibrado.' if 0.4 <= results['accuracy'] <= 0.6 else 'O threshold pode ser ajustado para equilibrar melhor aprovação e risco.'}
* {'O modelo tem boa capacidade de discriminação (AUC > 0.8).' if results['auc'] > 0.8 else 'A capacidade discriminativa do modelo pode ser melhorada.'}

---

Relatório gerado automaticamente em {datetime.now().strftime('%d/%m/%Y %H:%M')}
"""
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
            opportunity_loss = results['fp'] * avg_loan * avg_interest_rate * 0.5
        else:
            revenue = results['tp'] * avg_loan * avg_interest_rate
            loss = results['fp'] * avg_loan * default_loss_rate
            opportunity_loss = results['fn'] * avg_loan * avg_interest_rate * 0.5

        # Lucro líquido
        net_profit = revenue - loss

        # Calcular ROI
        roi = (net_profit / (approved_clients * avg_loan)) * 100 if approved_clients > 0 else 0

        txt = f"""RELATÓRIO DE NEGÓCIO - MODELO DE INADIMPLÊNCIA
======================================

Modelo: {name}
Data: {datetime.now().strftime('%d/%m/%Y %H:%M')}
Threshold de Classificação: {threshold:.4f}
Classe Positiva: {self.positive_class}

RESUMO EXECUTIVO
---------------

Este relatório apresenta os resultados e o impacto de negócio do modelo de predição de inadimplência {name}.
O modelo foi avaliado com um threshold de {threshold:.4f} para classificação.

- Taxa de Aprovação: {results['aprovacao_rate']:.1%} ({approved_clients} de {total_clients} clientes)
- Taxa de Inadimplência na Carteira: {results['inadimplencia_portfolio']:.1%} (Inadimplentes entre aprovados)
- AUC-ROC: {results['auc']:.3f}
- ROI Estimado: {roi:.1f}%

PERFORMANCE DE CLASSIFICAÇÃO
--------------------------

Matriz de Confusão:
                      | Predito: Não {self.positive_class} | Predito: {self.positive_class}
----------------------|---------------------------|----------------------
Real: Não {self.positive_class}| {results['tn']} (Verdadeiro Negativo) | {results['fp']} (Falso Positivo)
Real: {self.positive_class}    | {results['fn']} (Falso Negativo) | {results['tp']} (Verdadeiro Positivo)

Métricas de Performance:
- Acurácia: {results['accuracy']:.4f}
- Precisão: {results['precision']:.4f}
- Recall: {results['recall']:.4f}
- Especificidade: {results['specificity']:.4f}
- F1-Score: {results['f1_score']:.4f}

IMPACTO DE NEGÓCIO
----------------

Análise de Custo-Benefício:
- Receita Estimada: R$ {revenue:,.2f}
- Perda por Inadimplência: R$ {loss:,.2f}
- Perda de Oportunidade: R$ {opportunity_loss:,.2f}
- Lucro Líquido Estimado: R$ {net_profit:,.2f}

Segmentação de Clientes:
- Aprovados: {approved_clients} ({approved_clients / total_clients:.1%})
- Rejeitados: {rejected_clients} ({rejected_clients / total_clients:.1%})
- Adimplentes Aprovados: {results['tn'] if self.positive_class == 'inadimplente' else results['tp']} ({(results['tn'] if self.positive_class == 'inadimplente' else results['tp']) / total_clients:.1%})
- Inadimplentes Rejeitados: {results['tp'] if self.positive_class == 'inadimplente' else results['tn']} ({(results['tp'] if self.positive_class == 'inadimplente' else results['tn']) / total_clients:.1%})
- Adimplentes Rejeitados (Oportunidade Perdida): {results['fp'] if self.positive_class == 'inadimplente' else results['fn']} ({(results['fp'] if self.positive_class == 'inadimplente' else results['fn']) / total_clients:.1%})
- Inadimplentes Aprovados (Risco): {results['fn'] if self.positive_class == 'inadimplente' else results['fp']} ({(results['fn'] if self.positive_class == 'inadimplente' else results['fp']) / total_clients:.1%})

RECOMENDAÇÕES
------------

- O modelo apresenta uma taxa de aprovação de {results['aprovacao_rate']:.1%}, o que {'está dentro da faixa desejada' if 0.4 <= results['aprovacao_rate'] <= 0.6 else 'pode ser ajustado para atingir a faixa desejada de 40-60%'}.
- A taxa de inadimplência na carteira é de {results['inadimplencia_portfolio']:.1%}, o que {'está dentro do limite aceitável (<10%)' if results['inadimplencia_portfolio'] < 0.1 else 'está acima do limite aceitável e deve ser reduzida'}.
- {'O threshold atual de classificação está bem calibrado.' if 0.4 <= results['accuracy'] <= 0.6 else 'O threshold pode ser ajustado para equilibrar melhor aprovação e risco.'}
- {'O modelo tem boa capacidade de discriminação (AUC > 0.8).' if results['auc'] > 0.8 else 'A capacidade discriminativa do modelo pode ser melhorada.'}

-------------------------------
Relatório gerado automaticamente em {datetime.now().strftime('%d/%m/%Y %H:%M')}
"""
        return txt

    def _plot_roc_curve(self, y_true, y_proba, model_name, save_path=None):
        """Plota a curva ROC."""
        plt.figure(figsize=(8, 6))

        # Calcular curva ROC
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        auc = roc_auc_score(y_true, y_proba)

        # Plotar curva
        plt.plot(fpr, tpr, label=f'AUC = {auc:.4f}')

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

    def _plot_precision_recall_curve(self, y_true, y_proba, model_name, save_path=None):
        """Plota a curva Precision-Recall."""
        plt.figure(figsize=(8, 6))

        # Calcular curva PR
        precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
        avg_precision = average_precision_score(y_true, y_proba)

        # Eixo limite inferior para Precision
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])

        # Referência para classificador aleatório
        no_skill = np.sum(y_true) / len(y_true)
        plt.plot([0, 1], [no_skill, no_skill], 'k--', alpha=0.8, label='Aleatório')

        # Plotar curva
        plt.plot(recall, precision, label=f'AP = {avg_precision:.4f}')

        # Configurações do gráfico
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Curva Precision-Recall - {model_name}')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)

        # Salvar se path fornecido
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def _plot_confusion_matrix(self, y_true, y_pred, model_name, save_path=None):
        """Plota a matriz de confusão."""
        plt.figure(figsize=(8, 6))

        # Calcular matriz de confusão
        cm = confusion_matrix(y_true, y_pred)

        # Normalizar para percentuais
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        # Plotar heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)

        # Configurações do gráfico
        plt.xlabel('Predito')
        plt.ylabel('Real')
        plt.title(f'Matriz de Confusão - {model_name}')
        plt.xticks([0.5, 1.5], [f'Não {self.positive_class.title()}', self.positive_class.title()])
        plt.yticks([0.5, 1.5], [f'Não {self.positive_class.title()}', self.positive_class.title()])

        # Salvar se path fornecido
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

        # Plotar também a versão normalizada
        if save_path:
            norm_path = save_path.replace('.png', '_norm.png')
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm_percent, annot=True, fmt='.1%', cmap='Blues', cbar=False)
            plt.xlabel('Predito')
            plt.ylabel('Real')
            plt.title(f'Matriz de Confusão Normalizada - {model_name}')
            plt.xticks([0.5, 1.5], [f'Não {self.positive_class.title()}', self.positive_class.title()])
            plt.yticks([0.5, 1.5], [f'Não {self.positive_class.title()}', self.positive_class.title()])
            plt.savefig(norm_path, dpi=300, bbox_inches='tight')
            plt.close()

    def _plot_score_distribution(self, y_true, y_proba, threshold, model_name, save_path=None):
        """Plota a distribuição de scores por classe."""
        plt.figure(figsize=(10, 6))

        # Criar DataFrame para facilitar a plotagem
        df = pd.DataFrame({
            'Score': y_proba,
            'Classe': y_true
        })

        # Converter para texto para melhor visualização
        df['Classe'] = df['Classe'].map({0: f'Não {self.positive_class.title()}', 1: self.positive_class.title()})

        # Plotar histogramas
        sns.histplot(data=df, x='Score', hue='Classe', bins=50, alpha=0.7, kde=True, element="step")

        # Adicionar linha de threshold
        plt.axvline(x=threshold, color='red', linestyle='--', alpha=0.8, label=f'Threshold = {threshold:.4f}')

        # Configurações do gráfico
        plt.xlabel('Score de Probabilidade')
        plt.ylabel('Contagem')
        plt.title(f'Distribuição de Scores por Classe - {model_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Salvar se path fornecido
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def _plot_calibration_curve(self, y_true, y_proba, model_name, save_path=None):
        """Plota a curva de calibração usando scikit-learn."""
        plt.figure(figsize=(10, 6))

        # Calcular curva de calibração
        prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=10)

        # Plotar curva de calibração
        plt.plot(prob_pred, prob_true, 'o-', label='Observado')
        plt.plot([0, 1], [0, 1], 'k--', label='Perfeitamente Calibrado')

        # Configurações do gráfico
        plt.xlabel('Probabilidade Prevista')
        plt.ylabel('Frequência Observada')
        plt.title(f'Curva de Calibração - {model_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Salvar se path fornecido
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def _plot_threshold_impact(self, y_true, y_proba, model_name, save_path=None):
        """Plota o impacto do threshold nas métricas."""
        plt.figure(figsize=(10, 8))

        # Gerar thresholds
        thresholds = np.linspace(0.01, 0.99, 99)

        # Inicializar arrays para armazenar métricas
        precision_values = []
        recall_values = []
        f1_values = []
        accuracy_values = []
        approval_rate_values = []
        default_rate_values = []

        # Calcular métricas para cada threshold
        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)

            # Matriz de confusão
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

            # Métricas de classificação
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            accuracy = (tp + tn) / (tp + tn + fp + fn)

            # Métricas de negócio - CORRIGIDAS
            if self.positive_class == 'inadimplente':
                approval_rate = (tn + fn) / (tp + tn + fp + fn)
                default_rate = fn / (tn + fn) if (tn + fn) > 0 else 0
            else:
                approval_rate = (tp + fp) / (tp + tn + fp + fn)
                default_rate = fp / (tp + fp) if (tp + fp) > 0 else 0

            # Armazenar valores
            precision_values.append(precision)
            recall_values.append(recall)
            f1_values.append(f1)
            accuracy_values.append(accuracy)
            approval_rate_values.append(approval_rate)
            default_rate_values.append(default_rate)

        # Plotar gráficos
        plt.subplot(2, 2, 1)
        plt.plot(thresholds, precision_values, label='Precision')
        plt.plot(thresholds, recall_values, label='Recall')
        plt.plot(thresholds, f1_values, label='F1')
        plt.xlabel('Threshold')
        plt.ylabel('Valor')
        plt.title('Métricas de Classificação')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(2, 2, 2)
        plt.plot(thresholds, accuracy_values, label='Acurácia')
        plt.xlabel('Threshold')
        plt.ylabel('Acurácia')
        plt.title('Acurácia vs Threshold')
        plt.grid(True, alpha=0.3)

        plt.subplot(2, 2, 3)
        plt.plot(thresholds, approval_rate_values, label='Taxa de Aprovação')
        plt.xlabel('Threshold')
        plt.ylabel('Taxa de Aprovação')
        plt.title('Aprovação vs Threshold')
        plt.grid(True, alpha=0.3)

        plt.subplot(2, 2, 4)
        plt.plot(thresholds, default_rate_values, label='Taxa de Inadimplência')
        plt.xlabel('Threshold')
        plt.ylabel('Taxa de Inadimplência')
        plt.title('Inadimplência vs Threshold')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.suptitle(f'Impacto do Threshold nas Métricas - {model_name}', fontsize=16, y=1.05)

        # Salvar se path fornecido
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def _plot_models_comparison(self, comparison_df):
        """Plota gráfico comparativo entre modelos."""
        # Plot para AUC e F1-Score
        plt.figure(figsize=(12, 6))
        metrics = ['AUC', 'F1-Score', 'Precisão', 'Recall']
        comparison_df.set_index('Modelo')[metrics].plot(kind='bar')
        plt.title('Comparação de Métricas de Classificação entre Modelos')
        plt.ylabel('Valor')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # Salvar gráfico
        plt.savefig(os.path.join(self.output_dir, f"models_metrics_comparison_{self.timestamp}.png"),
                    dpi=300, bbox_inches='tight')
        plt.close()

        # Plot para Taxa de Aprovação e Taxa de Inadimplência
        plt.figure(figsize=(12, 6))
        metrics = ['Taxa de Aprovação', 'Taxa de Inadimplência', 'Custo de Negócio']
        comparison_df.set_index('Modelo')[metrics].plot(kind='bar')
        plt.title('Comparação de Métricas de Negócio entre Modelos')
        plt.ylabel('Valor')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # Salvar gráfico
        plt.savefig(os.path.join(self.output_dir, f"models_business_comparison_{self.timestamp}.png"),
                    dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_roc_curves_comparison(self):
        """Plota comparação das curvas ROC de todos os modelos avaliados."""
        plt.figure(figsize=(10, 8))

        for name, results in self.model_results.items():
            if 'y_true' not in results or 'y_proba' not in results:
                logger.warning(f"Dados de previsão não disponíveis para o modelo '{name}'. Execute evaluate_model com store_predictions=True.")
                continue

            y_true = results['y_true']
            y_proba = results['y_proba']

            # Calcular curva ROC
            fpr, tpr, _ = roc_curve(y_true, y_proba)
            auc = results['auc']

            # Plotar curva
            plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.4f})')

        # Plotar linha de referência
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.8)

        # Configurações do gráfico
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Taxa de Falsos Positivos')
        plt.ylabel('Taxa de Verdadeiros Positivos')
        plt.title('Comparação de Curvas ROC')
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)

        # Salvar gráfico
        plt.savefig(os.path.join(self.output_dir, f"roc_curves_comparison_{self.timestamp}.png"),
                    dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_pr_curves_comparison(self):
        """Plota comparação das curvas Precision-Recall de todos os modelos avaliados."""
        plt.figure(figsize=(10, 8))

        # Referência para classificador aleatório
        no_skill = None

        for name, results in self.model_results.items():
            if 'y_true' not in results or 'y_proba' not in results:
                logger.warning(f"Dados de previsão não disponíveis para o modelo '{name}'. Execute evaluate_model com store_predictions=True.")
                continue

            y_true = results['y_true']
            y_proba = results['y_proba']

            # Calcular curva PR
            precision, recall, _ = precision_recall_curve(y_true, y_proba)
            avg_precision = results['avg_precision']

            # Plotar curva
            plt.plot(recall, precision, label=f'{name} (AP = {avg_precision:.4f})')

            # Calcular linha de referência apenas uma vez
            if no_skill is None:
                no_skill = np.sum(y_true) / len(y_true)

        # Plotar linha de referência
        if no_skill is not None:
            plt.plot([0, 1], [no_skill, no_skill], 'k--', alpha=0.8, label=f'Aleatório ({no_skill:.4f})')

        # Configurações do gráfico
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Comparação de Curvas Precision-Recall')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)

        # Salvar gráfico
        plt.savefig(os.path.join(self.output_dir, f"pr_curves_comparison_{self.timestamp}.png"),
                    dpi=300, bbox_inches='tight')
        plt.close()


if __name__ == "__main__":
    """
    Demonstração de uso da classe ModelEvaluator.
    Esta função principal executa quando o script é chamado diretamente.
    """
    import argparse
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

    # Configurar parser de argumentos
    parser = argparse.ArgumentParser(description='Demonstração de avaliação de modelos de inadimplência')
    parser.add_argument('--model_path', type=str, help='Caminho para um modelo salvo (opcional)')
    parser.add_argument('--data_path', type=str, help='Caminho para dados de teste (opcional)')
    parser.add_argument('--demo', action='store_true', default=True, help='Executar demonstração com dados sintéticos')
    parser.add_argument('--report_format', type=str, default='html', choices=['html', 'md', 'txt'],
                        help='Formato do relatório de negócio')

    args = parser.parse_args()

    # Verificar modo de execução
    if args.model_path and args.data_path:
        # Modo de execução com arquivos externos
        logger.info(f"Executando avaliação com modelo: {args.model_path} e dados: {args.data_path}")

        # Implementar carregar modelo e dados reais aqui
        # ...

    elif args.demo:
        # Modo de demonstração com dados sintéticos
        logger.info("Executando demonstração com dados sintéticos...")

        # Gerar dados de exemplo
        X, y = make_classification(n_samples=1000, n_features=10, n_informative=5,
                                   n_redundant=2, n_classes=2, weights=[0.8, 0.2],
                                   random_state=42)

        # Dividir em conjuntos de treinamento, validação e teste
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

        # Treinar modelos de exemplo
        rf_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        rf_model.fit(X_train, y_train)

        # Criar avaliador
        evaluator = ModelEvaluator(cost_fn_ratio=4.0, default_threshold=0.5, positive_class='inadimplente')

        # Adicionar modelo
        evaluator.add_model("RandomForest_Demo", rf_model)

        # Otimizar threshold
        logger.info("Otimizando threshold usando conjunto de validação...")
        evaluator.find_optimal_threshold("RandomForest_Demo", X_val, y_val, optimization_metric='cost')

        # Avaliar modelo
        logger.info("Avaliando modelo no conjunto de teste...")
        evaluator.evaluate_model("RandomForest_Demo", X_test, y_test, store_predictions=True)

        # Gerar relatório de negócio
        logger.info(f"Gerando relatório de negócio no formato {args.report_format}...")
        report_path = evaluator.generate_business_report("RandomForest_Demo", output_format=args.report_format)

        logger.info(f"Demonstração concluída! Relatório salvo em: {report_path}")
        logger.info(f"Gráficos e resultados salvos em: {evaluator.output_dir}")

        print(f"\nDemonstração concluída com sucesso!")
        print(f"Relatório de negócio salvo em: {report_path}")
        print(f"Gráficos e resultados salvos em: {evaluator.output_dir}")

    else:
        logger.error("Nenhum modelo/dados fornecidos e modo de demonstração desativado.")
        parser.print_help()