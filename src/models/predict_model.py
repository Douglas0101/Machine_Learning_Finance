"""
Módulo para fazer predições usando modelos treinados.
Permite carregar modelos salvos, aplicar aos dados e analisar resultados.
"""

import os
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional, Union
import logging

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


class ModelPredictor:
    """
    Classe para fazer predições usando modelos treinados de inadimplência.
    """

    def __init__(self, model_path=None, feature_engineer_path=None):
        """
        Inicializa o preditor.

        Args:
            model_path: Caminho para o modelo treinado
            feature_engineer_path: Caminho para o engenheiro de features
        """
        self.model = None
        self.feature_engineer = None
        self.model_metadata = {}
        self.threshold = 0.5

        if model_path:
            self.load_model(model_path)

        if feature_engineer_path:
            self.load_feature_engineer(feature_engineer_path)

    def load_model(self, model_path):
        """
        Carrega um modelo salvo.

        Args:
            model_path: Caminho para o modelo

        Returns:
            self
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Arquivo não encontrado: {model_path}")

        logger.info(f"Carregando modelo de: {model_path}")
        self.model = joblib.load(model_path)

        # Tentar carregar metadados
        model_dir = os.path.dirname(model_path)
        model_name = os.path.basename(model_path).split('_')[0]
        timestamp = '_'.join(os.path.basename(model_path).split('_')[1:]).replace('.joblib', '')

        metadata_path = os.path.join(model_dir, f"model_metadata_{timestamp}.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                self.model_metadata = json.load(f)

            # Extrair threshold para este modelo
            if 'thresholds' in self.model_metadata and model_name in self.model_metadata['thresholds']:
                self.threshold = self.model_metadata['thresholds'][model_name]
                logger.info(f"Threshold carregado: {self.threshold}")

        # Se o modelo tem atributo threshold, usar esse
        if hasattr(self.model, 'threshold'):
            self.threshold = self.model.threshold
            logger.info(f"Usando threshold interno do modelo: {self.threshold}")

        return self

    def load_feature_engineer(self, path):
        """
        Carrega um engenheiro de features salvo.

        Args:
            path: Caminho para o engenheiro de features

        Returns:
            self
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Arquivo não encontrado: {path}")

        logger.info(f"Carregando engenheiro de features de: {path}")
        self.feature_engineer = joblib.load(path)
        return self

    def find_latest_model(self, model_dir=None, model_type="best_model"):
        """
        Encontra o modelo mais recente do tipo especificado.

        Args:
            model_dir: Diretório dos modelos
            model_type: Tipo de modelo a procurar ("best_model", "LogisticRegression", etc.)

        Returns:
            Caminho para o modelo mais recente
        """
        if model_dir is None:
            model_dir = os.path.join(get_project_root(), 'models', 'trained_models')

        # Procurar modelos
        model_files = [f for f in os.listdir(model_dir) if f.startswith(model_type) and f.endswith('.joblib')]

        if not model_files:
            raise FileNotFoundError(f"Nenhum modelo '{model_type}' encontrado em {model_dir}")

        # Ordenar por timestamp (assumindo formato: tipo_YYYYMMDD_HHMMSS.joblib)
        model_files.sort(reverse=True)
        latest_model = os.path.join(model_dir, model_files[0])

        logger.info(f"Modelo mais recente encontrado: {latest_model}")
        return latest_model

    def find_matching_feature_engineer(self, model_path):
        """
        Encontra o engenheiro de features correspondente ao modelo.

        Args:
            model_path: Caminho para o modelo

        Returns:
            Caminho para o engenheiro de features correspondente
        """
        # Extrair timestamp do modelo
        model_name = os.path.basename(model_path)
        parts = model_name.split('_')

        if len(parts) < 2:
            raise ValueError(f"Nome do modelo não segue formato esperado: {model_name}")

        timestamp = '_'.join(parts[1:]).replace('.joblib', '')

        # Procurar feature engineer com o mesmo timestamp
        feature_dir = os.path.join(get_project_root(), 'models', 'preprocessing')
        if not os.path.exists(feature_dir):
            feature_dir = os.path.join(get_project_root(), 'models')

        feature_path = os.path.join(feature_dir, f"feature_engineer_{timestamp}.joblib")

        if os.path.exists(feature_path):
            logger.info(f"Feature engineer correspondente encontrado: {feature_path}")
            return feature_path

        # Se não encontrar com o timestamp específico, procurar o mais recente
        feature_files = [f for f in os.listdir(feature_dir) if
                         f.startswith('feature_engineer_') and f.endswith('.joblib')]

        if not feature_files:
            logger.warning("Nenhum feature engineer encontrado. A preparação básica dos dados será usada.")
            return None

        # Ordenar por timestamp e pegar o mais recente
        feature_files.sort(reverse=True)
        latest_feature = os.path.join(feature_dir, feature_files[0])

        logger.info(f"Usando feature engineer mais recente: {latest_feature}")
        return latest_feature

    def prepare_data(self, data, target_col=None):
        """
        Prepara dados para predição.

        Args:
            data: DataFrame com dados para predição
            target_col: Nome da coluna alvo (se disponível)

        Returns:
            X: Features preparadas para predição
            y: Target (se disponível)
        """
        logger.info("Preparando dados para predição...")

        df = data.copy()

        # Separar target (se disponível)
        y = None
        if target_col and target_col in df.columns:
            y = df[target_col]
            df = df.drop(columns=[target_col])

        # Remover colunas que possam causar vazamento
        columns_to_exclude = [
            'ID_Cliente', 'Nome', 'CPF', 'Email', 'Telefone', 'Data_Referencia',
            'Nome_Completo', 'RG', 'CEP', 'Endereco'
        ]

        for col in columns_to_exclude:
            if col in df.columns:
                df = df.drop(columns=[col])

        # Aplicar feature engineering (se disponível)
        if self.feature_engineer:
            logger.info("Aplicando transformações de features...")
            X = self.feature_engineer.transform(df)
        else:
            logger.info("Feature engineer não disponível. Usando dados originais.")
            X = df

        logger.info(f"Dados preparados: {X.shape[0]} exemplos, {X.shape[1]} features")
        return X, y

    def predict(self, data, target_col=None, output_probabilities=False):
        """
        Faz predições usando o modelo carregado.

        Args:
            data: DataFrame com dados para predição
            target_col: Nome da coluna alvo (se disponível para avaliação)
            output_probabilities: Se True, retorna probabilidades além das classes

        Returns:
            DataFrame com predições
        """
        if self.model is None:
            raise ValueError("Nenhum modelo carregado. Use load_model() primeiro.")

        # Preparar dados
        X, y = self.prepare_data(data, target_col)

        # Fazer predições
        logger.info("Fazendo predições...")
        y_proba = self.model.predict_proba(X)[:, 1]
        y_pred = (y_proba >= self.threshold).astype(int)

        # Criar DataFrame de resultados
        if 'ID_Cliente' in data.columns:
            id_col = data['ID_Cliente']
        else:
            id_col = pd.Series(range(len(data)), name='ID')

        results = pd.DataFrame({
            'ID': id_col,
            'Inadimplencia_Predicao': y_pred,
            'Probabilidade': y_proba
        })

        # Categorizar risco
        results['Nivel_Risco'] = pd.cut(
            results['Probabilidade'],
            bins=[0, 0.25, 0.5, 0.75, 1],
            labels=['Baixo', 'Médio-Baixo', 'Médio-Alto', 'Alto']
        )

        # Adicionar avaliação (se target disponível)
        if y is not None:
            # Adicionar valores reais
            results['Inadimplencia_Real'] = y.reset_index(drop=True)

            # Calcular métricas
            from sklearn.metrics import (
                accuracy_score, precision_score, recall_score, f1_score,
                roc_auc_score, confusion_matrix
            )

            accuracy = accuracy_score(y, y_pred)
            precision = precision_score(y, y_pred)
            recall = recall_score(y, y_pred)
            f1 = f1_score(y, y_pred)
            auc = roc_auc_score(y, y_proba)
            cm = confusion_matrix(y, y_pred)

            logger.info("\nAvaliação das predições:")
            logger.info(f"Acurácia: {accuracy:.4f}")
            logger.info(f"Precisão: {precision:.4f}")
            logger.info(f"Recall: {recall:.4f}")
            logger.info(f"F1-Score: {f1:.4f}")
            logger.info(f"AUC-ROC: {auc:.4f}")
            logger.info("\nMatriz de Confusão:")
            logger.info(f"{cm}")

            # Adicionar flags de erro
            results['Falso_Positivo'] = ((y_pred == 1) & (y == 0)).astype(int)
            results['Falso_Negativo'] = ((y_pred == 0) & (y == 1)).astype(int)

            # Custo de negócio
            # Assumindo que falsos negativos custam 5x mais que falsos positivos
            cost_ratio = 5.0
            results['Custo'] = results['Falso_Positivo'] * 1 + results['Falso_Negativo'] * cost_ratio

            logger.info(f"\nCusto total: {results['Custo'].sum()}")
            logger.info(f"Custo médio por cliente: {results['Custo'].mean():.4f}")

        # Se não quiser retornar probabilidades
        if not output_probabilities:
            results = results.drop(columns=['Probabilidade'])

        logger.info(f"Predições concluídas para {len(results)} registros.")
        return results

    def predict_and_explain(self, data, target_col=None, num_features=10):
        """
        Faz predições e gera explicações simplificadas.

        Args:
            data: DataFrame com dados para predição
            target_col: Nome da coluna alvo (se disponível)
            num_features: Número de features para incluir na explicação

        Returns:
            DataFrame com predições e explicações
        """
        # Fazer predições normais
        results = self.predict(data, target_col, output_probabilities=True)

        # Se o modelo não suporta importância de features, retornar apenas predições
        if not hasattr(self.model, 'feature_importances_') and not hasattr(self.model, 'coef_'):
            logger.warning("Este modelo não suporta explicações baseadas em importância de features.")
            return results

        # Preparar dados
        X, _ = self.prepare_data(data, target_col)

        # Extrair importância de features
        if hasattr(self.model, 'feature_importances_'):
            # Para modelos baseados em árvores
            feature_importance = self.model.feature_importances_
        else:
            # Para modelos lineares
            feature_importance = np.abs(self.model.coef_[0])

        # Certificar que temos nomes de features
        if isinstance(X, pd.DataFrame):
            feature_names = X.columns
        else:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]

        # Criar DataFrame com importâncias
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': feature_importance
        })
        importance_df = importance_df.sort_values('Importance', ascending=False)

        # Gerar explicações para cada exemplo
        explanations = []

        for i in range(len(X)):
            # Obter valores do exemplo atual
            if isinstance(X, pd.DataFrame):
                example_values = X.iloc[i]
            else:
                example_values = X[i]

            # Criar DataFrame com valores e importâncias
            explanation_df = pd.DataFrame({
                'Feature': feature_names,
                'Value': example_values,
                'Importance': feature_importance
            })

            # Ordenar por importância absoluta
            explanation_df = explanation_df.sort_values('Importance', ascending=False)

            # Pegar top features
            top_features = explanation_df.head(num_features)

            # Gerar explicação em texto
            if results.iloc[i]['Inadimplencia_Predicao'] == 1:
                explanation = f"Cliente classificado como RISCO DE INADIMPLÊNCIA (probabilidade: {results.iloc[i]['Probabilidade']:.2%}). "
                explanation += "Principais fatores: "

                for j, row in top_features.iterrows():
                    explanation += f"{row['Feature']}={row['Value']:.2f}, "

                explanation = explanation[:-2] + "."
            else:
                explanation = f"Cliente classificado como BOM PAGADOR (probabilidade de inadimplência: {results.iloc[i]['Probabilidade']:.2%}). "
                explanation += "Principais fatores: "

                for j, row in top_features.iterrows():
                    explanation += f"{row['Feature']}={row['Value']:.2f}, "

                explanation = explanation[:-2] + "."

            explanations.append(explanation)

        # Adicionar explicações ao DataFrame de resultados
        results['Explicacao'] = explanations

        return results

    def batch_predict(self, data_path, output_path=None, target_col=None):
        """
        Faz predições em lote para um arquivo de dados.

        Args:
            data_path: Caminho para arquivo de dados
            output_path: Caminho para salvar resultados
            target_col: Nome da coluna alvo (se disponível)

        Returns:
            DataFrame com predições
        """
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
        results = self.predict(data, target_col, output_probabilities=True)

        # Salvar resultados (se caminho fornecido)
        if output_path:
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

    def plot_prediction_distribution(self, results, output_dir=None):
        """
        Plota a distribuição das predições.

        Args:
            results: DataFrame com resultados de predições
            output_dir: Diretório para salvar gráficos

        Returns:
            None
        """
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # 1. Distribuição de probabilidades
        plt.figure(figsize=(10, 6))
        sns.histplot(results['Probabilidade'], bins=50, kde=True)
        plt.axvline(x=self.threshold, color='red', linestyle='--',
                    label=f'Threshold = {self.threshold:.2f}')
        plt.title('Distribuição de Probabilidades de Inadimplência')
        plt.xlabel('Probabilidade')
        plt.ylabel('Contagem')
        plt.legend()
        plt.grid(True, alpha=0.3)

        if output_dir:
            plt.savefig(os.path.join(output_dir, 'distribuicao_probabilidades.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # 2. Gráfico de pizza com distribuição de classes previstas
        plt.figure(figsize=(8, 8))
        class_counts = results['Inadimplencia_Predicao'].value_counts()
        labels = ['Bom Pagador', 'Risco de Inadimplência']
        plt.pie(class_counts, labels=labels, autopct='%1.1f%%', startangle=90,
                colors=['#4CAF50', '#F44336'], explode=(0, 0.1))
        plt.title('Distribuição das Classificações de Risco')

        if output_dir:
            plt.savefig(os.path.join(output_dir, 'distribuicao_classes.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # 3. Gráfico de barras com níveis de risco
        plt.figure(figsize=(10, 6))
        risk_counts = results['Nivel_Risco'].value_counts().sort_index()
        sns.barplot(x=risk_counts.index, y=risk_counts.values, palette='YlOrRd')
        plt.title('Distribuição por Nível de Risco')
        plt.xlabel('Nível de Risco')
        plt.ylabel('Contagem')
        plt.grid(True, alpha=0.3, axis='y')

        if output_dir:
            plt.savefig(os.path.join(output_dir, 'distribuicao_niveis_risco.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # 4. Se tiver target real, plotar matriz de confusão
        if 'Inadimplencia_Real' in results.columns:
            plt.figure(figsize=(8, 6))
            cm = pd.crosstab(results['Inadimplencia_Real'], results['Inadimplencia_Predicao'],
                             rownames=['Real'], colnames=['Predito'], normalize='all')

            sns.heatmap(cm, annot=True, fmt='.2%', cmap='Blues', cbar=False)
            plt.title('Matriz de Confusão Normalizada')

            if output_dir:
                plt.savefig(os.path.join(output_dir, 'matriz_confusao.png'), dpi=300, bbox_inches='tight')
            plt.close()

        logger.info("Gráficos de distribuição gerados com sucesso.")


def main():
    """
    Função principal para fazer predições usando modelos treinados.
    """
    import argparse

    # Definir argumentos do comando
    parser = argparse.ArgumentParser(description="Fazer predições usando modelos treinados de inadimplência")
    parser.add_argument('--data', type=str, required=True,
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

    args = parser.parse_args()

    try:
        # Criar preditor
        predictor = ModelPredictor()

        # Carregar modelo
        if args.model:
            predictor.load_model(args.model)
        else:
            # Encontrar e carregar o melhor modelo mais recente
            model_path = predictor.find_latest_model()
            predictor.load_model(model_path)

        # Carregar feature engineer
        if args.feature_engineer:
            predictor.load_feature_engineer(args.feature_engineer)
        else:
            # Tentar encontrar feature engineer correspondente
            feature_path = predictor.find_matching_feature_engineer(predictor.model.__class__.__name__)
            if feature_path:
                predictor.load_feature_engineer(feature_path)

        # Definir caminho de saída padrão se não fornecido
        if not args.output:
            project_root = get_project_root()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            args.output = os.path.join(project_root, 'reports', 'predictions', f"predctions_{timestamp}.csv")

        # Fazer predições
        if args.explain:
            results = predictor.predict_and_explain(
                data=pd.read_csv(args.data) if args.data.endswith('.csv') else pd.read_excel(args.data),
                target_col=args.target
            )
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
        logger.error(f"Erro durante a predição: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()