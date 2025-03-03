"""
Módulo para integrar modelos calibrados com o sistema de predição de inadimplência.
Permite usar modelos calibrados no pipeline de predição e comparar resultados.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
import re
from datetime import datetime
from typing import Dict, Any, Optional, Union

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

# Classe de compatibilidade para carregar objetos FeatureEngineer serializados
class FeatureEngineer:
    """
    Classe de compatibilidade para deserialização de objetos FeatureEngineer.
    Fornece a estrutura mínima para que o objeto serializado possa ser carregado.
    """

    def __init__(self):
        self.transformers = {}
        self.pipeline_config = {}
        self.selected_features = []

    def transform(self, X):
        """Aplica transformações aos dados."""
        # Implementação simplificada
        return X

    def fit_transform(self, X, y=None):
        """Ajusta e transforma dados."""
        return self.transform(X)

# Tentar importar módulos relevantes
try:
    from src.models.predict_model import ModelPredictor, PathManager
except ImportError:
    logger.error("Não foi possível importar ModelPredictor ou PathManager.")
    logger.error("Verifique a instalação e o PYTHONPATH.")
    sys.exit(1)


def encontrar_dados_teste_correspondentes(timestamp_modelo=None):
    """
    Encontra automaticamente o arquivo de dados de teste apropriado com base no timestamp do modelo.

    Os arquivos são procurados nas seguintes localizações específicas:
    - data/interim
    - data/processed
    - data/processed/dados_processados

    O algoritmo tenta encontrar um arquivo correspondente ao timestamp do modelo.
    Se não for encontrado, usa o arquivo de teste mais recente.

    Args:
        timestamp_modelo: Timestamp opcional do modelo para correspondência com arquivos de dados

    Returns:
        Caminho para o arquivo de dados de teste mais apropriado ou None se nenhum for encontrado
    """
    project_root = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

    # Diretórios específicos onde procurar dados de teste, conforme instruções
    diretorios_dados = [
        os.path.join(project_root, 'data', 'interim'),
        os.path.join(project_root, 'data', 'processed'),
        os.path.join(project_root, 'data', 'processed', 'dados_processados')
    ]

    logger.info(f"Buscando dados de teste nos diretórios: {diretorios_dados}")

    arquivos_candidatos = []

    # Buscar arquivos de dados de teste em todos os diretórios especificados
    for diretorio in diretorios_dados:
        if not os.path.exists(diretorio):
            logger.info(f"Diretório {diretorio} não encontrado, pulando...")
            continue

        logger.info(f"Verificando arquivos em: {diretorio}")

        # Procurar arquivos de dados de teste com padrões comuns
        for arquivo in os.listdir(diretorio):
            arquivo_path = os.path.join(diretorio, arquivo)

            # Verificar se é um arquivo (não diretório) e termina com .csv
            if not os.path.isfile(arquivo_path) or not arquivo.endswith('.csv'):
                continue

            # Verificar padrões comuns de arquivos de teste
            if (arquivo.startswith('test_') or
                arquivo.startswith('teste_') or
                'test' in arquivo.lower() or
                'teste' in arquivo.lower()):

                # Verificar se o arquivo não está vazio (pelo menos 100 bytes)
                if os.path.getsize(arquivo_path) > 100:
                    arquivos_candidatos.append(arquivo_path)
                    logger.info(f"Arquivo candidato encontrado: {arquivo}")

    if not arquivos_candidatos:
        logger.warning("Nenhum arquivo de dados de teste encontrado nos diretórios especificados")
        return None

    # Se tivermos um timestamp de modelo, tente encontrar um arquivo de teste correspondente
    timestamp_correspondentes = []
    if timestamp_modelo:
        logger.info(f"Procurando arquivos que correspondam ao timestamp do modelo: {timestamp_modelo}")
        for arquivo in arquivos_candidatos:
            if timestamp_modelo in os.path.basename(arquivo):
                timestamp_correspondentes.append(arquivo)

        if timestamp_correspondentes:
            # Se houver múltiplos arquivos correspondentes, use o maior (provavelmente mais completo)
            if len(timestamp_correspondentes) > 1:
                arquivo_selecionado = max(timestamp_correspondentes, key=os.path.getsize)
                logger.info(f"Múltiplos arquivos encontrados com timestamp correspondente. Selecionando o maior: {arquivo_selecionado}")
            else:
                arquivo_selecionado = timestamp_correspondentes[0]
                logger.info(f"Encontrado dados de teste correspondentes ao timestamp: {arquivo_selecionado}")

            return arquivo_selecionado
        else:
            logger.info(f"Nenhum arquivo com timestamp {timestamp_modelo} encontrado. Buscando alternativas...")

    # Ordenar candidatos por data de modificação (mais recente primeiro)
    arquivos_por_data = sorted(arquivos_candidatos, key=os.path.getmtime, reverse=True)

    # Se não encontrar correspondência exata, usar o arquivo mais recente
    arquivo_mais_recente = arquivos_por_data[0] if arquivos_por_data else None

    if arquivo_mais_recente:
        logger.info(f"Usando arquivo de teste mais recente: {arquivo_mais_recente}")
        return arquivo_mais_recente
    else:
        logger.warning("Não foi possível encontrar um arquivo de dados adequado")
        return None


class CalibratedModelPredictor(ModelPredictor):
    """
    Extende ModelPredictor para trabalhar especificamente com modelos calibrados.
    """

    def __init__(self, model_path: Optional[str] = None, feature_engineer_path: Optional[str] = None):
        """
        Inicializa o preditor de modelos calibrados.

        Args:
            model_path: Caminho para o modelo treinado
            feature_engineer_path: Caminho para o engenheiro de features
        """
        # Inicializa a classe base
        super().__init__(model_path, feature_engineer_path)

        # Atributos específicos para modelos calibrados
        self.calibration_metrics = {}
        self.default_threshold = 0.8  # Threshold mais conservador para modelos calibrados
        self.threshold = 0.8

    def find_latest_calibrated_model(self) -> str:
        """
        Encontra o modelo calibrado mais recente.

        Returns:
            Caminho para o modelo calibrado mais recente
        """
        # Diretórios onde procurar modelos calibrados, em ordem de prioridade
        dirs_to_check = [
            self.path_manager.get_model_path("calibrated_models"),
            os.path.join(self.path_manager.project_root, "models", "calibrated_models"),
            self.path_manager.get_model_path("trained_models"),
            os.path.join(self.path_manager.project_root, "models", "trained")
        ]

        # Padrões de busca para modelos calibrados
        patterns = [
            "best_calibrated_model_*.joblib",
            "CalibratedEnsemble_*.joblib",
            "Calibrated*.joblib"
        ]

        import glob

        for directory in dirs_to_check:
            if not os.path.exists(directory):
                continue

            for pattern in patterns:
                matching_files = glob.glob(os.path.join(directory, pattern))

                if matching_files:
                    # Ordenar por data de modificação (mais recente primeiro)
                    matching_files.sort(key=os.path.getmtime, reverse=True)
                    latest_model = matching_files[0]

                    logger.info(f"Modelo calibrado mais recente encontrado: {latest_model}")
                    return latest_model

        # Se não encontrou modelos calibrados, procurar qualquer modelo
        logger.warning("Não foram encontrados modelos calibrados específicos.")
        return self.find_latest_model()

    def load_calibrated_model(self, model_path: str) -> 'CalibratedModelPredictor':
        """
        Carrega um modelo calibrado salvo.

        Args:
            model_path: Caminho para o modelo calibrado

        Returns:
            self para encadeamento de métodos
        """
        # Usar o método da classe base para carregar o modelo
        self.load_model(model_path)

        # Verificar se é realmente um modelo calibrado (lógica melhorada)
        is_calibrated = False
        model_name = os.path.basename(model_path)

        # Verificar pelo nome do arquivo
        if "Calibrated" in model_name or "calibrated" in model_name:
            is_calibrated = True

        # Verificar pelo tipo do modelo
        if hasattr(self.model, '__class__'):
            class_name = self.model.__class__.__name__
            if 'Calibrated' in class_name or class_name in ['CalibratedClassifierCV']:
                is_calibrated = True

        # Verificar pelo método predict_proba (todos os modelos calibrados implementam isso)
        if hasattr(self.model, 'predict_proba'):
            is_calibrated = True

        if is_calibrated:
            logger.info(f"Modelo calibrado '{model_name}' carregado com sucesso.")

            # Configurar threshold específico para modelos calibrados
            if not hasattr(self.model, 'threshold'):
                self.threshold = self.default_threshold
                logger.info(f"Usando threshold padrão para modelos calibrados: {self.threshold}")
        else:
            logger.warning(f"O modelo carregado '{model_name}' pode não ser um modelo calibrado.")

        return self

    def analyze_calibration(self, data: pd.DataFrame, target_col: Optional[str] = None) -> Dict[str, Any]:
        """
        Analisa a calibração do modelo em um conjunto de dados.

        Args:
            data: DataFrame com features
            target_col: Nome da coluna alvo (se disponível)

        Returns:
            Dicionário com métricas de calibração
        """
        if self.model is None:
            raise ValueError("Modelo não foi carregado. Use load_model() primeiro.")

        # Verificar se target_col está disponível
        if target_col is None or target_col not in data.columns:
            logger.warning("Coluna alvo não disponível. Não é possível calcular métricas de calibração.")
            return {}

        # Preparar features
        X = self._prepare_features(data, target_col)
        y_true = data[target_col].values

        # Obter probabilidades
        try:
            y_proba = self.model.predict_proba(X)[:, 1]
        except Exception as e:
            logger.error(f"Erro ao calcular probabilidades: {str(e)}")
            return {}

        # Calcular métricas de calibração
        try:
            from sklearn.calibration import calibration_curve
            from sklearn.metrics import brier_score_loss, log_loss

            # Curva de calibração
            prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=10)

            # Brier score (erro quadrático médio)
            brier_score = brier_score_loss(y_true, y_proba)

            # Log-loss (entropia cruzada)
            log_loss_score = log_loss(y_true, y_proba)

            # ECE - Expected Calibration Error (simplificado)
            ece = np.mean(np.abs(prob_true - prob_pred))

            # Outras métricas
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(y_true, y_proba)

            # Armazenar resultados
            self.calibration_metrics = {
                'brier_score': brier_score,
                'log_loss': log_loss_score,
                'ece': ece,
                'auc': auc,
                'prob_true': prob_true.tolist(),
                'prob_pred': prob_pred.tolist()
            }

            logger.info("Métricas de calibração:")
            logger.info(f"  Brier Score: {brier_score:.4f} (menor é melhor)")
            logger.info(f"  Log Loss: {log_loss_score:.4f} (menor é melhor)")
            logger.info(f"  ECE: {ece:.4f} (menor é melhor)")
            logger.info(f"  AUC: {auc:.4f}")

            return self.calibration_metrics

        except ImportError:
            logger.error("Bibliotecas necessárias para análise de calibração não disponíveis.")
            return {}
        except Exception as e:
            logger.error(f"Erro ao calcular métricas de calibração: {str(e)}")
            return {}

    def plot_calibration(self, output_dir: Optional[str] = None) -> None:
        """
        Gera visualizações da calibração do modelo.

        Args:
            output_dir: Diretório para salvar visualizações
        """
        if not self.calibration_metrics:
            logger.warning("Nenhuma métrica de calibração disponível. Execute analyze_calibration() primeiro.")
            return

        try:
            import matplotlib.pyplot as plt

            # Verificar diretório de saída
            if output_dir is None:
                path_manager = PathManager()
                output_dir = path_manager.get_report_path("plots", "calibration")

            os.makedirs(output_dir, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Plotar curva de calibração
            plt.figure(figsize=(10, 8))

            # Extrair dados da curva de calibração
            prob_true = self.calibration_metrics['prob_true']
            prob_pred = self.calibration_metrics['prob_pred']
            brier_score = self.calibration_metrics['brier_score']

            # Plotar curva de calibração
            plt.plot(prob_pred, prob_true, marker='o', linewidth=2, label='Modelo')

            # Linha de calibração perfeita
            plt.plot([0, 1], [0, 1], 'k--', label='Calibração Perfeita')

            # Configurações do gráfico
            plt.xlabel('Probabilidade Média Predita')
            plt.ylabel('Fração de Positivos')
            plt.title(f'Curva de Calibração\nBrier Score: {brier_score:.4f} (menor é melhor)')
            plt.legend(loc='best')
            plt.grid(True, alpha=0.3)

            # Salvar gráfico
            calibration_path = os.path.join(output_dir, f'calibration_curve_{timestamp}.png')
            plt.savefig(calibration_path, dpi=300, bbox_inches='tight')
            plt.close()

            logger.info(f"Curva de calibração salva em: {calibration_path}")

        except ImportError:
            logger.error("Bibliotecas necessárias para visualização não disponíveis.")
            return
        except Exception as e:
            logger.error(f"Erro ao gerar visualizações de calibração: {str(e)}")
            return

    def predict_with_reliability(self, data: pd.DataFrame, target_col: Optional[str] = None) -> pd.DataFrame:
        """
        Faz predições incluindo informações sobre confiabilidade/calibração.

        Args:
            data: DataFrame com features
            target_col: Nome da coluna alvo (se disponível)

        Returns:
            DataFrame com predições e métricas de confiabilidade
        """
        # Obter predições básicas
        results = self.predict(data, target_col, output_probabilities=True)

        # Adicionar informações de confiabilidade
        probabilities = results['probabilidade_inadimplencia']

        # 1. Calcular distância do threshold (confiança na decisão)
        results['confianca_decisao'] = np.abs(probabilities - self.threshold)

        # 2. Categorizar confiança em faixas
        conditions = [
            (results['confianca_decisao'] >= 0.4),
            (results['confianca_decisao'] >= 0.2) & (results['confianca_decisao'] < 0.4),
            (results['confianca_decisao'] < 0.2)
        ]

        choices = ['Alta', 'Média', 'Baixa']
        results['nivel_confianca'] = np.select(conditions, choices, default='Média')

        # 3. Adicionar flag para casos de revisão manual (baixa confiança)
        results['revisao_manual'] = results['nivel_confianca'] == 'Baixa'

        logger.info(f"Predições realizadas com informações de confiabilidade.")
        logger.info(f"Distribuição de níveis de confiança:")
        for nivel in ['Alta', 'Média', 'Baixa']:
            count = (results['nivel_confianca'] == nivel).sum()
            percent = 100 * count / len(results)
            logger.info(f"  {nivel}: {count} ({percent:.1f}%)")

        logger.info(
            f"Casos para revisão manual: {results['revisao_manual'].sum()} ({100 * results['revisao_manual'].sum() / len(results):.1f}%)")

        return results

    def compare_with_original_model(self, data: pd.DataFrame, target_col: Optional[str] = None,
                                   original_model_path: Optional[str] = None) -> pd.DataFrame:
        """
        Compara predições do modelo calibrado com um modelo original.

        Args:
            data: DataFrame com features
            target_col: Nome da coluna alvo (se disponível)
            original_model_path: Caminho para o modelo original não calibrado

        Returns:
            DataFrame com comparação de predições
        """
        # Obter predições do modelo calibrado
        calibrated_results = self.predict(data, target_col, output_probabilities=True)
        calibrated_results = calibrated_results.rename(columns={
            'probabilidade_inadimplencia': 'prob_calibrado',
            'inadimplente_previsto': 'pred_calibrado'
        })

        # Carregar e obter predições do modelo original
        original_predictor = ModelPredictor()

        if original_model_path is None:
            # Tentar encontrar modelo não calibrado
            try:
                original_model_path = original_predictor.find_latest_model("best_model")
            except:
                # Se não encontrar best_model, procurar qualquer modelo que não seja calibrado
                path_manager = PathManager()
                model_dir = path_manager.get_model_path("trained_models")
                import glob

                # Excluir modelos calibrados
                models = [m for m in glob.glob(os.path.join(model_dir, "*.joblib"))
                          if "Calibrated" not in os.path.basename(m)]

                if models:
                    models.sort(key=os.path.getmtime, reverse=True)
                    original_model_path = models[0]
                else:
                    logger.error("Não foi possível encontrar um modelo original para comparação.")
                    return calibrated_results

        # Carregar modelo original
        try:
            original_predictor.load_model(original_model_path)
            logger.info(f"Modelo original carregado: {os.path.basename(original_model_path)}")

            # Predições do modelo original
            original_results = original_predictor.predict(data, target_col, output_probabilities=True)
            original_results = original_results.rename(columns={
                'probabilidade_inadimplencia': 'prob_original',
                'inadimplente_previsto': 'pred_original'
            })

            # CORREÇÃO: Usar índice do DataFrame de dados como referência consistente
            # Preservar o índice original de data
            idx = data.index.copy()

            # Preparar DataFrames com o mesmo índice
            calibrated_subset = calibrated_results[['prob_calibrado', 'pred_calibrado']]
            calibrated_subset.index = idx

            original_subset = original_results[['prob_original', 'pred_original']]
            original_subset.index = idx

            # Manter data com seu índice original e fazer join com os outros DataFrames
            comparison = data.copy()
            comparison = comparison.join(calibrated_subset)
            comparison = comparison.join(original_subset)

            # Adicionar coluna de diferença
            comparison['diferenca_prob'] = comparison['prob_calibrado'] - comparison['prob_original']
            comparison['mudanca_decisao'] = comparison['pred_calibrado'] != comparison['pred_original']

            # Adicionar target quando disponível
            if target_col and target_col in data.columns:
                comparison['target_real'] = data[target_col]
                comparison['acerto_calibrado'] = comparison['pred_calibrado'] == comparison['target_real']
                comparison['acerto_original'] = comparison['pred_original'] == comparison['target_real']
                comparison['melhoria'] = comparison['acerto_calibrado'] & ~comparison['acerto_original']
                comparison['piora'] = ~comparison['acerto_calibrado'] & comparison['acerto_original']

            # Estatísticas de comparação
            n_decisoes_diferentes = comparison['mudanca_decisao'].sum()
            pct_diferente = 100 * n_decisoes_diferentes / len(comparison)

            logger.info(f"\nComparação entre modelo calibrado e original:")
            logger.info(f"  Decisões diferentes: {n_decisoes_diferentes} ({pct_diferente:.1f}%)")

            if target_col and target_col in data.columns:
                acertos_calibrado = comparison['acerto_calibrado'].mean() * 100
                acertos_original = comparison['acerto_original'].mean() * 100
                melhoria = comparison['melhoria'].sum()
                piora = comparison['piora'].sum()

                logger.info(f"  Acurácia modelo calibrado: {acertos_calibrado:.1f}%")
                logger.info(f"  Acurácia modelo original: {acertos_original:.1f}%")
                logger.info(f"  Casos com melhoria: {melhoria} ({100 * melhoria / len(comparison):.1f}%)")
                logger.info(f"  Casos com piora: {piora} ({100 * piora / len(comparison):.1f}%)")

            return comparison

        except Exception as e:
            logger.error(f"Erro ao comparar modelos: {str(e)}")
            return calibrated_results


def find_and_load_best_model() -> Union[CalibratedModelPredictor, ModelPredictor]:
    """
    Encontra e carrega o melhor modelo disponível, com preferência para modelos calibrados.

    Returns:
        Preditor com o melhor modelo carregado
    """
    try:
        # Primeiro, tentar encontrar e carregar modelo calibrado
        calibrated_predictor = CalibratedModelPredictor()

        try:
            calibrated_model_path = calibrated_predictor.find_latest_calibrated_model()
            calibrated_predictor.load_calibrated_model(calibrated_model_path)
            logger.info(f"Modelo calibrado carregado: {os.path.basename(calibrated_model_path)}")
            return calibrated_predictor
        except Exception as e:
            logger.warning(f"Não foi possível carregar modelo calibrado: {str(e)}")

        # Se não conseguir carregar modelo calibrado, tentar modelo regular
        standard_predictor = ModelPredictor()

        try:
            model_path = standard_predictor.find_latest_model()
            standard_predictor.load_model(model_path)
            logger.info(f"Modelo padrão carregado: {os.path.basename(model_path)}")
            return standard_predictor
        except Exception as e:
            logger.error(f"Não foi possível carregar nenhum modelo: {str(e)}")
            raise ValueError("Nenhum modelo disponível para carregamento.")

    except Exception as e:
        logger.error(f"Erro ao procurar e carregar modelos: {str(e)}")
        raise


def main():
    """
    Função principal para integrar modelos calibrados com o sistema de predição.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Integração de modelos calibrados para predição de inadimplência")
    parser.add_argument('--data', type=str, help='Caminho para arquivo de dados (CSV ou Excel)')
    parser.add_argument('--calibrated_model', type=str, default=None,
                        help='Caminho para modelo calibrado (se None, usa o mais recente)')
    parser.add_argument('--original_model', type=str, default=None,
                        help='Caminho para modelo original (se None, usa o mais recente)')
    parser.add_argument('--target', type=str, default=None,
                        help='Nome da coluna alvo (se disponível, para avaliação)')
    parser.add_argument('--output', type=str, default=None,
                        help='Caminho para salvar resultados (se None, usa diretório padrão)')
    parser.add_argument('--compare', action='store_true',
                        help='Comparar modelo calibrado com modelo original')
    parser.add_argument('--analyze', action='store_true',
                        help='Analisar calibração do modelo')
    parser.add_argument('--reliability', action='store_true',
                        help='Incluir métricas de confiabilidade nas predições')
    parser.add_argument('--threshold', type=float, default=None,
                        help='Threshold para classificação (se None, usa o definido no modelo ou padrão)')

    args = parser.parse_args()

    try:
        # 1. Carregar modelo calibrado
        if args.calibrated_model:
            predictor = CalibratedModelPredictor(args.calibrated_model)
        else:
            # Encontrar o melhor modelo disponível
            predictor = find_and_load_best_model()

        # Extrair timestamp do modelo para correspondência de dados
        caminho_modelo = predictor.model_path if hasattr(predictor, 'model_path') else None
        timestamp_modelo = None
        if caminho_modelo:
            # Extrair timestamp do nome do arquivo do modelo
            nome_arquivo = os.path.basename(caminho_modelo)
            # Procurar padrões como '20250303_004036' no nome do arquivo
            timestamp_match = re.search(r'(\d{8}_\d{6})', nome_arquivo)
            if timestamp_match:
                timestamp_modelo = timestamp_match.group(1)

        # Ajustar threshold se especificado
        if args.threshold is not None:
            predictor.threshold = args.threshold
            logger.info(f"Threshold definido para: {args.threshold}")

        # 2. Verificar se o arquivo de dados foi fornecido, caso contrário, encontrá-lo automaticamente
        if not args.data:
            logger.info("Nenhum arquivo de dados fornecido, buscando automaticamente...")
            args.data = encontrar_dados_teste_correspondentes(timestamp_modelo)

            if not args.data:
                logger.error("Nenhum arquivo de dados adequado encontrado automaticamente. Por favor, especifique com --data")
                return

            logger.info(f"Arquivo de dados selecionado automaticamente: {args.data}")

        # 3. Carregar dados
        path_manager = PathManager()

        # Verificar se o arquivo existe diretamente
        if not os.path.exists(args.data):
            # Tentar encontrar o arquivo nos diretórios de dados
            data_path = path_manager.find_data_file(args.data)
            if data_path:
                args.data = data_path
            else:
                logger.error(f"Arquivo de dados não encontrado: {args.data}")
                return

        # Carregar dados
        logger.info(f"Carregando dados de: {args.data}")

        if args.data.endswith('.csv'):
            data = pd.read_csv(args.data)
        elif args.data.endswith(('.xls', '.xlsx')):
            data = pd.read_excel(args.data)
        else:
            logger.error(f"Formato de arquivo não suportado: {args.data}")
            return

        logger.info(f"Dados carregados: {data.shape[0]} registros, {data.shape[1]} colunas")

        # 4. Definir caminho de saída padrão se não fornecido
        if not args.output:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            args.output = path_manager.get_report_path("predictions", f"calibrated_predictions_{timestamp}.csv")

        # 5. Fazer predições baseadas nas opções fornecidas
        if args.reliability:
            logger.info("Gerando predições com métricas de confiabilidade...")
            if isinstance(predictor, CalibratedModelPredictor):
                results = predictor.predict_with_reliability(data, args.target)
            else:
                logger.warning("Métricas de confiabilidade só estão disponíveis com modelos calibrados.")
                logger.info("Usando predição padrão...")
                results = predictor.predict(data, args.target)
        elif args.compare:
            logger.info("Comparando modelo calibrado com modelo original...")
            if isinstance(predictor, CalibratedModelPredictor):
                results = predictor.compare_with_original_model(data, args.target, args.original_model)
            else:
                logger.warning("A comparação com modelo original requer um preditor calibrado.")
                logger.info("Usando predição padrão...")
                results = predictor.predict(data, args.target)
        else:
            logger.info("Gerando predições padrão...")
            results = predictor.predict(data, args.target)

        # 6. Analisar calibração se solicitado
        if args.analyze and args.target and args.target in data.columns:
            logger.info("Analisando calibração do modelo...")
            if isinstance(predictor, CalibratedModelPredictor):
                calibration_metrics = predictor.analyze_calibration(data, args.target)
                predictor.plot_calibration()
            else:
                logger.warning("Análise de calibração só está disponível com modelos calibrados.")

        # 7. Salvar resultados
        os.makedirs(os.path.dirname(args.output), exist_ok=True)

        if args.output.endswith('.csv'):
            results.to_csv(args.output, index=False)
        elif args.output.endswith(('.xls', '.xlsx')):
            results.to_excel(args.output, index=False)
        else:
            # Padrão para CSV
            if not args.output.endswith(('.csv', '.xls', '.xlsx')):
                args.output += '.csv'
            results.to_csv(args.output, index=False)

        logger.info(f"Resultados salvos em: {args.output}")

        # 8. Exibir resumo das predições
        n_total = len(results)

        if 'pred_calibrado' in results.columns:
            n_inadimplentes = results['pred_calibrado'].sum()
            col_name = 'pred_calibrado'
        elif 'inadimplente_previsto' in results.columns:
            n_inadimplentes = results['inadimplente_previsto'].sum()
            col_name = 'inadimplente_previsto'
        else:
            logger.warning("Coluna de predição não encontrada nos resultados.")
            return

        percent_inadimplentes = 100 * n_inadimplentes / n_total

        logger.info(f"\nResumo das Predições:")
        logger.info(f"Total de registros: {n_total}")
        logger.info(f"Classificados como inadimplentes: {n_inadimplentes} ({percent_inadimplentes:.2f}%)")
        logger.info(f"Threshold utilizado: {predictor.threshold:.4f}")

        # Se houver comparação entre modelos, mostrar estatísticas adicionais
        if 'mudanca_decisao' in results.columns:
            mudancas = results['mudanca_decisao'].sum()
            pct_mudancas = 100 * mudancas / n_total
            logger.info(f"Decisões diferentes entre modelos: {mudancas} ({pct_mudancas:.2f}%)")

        # Se houver target real, mostrar métricas de desempenho
        if args.target and args.target in results.columns:
            try:
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

                # Determinar coluna de predição correta
                y_true = results[args.target]
                y_pred = results[col_name]

                accuracy = accuracy_score(y_true, y_pred)
                precision = precision_score(y_true, y_pred, zero_division=0)
                recall = recall_score(y_true, y_pred, zero_division=0)
                f1 = f1_score(y_true, y_pred, zero_division=0)

                logger.info(f"\nMétricas de desempenho:")
                logger.info(f"Acurácia: {accuracy:.4f}")
                logger.info(f"Precisão: {precision:.4f}")
                logger.info(f"Recall: {recall:.4f}")
                logger.info(f"F1-Score: {f1:.4f}")

                # Se tiver probabilidades disponíveis, calcular AUC
                if 'prob_calibrado' in results.columns:
                    auc = roc_auc_score(y_true, results['prob_calibrado'])
                    logger.info(f"AUC-ROC: {auc:.4f}")
                elif 'probabilidade_inadimplencia' in results.columns:
                    auc = roc_auc_score(y_true, results['probabilidade_inadimplencia'])
                    logger.info(f"AUC-ROC: {auc:.4f}")

            except ImportError:
                logger.warning("Biblioteca sklearn não disponível. Métricas de desempenho não calculadas.")
            except Exception as e:
                logger.warning(f"Erro ao calcular métricas de desempenho: {str(e)}")

        print(f"\nPredição concluída com sucesso! Resultados salvos em: {args.output}")

    except Exception as e:
        logger.error(f"Erro durante a integração de modelos: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()