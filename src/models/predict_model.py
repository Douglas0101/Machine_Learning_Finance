"""
Módulo para fazer predições usando modelos treinados.
Permite carregar modelos salvos, aplicar aos dados e analisar resultados.
"""

import os
import pandas as pd
import joblib
import json
from datetime import datetime
import logging
import glob

# Adicionar importação do PathManager
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from src.utils.path_manager import PathManager

# Configurar logger
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


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

        # Inicializar o gerenciador de caminhos
        self.path_manager = PathManager()

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
            with open(metadata_path, 'r') as f:
                self.model_metadata = json.load(f)

            # Extrair threshold para este modelo
            if 'thresholds' in self.model_metadata and model_name in self.model_metadata['thresholds']:
                self.threshold = self.model_metadata['thresholds'][model_name]
                logger.info(f"Threshold carregado: {self.threshold}")

        # Se o modelo tiver atributo threshold, usar esse
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
        # Verificar se é um caminho completo ou apenas nome de arquivo
        if not os.path.exists(path):
            # Tentar encontrar o arquivo no diretório de modelos/preprocessing
            feature_file = self.path_manager.find_model_file(path)
            if feature_file:
                path = feature_file
            else:
                raise FileNotFoundError(f"Engenheiro de features não encontrado: {path}")

        logger.info(f"Carregando engenheiro de features de: {path}")
        self.feature_engineer = joblib.load(path)
        return self

    def find_latest_model(self, model_type="best_model"):
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
        preprocessing_dir = self.path_manager.get_model_path("preprocessing")
        feature_path = os.path.join(preprocessing_dir, f"feature_engineer_{timestamp}.joblib")

        if os.path.exists(feature_path):
            logger.info(f"Feature engineer correspondente encontrado: {feature_path}")
            return feature_path

        # Se não encontrar com o timestamp específico, procurar o mais recente
        feature_files = [f for f in os.listdir(preprocessing_dir) if
                         f.startswith('feature_engineer_') and f.endswith('.joblib')]

        if not feature_files:
            logger.warning("Nenhum feature engineer encontrado. A preparação básica dos dados será usada.")
            return None

        # Ordenar por timestamp e pegar o mais recente
        feature_files.sort(reverse=True)
        latest_feature = os.path.join(preprocessing_dir, feature_files[0])

        logger.info(f"Usando feature engineer mais recente: {latest_feature}")
        return latest_feature

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
        results = self.predict(data, target_col, output_probabilities=True)

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

# ... outras funções sem alteração ...

def main():
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
        test_files = glob.glob(os.path.join(path_manager.get_data_path(subdir), "test_*.csv"))
        if test_files:
            default_data_file = max(test_files, key=os.path.getmtime)  # O mais recente
            break

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

    args = parser.parse_args()

    try:
        # Inicializar path manager
        path_manager = PathManager()

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
        logger.error(f"Erro durante a predição: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()