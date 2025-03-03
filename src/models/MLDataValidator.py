"""
Módulo MLDataValidator: Validador de Arquivos de Dados para Projetos de Machine Learning.

Fornece uma solução robusta para descoberta, localização e validação
de arquivos de dados em projetos de ciência de dados e machine learning.
"""

import os
import glob
import logging
from typing import List, Optional

class MLDataValidator:
    """
    Validador especializado de arquivos de dados para projetos de Machine Learning.

    Características principais:
    - Descoberta inteligente de arquivos de dados
    - Validação abrangente de estrutura e integridade
    - Suporte a múltiplos formatos e estruturas de projeto
    """

    def __init__(self, project_root: Optional[str] = None, logger: Optional[logging.Logger] = None):
        """
        Inicializa o validador de arquivos de dados para Machine Learning.

        Args:
            project_root: Diretório raiz do projeto.
                          Se None, tenta descobrir automaticamente.
            logger: Logger personalizado.
                    Se None, cria um logger padrão.
        """
        # Configurar logger
        self.logger = logger or self._configure_default_logger()

        # Definir diretório raiz do projeto
        self.project_root = project_root or self._discover_project_root()

        # Diretórios padrão para busca de dados
        self.data_search_paths = [
            'data/processed',
            'data/raw',
            'data/external',
            'datasets',
            'data',
            '../data/processed',
            '../data/raw',
            '../data/external',
            '../../data/processed',
            '../../data/raw',
            '../../data/external'
        ]

    def _configure_default_logger(self) -> logging.Logger:
        """
        Configura um logger padrão com formatação clara.

        Returns:
            Logger configurado
        """
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

        # Configurar handler de console
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Formato de log detalhado
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(formatter)

        # Limpar handlers existentes para evitar duplicação
        logger.handlers.clear()
        logger.addHandler(console_handler)

        return logger

    def _discover_project_root(self) -> str:
        """
        Descobre o diretório raiz do projeto de forma inteligente.

        Returns:
            Caminho absoluto para o diretório raiz do projeto
        """
        # Diretório atual do script
        current_dir = os.path.abspath(os.path.dirname(__file__))

        # Marcadores de diretório raiz
        root_markers = [
            'src',           # Diretório de código fonte
            'models',        # Diretório de modelos
            'data',          # Diretório de dados
            '.git',          # Repositório Git
            'pyproject.toml', # Configuração Python moderna
            'requirements.txt' # Requisitos tradicionais
        ]

        # Subir na hierarquia de diretórios
        while current_dir != os.path.dirname(current_dir):
            # Verificar marcadores de diretório raiz
            if any(os.path.exists(os.path.join(current_dir, marker)) for marker in root_markers):
                # Verificação adicional para garantir diretório correto
                if os.path.basename(current_dir) in ['src', 'models']:
                    return os.path.dirname(current_dir)
                return current_dir

            current_dir = os.path.dirname(current_dir)

        # Fallback para diretório atual
        return os.getcwd()

    def _get_full_search_paths(self) -> List[str]:
        """
        Gera caminhos completos para busca de arquivos de dados.

        Returns:
            Lista de caminhos absolutos para busca
        """
        # Combinar caminhos relativos com o diretório raiz
        full_paths = [
            os.path.join(self.project_root, path)
            for path in self.data_search_paths
        ]

        # Adicionar diretório atual e diretório do script
        current_dir = os.path.abspath(os.path.dirname(__file__))
        full_paths.extend([
            current_dir,
            os.path.join(current_dir, 'data'),
            os.path.join(current_dir, '..', 'data')
        ])

        # Remover duplicatas e caminhos inválidos
        return list(set(path for path in full_paths if os.path.exists(path)))

    def find_data_file(
        self,
        filename: Optional[str] = None,
        extensions: List[str] = ['.csv', '.xlsx', '.xls', '.parquet']
    ) -> Optional[str]:
        """
        Encontra um arquivo de dados usando múltiplas estratégias de busca.

        Args:
            filename: Nome do arquivo (opcional)
            extensions: Extensões de arquivo a considerar

        Returns:
            Caminho completo para o arquivo encontrado
        """
        # Validar caminho absoluto
        if filename and os.path.isfile(filename):
            self.logger.info(f"Arquivo encontrado diretamente: {filename}")
            return filename

        # Padrões de nomenclatura para arquivos de dados
        timestamp_patterns = [
            'train_*', 'test_*', 'val_*',
            '*_train_*', '*_test_*', '*_val_*',
            'engineered_*'  # Adicionado padrão de arquivo de dados engenheirados
        ]

        # Estratégias de busca
        search_paths = self._get_full_search_paths()

        # Busca por nome de arquivo específico
        if filename:
            for path in search_paths:
                for ext in extensions:
                    # Busca com caminho completo
                    full_path = os.path.join(path, filename)
                    if os.path.isfile(full_path):
                        self.logger.info(f"Arquivo encontrado em: {full_path}")
                        return full_path

                    # Busca com wildcard
                    wildcard_path = os.path.join(path, f"*{filename}*{ext}")
                    matches = glob.glob(wildcard_path)
                    if matches:
                        self.logger.info(f"Arquivo encontrado por wildcard: {matches[0]}")
                        return matches[0]

        # Busca por arquivos mais recentes
        most_recent_file = None
        most_recent_time = 0

        for path in search_paths:
            for ext in extensions:
                # Buscar por padrões de timestamp
                for pattern in timestamp_patterns:
                    search_pattern = os.path.join(path, f"{pattern}{ext}")
                    matches = glob.glob(search_pattern)

                    for match in matches:
                        mod_time = os.path.getmtime(match)
                        if mod_time > most_recent_time:
                            most_recent_file = match
                            most_recent_time = mod_time

                # Busca geral se não encontrou por timestamp
                if not most_recent_file:
                    search_pattern = os.path.join(path, f"*{ext}")
                    matches = glob.glob(search_pattern)

                    for match in matches:
                        mod_time = os.path.getmtime(match)
                        if mod_time > most_recent_time:
                            most_recent_file = match
                            most_recent_time = mod_time

        if most_recent_file:
            self.logger.info(f"Arquivo de dados mais recente encontrado: {most_recent_file}")
            return most_recent_file

        # Nenhum arquivo encontrado
        self.logger.warning("Nenhum arquivo de dados encontrado.")
        return None

    def validate_data_file(self, filepath: str) -> bool:
        """
        Valida um arquivo de dados com verificações abrangentes.

        Args:
            filepath: Caminho completo para o arquivo

        Returns:
            Booleano indicando validade do arquivo
        """
        try:
            # 1. Verificar existência
            if not os.path.exists(filepath):
                self.logger.error(f"Arquivo não existe: {filepath}")
                return False

            # 2. Verificar tamanho mínimo
            if os.path.getsize(filepath) == 0:
                self.logger.error(f"Arquivo vazio: {filepath}")
                return False

            # 3. Verificar permissões de leitura
            if not os.access(filepath, os.R_OK):
                self.logger.error(f"Sem permissão de leitura: {filepath}")
                return False

            # 4. Verificações específicas por extensão
            _, ext = os.path.splitext(filepath)
            ext = ext.lower()

            if ext in ['.csv', '.xlsx', '.xls']:
                import pandas as pd
                try:
                    # Ler primeiras linhas para validação
                    if ext == '.csv':
                        df = pd.read_csv(filepath, nrows=5)
                    else:
                        df = pd.read_excel(filepath, nrows=5)

                    # Verificar número mínimo de colunas
                    if df.shape[1] < 2:
                        self.logger.error(f"Arquivo tem menos de 2 colunas: {filepath}")
                        return False

                    # Verificar tipos de dados
                    if df.empty:
                        self.logger.warning(f"Arquivo de dados está vazio: {filepath}")
                        return False

                except Exception as e:
                    self.logger.error(f"Erro ao ler arquivo: {filepath}. Erro: {str(e)}")
                    return False

            self.logger.info(f"Arquivo validado com sucesso: {filepath}")
            return True

        except Exception as e:
            self.logger.error(f"Erro inesperado na validação: {str(e)}")
            return False

def main():
    """
    Demonstração do uso do MLDataValidator.
    """
    # Configurar logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Criar validador
    validator = MLDataValidator()

    # Exemplos de uso
    print("\n===== Exemplos de Busca de Arquivos =====")

    # 1. Buscar arquivo específico
    specific_file = validator.find_data_file('test_data.csv')
    if specific_file:
        validator.validate_data_file(specific_file)

    # 2. Buscar arquivo mais recente
    recent_file = validator.find_data_file()
    if recent_file:
        validator.validate_data_file(recent_file)

if __name__ == "__main__":
    main()