"""
Gerenciador de caminhos para o projeto de ML para inadimplência.
Facilita a localização de arquivos em qualquer contexto de execução.
"""

import os
import glob
import logging
from typing import Optional, List

logger = logging.getLogger(__name__)


class PathManager:
    """Gerencia caminhos de arquivos para o projeto de Machine Learning."""

    def __init__(self):
        """Inicializa o gerenciador de caminhos com base na estrutura do projeto."""
        # Determinar raiz do projeto
        current_file = os.path.abspath(__file__)
        # Se estamos em src/utils/path_manager.py, subir dois níveis
        self.project_root = os.path.abspath(os.path.join(os.path.dirname(current_file), "../../"))

        # Verificar se a estrutura corresponde a um projeto ML
        if not (os.path.isdir(os.path.join(self.project_root, "data")) and
                os.path.isdir(os.path.join(self.project_root, "src"))):
            # Tentar detectar automaticamente
            current_dir = os.getcwd()
            while True:
                if (os.path.isdir(os.path.join(current_dir, "data")) and
                        os.path.isdir(os.path.join(current_dir, "src"))):
                    self.project_root = current_dir
                    break

                parent_dir = os.path.dirname(current_dir)
                if parent_dir == current_dir:  # Chegou à raiz do sistema
                    # Usar diretório atual como fallback
                    self.project_root = os.getcwd()
                    logger.warning(f"Não foi possível determinar a raiz do projeto. Usando: {self.project_root}")
                    break

                current_dir = parent_dir

        logger.info(f"Raiz do projeto: {self.project_root}")

        # Definir diretórios do projeto
        self.data_dir = os.path.join(self.project_root, "data")
        self.models_dir = os.path.join(self.project_root, "models")
        self.reports_dir = os.path.join(self.project_root, "reports")

        # Subdiretorios específicos
        self.data_paths = {
            "raw": os.path.join(self.data_dir, "raw"),
            "processed": os.path.join(self.data_dir, "processed"),
            "interim": os.path.join(self.data_dir, "interim"),
            "external": os.path.join(self.data_dir, "external")
        }

        self.models_paths = {
            "trained": os.path.join(self.models_dir, "trained_models"),
            "preprocessing": os.path.join(self.models_dir, "preprocessing")
        }

        self.reports_paths = {
            "figures": os.path.join(self.reports_dir, "figures"),
            "predictions": os.path.join(self.reports_dir, "predictions")
        }

        # Criar diretórios se necessário
        for paths in [self.data_paths, self.models_paths, self.reports_paths]:
            for path in paths.values():
                if not os.path.exists(path):
                    os.makedirs(path, exist_ok=True)

    def find_file(self, filename: str, search_dirs: List[str] = None) -> Optional[str]:
        """
        Procura por um arquivo em vários diretórios do projeto.

        Args:
            filename: Nome do arquivo ou parte dele
            search_dirs: Lista de diretórios para buscar (se None, busca em todos)

        Returns:
            Caminho completo para o arquivo encontrado ou None
        """
        if search_dirs is None:
            # Buscar em todos os diretórios por padrão
            search_dirs = []
            for paths in [self.data_paths, self.models_paths, self.reports_paths]:
                search_dirs.extend(paths.values())

        # Verificar primeiro por nome exato
        for directory in search_dirs:
            file_path = os.path.join(directory, filename)
            if os.path.isfile(file_path):
                return file_path

        # Buscar de forma mais flexível
        for directory in search_dirs:
            # Verificar se filename é um padrão glob ou apenas parte do nome
            if "*" in filename:
                pattern = os.path.join(directory, filename)
            else:
                pattern = os.path.join(directory, f"*{filename}*")

            matching_files = glob.glob(pattern)

            if matching_files:
                # Retornar o arquivo mais recente
                return max(matching_files, key=os.path.getmtime)

        return None

    def find_data_file(self, filename: str) -> Optional[str]:
        """Procura um arquivo nos diretórios de dados."""
        return self.find_file(filename, list(self.data_paths.values()))

    def find_model_file(self, filename: str) -> Optional[str]:
        """Procura um arquivo de modelo nos diretórios de modelos."""
        return self.find_file(filename, list(self.models_paths.values()))

    def get_data_path(self, subdir: str, filename: Optional[str] = None) -> str:
        """Retorna o caminho para um diretório/arquivo de dados."""
        if subdir not in self.data_paths:
            raise ValueError(f"Subdiretório de dados inválido: {subdir}")

        if filename:
            return os.path.join(self.data_paths[subdir], filename)
        return self.data_paths[subdir]

    def get_model_path(self, subdir: str, filename: Optional[str] = None) -> str:
        """Retorna o caminho para um diretório/arquivo de modelo."""
        if subdir not in self.models_paths:
            raise ValueError(f"Subdiretório de modelos inválido: {subdir}")

        if filename:
            return os.path.join(self.models_paths[subdir], filename)
        return self.models_paths[subdir]

    def get_report_path(self, subdir: str, filename: Optional[str] = None) -> str:
        """Retorna o caminho para um diretório/arquivo de relatório."""
        if subdir not in self.reports_paths:
            raise ValueError(f"Subdiretório de relatórios inválido: {subdir}")

        if filename:
            return os.path.join(self.reports_paths[subdir], filename)
        return self.reports_paths[subdir]