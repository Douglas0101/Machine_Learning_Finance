"""
API para servir predições do modelo de inadimplência.
Fornece endpoints para predição em tempo real, monitoramento
e feedback de resultados para atualização contínua.
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
import json
import logging
from typing import List, Dict, Optional, Any, Union
from datetime import datetime
from fastapi import FastAPI, HTTPException, Query, Depends, BackgroundTasks, status, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, validator, root_validator
from enum import Enum

# Adicionar diretório raiz ao path para importações relativas
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
sys.path.insert(0, project_root)

# Importar módulos do projeto
try:
    from src.models.monitor_model import ModelMonitor
except ImportError:
    # Fallback se a importação falhar
    ModelMonitor = None

# Configurar logger
logger = logging.getLogger(__name__)
log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler = logging.StreamHandler()
handler.setFormatter(log_format)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


# Classes de enumeração para campos com opções fixas
class EstadoCivil(str, Enum):
    SOLTEIRO = "solteiro"
    CASADO = "casado"
    DIVORCIADO = "divorciado"
    VIUVO = "viuvo"


class Sexo(str, Enum):
    MASCULINO = "masculino"
    FEMININO = "feminino"


class TipoResidencia(str, Enum):
    PROPRIA = "propria"
    ALUGADA = "alugada"
    FINANCIADA = "financiada"
    OUTROS = "outros"


class NivelEducacao(str, Enum):
    FUNDAMENTAL = "fundamental"
    MEDIO = "medio"
    SUPERIOR = "superior"
    POS_GRADUACAO = "pos_graduacao"


class StatusInadimplencia(str, Enum):
    ADIMPLENTE = "adimplente"
    INADIMPLENTE = "inadimplente"
    DESCONHECIDO = "desconhecido"


# Modelo para os dados do cliente
class ClienteData(BaseModel):
    """Modelo de dados do cliente para predição de inadimplência."""

    # Informações pessoais
    idade: int = Field(..., ge=18, le=100, description="Idade do cliente em anos")
    sexo: Sexo = Field(..., description="Sexo do cliente")
    estado_civil: EstadoCivil = Field(..., description="Estado civil do cliente")
    num_dependentes: int = Field(..., ge=0, le=10, description="Número de dependentes")
    educacao: NivelEducacao = Field(..., description="Nível de educação do cliente")

    # Informações financeiras
    renda_mensal: float = Field(..., ge=0, description="Renda mensal do cliente em reais")
    valor_patrimonio: Optional[float] = Field(None, ge=0, description="Valor total do patrimônio do cliente em reais")

    # Informações de habitação
    tipo_residencia: TipoResidencia = Field(..., description="Tipo de residência do cliente")
    tempo_residencia: int = Field(..., ge=0, description="Tempo na residência atual em meses")

    # Histórico de crédito
    tempo_emprego_atual: int = Field(..., ge=0, description="Tempo no emprego atual em meses")
    num_contas_bancarias: int = Field(..., ge=0, description="Número de contas bancárias")
    num_cartoes_credito: int = Field(..., ge=0, description="Número de cartões de crédito")

    # Dívidas e empréstimos
    valor_emprestimo: float = Field(..., gt=0, description="Valor do empréstimo solicitado em reais")
    taxa_juros: float = Field(..., ge=0, le=100, description="Taxa de juros anual do empréstimo em percentual")
    prazo_emprestimo: int = Field(..., gt=0, description="Prazo do empréstimo em meses")
    valor_entrada: Optional[float] = Field(0, ge=0, description="Valor de entrada/sinal pago em reais")

    # Histórico de pagamentos
    divida_atual_total: Optional[float] = Field(0, ge=0, description="Valor total da dívida atual em reais")
    num_emprestimos_ativos: Optional[int] = Field(0, ge=0, description="Número de empréstimos ativos")
    num_pagamentos_atrasados_30d: Optional[int] = Field(0, ge=0,
                                                        description="Número de pagamentos atrasados 30+ dias nos últimos 12 meses")
    num_pagamentos_atrasados_60d: Optional[int] = Field(0, ge=0,
                                                        description="Número de pagamentos atrasados 60+ dias nos últimos 12 meses")
    num_pagamentos_atrasados_90d: Optional[int] = Field(0, ge=0,
                                                        description="Número de pagamentos atrasados 90+ dias nos últimos 12 meses")

    # Informações adicionais
    valor_limite_credito: Optional[float] = Field(None, ge=0, description="Valor total do limite de crédito disponível")
    valor_utilizado_credito: Optional[float] = Field(None, ge=0, description="Valor utilizado do limite de crédito")

    # Validações complexas
    @validator('valor_utilizado_credito')
    def validar_credito_utilizado(cls, v, values):
        limite = values.get('valor_limite_credito')
        if limite is not None and v is not None and v > limite:
            raise ValueError('Valor utilizado de crédito não pode ser maior que o limite')
        return v

    @root_validator
    def calcular_campos_derivados(cls, values):
        # Adicionar campos derivados que podem ser úteis para o modelo
        if 'valor_emprestimo' in values and 'renda_mensal' in values and values['renda_mensal'] > 0:
            values['relacao_emprestimo_renda'] = values['valor_emprestimo'] / values['renda_mensal']

        if 'divida_atual_total' in values and 'renda_mensal' in values and values['renda_mensal'] > 0:
            values['relacao_divida_renda'] = values['divida_atual_total'] / values['renda_mensal']

        if 'valor_limite_credito' in values and values[
            'valor_limite_credito'] and 'valor_utilizado_credito' in values and values['valor_utilizado_credito']:
            values['utilizacao_credito'] = values['valor_utilizado_credito'] / values['valor_limite_credito']

        # Calcular score de atrasos
        atrasos_fields = ['num_pagamentos_atrasados_30d', 'num_pagamentos_atrasados_60d',
                          'num_pagamentos_atrasados_90d']
        atrasos_score = 0

        for field in atrasos_fields:
            if field in values and values[field]:
                # Ponderar por severidade do atraso
                if field == 'num_pagamentos_atrasados_30d':
                    atrasos_score += values[field] * 1
                elif field == 'num_pagamentos_atrasados_60d':
                    atrasos_score += values[field] * 2
                elif field == 'num_pagamentos_atrasados_90d':
                    atrasos_score += values[field] * 3

        values['score_atrasos'] = atrasos_score

        return values


# Modelo para resposta de predição
class PredictionResponse(BaseModel):
    """Resposta da predição de inadimplência."""

    cliente_id: Optional[str] = Field(None, description="ID do cliente (se fornecido)")
    probabilidade: float = Field(..., description="Probabilidade de inadimplência")
    predicao: StatusInadimplencia = Field(..., description="Predição de inadimplência")
    threshold: float = Field(..., description="Threshold utilizado para a classificação")
    timestamp: datetime = Field(default_factory=datetime.now, description="Timestamp da predição")
    score_atrasos: Optional[float] = Field(None, description="Score calculado de atrasos")
    relacao_divida_renda: Optional[float] = Field(None, description="Relação dívida/renda")
    principais_fatores: Optional[List[Dict[str, Any]]] = Field(None,
                                                               description="Principais fatores que influenciaram a predição")


# Modelo para feedback de predição
class PredictionFeedback(BaseModel):
    """Feedback sobre uma predição para monitoramento."""

    cliente_id: str = Field(..., description="ID do cliente")
    prediction_id: str = Field(..., description="ID da predição")
    resultado_real: StatusInadimplencia = Field(..., description="Resultado real observado")
    data_resultado: datetime = Field(..., description="Data em que o resultado foi observado")
    comentarios: Optional[str] = Field(None, description="Comentários adicionais")


# Modelo para configuração do serviço
class ServiceConfig:
    """Configuração do serviço de predição."""

    def __init__(
            self,
            model_path: str = os.environ.get("MODEL_PATH", "models/deployed_model/model.joblib"),
            feature_builder_path: str = os.environ.get("FEATURE_BUILDER_PATH",
                                                       "models/deployed_model/feature_builder.joblib"),
            metadata_path: str = os.environ.get("METADATA_PATH", "models/deployed_model/metadata.json"),
            threshold: float = float(os.environ.get("THRESHOLD", "0.5")),
            monitoring_enabled: bool = os.environ.get("MONITORING_ENABLED", "False").lower() == "true",
            feedback_storage_path: str = os.environ.get("FEEDBACK_STORAGE_PATH", "data/feedback"),
            prediction_storage_path: str = os.environ.get("PREDICTION_STORAGE_PATH", "data/predictions"),
            monitoring_interval: int = int(os.environ.get("MONITORING_INTERVAL", "24")),  # Em horas
            reference_data_path: str = os.environ.get("REFERENCE_DATA_PATH", "data/reference/reference_data.csv"),
    ):
        self.model_path = os.path.join(project_root, model_path)
        self.feature_builder_path = os.path.join(project_root, feature_builder_path)
        self.metadata_path = os.path.join(project_root, metadata_path)
        self.threshold = threshold
        self.monitoring_enabled = monitoring_enabled
        self.feedback_storage_path = os.path.join(project_root, feedback_storage_path)
        self.prediction_storage_path = os.path.join(project_root, prediction_storage_path)
        self.monitoring_interval = monitoring_interval
        self.reference_data_path = os.path.join(project_root, reference_data_path)

        # Garantir que os diretórios existam
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        os.makedirs(self.feedback_storage_path, exist_ok=True)
        os.makedirs(self.prediction_storage_path, exist_ok=True)


# Classe principal do serviço de predição
class PredictionService:
    """Serviço de predição de inadimplência."""

    def __init__(self, config: ServiceConfig):
        """
        Inicializa o serviço de predição.

        Args:
            config: Configuração do serviço
        """
        self.config = config
        self.model = self._load_model()
        self.feature_builder = self._load_feature_builder()
        self.metadata = self._load_metadata()
        self.monitor = self._setup_monitoring() if config.monitoring_enabled else None
        self.last_monitoring_check = datetime.now()

    def _load_model(self) -> Any:
        """
        Carrega o modelo de classificação.

        Returns:
            Modelo carregado
        """
        if not os.path.exists(self.config.model_path):
            logger.warning(f"Arquivo de modelo não encontrado: {self.config.model_path}")
            raise FileNotFoundError(f"Modelo não encontrado: {self.config.model_path}")

        logger.info(f"Carregando modelo de: {self.config.model_path}")
        try:
            model = joblib.load(self.config.model_path)
            return model
        except Exception as e:
            logger.error(f"Erro ao carregar modelo: {str(e)}")
            raise RuntimeError(f"Erro ao carregar modelo: {str(e)}")

    def _load_feature_builder(self) -> Optional[Any]:
        """
        Carrega o feature builder, se disponível.

        Returns:
            Feature builder carregado ou None
        """
        if not self.config.feature_builder_path or not os.path.exists(self.config.feature_builder_path):
            logger.warning(f"Feature builder não encontrado: {self.config.feature_builder_path}")
            return None

        logger.info(f"Carregando feature builder de: {self.config.feature_builder_path}")
        try:
            feature_builder = joblib.load(self.config.feature_builder_path)
            return feature_builder
        except Exception as e:
            logger.warning(f"Erro ao carregar feature builder: {str(e)}")
            return None

    def _load_metadata(self) -> Dict:
        """
        Carrega metadados do modelo, se disponíveis.

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
                'default': self.config.threshold
            },
            'feature_importance': {}
        }

        if not self.config.metadata_path or not os.path.exists(self.config.metadata_path):
            logger.warning(f"Arquivo de metadados não encontrado: {self.config.metadata_path}")
            return default_metadata

        logger.info(f"Carregando metadados de: {self.config.metadata_path}")
        try:
            with open(self.config.metadata_path, 'r') as f:
                metadata = json.load(f)
            return metadata
        except Exception as e:
            logger.warning(f"Erro ao carregar metadados: {str(e)}")
            return default_metadata

    def _setup_monitoring(self) -> Optional[Any]:
        """
        Configura o monitor de modelo, se disponível.

        Returns:
            Monitor de modelo ou None
        """
        if not ModelMonitor:
            logger.warning("Módulo de monitoramento não disponível.")
            return None

        if not os.path.exists(self.config.reference_data_path):
            logger.warning(f"Dados de referência não encontrados: {self.config.reference_data_path}")
            return None

        logger.info("Configurando monitoramento de modelo...")
        try:
            monitor = ModelMonitor(
                model_path=self.config.model_path,
                reference_data_path=self.config.reference_data_path,
                feature_builder_path=self.config.feature_builder_path,
                model_metadata_path=self.config.metadata_path,
                threshold=self.config.threshold
            )
            return monitor
        except Exception as e:
            logger.warning(f"Erro ao configurar monitoramento: {str(e)}")
            return None

    def predict(self, client_data: ClienteData, cliente_id: Optional[str] = None) -> PredictionResponse:
        """
        Realiza predição de inadimplência para um cliente.

        Args:
            client_data: Dados do cliente
            cliente_id: ID opcional do cliente

        Returns:
            Resposta com a predição
        """
        try:
            # Converter dados do cliente para DataFrame
            df = pd.DataFrame([client_data.dict()])

            # Aplicar transformações se feature builder disponível
            X = df
            if self.feature_builder:
                try:
                    X = self.feature_builder.transform(df)
                except Exception as e:
                    logger.warning(f"Erro ao aplicar feature builder: {str(e)}. Usando dados originais.")

            # Fazer predição
            y_proba = self.model.predict_proba(X)[0, 1]
            prediction = StatusInadimplencia.INADIMPLENTE if y_proba >= self.config.threshold else StatusInadimplencia.ADIMPLENTE

            # Extrair principais fatores (se disponível)
            principais_fatores = self._extrair_principais_fatores(df, y_proba)

            # Criar resposta
            response = PredictionResponse(
                cliente_id=cliente_id,
                probabilidade=float(y_proba),
                predicao=prediction,
                threshold=self.config.threshold,
                timestamp=datetime.now(),
                score_atrasos=client_data.score_atrasos if hasattr(client_data, 'score_atrasos') else None,
                relacao_divida_renda=client_data.relacao_divida_renda if hasattr(client_data,
                                                                                 'relacao_divida_renda') else None,
                principais_fatores=principais_fatores
            )

            # Armazenar predição para monitoramento futuro
            self._store_prediction(response, client_data)

            return response

        except Exception as e:
            logger.error(f"Erro ao realizar predição: {str(e)}")
            raise RuntimeError(f"Erro ao realizar predição: {str(e)}")

    def _extrair_principais_fatores(self, df: pd.DataFrame, probabilidade: float) -> List[Dict[str, Any]]:
        """
        Extrai os principais fatores que influenciaram a predição.

        Args:
            df: DataFrame com os dados do cliente
            probabilidade: Probabilidade prevista

        Returns:
            Lista de fatores importantes
        """
        fatores = []

        # Verificar se temos informações de importância de features
        if 'feature_importance' in self.metadata and self.metadata['feature_importance']:
            # Mapear importâncias de features
            importances = self.metadata['feature_importance']

            # Extrair as top features disponíveis no DataFrame
            top_features = []
            for feature, importance in sorted(importances.items(), key=lambda x: x[1], reverse=True):
                if feature in df.columns:
                    top_features.append({
                        'feature': feature,
                        'importance': importance,
                        'value': df[feature].iloc[0]
                    })

                if len(top_features) >= 5:  # Limitar a top 5 features
                    break

            # Adicionar features à lista de fatores
            for feature_info in top_features:
                feature_name = feature_info['feature']
                feature_value = feature_info['value']

                # Tentar obter uma descrição mais amigável da feature
                feature_display = feature_name.replace('_', ' ').title()

                fatores.append({
                    'feature': feature_name,
                    'display_name': feature_display,
                    'value': feature_value,
                    'importance': feature_info['importance']
                })

        # Se não temos informações de importância, usar heurísticas para features comuns
        elif df is not None:
            # Lista de features críticas comuns para modelos de crédito
            critical_features = {
                'score_atrasos': 'Score de Atrasos',
                'relacao_divida_renda': 'Relação Dívida/Renda',
                'utilizacao_credito': 'Utilização de Crédito',
                'num_pagamentos_atrasados_90d': 'Pagamentos Atrasados 90+ dias',
                'renda_mensal': 'Renda Mensal',
                'idade': 'Idade'
            }

            for feature, display_name in critical_features.items():
                if feature in df.columns:
                    fatores.append({
                        'feature': feature,
                        'display_name': display_name,
                        'value': df[feature].iloc[0],
                        'importance': None  # Não temos informação real de importância
                    })

            # Limitar a top 5 features
            fatores = fatores[:5]

        return fatores

    def _store_prediction(self, response: PredictionResponse, client_data: ClienteData) -> None:
        """
        Armazena os dados da predição para monitoramento futuro.

        Args:
            response: Resposta da predição
            client_data: Dados do cliente
        """
        try:
            # Gerar ID único para a predição se não fornecido
            prediction_id = response.cliente_id or f"pred_{datetime.now().strftime('%Y%m%d%H%M%S')}_{hash(str(client_data))}"

            # Combinar dados do cliente e resposta
            data = {
                "prediction_id": prediction_id,
                "cliente_id": response.cliente_id,
                "timestamp": response.timestamp.isoformat(),
                "probabilidade": response.probabilidade,
                "predicao": response.predicao,
                "threshold": response.threshold
            }

            # Adicionar dados do cliente
            data.update(client_data.dict())

            # Salvar como JSON
            filename = os.path.join(
                self.config.prediction_storage_path,
                f"prediction_{prediction_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}.json"
            )

            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)

            logger.debug(f"Predição armazenada em: {filename}")

        except Exception as e:
            logger.warning(f"Erro ao armazenar predição: {str(e)}")

    def store_feedback(self, feedback: PredictionFeedback) -> bool:
        """
        Armazena feedback sobre uma predição.

        Args:
            feedback: Dados de feedback

        Returns:
            True se o feedback foi armazenado com sucesso
        """
        try:
            # Salvar como JSON
            filename = os.path.join(
                self.config.feedback_storage_path,
                f"feedback_{feedback.prediction_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}.json"
            )

            with open(filename, 'w') as f:
                json.dump(feedback.dict(), f, indent=2)

            logger.info(f"Feedback armazenado em: {filename}")

            return True

        except Exception as e:
            logger.error(f"Erro ao armazenar feedback: {str(e)}")
            return False

    def check_model_health(self) -> Dict[str, Any]:
        """
        Verifica a saúde do modelo com base nos dados recentes.

        Returns:
            Relatório de saúde do modelo
        """
        if not self.monitor:
            return {"status": "Monitoramento não configurado"}

        # Verificar se passamos do intervalo de monitoramento
        now = datetime.now()
        hours_since_last_check = (now - self.last_monitoring_check).total_seconds() / 3600

        if hours_since_last_check < self.config.monitoring_interval:
            return {
                "status": "Monitoramento em espera",
                "last_check": self.last_monitoring_check.isoformat(),
                "next_check": (self.last_monitoring_check +
                               pd.Timedelta(hours=self.config.monitoring_interval)).isoformat()
            }

        try:
            # Carregar predições recentes
            df_predictions = self._load_recent_predictions_with_feedback()

            if df_predictions.empty:
                return {"status": "Sem dados suficientes para monitoramento"}

            # Monitorar saúde do modelo
            health_metrics = self.monitor.monitor_model_health(df_predictions)

            # Atualizar timestamp de última verificação
            self.last_monitoring_check = now

            # Retornar métricas
            return {
                "status": health_metrics.health_status,
                "timestamp": health_metrics.timestamp.isoformat(),
                "stability_score": health_metrics.stability_score,
                "data_drift_score": health_metrics.data_drift_score,
                "target_drift_score": health_metrics.target_drift_score,
                "prediction_drift_score": health_metrics.prediction_drift_score,
                "auc": health_metrics.auc,
                "precision": health_metrics.precision,
                "recall": health_metrics.recall,
                "f1": health_metrics.f1,
                "drifted_features": health_metrics.data_drift_features
            }

        except Exception as e:
            logger.error(f"Erro ao verificar saúde do modelo: {str(e)}")
            return {"status": "Erro", "message": str(e)}

    def _load_recent_predictions_with_feedback(self) -> pd.DataFrame:
        """
        Carrega predições recentes que possuem feedback.

        Returns:
            DataFrame com predições e feedback
        """
        # Carregar todos os arquivos de feedback
        feedback_files = [f for f in os.listdir(self.config.feedback_storage_path)
                          if f.startswith('feedback_') and f.endswith('.json')]

        if not feedback_files:
            logger.warning("Nenhum feedback encontrado.")
            return pd.DataFrame()

        # Carregar dados de feedback
        feedback_data = []
        for file in feedback_files:
            try:
                with open(os.path.join(self.config.feedback_storage_path, file), 'r') as f:
                    data = json.load(f)
                    feedback_data.append(data)
            except Exception as e:
                logger.warning(f"Erro ao carregar arquivo de feedback {file}: {str(e)}")

        # Criar DataFrame de feedback
        df_feedback = pd.DataFrame(feedback_data)

        if df_feedback.empty:
            logger.warning("Nenhum feedback válido encontrado.")
            return pd.DataFrame()

        # Carregar predições relacionadas
        prediction_files = [f for f in os.listdir(self.config.prediction_storage_path)
                            if f.startswith('prediction_') and f.endswith('.json')]

        prediction_data = []
        for file in prediction_files:
            try:
                with open(os.path.join(self.config.prediction_storage_path, file), 'r') as f:
                    data = json.load(f)
                    prediction_data.append(data)
            except Exception as e:
                logger.warning(f"Erro ao carregar arquivo de predição {file}: {str(e)}")

        # Criar DataFrame de predições
        df_predictions = pd.DataFrame(prediction_data)

        if df_predictions.empty:
            logger.warning("Nenhuma predição válida encontrada.")
            return pd.DataFrame()

        # Mesclar predições com feedback
        df = pd.merge(
            df_predictions,
            df_feedback[['prediction_id', 'resultado_real']],
            on='prediction_id',
            how='inner'
        )

        if df.empty:
            logger.warning("Nenhum match entre predições e feedback.")
            return pd.DataFrame()

        # Converter 'resultado_real' para numérico (0/1)
        df['inadimplente'] = df['resultado_real'].apply(
            lambda x: 1 if x == StatusInadimplencia.INADIMPLENTE else 0
        )

        return df


# Instanciar API
app = FastAPI(
    title="API de Predição de Inadimplência",
    description="API para predição de inadimplência de clientes",
    version="1.0.0"
)

# Adicionar middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permite qualquer origem em ambiente de desenvolvimento
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inicializar serviço de predição
service_config = ServiceConfig()
prediction_service = PredictionService(service_config)


# Função de dependência para obter o serviço
def get_prediction_service():
    """Retorna o serviço de predição."""
    return prediction_service


# Endpoints
@app.get("/", include_in_schema=False)
async def root():
    """Redireciona para a documentação da API."""
    return {"message": "API de Predição de Inadimplência", "docs": "/docs"}


@app.post("/predict", response_model=PredictionResponse, status_code=status.HTTP_200_OK)
async def predict(
        client: ClienteData,
        cliente_id: Optional[str] = Query(None, description="ID do cliente (opcional)"),
        service: PredictionService = Depends(get_prediction_service)
):
    """
    Prediz a probabilidade de inadimplência para um cliente.

    - **client**: Dados do cliente
    - **cliente_id**: ID opcional do cliente

    Retorna a probabilidade de inadimplência e a classificação.
    """
    try:
        response = service.predict(client, cliente_id)
        return response
    except Exception as e:
        logger.error(f"Erro ao processar predição: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/feedback", status_code=status.HTTP_201_CREATED)
async def submit_feedback(
        feedback: PredictionFeedback,
        service: PredictionService = Depends(get_prediction_service)
):
    """
    Envia feedback sobre uma predição para melhorar o monitoramento.

    - **feedback**: Dados do feedback, incluindo o resultado real observado

    Retorna confirmação de recebimento do feedback.
    """
    try:
        success = service.store_feedback(feedback)
        if success:
            return {"message": "Feedback recebido com sucesso", "status": "success"}
        else:
            raise HTTPException(status_code=500, detail="Erro ao processar feedback")
    except Exception as e:
        logger.error(f"Erro ao processar feedback: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check(
        background_tasks: BackgroundTasks,
        service: PredictionService = Depends(get_prediction_service)
):
    """
    Verifica a saúde da API e do modelo.

    Retorna status da API e estatísticas básicas do modelo.
    """
    api_status = {
        "status": "online",
        "timestamp": datetime.now().isoformat(),
        "model": service.metadata.get('model_name', 'unknown'),
        "model_version": service.metadata.get('model_version', 'unknown'),
        "model_creation_date": service.metadata.get('creation_date', 'unknown')
    }

    # Adicionar informações de monitoramento em background para não bloquear a resposta
    if service.config.monitoring_enabled:
        background_tasks.add_task(service.check_model_health)
        api_status["monitoring"] = {
            "enabled": True,
            "last_check": service.last_monitoring_check.isoformat()
        }
    else:
        api_status["monitoring"] = {"enabled": False}

    return api_status


@app.get("/monitor", status_code=status.HTTP_200_OK)
async def model_health(
        service: PredictionService = Depends(get_prediction_service)
):
    """
    Verifica a saúde e desempenho do modelo.

    Retorna métricas detalhadas sobre drift e degradação de performance.
    """
    if not service.config.monitoring_enabled:
        return {"status": "Monitoramento desabilitado"}

    health_report = service.check_model_health()
    return health_report


@app.get("/model-info", status_code=status.HTTP_200_OK)
async def model_info(
        service: PredictionService = Depends(get_prediction_service)
):
    """
    Retorna informações sobre o modelo em uso.

    Inclui nome, tipo, data de criação, features e métricas.
    """
    # Filtrar informações sensíveis
    safe_metadata = {
        "model_name": service.metadata.get('model_name', 'unknown'),
        "model_type": service.metadata.get('model_type', 'unknown'),
        "creation_date": service.metadata.get('creation_date', 'unknown'),
        "metrics": service.metadata.get('metrics', {}),
        "threshold": service.config.threshold
    }

    # Adicionar top features por importância se disponível
    if 'feature_importance' in service.metadata and service.metadata['feature_importance']:
        # Ordenar por importância e pegar top 10
        feature_importance = service.metadata['feature_importance']
        top_features = dict(sorted(
            feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10])

        safe_metadata["top_features"] = top_features

    return safe_metadata


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handler para exceções não tratadas."""
    logger.error(f"Erro não tratado: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"message": "Erro interno do servidor", "detail": str(exc)}
    )


# Iniciar a aplicação quando executada diretamente
if __name__ == "__main__":
    import uvicorn

    # Carregar variáveis de ambiente, se existirem
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    # Iniciar servidor com Uvicorn
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "127.0.0.1")

    logger.info(f"Iniciando API de Predição de Inadimplência em http://{host}:{port}")
    uvicorn.run("prediction_service:app", host=host, port=port, reload=True)