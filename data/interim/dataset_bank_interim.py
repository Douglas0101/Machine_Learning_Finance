import pandas as pd
import logging
import os
from sklearn.preprocessing import LabelEncoder
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)

FERIADOS_BRASIL = {
    "01-01",  # Confraternização Universal
    "04-21",  # Tiradentes
    "05-01",  # Dia do Trabalhador
    "09-07",  # Independência do Brasil
    "10-12",  # Nossa Senhora Aparecida
    "11-02",  # Finados
    "11-15",  # Proclamação da República
    "12-25"  # Natal
}


# ==============================================================
# FUNÇÕES AUXILIARES
# ==============================================================

def ler_dataset(caminho_csv: str) -> pd.DataFrame:
    """
    Lê o dataset financeiro sintético em CSV e converte data_transacao para datetime.
    """
    try:
        df = pd.read_csv(caminho_csv, parse_dates=['data_transacao'])
        logging.info(f"Dataset lido com {len(df)} linhas e {len(df.columns)} colunas.")

        # Converter data_transacao para datetime se necessário
        if df['data_transacao'].dtype != 'datetime64[ns]':
            df['data_transacao'] = pd.to_datetime(df['data_transacao'], errors='coerce')
        return df
    except Exception as e:
        logging.error(f"Erro ao ler o arquivo {caminho_csv}: {e}")
        raise


def tratar_missing_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Imputa dados faltantes em 'categoria_merchant' e 'localizacao_transacao'
    de forma condicional, criando colunas booleanas para indicar imputação.
    """
    logging.info("Iniciando tratamento de missing data avançado.")
    df['categoria_merchant_imputada'] = False
    df['localizacao_transacao_imputada'] = False

    # Imputação de categoria_merchant
    mode_by_merchant = (
        df.dropna(subset=['merchant', 'categoria_merchant'])
        .groupby('merchant')['categoria_merchant']
        .agg(lambda x: x.value_counts().index[0])
        .to_dict()
    )
    global_mode_categoria = df['categoria_merchant'].dropna().mode()
    global_mode_categoria = global_mode_categoria[0] if len(global_mode_categoria) > 0 else 'outros'

    def imputar_categoria(row):
        if pd.isnull(row['categoria_merchant']):
            cat_ = mode_by_merchant.get(row['merchant'], None)
            return (cat_ if cat_ is not None else global_mode_categoria, True)
        return (row['categoria_merchant'], False)

    df['categoria_merchant'], df['categoria_merchant_imputada'] = zip(*df.apply(imputar_categoria, axis=1))

    # Imputação de localizacao_transacao
    mode_by_user_loc = (
        df.dropna(subset=['id_usuario', 'localizacao_transacao'])
        .groupby('id_usuario')['localizacao_transacao']
        .agg(lambda x: x.value_counts().index[0])
        .to_dict()
    )
    global_mode_local = df['localizacao_transacao'].dropna().mode()
    global_mode_local = global_mode_local[0] if len(global_mode_local) > 0 else 'São Paulo, Brazil'

    def imputar_local(row):
        if pd.isnull(row['localizacao_transacao']):
            loc_ = mode_by_user_loc.get(row['id_usuario'], None)
            return (loc_ if loc_ is not None else global_mode_local, True)
        return (row['localizacao_transacao'], False)

    df['localizacao_transacao'], df['localizacao_transacao_imputada'] = zip(*df.apply(imputar_local, axis=1))
    logging.info("Tratamento de missing data concluído.")
    return df

# ==============================================================
# FUNÇÕES DE ENGENHARIA TEMPORAL (OTIMIZADAS)
# ==============================================================

def engenharia_features_temporais_otimizadas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula rolling windows (7D, 30D, 90D) para média e std do valor_transacao,
    bem como contagens de cada tipo_transacao, utilizando operações de transformação
    em grupo (groupby + transform) para evitar múltiplos merges.
    """
    logging.info("Iniciando engenharia de features temporais otimizada.")
    df = df.sort_values(['id_usuario', 'data_transacao']).copy()

    # Define data_transacao como índice para operações rolling baseadas em tempo
    df.set_index('data_transacao', inplace=True)

    # Cálculo de média e desvio padrão para cada janela
    for w in ['7D', '30D', '90D']:
        df[f'media_valor_transacao_{w}'] = (
            df.groupby('id_usuario')['valor_transacao']
            .transform(lambda x: x.rolling(w).mean())
        )
        df[f'std_valor_transacao_{w}'] = (
            df.groupby('id_usuario')['valor_transacao']
            .transform(lambda x: x.rolling(w).std())
        )

    # Cálculo de contagens para cada tipo_transacao usando uma Series temporária
    tipos_unicos = df['tipo_transacao'].unique()
    for t in tipos_unicos:
        # Cria uma Series binária para o tipo t
        temp_series = (df['tipo_transacao'] == t).astype(int)
        for w in ['7D', '30D', '90D']:
            col_name = f'contagem_{t}_{w}'
            df[col_name] = (
                temp_series.groupby(df['id_usuario'])
                .transform(lambda x: x.rolling(w).sum())
            )

    # Restaura o índice
    df.reset_index(inplace=True)
    logging.info("Engenharia de features temporais otimizada concluída.")
    return df

def criar_features_sazonalidade(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extrai dia da semana numérico, mês, flags de fim de semana e feriado.
    """
    logging.info("Criando features de sazonalidade.")
    df['dia_semana_transacao_num'] = df['data_transacao'].dt.weekday  # segunda=0, domingo=6
    df['mes_transacao'] = df['data_transacao'].dt.month
    df['fim_de_semana'] = df['dia_semana_transacao_num'].isin([5, 6])
    df['feriado'] = df['data_transacao'].dt.strftime('%m-%d').isin(FERIADOS_BRASIL)
    return df

def segmentar_usuarios_risco(df: pd.DataFrame) -> pd.DataFrame:
    """
    Segmenta usuários em categorias de risco com base em:
      - Soma de flag_fraude nos últimos 90 dias
      - Média de score_fraude nos últimos 30 dias
      - Proporção de transações de alto valor nos últimos 90 dias
    Utiliza transformações para calcular as métricas rolling e, em seguida, aplica regras de negócio.
    """
    logging.info("Segmentando usuários em categorias de risco.")
    df = df.sort_values(['id_usuario', 'data_transacao']).copy()

    # Definir data_transacao como índice para operações rolling
    df.set_index('data_transacao', inplace=True)

    # Rolling 90D para flag_fraude (soma)
    df['rolling_flag_fraude_90d'] = (
        df.groupby('id_usuario')['flag_fraude']
        .transform(lambda x: x.rolling('90D').sum())
    )

    # Rolling 30D para score_fraude (média)
    df['rolling_score_fraude_30d'] = (
        df.groupby('id_usuario')['score_fraude']
        .transform(lambda x: x.rolling('30D').mean())
    )

    # Percentil 90 de valor_transacao por tipo_conta
    percentil_90 = df.groupby('tipo_conta')['valor_transacao'].transform(lambda x: x.quantile(0.90))
    df['limite_alto_valor'] = percentil_90
    df['is_alto_valor'] = (df['valor_transacao'] > df['limite_alto_valor']).astype(int)

    # Rolling 90D para proporção de transações de alto valor (média de is_alto_valor)
    df['rolling_alto_valor_90d'] = (
        df.groupby('id_usuario')['is_alto_valor']
        .transform(lambda x: x.rolling('90D').mean())
    )

    # Restaura o índice
    df.reset_index(inplace=True)

    # Definir segmento de risco utilizando np.select para evitar .apply (mais performático)
    conditions = [
        (df['rolling_flag_fraude_90d'] > 2) | (df['rolling_score_fraude_30d'] > 0.7) | (
                    df['rolling_alto_valor_90d'] > 0.5),
        (df['rolling_flag_fraude_90d'] == 0) & (df['rolling_score_fraude_30d'] < 0.3) & (
                    df['rolling_alto_valor_90d'] < 0.1)
    ]
    choices = ['alto', 'baixo']
    df['segmento_risco'] = np.select(conditions, choices, default='medio')

    logging.info("Segmentação de risco concluída.")
    return df


def transformar_localizacao_dispositivo(df: pd.DataFrame) -> pd.DataFrame:
    """
    - Extrai cidade e país de localizacao_transacao.
    - Cria flag transacao_internacional (True se país não for Brasil).
    - Aplica Label Encoding em dispositivo_transacao e canal_transacao,
      preservando as colunas originais.
    """
    logging.info("Transformando localização e codificando dispositivo/canal.")

    # Extração de cidade e país
    mask_local = df['localizacao_transacao'].notnull()
    splits = df.loc[mask_local, 'localizacao_transacao'].str.split(',', n=1, expand=True)
    df.loc[mask_local, 'cidade_transacao'] = splits[0].str.strip()
    df.loc[mask_local, 'pais_transacao'] = splits[1].str.strip()

    # Flag para transação internacional
    df['transacao_internacional'] = df['pais_transacao'].str.lower().apply(
        lambda x: True if pd.notnull(x) and x not in ['brazil', 'brasil'] else False
    )

    # Manter colunas originais
    df['dispositivo_transacao_original'] = df['dispositivo_transacao']
    df['canal_transacao_original'] = df['canal_transacao']

    # Preencher missing e aplicar Label Encoding
    df['dispositivo_transacao'] = df['dispositivo_transacao'].fillna('desconhecido')
    df['canal_transacao'] = df['canal_transacao'].fillna('desconhecido')

    enc_disp = LabelEncoder()
    enc_canal = LabelEncoder()
    df['dispositivo_transacao'] = enc_disp.fit_transform(df['dispositivo_transacao'])
    df['canal_transacao'] = enc_canal.fit_transform(df['canal_transacao'])

    logging.info("Transformações de localização e dispositivo/canal concluídas.")
    return df


def criar_flags_anomalia(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria flags de anomalia:
      - flag_anomalia_valor: valor_transacao maior que (média + 3*std) do respectivo tipo_conta.
      - flag_anomalia_horario: transação com horário fora de +/-4h do horário mais comum (moda) para cada (id_usuario, tipo_conta).
    """
    logging.info("Calculando flags de anomalia.")

    # Anomalia de valor: uso de transform para média e std por grupo
    df['media_conta'] = df.groupby('tipo_conta')['valor_transacao'].transform('mean')
    df['std_conta'] = df.groupby('tipo_conta')['valor_transacao'].transform('std')
    df['flag_anomalia_valor'] = ((df['valor_transacao'] > (df['media_conta'] + 3 * df['std_conta'])) &
                                 (df['std_conta'] > 0))

    # Anomalia de horário: calcular moda do horário para cada (id_usuario, tipo_conta)
    mode_horario = df.groupby(['id_usuario', 'tipo_conta'])['horario_transacao'] \
        .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan) \
        .reset_index() \
        .rename(columns={'horario_transacao': 'horario_moda'})

    df = pd.merge(df, mode_horario, on=['id_usuario', 'tipo_conta'], how='left')
    # Calcular a diferença cíclica em relação a 24h
    diff = abs(df['horario_transacao'] - df['horario_moda'])
    diff = diff.apply(lambda d: min(d, 24 - d))
    df['flag_anomalia_horario'] = diff > 4
    df.drop(columns=['horario_moda', 'media_conta', 'std_conta'], inplace=True)

    logging.info("Flags de anomalia criadas.")
    return df


# ==============================================================
# PIPELINE PRINCIPAL
# ==============================================================
def processar_dataset_financeiro(caminho_csv_entrada: str) -> pd.DataFrame:
    """
    Executa todo o pipeline de processamento intermediário do dataset financeiro sintético.

    Etapas:
      1) Ler dataset
      2) Tratar missing data
      3) Engenharia de features temporais otimizada
      4) Criar features de sazonalidade
      5) Segmentar usuários em categorias de risco
      6) Transformar localização, dispositivo e canal
      7) Criar flags de anomalia
    """
    logging.info("=== Início do Pipeline de Processamento Financeiro ===")

    df = ler_dataset(caminho_csv_entrada)
    df = tratar_missing_data(df)
    df = engenharia_features_temporais_otimizadas(df)
    df = criar_features_sazonalidade(df)
    df = segmentar_usuarios_risco(df)
    df = transformar_localizacao_dispositivo(df)
    df = criar_flags_anomalia(df)

    logging.info("=== Fim do Pipeline de Processamento. Dataset pronto para análise. ===")
    return df


# ==============================================================
# MAIN (EXEMPLO DE EXECUÇÃO)
# ==============================================================
if __name__ == "__main__":
    CAMINHO_INPUT = "../raw/dataset_financeiro_simulado.csv"

    if os.path.exists(CAMINHO_INPUT):
        try:
            df_final = processar_dataset_financeiro(CAMINHO_INPUT)
            df_final.to_csv("dataset_financeiro_intermediario.csv", index=False, encoding='utf-8')
            logging.info(
                f"Dataset intermediário salvo com {len(df_final)} linhas e {len(df_final.columns)} colunas."
            )
        except Exception as e:
            logging.error(f"Falha no pipeline: {e}")
    else:
        logging.error(f"Arquivo {CAMINHO_INPUT} não encontrado.")
