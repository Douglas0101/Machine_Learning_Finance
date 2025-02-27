# ------------------------------------------------------------------
# Algoritmo Avançado de Previsão de Risco de Inadimplência Bancária
# Objetivo: Classificar clientes bancários quanto ao risco de inadimplência
# Versão: 1.0 (Aprimorada)
# ------------------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, RandomizedSearchCV, \
    TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
import time
import warnings
import joblib
from datetime import datetime
from sklearn.metrics import (classification_report, confusion_matrix, roc_curve,
                            roc_auc_score, precision_recall_curve, average_precision_score,
                            precision_score, recall_score)


# Bibliotecas adicionais para as melhorias
try:
    import xgboost as xgb

    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("Biblioteca XGBoost não encontrada. Alguns modelos não estarão disponíveis.")

try:
    import lightgbm as lgb

    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False
    print("Biblioteca LightGBM não encontrada. Alguns modelos não estarão disponíveis.")

try:
    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.over_sampling import SMOTE, ADASYN
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.combine import SMOTEENN, SMOTETomek

    IMBALANCED_AVAILABLE = True
except ImportError:
    IMBALANCED_AVAILABLE = False
    print("Biblioteca imbalanced-learn não encontrada. Técnicas de balanceamento não estarão disponíveis.")

try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Biblioteca SHAP não encontrada. Explicabilidade avançada não estará disponível.")

warnings.filterwarnings('ignore')


# ------------------------------------------------------------------
# 1. Carregamento e exploração inicial dos dados
# ------------------------------------------------------------------

def carregar_dados(caminho_arquivo, caminho_dados_macro=None):
    """
    Carrega o dataset bancário e exibe informações básicas
    Também carrega dados macroeconômicos se o caminho for fornecido

    Args:
        caminho_arquivo: Caminho para o arquivo CSV do dataset
        caminho_dados_macro: Caminho opcional para arquivo CSV com dados macroeconômicos

    Returns:
        DataFrame contendo os dados bancários
    """
    print("Carregando dataset bancário...")
    inicio = time.time()

    try:
        df = pd.read_csv(caminho_arquivo)
        fim = time.time()

        print(f"Dataset carregado com sucesso em {fim - inicio:.2f} segundos")
        print(f"Dimensões do dataset: {df.shape[0]} linhas x {df.shape[1]} colunas")
        print("\nInformações dos tipos de dados:")
        print(df.dtypes.value_counts())
        print("\nPrimeiras 5 linhas do dataset:")
        print(df.head())

        # Verificar valores ausentes
        pct_ausentes = df.isnull().mean() * 100
        colunas_ausentes = pct_ausentes[pct_ausentes > 0].sort_values(ascending=False)

        if not colunas_ausentes.empty:
            print("\nColunas com valores ausentes:")
            print(colunas_ausentes)

        # MELHORIA: Carregar dados macroeconômicos se disponíveis
        if caminho_dados_macro:
            df = incorporar_dados_macroeconomicos(df, caminho_dados_macro)
            print(
                f"\nDados macroeconômicos incorporados. Novas dimensões: {df.shape[0]} linhas x {df.shape[1]} colunas")

        # MELHORIA: Detecção de colunas temporais e conversão para datetime
        converter_colunas_temporais(df)

        return df

    except Exception as e:
        print(f"Erro ao carregar o dataset: {e}")
        return None


def incorporar_dados_macroeconomicos(df, caminho_dados_macro):
    """
    Incorpora variáveis macroeconômicas ao dataset principal

    Args:
        df: DataFrame com dados bancários
        caminho_dados_macro: Caminho para arquivo com dados macroeconômicos

    Returns:
        DataFrame com dados macroeconômicos incorporados
    """
    try:
        # Carregar dados macroeconômicos
        print("\nCarregando dados macroeconômicos...")
        df_macro = pd.read_csv(caminho_dados_macro)

        # Converter coluna de data para datetime
        data_cols = [col for col in df_macro.columns if 'data' in col.lower() or 'date' in col.lower()]
        if data_cols:
            data_col = data_cols[0]
            df_macro[data_col] = pd.to_datetime(df_macro[data_col])

            # Criar coluna de mês/ano para junção
            df_macro['Mes_Ano'] = df_macro[data_col].dt.to_period('M')

            # Verificar se há alguma coluna de data no dataframe principal
            df_data_cols = [col for col in df.columns if 'data' in col.lower() or 'date' in col.lower()]

            if df_data_cols:
                # Converter para datetime e criar coluna de mês/ano
                df_data_col = df_data_cols[0]
                df[df_data_col] = pd.to_datetime(df[df_data_col])
                df['Mes_Ano'] = df[df_data_col].dt.to_period('M')

                # Juntar dataframes
                df = df.merge(df_macro.drop(columns=[data_col]), on='Mes_Ano', how='left')

                # Preencher valores ausentes com forward fill (usando valores do período anterior)
                for col in df_macro.columns:
                    if col != data_col and col != 'Mes_Ano' and col in df.columns:
                        df[col] = df[col].ffill()

                print(
                    f"Variáveis macroeconômicas adicionadas: {list(set(df_macro.columns) - set([data_col, 'Mes_Ano']))}")
            else:
                print("Não foi possível encontrar uma coluna de data no dataset principal para junção com dados macro.")

        return df

    except Exception as e:
        print(f"Erro ao incorporar dados macroeconômicos: {e}")
        return df


def converter_colunas_temporais(df):
    """
    Identifica e converte colunas temporais para datetime

    Args:
        df: DataFrame com os dados

    Returns:
        None (modifica o DataFrame in-place)
    """
    # Identificar possíveis colunas de data
    colunas_data = [col for col in df.columns if 'data' in col.lower() or 'date' in col.lower()]

    for col in colunas_data:
        try:
            df[col] = pd.to_datetime(df[col])
            print(f"Coluna {col} convertida para datetime")
        except:
            print(f"Não foi possível converter a coluna {col} para datetime")


# ------------------------------------------------------------------
# 2. Definição da variável alvo e preparação dos dados
# ------------------------------------------------------------------

def definir_variavel_alvo(df):
    """
    Define a variável alvo para o modelo de previsão de inadimplência

    Args:
        df: DataFrame com os dados bancários

    Returns:
        DataFrame com a variável alvo adicionada
    """
    print("\nDefinindo variável alvo para previsão de inadimplência...")

    # Criar variável alvo baseada em múltiplos fatores de risco
    # Nesta versão interim, usamos um critério simplificado

    # Opção 1: Usar status de empréstimo (cliente já inadimplente)
    if 'Status_Emprestimo' in df.columns:
        df['Inadimplente'] = df['Status_Emprestimo'].apply(
            lambda x: 1 if x == 'Inadimplente' else 0
        )

    # Opção 2: Combinação de indicadores (para aumentar o número de casos positivos)
    # Para clientes sem empréstimo, usar outros indicadores de risco
    else:
        df['Inadimplente'] = 0

    # Para quem não tem empréstimo, usar o score de risco + outros indicadores
    if 'Risco_Inadimplencia' in df.columns:
        # Alto risco + alto comprometimento de renda + atrasos
        mask_sem_emprestimo = df['Tem_Emprestimo_Ativo'] == 'Não'
        mask_alto_risco = df['Risco_Inadimplencia'] > 70

        if 'Percentual_Comprometimento_Renda' in df.columns:
            mask_comprometimento = df['Percentual_Comprometimento_Renda'] > 60
        else:
            mask_comprometimento = True

        if 'Atraso_Medio_Pagamentos_Dias' in df.columns:
            mask_atrasos = df['Atraso_Medio_Pagamentos_Dias'] > 10
        else:
            mask_atrasos = True

        # Combinar condições
        df.loc[mask_sem_emprestimo & mask_alto_risco &
               (mask_comprometimento | mask_atrasos), 'Inadimplente'] = 1

    # Verificar distribuição da variável alvo
    distribuicao = df['Inadimplente'].value_counts(normalize=True) * 100
    print("\nDistribuição da variável alvo (Inadimplente):")
    print(distribuicao)

    # Alerta se dados muito desbalanceados
    if distribuicao.min() < 10:
        print("\nAVISO: Classes muito desbalanceadas! Serão aplicadas técnicas de balanceamento.")

    return df


# ------------------------------------------------------------------
# 3. Análise exploratória e engenharia de features avançada
# ------------------------------------------------------------------

def analise_exploratoria(df, var_alvo='Inadimplente'):
    """
    Realiza análise exploratória básica dos dados

    Args:
        df: DataFrame com os dados bancários
        var_alvo: Nome da variável alvo (default: 'Inadimplente')

    Returns:
        DataFrame original (análises são exibidas, não retornadas)
    """
    print("\nRealizando análise exploratória dos dados...")

    # Estatísticas descritivas básicas das variáveis numéricas
    print("\nEstatísticas descritivas das variáveis numéricas principais:")
    colunas_numericas = df.select_dtypes(include=['int64', 'float64']).columns
    colunas_interesse = [col for col in colunas_numericas if col != var_alvo
                         and df[col].isnull().sum() / len(df) < 0.3]  # Menos de 30% valores nulos

    if len(colunas_interesse) > 10:
        colunas_interesse = colunas_interesse[:10]  # Limitar a 10 colunas para clareza

    print(df[colunas_interesse].describe())

    # Análise da correlação com a variável alvo
    if var_alvo in df.columns:
        print("\nCorrelação das variáveis numéricas com a variável alvo:")
        correlacoes = df[colunas_numericas].corr()[var_alvo].sort_values(ascending=False)
        print(correlacoes)

        # Visualizar distribuição da variável alvo
        plt.figure(figsize=(10, 6))
        counts = df[var_alvo].value_counts()
        plt.bar(counts.index.astype(str), counts.values)
        plt.title(f'Distribuição da Variável Alvo: {var_alvo}')
        plt.xlabel(var_alvo)
        plt.ylabel('Contagem')
        plt.xticks(counts.index.astype(str))
        for i, v in enumerate(counts.values):
            plt.text(i, v + 50, str(v), ha='center')
        plt.tight_layout()
        plt.savefig('distribuicao_variavel_alvo.png')
        print("\nGráfico de distribuição da variável alvo salvo como 'distribuicao_variavel_alvo.png'")

        # Analisar relação entre variáveis importantes e o alvo
        top_correlacoes = correlacoes.drop(var_alvo, errors='ignore').abs().nlargest(5).index

        for coluna in top_correlacoes:
            plt.figure(figsize=(10, 6))

            if df[coluna].nunique() > 10:  # Variável contínua
                for target_val in sorted(df[var_alvo].unique()):
                    subset = df[df[var_alvo] == target_val]
                    sns.kdeplot(subset[coluna].dropna(), label=f"{var_alvo}={target_val}")

                plt.title(f'Distribuição de {coluna} por {var_alvo}')
                plt.legend()
            else:  # Variável categórica ou discreta
                sns.countplot(x=coluna, hue=var_alvo, data=df)
                plt.title(f'Contagem de {coluna} por {var_alvo}')
                plt.xticks(rotation=45)

            plt.tight_layout()
            plt.savefig(f'relacao_{coluna}_vs_{var_alvo}.png')

        print(f"\nGráficos de relação entre as 5 variáveis mais correlacionadas e {var_alvo} foram salvos.")

        # MELHORIA: Análise multivariada usando pairplot
        try:
            top_vars = top_correlacoes[:3].tolist() + [var_alvo]
            plt.figure(figsize=(12, 10))
            sns.pairplot(df[top_vars], hue=var_alvo, corner=True)
            plt.tight_layout()
            plt.savefig('analise_multivariada.png')
            print("\nAnálise multivariada salva como 'analise_multivariada.png'")
        except Exception as e:
            print(f"Não foi possível gerar o gráfico de análise multivariada: {e}")

    # MELHORIA: Análise de outliers
    print("\nDetectando outliers em variáveis numéricas principais...")
    for col in colunas_interesse[:5]:  # Limitar a 5 colunas para não sobrecarregar
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers_count = df[(df[col] < lower_bound) | (df[col] > upper_bound)].shape[0]
        outliers_pct = outliers_count / df.shape[0] * 100

        print(f"Outliers em {col}: {outliers_count} ({outliers_pct:.2f}%)")

        if outliers_pct > 5:
            plt.figure(figsize=(10, 6))
            sns.boxplot(x=df[col])
            plt.title(f'Boxplot de {col} - {outliers_pct:.2f}% outliers')
            plt.tight_layout()
            plt.savefig(f'outliers_{col}.png')

    return df


def engenharia_features(df):
    """
    Realiza engenharia de features para melhorar o modelo

    Args:
        df: DataFrame com os dados bancários

    Returns:
        DataFrame com novas features adicionadas
    """
    print("\nRealizando engenharia de features avançada...")

    # Cópia para não modificar o original
    df_features = df.copy()

    # ---------- FEATURES BÁSICAS (ORIGINAL) ----------

    # 1. Razão entre saldo e renda
    if 'Saldo_Atual' in df.columns and 'Renda_Mensal' in df.columns:
        df_features['Razao_Saldo_Renda'] = df['Saldo_Atual'] / df['Renda_Mensal'].replace(0, 0.01)
        print("Feature criada: Razao_Saldo_Renda")

    # 2. Utilização de limite de crédito (cheque especial)
    if 'Saldo_Atual' in df.columns and 'Limite_Cheque_Especial' in df.columns:
        df_features['Utilizacao_Cheque_Especial'] = 0
        mask = (df['Limite_Cheque_Especial'] > 0) & (df['Saldo_Atual'] < 0)
        df_features.loc[mask, 'Utilizacao_Cheque_Especial'] = abs(df.loc[mask, 'Saldo_Atual']) / df.loc[
            mask, 'Limite_Cheque_Especial']
        print("Feature criada: Utilizacao_Cheque_Especial")

    # 3. Razão entre valor do empréstimo e renda (capacidade de pagamento)
    if all(col in df.columns for col in ['Tem_Emprestimo_Ativo', 'Valor_Emprestimo', 'Renda_Mensal']):
        df_features['Razao_Emprestimo_Renda'] = 0
        mask = df['Tem_Emprestimo_Ativo'] == 'Sim'
        df_features.loc[mask, 'Razao_Emprestimo_Renda'] = df.loc[mask, 'Valor_Emprestimo'] / (
                12 * df.loc[mask, 'Renda_Mensal'].replace(0, 0.01))
        print("Feature criada: Razao_Emprestimo_Renda")

    # 4. Idade da conta (em anos)
    if 'Tempo_Relacionamento_Anos' in df.columns:
        df_features['Faixa_Tempo_Relacionamento'] = pd.cut(
            df['Tempo_Relacionamento_Anos'],
            bins=[0, 1, 3, 5, 10, 100],
            labels=['<1 ano', '1-3 anos', '3-5 anos', '5-10 anos', '>10 anos']
        )
        print("Feature criada: Faixa_Tempo_Relacionamento")

    # 5. Faixa etária
    if 'Idade' in df.columns:
        df_features['Faixa_Etaria'] = pd.cut(
            df['Idade'],
            bins=[0, 25, 35, 45, 55, 65, 100],
            labels=['<25', '25-35', '35-45', '45-55', '55-65', '>65']
        )
        print("Feature criada: Faixa_Etaria")

    # 6. Indicador de múltiplos produtos
    produtos = ['Possui_Cartao_Credito', 'Possui_Seguro_Vida', 'Possui_Previdencia', 'Possui_Investimentos']
    produtos_existentes = [col for col in produtos if col in df.columns]

    if produtos_existentes:
        df_features['Num_Produtos'] = 0
        for produto in produtos_existentes:
            df_features['Num_Produtos'] += (df[produto] == 'Sim').astype(int)
        print("Feature criada: Num_Produtos")

    # 7. Combinação de risco (score de crédito + atraso + reclamações)
    features_risco = []
    if 'Score_Credito' in df.columns:
        df_features['Baixo_Score'] = (df['Score_Credito'] < 600).astype(int)
        features_risco.append('Baixo_Score')
        print("Feature criada: Baixo_Score")

    if 'Atraso_Medio_Pagamentos_Dias' in df.columns:
        df_features['Atraso_Frequente'] = (df['Atraso_Medio_Pagamentos_Dias'] > 5).astype(int)
        features_risco.append('Atraso_Frequente')
        print("Feature criada: Atraso_Frequente")

    if 'Numero_Reclamacoes_Ultimo_Ano' in df.columns:
        df_features['Tem_Reclamacoes'] = (df['Numero_Reclamacoes_Ultimo_Ano'] > 0).astype(int)
        features_risco.append('Tem_Reclamacoes')
        print("Feature criada: Tem_Reclamacoes")

    if len(features_risco) >= 2:
        df_features['Indicadores_Risco'] = df_features[features_risco].sum(axis=1)
        print("Feature criada: Indicadores_Risco")

    # ---------- FEATURES AVANÇADAS (MELHORIAS) ----------

    # 8. MELHORIA: Capacidade de pagamento ajustada
    if all(col in df.columns for col in ['Renda_Mensal', 'Valor_Parcela_Emprestimo']):
        df_features['Capacidade_Pagamento'] = 1 - (df['Valor_Parcela_Emprestimo'] / df['Renda_Mensal'].replace(0, 0.01))
        # Limitar a um intervalo razoável (0 a 1)
        df_features['Capacidade_Pagamento'] = df_features['Capacidade_Pagamento'].clip(0, 1)
        print("Feature criada: Capacidade_Pagamento")

    # 9. MELHORIA: Estabilidade financeira (baseada na variação de saldo)
    if 'Variacao_Saldo_Ultimos_3Meses' in df.columns:
        df_features['Estabilidade_Financeira'] = 1 / (1 + np.abs(df['Variacao_Saldo_Ultimos_3Meses']))
        print("Feature criada: Estabilidade_Financeira")

    # 10. MELHORIA: Índice de risco composto (usando pesos)
    if all(col in df.columns for col in ['Score_Credito', 'Atraso_Medio_Pagamentos_Dias']):
        # Normalizar score de crédito (assumindo entre 0-1000)
        score_norm = df['Score_Credito'] / 1000

        # Normalizar atrasos (assumindo que 30 dias é um valor alto)
        atrasos_norm = np.minimum(df['Atraso_Medio_Pagamentos_Dias'] / 30, 1)

        # Combinar com pesos (score_credito tem peso negativo: quanto maior, menor o risco)
        df_features['Indice_Risco_Composto'] = (1 - score_norm) * 0.7 + atrasos_norm * 0.3
        print("Feature criada: Indice_Risco_Composto")

    # 11. MELHORIA: Features de comportamento temporal (se houver histórico)
    for periodo in [3, 6, 12]:
        col_historico = f'Num_Atrasos_Ultimos_{periodo}Meses'
        if col_historico in df.columns:
            # Taxa de atrasos por período
            df_features[f'Taxa_Atrasos_{periodo}m'] = df[col_historico] / periodo
            print(f"Feature criada: Taxa_Atrasos_{periodo}m")

    # 12. MELHORIA: Interações entre variáveis importantes
    if 'Score_Credito' in df.columns and 'Percentual_Comprometimento_Renda' in df.columns:
        # Interação entre score de crédito e comprometimento da renda
        df_features['Score_X_Comprometimento'] = df['Score_Credito'] * (
                    100 - df['Percentual_Comprometimento_Renda']) / 100
        print("Feature criada: Score_X_Comprometimento")

    # 13. MELHORIA: Nível de atividade bancária
    if 'Num_Transacoes_Ultimos_30dias' in df.columns and 'Saldo_Atual' in df.columns:
        # Normalizar pelo saldo para capturar a "rotatividade" do dinheiro
        df_features['Atividade_Bancaria'] = df['Num_Transacoes_Ultimos_30dias'] / (np.abs(df['Saldo_Atual']) + 1)
        print("Feature criada: Atividade_Bancaria")

    # 14. MELHORIA: Transformações não-lineares de variáveis importantes
    if 'Score_Credito' in df.columns:
        # Transformações logarítmicas podem capturar relações não-lineares
        df_features['Log_Score_Credito'] = np.log1p(df['Score_Credito'])
        print("Feature criada: Log_Score_Credito")

    # 15. MELHORIA: Comportamento sazonal (se houver dados temporais)
    if 'Data_Ultima_Transacao' in df.columns:
        # Extrair mês da última transação
        df_features['Mes_Ultima_Transacao'] = pd.to_datetime(df['Data_Ultima_Transacao']).dt.month
        df_features['Trimestre_Ultima_Transacao'] = pd.to_datetime(df['Data_Ultima_Transacao']).dt.quarter
        print("Features criadas: Mes_Ultima_Transacao, Trimestre_Ultima_Transacao")

    # Exibir informações sobre as novas features
    novas_features = list(set(df_features.columns) - set(df.columns))
    print(f"\nTotal de {len(novas_features)} novas features criadas")

    return df_features


def engenharia_features_temporais(df):
    """
    Cria features baseadas em séries temporais, se houver dados transacionais

    Args:
        df: DataFrame com dados transacionais

    Returns:
        DataFrame com features temporais agregadas
    """
    print("\nVerificando possibilidade de criar features temporais...")

    # Verificar se existem colunas de transação com timestamps
    colunas_data = [col for col in df.columns if 'data' in col.lower() or 'date' in col.lower()]
    colunas_id = [col for col in df.columns if 'id_cliente' in col.lower() or 'cliente_id' in col.lower()]

    if not colunas_data or not colunas_id:
        print("Dados transacionais insuficientes para criar features temporais.")
        return df

    print("Criando features baseadas em séries temporais...")
    data_col = colunas_data[0]
    id_col = colunas_id[0]

    # Garantir que a coluna de data está em formato datetime
    df[data_col] = pd.to_datetime(df[data_col])

    # Ordenar por cliente e data
    df = df.sort_values([id_col, data_col])

    # Criar novas features baseadas em janelas temporais
    df_temporal = df.copy()

    # 1. Verificar se há colunas de saldo, valor de transação ou pagamentos
    colunas_valor = [col for col in df.columns if any(termo in col.lower()
                                                      for termo in ['saldo', 'valor', 'pagamento', 'transacao'])]

    if colunas_valor:
        valor_col = colunas_valor[0]

        # Agrupar por cliente e criar features temporais
        df_agg = df.groupby(id_col).agg({
            valor_col: ['mean', 'std', 'min', 'max',
                        lambda x: x.diff().fillna(0).mean(),  # Média de variação
                        lambda x: x.diff().fillna(0).std()],  # Volatilidade
            data_col: [
                lambda x: (x.max() - x.min()).days,  # Período de atividade em dias
                'count'  # Número de transações
            ]
        })

        # Renomear as colunas agregadas
        df_agg.columns = [
            f'Media_{valor_col}',
            f'Desvio_{valor_col}',
            f'Min_{valor_col}',
            f'Max_{valor_col}',
            f'Media_Variacao_{valor_col}',
            f'Volatilidade_{valor_col}',
            'Periodo_Atividade_Dias',
            'Num_Transacoes'
        ]

        # Calcular tendência (inclinação da linha de regressão)
        tendencia = {}
        volatilidade_rolling = {}
        sazonalidade = {}

        for cliente in df[id_col].unique():
            cliente_df = df[df[id_col] == cliente].sort_values(data_col)

            if len(cliente_df) >= 5:  # Verificar se há dados suficientes
                # Calcular tendência
                try:
                    x = np.arange(len(cliente_df))
                    y = cliente_df[valor_col].values
                    coef = np.polyfit(x, y, 1)
                    tendencia[cliente] = coef[0]
                except:
                    tendencia[cliente] = 0

                # Calcular volatilidade em janela móvel
                try:
                    rolling_std = cliente_df[valor_col].rolling(window=3, min_periods=1).std().mean()
                    volatilidade_rolling[cliente] = rolling_std
                except:
                    volatilidade_rolling[cliente] = 0

                # Verificar padrão sazonal (se tiver pelo menos 12 pontos de dados)
                if len(cliente_df) >= 12:
                    try:
                        # Verificar se há sazonalidade mensal
                        cliente_df['mes'] = cliente_df[data_col].dt.month
                        variacao_mensal = cliente_df.groupby('mes')[valor_col].std()
                        sazonalidade[cliente] = variacao_mensal.max() / (variacao_mensal.mean() + 0.001)
                    except:
                        sazonalidade[cliente] = 1
                else:
                    sazonalidade[cliente] = 1
            else:
                tendencia[cliente] = 0
                volatilidade_rolling[cliente] = 0
                sazonalidade[cliente] = 1

        # Adicionar as novas métricas ao dataframe agregado
        df_agg[f'Tendencia_{valor_col}'] = df_agg.index.map(lambda x: tendencia.get(x, 0))
        df_agg[f'Volatilidade_Rolling_{valor_col}'] = df_agg.index.map(lambda x: volatilidade_rolling.get(x, 0))
        df_agg[f'Indice_Sazonalidade_{valor_col}'] = df_agg.index.map(lambda x: sazonalidade.get(x, 1))

        # Juntar as features temporais com o dataset original
        # Vamos pegar apenas uma linha por cliente do dataset original
        df_unique = df.drop_duplicates(id_col, keep='last')
        df_temporal = df_unique.merge(df_agg, left_on=id_col, right_index=True)

        print(f"Features temporais criadas: {list(df_agg.columns)}")

    return df_temporal


# ------------------------------------------------------------------
# 4. Preparação para modelagem
# ------------------------------------------------------------------

def preparar_dados_modelagem(df, var_alvo='Inadimplente', test_size=0.25, random_state=42, validacao_temporal=False):
    """
    Prepara os dados para modelagem, separando features e alvo

    Args:
        df: DataFrame com os dados preparados
        var_alvo: Nome da variável alvo
        test_size: Proporção do conjunto de teste
        random_state: Semente para reprodutibilidade
        validacao_temporal: Se True, divide os dados respeitando a ordem temporal

    Returns:
        X_train, X_test, y_train, y_test, feature_names
    """
    print("\nPreparando dados para modelagem...")

    # Verificar se a variável alvo existe
    if var_alvo not in df.columns:
        print(f"ERRO: Variável alvo '{var_alvo}' não encontrada no DataFrame")
        return None, None, None, None, None

    # Separar variável alvo
    y = df[var_alvo]

    # Selecionar features (excluir variáveis não utilizáveis)
    colunas_excluir = [
        var_alvo, 'ID_Cliente', 'Nome_Completo', 'Data_Nascimento',
        'Data_Abertura_Conta', 'CEP', 'Status_Emprestimo', 'Mes_Ano'  # Adicionar Mes_Ano à lista de exclusão
    ]

    colunas_excluir = [col for col in colunas_excluir if col in df.columns]
    X = df.drop(columns=colunas_excluir)

    print(f"Conjunto de dados: {X.shape[0]} exemplos, {X.shape[1]} features")

    # Identificar tipos de colunas
    feature_names = X.columns.tolist()
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

    print(f"Features numéricas: {len(numeric_features)}")
    print(f"Features categóricas: {len(categorical_features)}")

    # Dividir em treino e teste
    if validacao_temporal and 'Data' in df.columns:
        # MELHORIA: Divisão temporal
        print("Usando divisão temporal para train/test split...")

        # Ordenar por data
        indices_ordenados = df.sort_values('Data').index

        # Calcular ponto de corte
        cutoff_idx = int(len(indices_ordenados) * (1 - test_size))

        # Dividir em treino e teste
        train_indices = indices_ordenados[:cutoff_idx]
        test_indices = indices_ordenados[cutoff_idx:]

        X_train, X_test = X.loc[train_indices], X.loc[test_indices]
        y_train, y_test = y.loc[train_indices], y.loc[test_indices]
    else:
        # Divisão aleatória estratificada (original)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

    print(f"Conjunto de treino: {X_train.shape[0]} exemplos")
    print(f"Conjunto de teste: {X_test.shape[0]} exemplos")

    # Verificar balanceamento da variável alvo em ambos os conjuntos
    print("\nDistribuição da variável alvo:")
    print(f"Treino: {y_train.value_counts().to_dict()}")
    print(f"Teste: {y_test.value_counts().to_dict()}")

    return X_train, X_test, y_train, y_test, feature_names


# ------------------------------------------------------------------
# 5. Construção do pipeline de modelagem
# ------------------------------------------------------------------

def criar_pipeline_modelagem(X_train, y_train=None, algoritmo='rf', balanceamento=None):
    """
    Cria um pipeline de pré-processamento e modelagem com diferentes algoritmos e
    opções de balanceamento

    Args:
        X_train: Conjunto de dados de treino
        y_train: Variável alvo (opcional, necessário para balanceamento)
        algoritmo: Tipo de algoritmo ('rf', 'xgb', 'lgb')
        balanceamento: Método de balanceamento (None, 'smote', 'adasyn', 'smoteenn', 'undersample')

    Returns:
        Pipeline de modelagem, preprocessador
    """
    print("\nCriando pipeline de modelagem...")
    print(f"Algoritmo selecionado: {algoritmo}")
    if balanceamento:
        print(f"Método de balanceamento: {balanceamento}")

    # Identificar tipos de colunas
    numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

    # Pipeline para features numéricas
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Pipeline para features categóricas
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Combinar os transformadores
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    # Selecionar algoritmo
    if algoritmo == 'rf':
        classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=10,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
    elif algoritmo == 'xgb' and XGB_AVAILABLE:
        # Configuração básica do XGBoost
        classifier = xgb.XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='binary:logistic',
            random_state=42,
            n_jobs=-1
        )

        # Ajustar scale_pos_weight para balanceamento (se tiver y_train)
        if y_train is not None:
            neg_pos_ratio = (y_train == 0).sum() / max(1, (y_train == 1).sum())
            classifier.set_params(scale_pos_weight=neg_pos_ratio)

    elif algoritmo == 'lgb' and LGBM_AVAILABLE:
        # Configuração básica do LightGBM
        classifier = lgb.LGBMClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='binary',
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
    else:
        print(f"Algoritmo {algoritmo} não disponível ou não encontrado. Usando Random Forest como fallback.")
        classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=10,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )

    # Criar pipeline com balanceamento (se solicitado e disponível)
    if balanceamento and IMBALANCED_AVAILABLE:
        if balanceamento == 'smote':
            sampler = SMOTE(random_state=42)
        elif balanceamento == 'adasyn':
            sampler = ADASYN(random_state=42)
        elif balanceamento == 'smoteenn':
            sampler = SMOTEENN(random_state=42)
        elif balanceamento == 'undersample':
            sampler = RandomUnderSampler(random_state=42)
        else:
            print(f"Método de balanceamento {balanceamento} não reconhecido. Prosseguindo sem balanceamento.")
            sampler = None

        if sampler:
            # Usar pipeline do imbalanced-learn com sampler
            pipeline = ImbPipeline(steps=[
                ('preprocessor', preprocessor),
                ('sampler', sampler),
                ('classifier', classifier)
            ])
        else:
            # Pipeline sem sampler
            pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('classifier', classifier)
            ])
    else:
        # Pipeline padrão sem balanceamento
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', classifier)
        ])

    print("Pipeline criado com sucesso!")
    return pipeline, preprocessor


# ------------------------------------------------------------------
# 6. Otimização de hiperparâmetros
# ------------------------------------------------------------------

def otimizar_hiperparametros(pipeline, X_train, y_train, algoritmo='rf', cv=5, n_iter=20):
    """
    Otimiza os hiperparâmetros do modelo usando busca randomizada

    Args:
        pipeline: Pipeline de modelagem
        X_train, y_train: Dados de treino
        algoritmo: Tipo de algoritmo ('rf', 'xgb', 'lgb')
        cv: Número de folds na validação cruzada
        n_iter: Número de iterações na busca randomizada

    Returns:
        Pipeline otimizado
    """
    print("\nOtimizando hiperparâmetros do modelo...")

    # Definir grid de hiperparâmetros para cada algoritmo
    if algoritmo == 'rf':
        param_distributions = {
            'classifier__n_estimators': [50, 100, 200, 300],
            'classifier__max_depth': [5, 10, 15, 20, None],
            'classifier__min_samples_split': [2, 5, 10, 15],
            'classifier__min_samples_leaf': [1, 2, 4, 8],
            'classifier__max_features': ['sqrt', 'log2', None]
        }
    elif algoritmo == 'xgb' and XGB_AVAILABLE:
        param_distributions = {
            'classifier__n_estimators': [50, 100, 200, 300],
            'classifier__learning_rate': [0.01, 0.05, 0.1, 0.2],
            'classifier__max_depth': [3, 5, 7, 9],
            'classifier__subsample': [0.6, 0.8, 1.0],
            'classifier__colsample_bytree': [0.6, 0.8, 1.0],
            'classifier__gamma': [0, 0.1, 0.2, 0.5],
            'classifier__min_child_weight': [1, 3, 5, 7]
        }
    elif algoritmo == 'lgb' and LGBM_AVAILABLE:
        param_distributions = {
            'classifier__n_estimators': [50, 100, 200, 300],
            'classifier__learning_rate': [0.01, 0.05, 0.1, 0.2],
            'classifier__max_depth': [3, 5, 7, 9, -1],
            'classifier__num_leaves': [15, 31, 63, 127],
            'classifier__subsample': [0.6, 0.8, 1.0],
            'classifier__colsample_bytree': [0.6, 0.8, 1.0],
            'classifier__min_child_samples': [10, 20, 30, 50]
        }
    else:
        # Fallback para Random Forest
        param_distributions = {
            'classifier__n_estimators': [50, 100, 200],
            'classifier__max_depth': [5, 10, 15, None],
            'classifier__min_samples_split': [2, 5, 10],
            'classifier__min_samples_leaf': [1, 2, 4]
        }

    # Criar validação cruzada estratificada
    cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    # Iniciar busca randomizada
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=cv_strategy,
        scoring='roc_auc',
        n_jobs=-1,
        random_state=42,
        verbose=1
    )

    print(f"Iniciando busca randomizada com {n_iter} iterações e {cv} folds...")
    search.fit(X_train, y_train)

    print("\nResultados da otimização de hiperparâmetros:")
    print(f"Melhor score (AUC-ROC): {search.best_score_:.4f}")
    print("Melhores hiperparâmetros:")
    for param, value in search.best_params_.items():
        print(f"  {param}: {value}")

    # Retornar o melhor modelo
    return search.best_estimator_


# ------------------------------------------------------------------
# 7. Treinamento e avaliação do modelo
# ------------------------------------------------------------------

def treinar_e_avaliar_modelo(pipeline, X_train, X_test, y_train, y_test, feature_names):
    """
    Treina e avalia o modelo de previsão de inadimplência

    Args:
        pipeline: Pipeline de modelagem
        X_train, X_test, y_train, y_test: Dados de treino e teste
        feature_names: Nomes das features

    Returns:
        Modelo treinado e métricas de avaliação
    """
    print("\nTreinando modelo...")
    inicio = time.time()

    # Treinar o modelo
    pipeline.fit(X_train, y_train)

    # Tempo de treinamento
    fim = time.time()
    print(f"Treinamento concluído em {fim - inicio:.2f} segundos")

    # Fazer previsões
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    # Avaliar modelo
    print("\nResultados da avaliação no conjunto de teste:")
    print("\nMatriz de Confusão:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    # Métricas de classificação
    print("\nRelatório de Classificação:")
    print(classification_report(y_test, y_pred))

    # AUC-ROC
    auc = roc_auc_score(y_test, y_prob)
    print(f"\nAUC-ROC: {auc:.4f}")

    # Average Precision (AP)
    ap = average_precision_score(y_test, y_prob)
    print(f"Average Precision Score: {ap:.4f}")

    # Validação cruzada (mais robusta para avaliar o modelo)
    print("\nRealizando validação cruzada...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
    print(f"AUC-ROC média (validação cruzada 5-fold): {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

    # Plotar matriz de confusão como heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Previsão')
    plt.ylabel('Valor Real')
    plt.title('Matriz de Confusão')
    plt.savefig('matriz_confusao.png')
    print("\nMatriz de confusão salva como 'matriz_confusao.png'")

    # Plotar curva ROC
    plt.figure(figsize=(10, 6))
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.plot(fpr, tpr, label=f'AUC = {auc:.4f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('Taxa de Falsos Positivos')
    plt.ylabel('Taxa de Verdadeiros Positivos')
    plt.title('Curva ROC - Previsão de Inadimplência')
    plt.legend()
    plt.savefig('curva_roc.png')
    print("Curva ROC salva como 'curva_roc.png'")

    # Curva de Precisão-Recall (importante para classes desbalanceadas)
    plt.figure(figsize=(10, 6))
    precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
    plt.plot(recall, precision, label=f'AP = {ap:.4f}')
    plt.xlabel('Recall')
    plt.ylabel('Precisão')
    plt.title('Curva Precisão-Recall - Previsão de Inadimplência')
    plt.legend()
    plt.savefig('curva_precisao_recall.png')
    print("Curva Precisão-Recall salva como 'curva_precisao_recall.png'")

    # MELHORIA: Análise de threshold ótimo
    plt.figure(figsize=(12, 6))

    # Calcular métricas para diferentes thresholds
    thresholds = np.linspace(0, 1, 100)
    f1_scores = []
    precision_scores = []
    recall_scores = []

    for threshold in thresholds:
        y_pred_t = (y_prob >= threshold).astype(int)
        precision_scores.append(np.mean(y_test[y_pred_t == 1] == 1))
        recall_scores.append(np.sum(y_pred_t[y_test == 1] == 1) / np.sum(y_test == 1))

        # Calcular F1 manualmente para evitar divisão por zero
        if np.sum(y_pred_t == 1) == 0 or np.sum(y_test == 1) == 0:
            f1_scores.append(0)
        else:
            prec = np.mean(y_test[y_pred_t == 1] == 1)
            rec = np.sum(y_pred_t[y_test == 1] == 1) / np.sum(y_test == 1)

            if prec + rec == 0:
                f1_scores.append(0)
            else:
                f1_scores.append(2 * prec * rec / (prec + rec))

    # Plotar métricas vs thresholds
    plt.plot(thresholds, precision_scores, label='Precisão')
    plt.plot(thresholds, recall_scores, label='Recall')
    plt.plot(thresholds, f1_scores, label='F1 Score')

    # Encontrar threshold ótimo para F1
    f1_array = np.array(f1_scores)
    optimal_idx = np.argmax(f1_array)
    optimal_threshold = thresholds[optimal_idx]

    plt.axvline(x=optimal_threshold, color='r', linestyle='--',
                label=f'Threshold Ótimo: {optimal_threshold:.2f}')

    plt.xlabel('Threshold')
    plt.ylabel('Valor da Métrica')
    plt.title('Métricas vs Threshold - Previsão de Inadimplência')
    plt.legend()
    plt.grid(True)
    plt.savefig('analise_threshold.png')
    print(f"Análise de threshold salva como 'analise_threshold.png'")
    print(f"Threshold ótimo para F1: {optimal_threshold:.4f}")

    return pipeline, optimal_threshold


# ------------------------------------------------------------------
# 8. Interpretação do modelo (features importantes)
# ------------------------------------------------------------------

def interpretar_modelo(pipeline, X_train, y_train, X_test, y_test, feature_names):
    """
    Interpreta o modelo, identificando as features mais importantes e usando SHAP
    para explicabilidade avançada

    Args:
        pipeline: Pipeline de modelagem treinado
        X_train, y_train: Dados de treino
        X_test, y_test: Dados de teste
        feature_names: Nomes das features originais

    Returns:
        DataFrame com a importância das features
    """
    print("\nInterpretando o modelo...")

    # Extrair o modelo do pipeline
    modelo = pipeline.named_steps['classifier']

    # Para Random Forest, podemos usar feature_importances_
    if hasattr(modelo, 'feature_importances_'):
        print("\nImportância das features (baseada no modelo):")

        # Obter nomes das features após transformação
        preprocessor = pipeline.named_steps['preprocessor']
        cat_features = preprocessor.transformers_[1][2]  # Features categóricas

        # Tentativa de obter nomes de features transformadas
        try:
            # Para OneHotEncoder, tente obter as categorias
            cat_encoder = preprocessor.transformers_[1][1].named_steps['onehot']
            cat_transformed = []

            for i, col in enumerate(cat_features):
                categories = cat_encoder.categories_[i]
                cat_transformed.extend([f"{col}_{cat}" for cat in categories])

            # Features numéricas mantêm os nomes originais
            num_features = preprocessor.transformers_[0][2]
            feature_names_transformed = list(num_features) + cat_transformed

            # Limitar o tamanho para corresponder às importâncias
            if len(feature_names_transformed) > len(modelo.feature_importances_):
                feature_names_transformed = feature_names_transformed[:len(modelo.feature_importances_)]
            elif len(feature_names_transformed) < len(modelo.feature_importances_):
                # Preencher com nomes genéricos se necessário
                feature_names_transformed.extend([f"feature_{i}" for i in range(
                    len(feature_names_transformed), len(modelo.feature_importances_))])

            # Criar DataFrame de importâncias
            importances = pd.DataFrame({
                'feature': feature_names_transformed,
                'importance': modelo.feature_importances_
            })
        except:
            # Fallback: usar índices numéricos
            importances = pd.DataFrame({
                'feature': [f"feature_{i}" for i in range(len(modelo.feature_importances_))],
                'importance': modelo.feature_importances_
            })

        # Ordenar por importância
        importances = importances.sort_values('importance', ascending=False)

        # Mostrar top 20 features
        print(importances.head(20))

        # Plotar importância das features
        plt.figure(figsize=(12, 8))
        sns.barplot(x='importance', y='feature', data=importances.head(20))
        plt.title('Top 20 Features Mais Importantes')
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        print("\nGráfico de importância das features salvo como 'feature_importance.png'")

        # MELHORIA: Análise de importância por grupos de features
        try:
            # Agrupar features por prefixo (antes do primeiro underscore)
            importances['grupo'] = importances['feature'].apply(
                lambda x: x.split('_')[0] if '_' in x else x)

            # Importância por grupo
            importancia_grupos = importances.groupby('grupo')['importance'].sum().sort_values(ascending=False)

            plt.figure(figsize=(12, 6))
            importancia_grupos.plot(kind='bar')
            plt.title('Importância por Grupo de Features')
            plt.xlabel('Grupo de Features')
            plt.ylabel('Importância Total')
            plt.tight_layout()
            plt.savefig('importancia_grupos.png')
            print("Importância por grupos de features salva como 'importancia_grupos.png'")
        except Exception as e:
            print(f"Não foi possível gerar gráfico de importância por grupos: {e}")

        # MELHORIA: Análise de permutação (mais robusta que feature_importances_)
        try:
            print("\nCalculando importância por permutação (mais robusta)...")
            perm_importance = permutation_importance(
                pipeline, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1
            )

            perm_importances = pd.DataFrame({
                'feature': feature_names_transformed[:len(perm_importance.importances_mean)],
                'importance': perm_importance.importances_mean,
                'std': perm_importance.importances_std
            }).sort_values('importance', ascending=False)

            print("Top 10 features por importância de permutação:")
            print(perm_importances.head(10))

            plt.figure(figsize=(12, 8))
            sns.barplot(x='importance', y='feature', data=perm_importances.head(20))
            plt.title('Top 20 Features - Importância por Permutação')
            plt.tight_layout()
            plt.savefig('importancia_permutacao.png')
            print("Importância por permutação salva como 'importancia_permutacao.png'")

        except Exception as e:
            print(f"Não foi possível calcular importância por permutação: {e}")

        # MELHORIA: Usando SHAP para explicabilidade avançada
        if SHAP_AVAILABLE:
            try:
                print("\nRealizando análise SHAP para explicabilidade aprofundada...")

                # Para TreeExplainer, precisamos do modelo puro (sem o pipeline)
                # Preparar uma amostra de X_test para explicação
                X_test_processed = pipeline.named_steps['preprocessor'].transform(X_test)

                # Limitar número de amostras para SHAP (pode ser lento)
                n_samples = min(100, X_test.shape[0])

                # Criar explainer e calcular valores SHAP
                explainer = shap.TreeExplainer(modelo)
                shap_values = explainer.shap_values(X_test_processed[:n_samples])

                # Para modelos de classificação, shap_values pode ser uma lista
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]  # Valores para classe positiva

                # Gráfico resumo dos valores SHAP
                plt.figure(figsize=(12, 10))
                shap.summary_plot(
                    shap_values,
                    X_test_processed[:n_samples],
                    feature_names=feature_names_transformed[:X_test_processed.shape[1]],
                    show=False
                )
                plt.tight_layout()
                plt.savefig('shap_summary.png')

                # Gráfico de dependência para as 3 features mais importantes
                top_features = perm_importances['feature'].head(3).tolist()
                for idx, feature in enumerate(top_features):
                    try:
                        if idx < X_test_processed.shape[1]:
                            plt.figure(figsize=(10, 6))
                            feature_idx = feature_names_transformed.index(feature)
                            shap.dependence_plot(
                                feature_idx,
                                shap_values,
                                X_test_processed[:n_samples],
                                feature_names=feature_names_transformed[:X_test_processed.shape[1]],
                                show=False
                            )
                            plt.tight_layout()
                            plt.savefig(f'shap_dependencia_{feature}.png')
                    except Exception as e:
                        print(f"Erro ao gerar gráfico de dependência para {feature}: {e}")

                print("Análise SHAP concluída. Gráficos salvos.")

            except Exception as e:
                print(f"Erro na análise SHAP: {e}")
                print("Continuando sem análise SHAP...")

        return importances

    else:
        print("Modelo não suporta importância de features diretamente. Usando importância por permutação...")

        # Usa importância por permutação
        # Mais lento, mas funciona para qualquer modelo
        perm_importance = permutation_importance(
            pipeline, X_train, y_train, n_repeats=10, random_state=42, n_jobs=-1
        )

        importances = pd.DataFrame({
            'feature': feature_names,
            'importance': perm_importance.importances_mean
        }).sort_values('importance', ascending=False)

        print(importances.head(20))

        # Plotar importância das features
        plt.figure(figsize=(12, 8))
        sns.barplot(x='importance', y='feature', data=importances.head(20))
        plt.title('Top 20 Features Mais Importantes (Importância por Permutação)')
        plt.tight_layout()
        plt.savefig('feature_importance_permutation.png')
        print("\nGráfico de importância das features salvo como 'feature_importance_permutation.png'")

        return importances


# ------------------------------------------------------------------
# 9. Análise de Estabilidade do Modelo
# ------------------------------------------------------------------

def analisar_estabilidade_modelo(pipeline, X, y, feature_names, n_splits=5):
    """
    Avalia a estabilidade do modelo ao longo do tempo e diferentes segmentos

    Args:
        pipeline: Pipeline de modelagem
        X: Features completas
        y: Variável alvo
        feature_names: Nomes das features
        n_splits: Número de divisões para validação temporal

    Returns:
        DataFrame com resultados de estabilidade
    """
    print("\nAnalisando estabilidade do modelo...")

    resultados = []

    # 1. Estabilidade ao longo do tempo (se houver coluna de data)
    colunas_data = [col for col in X.columns if 'data' in col.lower() or 'date' in col.lower()]

    if colunas_data:
        print("\nVerificando estabilidade temporal do modelo...")
        data_col = colunas_data[0]

        # Ordenar por data
        indices_ordenados = np.argsort(X[data_col])
        X_ordenado = X.iloc[indices_ordenados].reset_index(drop=True)
        y_ordenado = y.iloc[indices_ordenados].reset_index(drop=True)

        # Validação temporal
        tscv = TimeSeriesSplit(n_splits=n_splits)

        periodos = []
        aucs = []
        precisoes = []
        recalls = []

        fig, ax = plt.subplots(figsize=(12, 6))

        for i, (train_idx, test_idx) in enumerate(tscv.split(X_ordenado)):
            X_train_split, X_test_split = X_ordenado.iloc[train_idx], X_ordenado.iloc[test_idx]
            y_train_split, y_test_split = y_ordenado.iloc[train_idx], y_ordenado.iloc[test_idx]

            # Treinar modelo
            pipeline.fit(X_train_split, y_train_split)

            # Fazer previsões
            y_pred = pipeline.predict(X_test_split)
            y_prob = pipeline.predict_proba(X_test_split)[:, 1]

            # Calcular métricas
            auc = roc_auc_score(y_test_split, y_prob)
            precision = precision_score(y_test_split, y_pred)
            recall = recall_score(y_test_split, y_pred)

            # Adicionar resultados
            periodo = f"Período {i + 1}"
            periodos.append(periodo)
            aucs.append(auc)
            precisoes.append(precision)
            recalls.append(recall)

            # Adicionar ao DataFrame de resultados
            data_inicio = X_test_split[data_col].min()
            data_fim = X_test_split[data_col].max()

            resultados.append({
                'tipo_validacao': 'temporal',
                'periodo': periodo,
                'data_inicio': data_inicio,
                'data_fim': data_fim,
                'auc': auc,
                'precision': precision,
                'recall': recall,
                'amostras': len(y_test_split),
                'taxa_positivos': y_test_split.mean()
            })

        # Plotar estabilidade temporal
        ax.plot(periodos, aucs, 'o-', label='AUC-ROC')
        ax.plot(periodos, precisoes, 's-', label='Precisão')
        ax.plot(periodos, recalls, '^-', label='Recall')

        ax.set_xlabel('Período')
        ax.set_ylabel('Valor da Métrica')
        ax.set_title('Estabilidade do Modelo ao Longo do Tempo')
        ax.legend()
        ax.grid(True)

        plt.tight_layout()
        plt.savefig('estabilidade_temporal.png')
        print("Gráfico de estabilidade temporal salvo como 'estabilidade_temporal.png'")

    # 2. Estabilidade por segmentos (exemplo: faixa etária)
    segmentos = []

    # Verificar possíveis segmentos de interesse
    if 'Faixa_Etaria' in X.columns:
        segmentos.append('Faixa_Etaria')

    if 'Faixa_Renda' in X.columns:
        segmentos.append('Faixa_Renda')

    if 'Genero' in X.columns:
        segmentos.append('Genero')

    if 'Regiao' in X.columns:
        segmentos.append('Regiao')

    # Analisar estabilidade por segmentos
    if segmentos:
        print("\nVerificando estabilidade do modelo por segmentos...")

        for segmento in segmentos:
            print(f"\nAnalisando estabilidade por {segmento}...")

            # Grupos únicos no segmento
            grupos = X[segmento].unique()

            segmento_aucs = []
            segmento_precisoes = []
            segmento_recalls = []
            segmento_grupos = []

            for grupo in grupos:
                # Selecionar apenas dados deste grupo
                mask = X[segmento] == grupo
                X_grupo = X[mask]
                y_grupo = y[mask]

                if len(y_grupo) < 20 or y_grupo.nunique() < 2:
                    print(f"  Grupo {grupo} tem amostras insuficientes ou sem variação. Ignorando.")
                    continue

                try:
                    # Divisão treino/teste para este grupo
                    X_train_grupo, X_test_grupo, y_train_grupo, y_test_grupo = train_test_split(
                        X_grupo, y_grupo, test_size=0.3, random_state=42, stratify=y_grupo
                    )

                    # Treinar modelo neste grupo
                    pipeline.fit(X_train_grupo, y_train_grupo)

                    # Fazer previsões
                    y_pred = pipeline.predict(X_test_grupo)
                    y_prob = pipeline.predict_proba(X_test_grupo)[:, 1]

                    # Calcular métricas
                    auc = roc_auc_score(y_test_grupo, y_prob)
                    precision = precision_score(y_test_grupo, y_pred)
                    recall = recall_score(y_test_grupo, y_pred)

                    # Adicionar resultados
                    segmento_grupos.append(str(grupo))
                    segmento_aucs.append(auc)
                    segmento_precisoes.append(precision)
                    segmento_recalls.append(recall)

                    # Adicionar ao DataFrame de resultados
                    resultados.append({
                        'tipo_validacao': f'segmento_{segmento}',
                        'periodo': str(grupo),
                        'data_inicio': None,
                        'data_fim': None,
                        'auc': auc,
                        'precision': precision,
                        'recall': recall,
                        'amostras': len(y_test_grupo),
                        'taxa_positivos': y_test_grupo.mean()
                    })

                    print(f"  Grupo {grupo}: AUC={auc:.4f}, Precision={precision:.4f}, Recall={recall:.4f}")

                except Exception as e:
                    print(f"  Erro ao analisar grupo {grupo}: {e}")

            # Plotar estabilidade por segmento
            if segmento_grupos:
                plt.figure(figsize=(12, 6))
                indices = np.arange(len(segmento_grupos))
                width = 0.25

                plt.bar(indices - width, segmento_aucs, width, label='AUC-ROC')
                plt.bar(indices, segmento_precisoes, width, label='Precisão')
                plt.bar(indices + width, segmento_recalls, width, label='Recall')

                plt.xlabel(segmento)
                plt.ylabel('Valor da Métrica')
                plt.title(f'Estabilidade do Modelo por {segmento}')
                plt.xticks(indices, segmento_grupos, rotation=45)
                plt.legend()
                plt.grid(True, axis='y')
                plt.tight_layout()
                plt.savefig(f'estabilidade_{segmento}.png')
                print(f"Gráfico de estabilidade por {segmento} salvo como 'estabilidade_{segmento}.png'")

    # Criar DataFrame com resultados de estabilidade
    df_estabilidade = pd.DataFrame(resultados)

    if not df_estabilidade.empty:
        print("\nResumo da estabilidade do modelo:")
        print(df_estabilidade.groupby('tipo_validacao')[['auc', 'precision', 'recall']].agg(
            ['mean', 'std', 'min', 'max']))

        df_estabilidade.to_csv('resultados_estabilidade.csv', index=False)
        print("Resultados de estabilidade salvos como 'resultados_estabilidade.csv'")

    return df_estabilidade if not df_estabilidade.empty else None


# ------------------------------------------------------------------
# 10. Função para fazer previsões em novos dados
# ------------------------------------------------------------------

def prever_inadimplencia(modelo, novos_dados, threshold=0.5):
    """
    Realiza previsões de inadimplência em novos dados

    Args:
        modelo: Pipeline treinado
        novos_dados: DataFrame com novos dados para previsão
        threshold: Limiar de probabilidade para classificação (default: 0.5)

    Returns:
        DataFrame com dados originais e previsões
    """
    print("\nRealizando previsões em novos dados...")

    # Fazer uma cópia dos dados
    dados_previsao = novos_dados.copy()

    # Realizar previsões
    try:
        # Probabilidades
        probabilidades = modelo.predict_proba(novos_dados)[:, 1]

        # Classificações baseadas no threshold
        classificacoes = (probabilidades >= threshold).astype(int)

        # Adicionar resultados ao DataFrame
        dados_previsao['Probabilidade_Inadimplencia'] = probabilidades
        dados_previsao['Classificacao_Inadimplencia'] = classificacoes

        # Categorizar risco
        dados_previsao['Categoria_Risco'] = pd.cut(
            probabilidades,
            bins=[0, 0.25, 0.5, 0.75, 1.0],
            labels=['Baixo', 'Médio-Baixo', 'Médio-Alto', 'Alto']
        )

        # MELHORIA: Adicionar flag de confiança da previsão
        # Quanto mais próximo de 0.5, menor a confiança
        dados_previsao['Confianca_Previsao'] = 1 - 2 * np.abs(probabilidades - 0.5)

        # Categorizar confiança
        dados_previsao['Nivel_Confianca'] = pd.cut(
            dados_previsao['Confianca_Previsao'],
            bins=[0, 0.33, 0.66, 1.0],
            labels=['Alta', 'Média', 'Baixa']
        )

        print(f"\nResumo das previsões (threshold={threshold}):")
        print(f"- Total de clientes analisados: {len(dados_previsao)}")
        print(f"- Clientes classificados como inadimplentes: {classificacoes.sum()}")
        print(f"- Percentual de inadimplência previsto: {classificacoes.mean() * 100:.2f}%")

        # Distribuição das categorias de risco
        print("\nDistribuição das categorias de risco:")
        print(dados_previsao['Categoria_Risco'].value_counts(normalize=True).sort_index() * 100)

        # Distribuição dos níveis de confiança
        print("\nDistribuição dos níveis de confiança:")
        print(dados_previsao['Nivel_Confianca'].value_counts(normalize=True) * 100)

        # MELHORIA: Análise de risco por segmentos
        segmentos = []
        if 'Faixa_Etaria' in dados_previsao.columns:
            segmentos.append('Faixa_Etaria')
        if 'Faixa_Renda' in dados_previsao.columns:
            segmentos.append('Faixa_Renda')

        if segmentos:
            print("\nTaxa de inadimplência prevista por segmentos:")
            for segmento in segmentos:
                print(f"\nSegmento: {segmento}")
                taxa_por_segmento = dados_previsao.groupby(segmento)['Classificacao_Inadimplencia'].mean() * 100
                print(taxa_por_segmento.sort_values(ascending=False))

        return dados_previsao

    except Exception as e:
        print(f"Erro ao realizar previsões: {e}")
        return None


# ------------------------------------------------------------------
# 11. Função para gerar perfis de risco e relatório avançado
# ------------------------------------------------------------------

def gerar_perfis_risco(dados_previsao, top_n=5):
    """
    Gera perfis de risco baseados nas previsões e cria relatório avançado

    Args:
        dados_previsao: DataFrame com os dados e previsões
        top_n: Número de perfis de alto risco para mostrar

    Returns:
        DataFrames com perfis de alto e baixo risco
    """
    if 'Probabilidade_Inadimplencia' not in dados_previsao.columns:
        print("ERRO: Dados de previsão não contêm probabilidades de inadimplência")
        return None, None

    print("\nGerando perfis de risco e relatório avançado...")

    # Selecionar colunas relevantes para o perfil
    colunas_perfil = [
        'Idade', 'Estado_Civil', 'Nivel_Educacional', 'Profissao',
        'Renda_Mensal', 'Numero_Dependentes', 'Residencia_Propria',
        'Tipo_Conta', 'Tempo_Relacionamento_Anos', 'Saldo_Atual',
        'Score_Credito', 'Possui_Cartao_Credito', 'Perfil_Investidor',
        'Probabilidade_Inadimplencia', 'Categoria_Risco'
    ]

    # Filtrar apenas colunas existentes
    colunas_perfil = [col for col in colunas_perfil if col in dados_previsao.columns]

    # Ordenar por probabilidade de inadimplência
    dados_ordenados = dados_previsao.sort_values('Probabilidade_Inadimplencia', ascending=False)

    # Selecionar perfis de alto risco
    perfis_alto_risco = dados_ordenados.head(top_n)[colunas_perfil]

    # Selecionar perfis de baixo risco
    perfis_baixo_risco = dados_ordenados.tail(top_n)[colunas_perfil]

    print("\nPerfis de Alto Risco (Top 5):")
    print(perfis_alto_risco)

    print("\nPerfis de Baixo Risco (Top 5):")
    print(perfis_baixo_risco)

    # MELHORIA: Criar relatório gráfico avançado
    print("\nGerando relatório gráfico avançado...")

    # 1. Distribuição de probabilidades
    plt.figure(figsize=(10, 6))
    sns.histplot(dados_previsao['Probabilidade_Inadimplencia'], bins=20, kde=True)
    plt.title('Distribuição de Probabilidades de Inadimplência')
    plt.xlabel('Probabilidade')
    plt.ylabel('Contagem')
    plt.axvline(x=0.5, color='r', linestyle='--', label='Threshold Padrão (0.5)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('distribuicao_probabilidades.png')

    # 2. Heatmap de correlação entre variáveis e probabilidade
    # Selecionar variáveis numéricas relevantes
    vars_numericas = dados_previsao.select_dtypes(include=['int64', 'float64']).columns
    vars_para_correlacao = [col for col in vars_numericas if col not in [
        'Classificacao_Inadimplencia', 'Confianca_Previsao']]

    if len(vars_para_correlacao) > 3:
        plt.figure(figsize=(12, 10))
        correlacoes = dados_previsao[vars_para_correlacao].corr()
        mask = np.triu(np.ones_like(correlacoes, dtype=bool))
        sns.heatmap(correlacoes, mask=mask, annot=True, cmap='coolwarm', vmin=-1, vmax=1,
                    fmt='.2f', linewidths=0.5)
        plt.title('Correlação entre Variáveis e Probabilidade de Inadimplência')
        plt.tight_layout()
        plt.savefig('correlacao_variaveis.png')

    # 3. Boxplots de variáveis importantes por categoria de risco
    vars_importantes = ['Renda_Mensal', 'Score_Credito', 'Tempo_Relacionamento_Anos']
    vars_existentes = [col for col in vars_importantes if col in dados_previsao.columns]

    if vars_existentes:
        fig, axes = plt.subplots(len(vars_existentes), 1, figsize=(10, 4 * len(vars_existentes)))
        if len(vars_existentes) == 1:
            axes = [axes]

        for i, var in enumerate(vars_existentes):
            sns.boxplot(x='Categoria_Risco', y=var, data=dados_previsao, ax=axes[i],
                        order=['Baixo', 'Médio-Baixo', 'Médio-Alto', 'Alto'])
            axes[i].set_title(f'{var} por Categoria de Risco')
            axes[i].set_xlabel('Categoria de Risco')
            axes[i].set_ylabel(var)

        plt.tight_layout()
        plt.savefig('variaveis_por_risco.png')

    # 4. Gráfico de dispersão para as duas variáveis mais importantes
    if len(vars_existentes) >= 2:
        plt.figure(figsize=(10, 8))
        scatter = sns.scatterplot(
            x=vars_existentes[0],
            y=vars_existentes[1],
            hue='Categoria_Risco',
            size='Probabilidade_Inadimplencia',
            sizes=(20, 200),
            palette='viridis',
            data=dados_previsao
        )
        plt.title(f'Relação entre {vars_existentes[0]} e {vars_existentes[1]} por Categoria de Risco')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('dispersao_variaveis.png')

    # Salvar perfis em CSV
    perfis_alto_risco.to_csv('perfis_alto_risco.csv', index=False)
    perfis_baixo_risco.to_csv('perfis_baixo_risco.csv', index=False)

    print("Relatório gráfico avançado gerado com sucesso!")
    print("Perfis salvos em 'perfis_alto_risco.csv' e 'perfis_baixo_risco.csv'")

    return perfis_alto_risco, perfis_baixo_risco


# ------------------------------------------------------------------
# 12. Monitoramento de Drift e Qualidade do Modelo
# ------------------------------------------------------------------

def monitorar_drift(modelo, X_ref, X_atual, threshold=0.1):
    """
    Monitora drift entre conjunto de referência e atual

    Args:
        modelo: Pipeline de modelagem treinado
        X_ref: Conjunto de dados de referência
        X_atual: Conjunto de dados atual
        threshold: Limiar para alerta de drift

    Returns:
        Boolean indicando se há drift significativo
    """
    print("\nMonitorando drift entre conjuntos de dados...")

    # 1. Calcular estatísticas descritivas para ambos os conjuntos
    colunas_numericas = X_ref.select_dtypes(include=['int64', 'float64']).columns

    stats_ref = X_ref[colunas_numericas].describe()
    stats_atual = X_atual[colunas_numericas].describe()

    # 2. Calcular diferença percentual para média e desvio padrão
    diff_mean = np.abs((stats_atual.loc['mean'] - stats_ref.loc['mean']) / stats_ref.loc['mean'])
    diff_std = np.abs((stats_atual.loc['std'] - stats_ref.loc['std']) / stats_ref.loc['std'])

    # 3. Identificar variáveis com drift significativo
    vars_com_drift_mean = diff_mean[diff_mean > threshold].index.tolist()
    vars_com_drift_std = diff_std[diff_std > threshold].index.tolist()

    vars_com_drift = list(set(vars_com_drift_mean + vars_com_drift_std))

    # 4. Calcular probabilidades para ambos os conjuntos
    probs_ref = modelo.predict_proba(X_ref)[:, 1]
    probs_atual = modelo.predict_proba(X_atual)[:, 1]

    # 5. Comparar distribuição de probabilidades (teste KS)
    from scipy import stats
    ks_statistic, ks_pvalue = stats.ks_2samp(probs_ref, probs_atual)

    # 6. Gerar relatório de drift
    print("\nRelatório de Monitoramento de Drift:")
    print(f"Variáveis com drift significativo na média (threshold={threshold}):")
    for var in vars_com_drift_mean:
        print(f"  - {var}: {diff_mean[var]:.4f}")

    print(f"\nVariáveis com drift significativo no desvio padrão (threshold={threshold}):")
    for var in vars_com_drift_std:
        print(f"  - {var}: {diff_std[var]:.4f}")

    print(f"\nTeste KS para distribuições de probabilidade:")
    print(f"  - Estatística KS: {ks_statistic:.4f}")
    print(f"  - p-valor: {ks_pvalue:.4f}")

    # Determinar se há drift significativo
    drift_significativo = (len(vars_com_drift) > len(colunas_numericas) * 0.2) or (ks_pvalue < 0.05)

    if drift_significativo:
        print("\nALERTA: Drift significativo detectado! Considere retreinar o modelo.")
    else:
        print("\nSem drift significativo. O modelo continua válido.")

    # 7. Plotar comparativo de distribuições
    # Probabilidades
    plt.figure(figsize=(10, 6))
    sns.kdeplot(probs_ref, label='Referência', color='blue')
    sns.kdeplot(probs_atual, label='Atual', color='red')
    plt.title('Comparação de Distribuições de Probabilidade')
    plt.xlabel('Probabilidade de Inadimplência')
    plt.ylabel('Densidade')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('drift_probabilidades.png')

    # Top variáveis com drift
    if vars_com_drift:
        fig, axes = plt.subplots(min(len(vars_com_drift), 3), 1, figsize=(10, 10))
        if len(vars_com_drift) == 1:
            axes = [axes]

        for i, var in enumerate(vars_com_drift[:3]):
            sns.kdeplot(X_ref[var], label='Referência', color='blue', ax=axes[i])
            sns.kdeplot(X_atual[var], label='Atual', color='red', ax=axes[i])
            axes[i].set_title(f'Drift em {var}')
            axes[i].set_xlabel(var)
            axes[i].set_ylabel('Densidade')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('drift_variaveis.png')

    print("Gráficos de monitoramento de drift gerados.")

    return drift_significativo


# ------------------------------------------------------------------
# 13. Função principal para executar o algoritmo completo
# ------------------------------------------------------------------

def executar_algoritmo_completo(caminho_arquivo, caminho_dados_macro=None, algoritmo='rf',
                                balanceamento='smote', otimizar=True, validacao_temporal=False):
    """
    Executa o algoritmo completo de previsão de inadimplência

    Args:
        caminho_arquivo: Caminho para o arquivo CSV do dataset
        caminho_dados_macro: Caminho opcional para arquivo com dados macroeconômicos
        algoritmo: Algoritmo a ser usado ('rf', 'xgb', 'lgb')
        balanceamento: Método de balanceamento (None, 'smote', 'smoteenn', 'undersample')
        otimizar: Se True, realiza otimização de hiperparâmetros
        validacao_temporal: Se True, usa validação temporal

    Returns:
        Modelo treinado e métricas de avaliação
    """
    print("=" * 80)
    print("ALGORITMO AVANÇADO DE PREVISÃO DE INADIMPLÊNCIA BANCÁRIA")
    print("=" * 80)

    print(f"Configuração:")
    print(f"- Algoritmo: {algoritmo}")
    print(f"- Balanceamento: {balanceamento}")
    print(f"- Otimização de hiperparâmetros: {'Sim' if otimizar else 'Não'}")
    print(f"- Validação temporal: {'Sim' if validacao_temporal else 'Não'}")
    print(f"- Dados macroeconômicos: {'Sim' if caminho_dados_macro else 'Não'}")

    # 1. Carregar dados
    df = carregar_dados(caminho_arquivo, caminho_dados_macro)
    if df is None:
        return None

    # 2. Definir variável alvo
    df = definir_variavel_alvo(df)

    # 3. Análise exploratória
    df = analise_exploratoria(df)

    # 4. Engenharia de features básicas
    df = engenharia_features(df)

    # 5. Engenharia de features temporais (se houver dados transacionais em outro arquivo)
    try:
        caminho_transacoes = "dados_transacionais.csv"  # Exemplo, ajuste conforme necessário
        if os.path.exists(caminho_transacoes):
            print(f"\nEncontramos dados transacionais em {caminho_transacoes}")
            df_transacoes = pd.read_csv(caminho_transacoes)

            # Criar features temporais
            df_temporal = engenharia_features_temporais(df_transacoes)

            # Juntar com o dataset principal (se tiver ID em comum)
            if 'ID_Cliente' in df_temporal.columns and 'ID_Cliente' in df.columns:
                df = df.merge(df_temporal.drop(columns=['ID_Cliente']), on='ID_Cliente', how='left')
                print(f"Features temporais incorporadas ao dataset principal.")
                print(f"Dataset resultante: {df.shape[0]} linhas x {df.shape[1]} colunas")
    except Exception as e:
        print(f"Aviso: Não foi possível processar dados transacionais: {e}")
        print("Continuando sem features temporais...")

        # 6. Preparar dados para modelagem
    X_train, X_test, y_train, y_test, feature_names = preparar_dados_modelagem(
        df, validacao_temporal=validacao_temporal
    )

    if X_train is None:
        return None

    # 7. Criar pipeline de modelagem
    pipeline, preprocessor = criar_pipeline_modelagem(
        X_train, y_train, algoritmo=algoritmo, balanceamento=balanceamento
    )

    # 8. Otimizar hiperparâmetros (se solicitado)
    if otimizar:
        print("\nIniciando otimização de hiperparâmetros...")
        pipeline = otimizar_hiperparametros(pipeline, X_train, y_train, algoritmo=algoritmo)

    # 9. Treinar e avaliar modelo
    modelo, threshold_otimo = treinar_e_avaliar_modelo(
        pipeline, X_train, X_test, y_train, y_test, feature_names
    )

    # 10. Interpretar modelo
    importancias = interpretar_modelo(pipeline, X_train, y_train, X_test, y_test, feature_names)

    # 11. Analisar estabilidade do modelo
    estabilidade = analisar_estabilidade_modelo(pipeline, X_train, y_train, feature_names)

    # 12. Salvar modelo e resultados
    print("\nSalvando modelo e resultados...")
    import joblib
    import datetime

    # Adicionar timestamp ao nome do arquivo para controle de versão
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    modelo_filename = f'modelo_inadimplencia_{algoritmo}_{timestamp}.pkl'

    # Salvar o modelo
    joblib.dump(pipeline, modelo_filename)
    print(f"Modelo salvo como '{modelo_filename}'")

    # Salvar threshold ótimo
    with open(f'threshold_otimo_{timestamp}.txt', 'w') as f:
        f.write(str(threshold_otimo))
    print(f"Threshold ótimo ({threshold_otimo:.4f}) salvo.")

    # Salvar importância das features
    if importancias is not None:
        importancias.to_csv(f'importancia_features_{timestamp}.csv', index=False)
        print(f"Importância das features salva como 'importancia_features_{timestamp}.csv'")

    # Criar e salvar metadados do modelo
    metadados = {
        'data_treinamento': timestamp,
        'algoritmo': algoritmo,
        'balanceamento': balanceamento,
        'threshold_otimo': threshold_otimo,
        'features_totais': len(feature_names),
        'amostras_treino': X_train.shape[0],
        'amostras_teste': X_test.shape[0],
        'distribuicao_treino': y_train.value_counts().to_dict(),
        'distribuicao_teste': y_test.value_counts().to_dict()
    }

    pd.DataFrame([metadados]).to_csv(f'metadados_modelo_{timestamp}.csv', index=False)
    print(f"Metadados do modelo salvos como 'metadados_modelo_{timestamp}.csv'")

    print("\nResumo do algoritmo:")
    print(f"- Dataset: {df.shape[0]} clientes, {df.shape[1]} atributos")
    print(f"- Conjunto de treino: {X_train.shape[0]} exemplos")
    print(f"- Conjunto de teste: {X_test.shape[0]} exemplos")
    print(f"- Distribuição da variável alvo: {df['Inadimplente'].value_counts().to_dict()}")

    # Exibir recomendações para próximos passos
    print("\nRecomendações para próximas etapas:")
    print("1. Realizar análise de feature drift e model drift para monitoramento contínuo")
    print("2. Implementar análise de fairness para identificar possíveis viéses do modelo")
    print("3. Desenvolver estratégias de intervenção baseadas nas previsões de inadimplência")
    print("4. Construir dashboards interativos para visualização dos resultados")
    print("5. Integrar o modelo a sistemas de decisão para uso em tempo real")

    print("\nAlgoritmo avançado concluído com sucesso!")
    return modelo, pipeline, importancias, threshold_otimo


# ------------------------------------------------------------------
# 14. Função para relatório de validação e interpretabilidade
# ------------------------------------------------------------------

def gerar_relatorio_validacao(pipeline, X_train, y_train, X_test, y_test, threshold=0.5, output_file=None):
    """
    Gera um relatório detalhado de validação do modelo com visualizações avançadas

    Args:
        pipeline: Pipeline treinado
        X_train, y_train: Dados de treino
        X_test, y_test: Dados de teste
        threshold: Threshold para classificação
        output_file: Arquivo para salvar o relatório

    Returns:
        DataFrame com métricas de validação
    """
    print("\nGerando relatório detalhado de validação do modelo...")

    # Estrutura para armazenar métricas
    resultados = []

    # Conjuntos a serem avaliados
    conjuntos = {'Treino': (X_train, y_train), 'Teste': (X_test, y_test)}

    # Avaliar em cada conjunto
    for nome, (X, y) in conjuntos.items():
        # Fazer previsões
        y_prob = pipeline.predict_proba(X)[:, 1]
        y_pred = (y_prob >= threshold).astype(int)

        # Calcular métricas
        from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                     f1_score, roc_auc_score, average_precision_score)

        metricas = {
            'conjunto': nome,
            'samples': len(y),
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred),
            'recall': recall_score(y, y_pred),
            'f1': f1_score(y, y_pred),
            'auc': roc_auc_score(y, y_prob),
            'avg_precision': average_precision_score(y, y_prob)
        }

        # Calcular matriz de confusão
        from sklearn.metrics import confusion_matrix
        tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()

        # Adicionar métricas derivadas da matriz de confusão
        metricas.update({
            'true_negatives': tn,
            'false_positives': fp,
            'false_negatives': fn,
            'true_positives': tp,
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'false_positive_rate': fp / (fp + tn) if (fp + tn) > 0 else 0,
            'false_negative_rate': fn / (fn + tp) if (fn + tp) > 0 else 0
        })

        resultados.append(metricas)

    # Criar DataFrame de resultados
    df_resultados = pd.DataFrame(resultados)

    # Exibir resultados
    print("\nResultados da validação:")
    print(df_resultados.set_index('conjunto').T)

    # Gerar visualizações
    # 1. Comparação de métricas entre treino e teste
    metricas_plot = ['accuracy', 'precision', 'recall', 'f1', 'auc', 'avg_precision']

    plt.figure(figsize=(12, 6))
    df_plot = df_resultados.set_index('conjunto')[metricas_plot].T
    df_plot.plot(kind='bar', rot=0)
    plt.title('Comparação de Métricas entre Treino e Teste')
    plt.ylabel('Valor')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('comparacao_metricas.png')

    # 2. Trade-off Precision-Recall
    plt.figure(figsize=(10, 6))

    # Para conjunto de teste
    y_prob_test = pipeline.predict_proba(X_test)[:, 1]
    from sklearn.metrics import precision_recall_curve
    precision, recall, thresholds = precision_recall_curve(y_test, y_prob_test)

    # Adicionar threshold atual
    idx_threshold = np.argmin(np.abs(thresholds - threshold))
    current_precision = precision[idx_threshold]
    current_recall = recall[idx_threshold]

    plt.plot(recall, precision, label='Precision-Recall curve')
    plt.scatter(current_recall, current_precision, color='red',
                label=f'Threshold atual ({threshold:.2f})', s=100)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Curva Precision-Recall com Threshold Atual')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig('precision_recall_tradeoff.png')

    # 3. Distribuição de probabilidades por classe real
    plt.figure(figsize=(10, 6))

    # Para conjunto de teste
    df_prob = pd.DataFrame({
        'Probabilidade': y_prob_test,
        'Classe Real': y_test
    })

    # Plotar histogramas sobrepostos
    sns.histplot(data=df_prob, x='Probabilidade', hue='Classe Real',
                 bins=20, element='step', common_norm=False)

    plt.axvline(x=threshold, color='red', linestyle='--',
                label=f'Threshold: {threshold:.2f}')

    plt.title('Distribuição de Probabilidades por Classe Real')
    plt.xlabel('Probabilidade Prevista')
    plt.ylabel('Contagem')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('distribuicao_probabilidades.png')

    # 4. Gráfico de calibração do modelo
    plt.figure(figsize=(10, 6))

    from sklearn.calibration import calibration_curve

    # Calcular curva de calibração
    prob_true, prob_pred = calibration_curve(y_test, y_prob_test, n_bins=10)

    # Plotar
    plt.plot(prob_pred, prob_true, marker='o', linewidth=1, label='Calibração do modelo')

    # Linha de referência (calibração perfeita)
    plt.plot([0, 1], [0, 1], linestyle='--', label='Calibração perfeita')

    plt.xlabel('Probabilidade Prevista')
    plt.ylabel('Fração de Positivos')
    plt.title('Diagrama de Calibração do Modelo')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('calibracao_modelo.png')

    # Salvar resultados em arquivo se especificado
    if output_file:
        df_resultados.to_csv(output_file, index=False)
        print(f"Resultados de validação salvos em '{output_file}'")

    print("Relatório de validação gerado com sucesso.")
    print(
        "Imagens salvas: 'comparacao_metricas.png', 'precision_recall_tradeoff.png', 'distribuicao_probabilidades.png', 'calibracao_modelo.png'")

    return df_resultados


# ------------------------------------------------------------------
# 15. Implementação de detecção de viés (Fairness)
# ------------------------------------------------------------------

def analisar_fairness(pipeline, df, var_alvo='Inadimplente', variaveis_protegidas=None):
    """
    Analisa viés do modelo em relação a variáveis protegidas

    Args:
        pipeline: Pipeline treinado
        df: DataFrame com dados completos
        var_alvo: Nome da variável alvo
        variaveis_protegidas: Lista de variáveis protegidas a verificar

    Returns:
        DataFrame com métricas de fairness
    """
    print("\nAnalisando fairness do modelo...")

    # Se não forem especificadas variáveis protegidas, identificar automaticamente
    if variaveis_protegidas is None:
        variaveis_protegidas = []

        # Possíveis variáveis demográficas ou protegidas
        candidatas = ['Genero', 'Faixa_Etaria', 'Estado_Civil', 'Regiao', 'Raca']

        for var in candidatas:
            if var in df.columns:
                variaveis_protegidas.append(var)

    if not variaveis_protegidas:
        print("Não foram encontradas variáveis protegidas para análise de fairness.")
        return None

    print(f"Variáveis protegidas para análise: {variaveis_protegidas}")

    # Remover variável alvo
    X = df.drop(columns=[var_alvo])
    y = df[var_alvo]

    # Fazer previsões
    y_prob = pipeline.predict_proba(X)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    # Criar DataFrame com previsões e variáveis protegidas
    df_fairness = pd.DataFrame({
        'real': y,
        'previsto': y_pred,
        'probabilidade': y_prob
    })

    # Adicionar variáveis protegidas
    for var in variaveis_protegidas:
        df_fairness[var] = df[var].values

    # Métricas de fairness para cada variável protegida
    resultados = []

    for var in variaveis_protegidas:
        # Calcular métricas para cada grupo
        grupos = df_fairness[var].unique()

        for grupo in grupos:
            subset = df_fairness[df_fairness[var] == grupo]

            # Calcular métricas básicas
            metricas = {
                'variavel': var,
                'grupo': grupo,
                'tamanho': len(subset),
                'taxa_positivos_real': subset['real'].mean(),
                'taxa_positivos_previsto': subset['previsto'].mean(),
                'probabilidade_media': subset['probabilidade'].mean()
            }

            # Calcular taxas de erro
            if subset['real'].sum() > 0:  # Evitar divisão por zero
                metricas['falso_negativo'] = (
                                                     (subset['real'] == 1) & (subset['previsto'] == 0)
                                             ).sum() / subset['real'].sum()
            else:
                metricas['falso_negativo'] = 0

            if (subset['real'] == 0).sum() > 0:  # Evitar divisão por zero
                metricas['falso_positivo'] = (
                                                     (subset['real'] == 0) & (subset['previsto'] == 1)
                                             ).sum() / (subset['real'] == 0).sum()
            else:
                metricas['falso_positivo'] = 0

            resultados.append(metricas)

    # Criar DataFrame de resultados
    df_resultados = pd.DataFrame(resultados)

    # Exibir resultados
    print("\nResultados da análise de fairness:")
    for var in variaveis_protegidas:
        print(f"\nAnálise para variável: {var}")
        subset = df_resultados[df_resultados['variavel'] == var].sort_values('grupo')
        print(subset[['grupo', 'tamanho', 'taxa_positivos_real', 'taxa_positivos_previsto',
                      'falso_positivo', 'falso_negativo']])

        # Calcular disparidade entre grupos
        max_fpr = subset['falso_positivo'].max()
        min_fpr = subset['falso_positivo'].min()
        disparidade_fpr = max_fpr - min_fpr

        max_fnr = subset['falso_negativo'].max()
        min_fnr = subset['falso_negativo'].min()
        disparidade_fnr = max_fnr - min_fnr

        print(f"Disparidade na taxa de falsos positivos: {disparidade_fpr:.4f}")
        print(f"Disparidade na taxa de falsos negativos: {disparidade_fnr:.4f}")

        # Visualizar resultados
        plt.figure(figsize=(12, 6))

        # Plot de barras para taxas de erro
        x = np.arange(len(subset))
        width = 0.35

        plt.bar(x - width / 2, subset['falso_positivo'], width, label='Taxa Falso Positivo')
        plt.bar(x + width / 2, subset['falso_negativo'], width, label='Taxa Falso Negativo')

        plt.xlabel(var)
        plt.ylabel('Taxa')
        plt.title(f'Taxas de Erro por {var}')
        plt.xticks(x, subset['grupo'])
        plt.legend()
        plt.grid(True, axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'fairness_{var}.png')

        # Plot da diferença entre real e previsto
        plt.figure(figsize=(12, 6))

        plt.bar(x - width / 2, subset['taxa_positivos_real'], width, label='Taxa Real')
        plt.bar(x + width / 2, subset['taxa_positivos_previsto'], width, label='Taxa Prevista')

        plt.xlabel(var)
        plt.ylabel('Taxa de Positivos')
        plt.title(f'Comparação Real vs. Previsto por {var}')
        plt.xticks(x, subset['grupo'])
        plt.legend()
        plt.grid(True, axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'disparidade_{var}.png')

    # Salvar resultados
    df_resultados.to_csv('analise_fairness.csv', index=False)
    print("\nResultados de fairness salvos em 'analise_fairness.csv'")

    return df_resultados


# ------------------------------------------------------------------
# 16. Função para criação de modelo de segmentação comportamental
# ------------------------------------------------------------------

def criar_segmentacao_comportamental(df, n_clusters=4):
    """
    Cria uma segmentação comportamental de clientes baseada em padrões financeiros

    Args:
        df: DataFrame com dados dos clientes
        n_clusters: Número de segmentos a criar

    Returns:
        DataFrame com segmentos atribuídos
    """
    print("\nCriando segmentação comportamental de clientes...")

    # Selecionar variáveis comportamentais/financeiras
    vars_comportamentais = [
        'Renda_Mensal', 'Score_Credito', 'Saldo_Atual', 'Tempo_Relacionamento_Anos',
        'Num_Produtos', 'Percentual_Comprometimento_Renda'
    ]

    # Filtrar apenas variáveis existentes
    vars_existentes = [col for col in vars_comportamentais if col in df.columns]

    if len(vars_existentes) < 3:
        print("Variáveis insuficientes para criar segmentação comportamental.")
        return df

    # Criar cópia para não modificar o original
    df_seg = df.copy()

    # Selecionar apenas variáveis numéricas para clusterização
    X_cluster = df_seg[vars_existentes].copy()

    # Tratar valores ausentes
    for col in X_cluster.columns:
        X_cluster[col] = X_cluster[col].fillna(X_cluster[col].median())

    # Normalizar dados
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)

    # Aplicar PCA para redução de dimensionalidade (opcional)
    from sklearn.decomposition import PCA
    n_components = min(len(vars_existentes), 3)  # No máximo 3 componentes
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    # Aplicar K-Means para segmentação
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    segmentos = kmeans.fit_predict(X_scaled)

    # Adicionar segmentos ao DataFrame
    df_seg['Segmento_Comportamental'] = segmentos

    # Análise dos segmentos
    print("\nAnálise dos segmentos comportamentais:")

    analise_segmentos = df_seg.groupby('Segmento_Comportamental')[vars_existentes].mean().round(2)
    print(analise_segmentos)

    # Calcular tamanho e taxa de inadimplência por segmento
    tamanho_segmentos = df_seg['Segmento_Comportamental'].value_counts()

    if 'Inadimplente' in df_seg.columns:
        taxa_inadimplencia = df_seg.groupby('Segmento_Comportamental')['Inadimplente'].mean() * 100

        print("\nTamanho e Taxa de Inadimplência por Segmento:")
        for seg in range(n_clusters):
            print(
                f"Segmento {seg}: {tamanho_segmentos[seg]} clientes - Taxa de Inadimplência: {taxa_inadimplencia[seg]:.2f}%")

    # Visualizar segmentos no espaço PCA
    plt.figure(figsize=(10, 8))

    # Cores para cada segmento
    cores = ['darkblue', 'green', 'red', 'orange', 'purple', 'brown', 'pink', 'gray']

    for i in range(n_clusters):
        # Selecionar pontos deste segmento
        mask = segmentos == i

        # Plotar em 2D ou 3D conforme o número de componentes
        if n_components >= 2:
            plt.scatter(
                X_pca[mask, 0], X_pca[mask, 1],
                label=f'Segmento {i}',
                color=cores[i % len(cores)],
                alpha=0.7
            )

    plt.title('Segmentação Comportamental de Clientes (PCA)')
    plt.xlabel('Componente Principal 1')
    plt.ylabel('Componente Principal 2')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('segmentacao_clientes.png')

    # Criar perfil descritivo de cada segmento
    print("\nPerfil descritivo dos segmentos:")

    for i in range(n_clusters):
        perfil = []
        for col in vars_existentes:
            # Comparar média do segmento com média global
            media_segmento = df_seg[df_seg['Segmento_Comportamental'] == i][col].mean()
            media_global = df_seg[col].mean()

            # Determinar se é alto, médio ou baixo
            ratio = media_segmento / media_global if media_global != 0 else 1

            if ratio > 1.3:
                perfil.append(f"{col}: Alto")
            elif ratio < 0.7:
                perfil.append(f"{col}: Baixo")
            else:
                perfil.append(f"{col}: Médio")

        # Adicionar taxa de inadimplência se disponível
        if 'Inadimplente' in df_seg.columns:
            tx_inadimplencia = df_seg[df_seg['Segmento_Comportamental'] == i]['Inadimplente'].mean() * 100

            if tx_inadimplencia > 15:
                perfil.append(f"Inadimplência: Alta ({tx_inadimplencia:.1f}%)")
            elif tx_inadimplencia < 5:
                perfil.append(f"Inadimplência: Baixa ({tx_inadimplencia:.1f}%)")
            else:
                perfil.append(f"Inadimplência: Média ({tx_inadimplencia:.1f}%)")

        print(f"Segmento {i}: {', '.join(perfil)}")

    # Salvar resultados
    if 'ID_Cliente' in df_seg.columns:
        df_seg[['ID_Cliente', 'Segmento_Comportamental']].to_csv('segmentos_clientes.csv', index=False)
        print("\nSegmentos de clientes salvos em 'segmentos_clientes.csv'")

    return df_seg


# ------------------------------------------------------------------
# 17. Execução do algoritmo quando o script é executado diretamente
# ------------------------------------------------------------------

if __name__ == "__main__":
    import os
    import argparse

    # Configurar parser de argumentos para mais flexibilidade
    parser = argparse.ArgumentParser(description='Algoritmo Avançado de Previsão de Inadimplência Bancária')

    # Argumentos principais
    parser.add_argument('--arquivo', type=str, default="../raw/dataset_bancario.csv",
                        help='Caminho para o arquivo CSV do dataset principal')

    parser.add_argument('--macro', type=str, default=None,
                        help='Caminho para arquivo CSV com dados macroeconômicos (opcional)')

    parser.add_argument('--modo', type=str, choices=['treinar', 'prever', 'otimizar', 'analisar'],
                        default='treinar', help='Modo de execução')

    parser.add_argument('--modelo', type=str, default=None,
                        help='Caminho para arquivo do modelo salvo (para modo prever ou analisar)')

    parser.add_argument('--algoritmo', type=str, choices=['rf', 'xgb', 'lgb'],
                        default='rf', help='Algoritmo de ML a utilizar')

    parser.add_argument('--balanceamento', type=str,
                        choices=['None', 'smote', 'adasyn', 'smoteenn', 'undersample'],
                        default='smote', help='Método de balanceamento de classes')

    parser.add_argument('--otimizar', action='store_true',
                        help='Realizar otimização de hiperparâmetros')

    parser.add_argument('--temporal', action='store_true',
                        help='Usar validação temporal (em vez de aleatória)')

    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Threshold para classificação (modo prever)')

    parser.add_argument('--segmentar', action='store_true',
                        help='Criar segmentação comportamental de clientes')

    parser.add_argument('--analisar_fairness', action='store_true',
                        help='Realizar análise de fairness do modelo')

    # Parsing de argumentos
    args = parser.parse_args()

    # Converter None de string para None real
    if args.balanceamento == 'None':
        args.balanceamento = None

    # Executar conforme o modo selecionado
    if args.modo == 'treinar':
        # Treinar novo modelo
        modelo, pipeline, importancias, threshold_otimo = executar_algoritmo_completo(
            args.arquivo,
            caminho_dados_macro=args.macro,
            algoritmo=args.algoritmo,
            balanceamento=args.balanceamento,
            otimizar=args.otimizar,
            validacao_temporal=args.temporal
        )

        # Criar segmentação comportamental se solicitado
        if args.segmentar and modelo is not None:
            df = pd.read_csv(args.arquivo)
            df_segmentado = criar_segmentacao_comportamental(df)

        # Análise de fairness se solicitado
        if args.analisar_fairness and modelo is not None:
            df = pd.read_csv(args.arquivo)
            analisar_fairness(pipeline, df)

    elif args.modo == 'prever':
        # Verificar se modelo foi especificado
        if args.modelo is None:
            print("ERRO: É necessário especificar um modelo com --modelo para o modo 'prever'")
        else:
            try:
                # Carregar modelo
                print(f"Carregando modelo de {args.modelo}...")
                modelo = joblib.load(args.modelo)

                # Carregar dados para previsão
                print(f"Carregando dados para previsão de {args.arquivo}...")
                novos_dados = pd.read_csv(args.arquivo)

                # Carregar threshold personalizado se disponível
                threshold = args.threshold
                threshold_file = args.modelo.replace('.pkl', '_threshold.txt')
                if os.path.exists(threshold_file):
                    with open(threshold_file, 'r') as f:
                        threshold = float(f.read().strip())
                    print(f"Threshold otimizado carregado: {threshold}")

                # Fazer previsões
                dados_previsao = prever_inadimplencia(modelo, novos_dados, threshold=threshold)

                # Gerar perfis de risco
                if dados_previsao is not None:
                    perfis_alto_risco, perfis_baixo_risco = gerar_perfis_risco(dados_previsao)

                    # Salvar resultados
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    dados_previsao.to_csv(f'resultados_previsao_{timestamp}.csv', index=False)
                    print(f"Resultados da previsão salvos como 'resultados_previsao_{timestamp}.csv'")

            except Exception as e:
                print(f"Erro ao carregar modelo ou fazer previsões: {e}")

    elif args.modo == 'otimizar':
        # Modo específico para otimização intensiva
        print("Iniciando modo de otimização intensiva...")

        # Carregar dados
        df = carregar_dados(args.arquivo)
        if df is None:
            exit(1)

        # Definir variável alvo
        df = definir_variavel_alvo(df)

        # Preparar dados para modelagem
        X_train, X_test, y_train, y_test, feature_names = preparar_dados_modelagem(
            df, validacao_temporal=args.temporal)

        if X_train is None:
            exit(1)

        # Criar pipeline básico
        pipeline, preprocessor = criar_pipeline_modelagem(
            X_train, y_train, algoritmo=args.algoritmo, balanceamento=args.balanceamento)

        # Otimização intensiva com mais iterações
        pipeline_otimizado = otimizar_hiperparametros(
            pipeline, X_train, y_train, algoritmo=args.algoritmo, cv=5, n_iter=50)

        # Avaliar modelo otimizado
        modelo, threshold_otimo = treinar_e_avaliar_modelo(
            pipeline_otimizado, X_train, X_test, y_train, y_test, feature_names)

        # Salvar modelo otimizado
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        joblib.dump(pipeline_otimizado, f'modelo_otimizado_{args.algoritmo}_{timestamp}.pkl')

        # Salvar threshold ótimo
        with open(f'threshold_otimo_{timestamp}.txt', 'w') as f:
            f.write(str(threshold_otimo))

        print(f"Modelo otimizado salvo como 'modelo_otimizado_{args.algoritmo}_{timestamp}.pkl'")
        print(f"Threshold ótimo ({threshold_otimo:.4f}) salvo.")

    elif args.modo == 'analisar':
        # Modo para análise aprofundada de um modelo existente
        if args.modelo is None:
            print("ERRO: É necessário especificar um modelo com --modelo para o modo 'analisar'")
        else:
            try:
                # Carregar modelo
                print(f"Carregando modelo de {args.modelo}...")
                pipeline = joblib.load(args.modelo)

                # Carregar dados
                print(f"Carregando dados de {args.arquivo}...")
                df = pd.read_csv(args.arquivo)

                # Preparar dados
                if 'Inadimplente' not in df.columns:
                    print("ERRO: Dataset deve conter a coluna 'Inadimplente' para análise")
                    exit(1)

                X = df.drop(columns=['Inadimplente'])
                y = df['Inadimplente']

                # Dividir em treino e teste
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.25, random_state=42, stratify=y)

                # Gerar relatório de validação
                gerar_relatorio_validacao(pipeline, X_train, y_train, X_test, y_test,
                                          threshold=args.threshold)

                # Análise de fairness
                if args.analisar_fairness:
                    analisar_fairness(pipeline, df)

                # Segmentação comportamental
                if args.segmentar:
                    criar_segmentacao_comportamental(df)

            except Exception as e:
                print(f"Erro na análise do modelo: {e}")

    print("\nProcesso concluído!")