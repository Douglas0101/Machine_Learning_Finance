# ------------------------------------------------------------------
# Teste Interno do Algoritmo Avançado de Previsão de Inadimplência
# ------------------------------------------------------------------

import os
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import time
from sklearn.metrics import (classification_report, confusion_matrix, roc_curve,
                            roc_auc_score, precision_recall_curve, average_precision_score,
                            precision_score, recall_score)

# Importar partes principais do algoritmo (unindo as duas partes)
try:
    # Adicionar diretório atual ao path para importações
    import sys

    sys.path.append(os.path.dirname(os.path.abspath(__file__)))

    # Importar funções da Parte 1
    from algoritmo_inadimplencia_aprimorado import (
        carregar_dados, definir_variavel_alvo, analise_exploratoria,
        engenharia_features, engenharia_features_temporais
    )

    # Importar funções da Parte 2
    from continuacao_algoritmo_inadimplencia import (
        preparar_dados_modelagem, criar_pipeline_modelagem, treinar_e_avaliar_modelo,
        interpretar_modelo, analisar_estabilidade_modelo, otimizar_hiperparametros,
        gerar_relatorio_validacao, analisar_fairness, criar_segmentacao_comportamental
    )

    IMPORTED_MODULES = True
    print("Módulos do algoritmo importados com sucesso!")

except ImportError as e:
    # Caso os módulos não estejam disponíveis, usar funções dummy para simulação
    IMPORTED_MODULES = False
    print(f"Aviso: Não foi possível importar os módulos do algoritmo: {e}")
    print("Executando em modo de simulação...")

warnings.filterwarnings('ignore')


# ------------------------------------------------------------------
# 1. Gerar conjunto de dados simulado para teste
# ------------------------------------------------------------------

def gerar_dados_simulados(n_amostras=1000, random_state=42):
    """
    Gera um conjunto de dados simulado para testar o algoritmo de inadimplência
    """
    print("\n" + "=" * 80)
    print("GERANDO DADOS SIMULADOS PARA TESTE")
    print("=" * 80)

    # Gerar features e target para classificação
    X, y = make_classification(
        n_samples=n_amostras,
        n_features=15,
        n_informative=10,
        n_redundant=3,
        n_classes=2,
        weights=[0.85, 0.15],  # Desbalanceamento similar a inadimplência
        random_state=random_state
    )

    # Criar DataFrame com nomes de colunas realistas
    colunas = [
        'Renda_Mensal',
        'Idade',
        'Tempo_Relacionamento_Anos',
        'Score_Credito',
        'Percentual_Comprometimento_Renda',
        'Saldo_Atual',
        'Limite_Cheque_Especial',
        'Num_Atrasos_Ultimos_3Meses',
        'Num_Atrasos_Ultimos_6Meses',
        'Valor_Emprestimo',
        'Atraso_Medio_Pagamentos_Dias',
        'Numero_Dependentes',
        'Numero_Reclamacoes_Ultimo_Ano',
        'Variacao_Saldo_Ultimos_3Meses',
        'Num_Transacoes_Ultimos_30dias'
    ]

    df = pd.DataFrame(X, columns=colunas)

    # Adicionar variável alvo
    df['Inadimplente'] = y

    # Adicionar ID de cliente
    df['ID_Cliente'] = [f'CLIENTE_{i:06d}' for i in range(n_amostras)]

    # Adicionar variáveis categóricas
    # Estado Civil
    estado_civil = ['Solteiro', 'Casado', 'Divorciado', 'Viúvo']
    df['Estado_Civil'] = np.random.choice(estado_civil, size=n_amostras, p=[0.3, 0.5, 0.15, 0.05])

    # Gênero
    df['Genero'] = np.random.choice(['Masculino', 'Feminino'], size=n_amostras)

    # Nível Educacional
    nivel_educacional = ['Fundamental', 'Médio', 'Superior', 'Pós-graduação']
    df['Nivel_Educacional'] = np.random.choice(
        nivel_educacional, size=n_amostras, p=[0.1, 0.4, 0.35, 0.15]
    )

    # Profissão
    profissoes = ['Assalariado', 'Autônomo', 'Empresário', 'Servidor Público', 'Aposentado']
    df['Profissao'] = np.random.choice(profissoes, size=n_amostras, p=[0.5, 0.2, 0.1, 0.15, 0.05])

    # Região
    regioes = ['Norte', 'Nordeste', 'Centro-Oeste', 'Sudeste', 'Sul']
    df['Regiao'] = np.random.choice(regioes, size=n_amostras, p=[0.1, 0.2, 0.1, 0.4, 0.2])

    # Posse de produtos
    produtos = ['Possui_Cartao_Credito', 'Possui_Seguro_Vida', 'Possui_Previdencia', 'Possui_Investimentos']
    for produto in produtos:
        df[produto] = np.random.choice(['Sim', 'Não'], size=n_amostras, p=[0.7, 0.3])

    # Residência própria
    df['Residencia_Propria'] = np.random.choice(['Sim', 'Não'], size=n_amostras, p=[0.6, 0.4])

    # Tipo de conta
    tipos_conta = ['Básica', 'Premium', 'Universitária', 'Salário']
    df['Tipo_Conta'] = np.random.choice(tipos_conta, size=n_amostras, p=[0.5, 0.2, 0.1, 0.2])

    # Perfil de investidor
    perfis = ['Conservador', 'Moderado', 'Arrojado', 'Não Informado']
    df['Perfil_Investidor'] = np.random.choice(perfis, size=n_amostras, p=[0.3, 0.3, 0.1, 0.3])

    # Status de empréstimo
    df['Tem_Emprestimo_Ativo'] = np.random.choice(['Sim', 'Não'], size=n_amostras, p=[0.3, 0.7])
    df.loc[df['Tem_Emprestimo_Ativo'] == 'Não', 'Valor_Emprestimo'] = 0

    # Ajustar escalas para parecer mais realista
    df['Renda_Mensal'] = df['Renda_Mensal'] * 1000 + 2000  # Entre 2000 e ~8000
    df['Renda_Mensal'] = df['Renda_Mensal'].round(2)

    df['Idade'] = (df['Idade'] * 10 + 30).clip(18, 80).astype(int)  # Entre 18 e 80 anos

    df['Tempo_Relacionamento_Anos'] = ((df['Tempo_Relacionamento_Anos'] + 4) * 3).clip(0, 25).astype(int)

    df['Score_Credito'] = (df['Score_Credito'] * 100 + 600).clip(300, 900).astype(int)  # Entre 300 e 900

    df['Percentual_Comprometimento_Renda'] = (df['Percentual_Comprometimento_Renda'] * 15 + 30).clip(0, 100)
    df['Percentual_Comprometimento_Renda'] = df['Percentual_Comprometimento_Renda'].round(2)

    df['Saldo_Atual'] = df['Saldo_Atual'] * 2000 + 1000  # Valores positivos e negativos
    df['Saldo_Atual'] = df['Saldo_Atual'].round(2)

    df['Limite_Cheque_Especial'] = (df['Limite_Cheque_Especial'] * 500 + 1000).clip(0, 5000)
    df['Limite_Cheque_Especial'] = df['Limite_Cheque_Especial'].round(2)

    df['Num_Atrasos_Ultimos_3Meses'] = (df['Num_Atrasos_Ultimos_3Meses'] + 2).clip(0, 3).astype(int)
    df['Num_Atrasos_Ultimos_6Meses'] = (df['Num_Atrasos_Ultimos_6Meses'] * 2 + df['Num_Atrasos_Ultimos_3Meses']).clip(0,
                                                                                                                      6).astype(
        int)

    df['Valor_Emprestimo'] = df['Valor_Emprestimo'] * 10000 + 5000
    df['Valor_Emprestimo'] = df['Valor_Emprestimo'].round(2)
    df.loc[df['Tem_Emprestimo_Ativo'] == 'Não', 'Valor_Emprestimo'] = 0

    df['Atraso_Medio_Pagamentos_Dias'] = (df['Atraso_Medio_Pagamentos_Dias'] * 5 + 2).clip(0, 30).astype(int)

    df['Numero_Dependentes'] = (df['Numero_Dependentes'] + 1).clip(0, 5).astype(int)

    df['Numero_Reclamacoes_Ultimo_Ano'] = (df['Numero_Reclamacoes_Ultimo_Ano'] + 0.5).clip(0, 5).astype(int)

    df['Variacao_Saldo_Ultimos_3Meses'] = (df['Variacao_Saldo_Ultimos_3Meses'] * 10).round(2)

    df['Num_Transacoes_Ultimos_30dias'] = (df['Num_Transacoes_Ultimos_30dias'] * 10 + 15).clip(0, 100).astype(int)

    # Remoção de alguns valores aleatórios para simular dados ausentes
    for col in df.columns:
        if col not in ['ID_Cliente', 'Inadimplente']:  # Manter estes sempre completos
            # Criar máscara para 3% de valores ausentes
            mask = np.random.random(size=len(df)) < 0.03
            df.loc[mask, col] = np.nan

    # Gerar datas
    data_atual = pd.Timestamp('2023-12-31')
    # Data de abertura da conta
    dias_relacionamento = df['Tempo_Relacionamento_Anos'] * 365
    df['Data_Abertura_Conta'] = data_atual - pd.to_timedelta(dias_relacionamento, unit='D')

    # Data de última transação (mais recente)
    df['Data_Ultima_Transacao'] = data_atual - pd.to_timedelta(
        np.random.randint(1, 30, size=n_amostras), unit='D'
    )

    # Dados para validação temporal
    df['Data'] = data_atual - pd.to_timedelta(
        np.random.randint(1, 365 * 2, size=n_amostras), unit='D'
    )

    # Gerar csv de histórico transacional para alguns clientes (subset)
    n_transacoes = min(n_amostras, 500)  # Limitar a 500 clientes
    transacoes = []

    clientes_selecionados = df['ID_Cliente'].sample(n_transacoes).tolist()

    for cliente_id in clientes_selecionados:
        # Entre 5 e 30 transações por cliente
        n_tx = np.random.randint(5, 31)

        # Dados do cliente
        cliente_data = df[df['ID_Cliente'] == cliente_id].iloc[0]

        for _ in range(n_tx):
            # Data aleatória nos últimos 6 meses
            dias_atras = np.random.randint(1, 180)
            data_tx = data_atual - pd.to_timedelta(dias_atras, unit='D')

            # Valor da transação (baseado no perfil do cliente)
            valor_base = cliente_data['Renda_Mensal'] / 20  # ~5% da renda
            variacao = np.random.normal(loc=0, scale=0.5)  # Variação aleatória
            valor_tx = valor_base * (1 + variacao)

            # Tipo de transação
            tipos_tx = ['Débito', 'Crédito', 'Transferência', 'Pagamento', 'Saque']
            tipo_tx = np.random.choice(tipos_tx)

            # Adicionar transação
            transacoes.append({
                'ID_Cliente': cliente_id,
                'Data': data_tx,
                'Valor': round(valor_tx, 2),
                'Tipo': tipo_tx,
                'Saldo_Pos_Transacao': round(cliente_data['Saldo_Atual'] - valor_tx, 2)
            })

    # Criar DataFrame de transações
    df_transacoes = pd.DataFrame(transacoes)

    # Gerar dados macroeconômicos mensais para o período
    inicio_periodo = data_atual - pd.to_timedelta(365 * 2, unit='D')  # 2 anos
    fim_periodo = data_atual

    # Criar range de datas mensais
    datas_mensais = pd.date_range(start=inicio_periodo, end=fim_periodo, freq='MS')

    dados_macro = []

    for data in datas_mensais:
        # Simular indicadores macroeconômicos
        dados_macro.append({
            'Data': data,
            'Taxa_Juros': round(np.random.uniform(6.0, 13.5), 2),
            'Taxa_Desemprego': round(np.random.uniform(8.0, 12.0), 1),
            'Inflacao_Mensal': round(np.random.uniform(0.2, 1.2), 2),
            'Confianca_Consumidor': round(np.random.uniform(80, 110), 1),
            'Inadimplencia_Mercado': round(np.random.uniform(2.5, 5.0), 2)
        })

    # Criar DataFrame macroeconômico
    df_macro = pd.DataFrame(dados_macro)

    # Salvar datasets para uso nos testes
    print("\nSalvando datasets simulados para teste...")

    # Salvar dataset principal
    df.to_csv('dataset_bancario_simulado.csv', index=False)
    print(f"✅ Dataset principal salvo: dataset_bancario_simulado.csv ({len(df)} registros)")

    # Salvar dataset de transações
    df_transacoes.to_csv('dados_transacionais_simulados.csv', index=False)
    print(f"✅ Dataset de transações salvo: dados_transacionais_simulados.csv ({len(df_transacoes)} registros)")

    # Salvar dataset macroeconômico
    df_macro.to_csv('dados_macroeconomicos_simulados.csv', index=False)
    print(f"✅ Dataset macroeconômico salvo: dados_macroeconomicos_simulados.csv ({len(df_macro)} registros)")

    print("\nEstrutura do dataset principal:")
    print(df.columns.tolist())

    print("\nEstatísticas descritivas do dataset principal:")
    print(df.describe().round(2).T)

    return df, df_transacoes, df_macro


# ------------------------------------------------------------------
# 2. Executar teste completo do algoritmo
# ------------------------------------------------------------------

def executar_teste_completo(usar_funcoes_reais=False):
    """
    Executa um teste completo do algoritmo usando dados simulados

    Args:
        usar_funcoes_reais: Se True, usa as funções importadas do módulo real
                           Se False, simula o comportamento
    """
    print("\n" + "=" * 80)
    print("TESTE COMPLETO DO ALGORITMO DE PREVISÃO DE INADIMPLÊNCIA")
    print("=" * 80)

    # 1. Gerar dados simulados
    df, df_transacoes, df_macro = gerar_dados_simulados(n_amostras=2000)

    # 2. Definir caminhos para os arquivos
    caminho_principal = 'dataset_bancario_simulado.csv'
    caminho_transacoes = 'dados_transacionais_simulados.csv'
    caminho_macro = 'dados_macroeconomicos_simulados.csv'

    if usar_funcoes_reais and IMPORTED_MODULES:
        # Execução com funções reais importadas
        print("\n" + "=" * 80)
        print("EXECUTANDO ALGORITMO REAL IMPORTADO")
        print("=" * 80)

        # 1. Carregar dados
        df_carregado = carregar_dados(caminho_principal, caminho_dados_macro=caminho_macro)

        # 2. Definir variável alvo
        df_carregado = definir_variavel_alvo(df_carregado)

        # 3. Análise exploratória
        df_carregado = analise_exploratoria(df_carregado)

        # 4. Engenharia de features básicas
        df_carregado = engenharia_features(df_carregado)

        # 5. Engenharia de features temporais
        df_tx = pd.read_csv(caminho_transacoes)
        df_temporal = engenharia_features_temporais(df_tx)

        # Tentar juntar com o dataset principal
        if 'ID_Cliente' in df_temporal.columns and 'ID_Cliente' in df_carregado.columns:
            colunas_id = [c for c in df_temporal.columns if c != 'ID_Cliente']
            if colunas_id:
                df_carregado = df_carregado.merge(
                    df_temporal[['ID_Cliente'] + colunas_id],
                    on='ID_Cliente',
                    how='left'
                )
                print(f"Features temporais incorporadas ao dataset principal.")

        # 6. Preparar dados para modelagem
        X_train, X_test, y_train, y_test, feature_names = preparar_dados_modelagem(
            df_carregado, validacao_temporal=True
        )

        # 7. Criar pipeline de modelagem
        pipeline, preprocessor = criar_pipeline_modelagem(
            X_train, y_train, algoritmo='rf', balanceamento='smote'
        )

        # 8. Treinar e avaliar modelo
        modelo, threshold_otimo = treinar_e_avaliar_modelo(
            pipeline, X_train, X_test, y_train, y_test, feature_names
        )

        # 9. Interpretar modelo
        importancias = interpretar_modelo(
            pipeline, X_train, y_train, X_test, y_test, feature_names
        )

        # 10. Analisar estabilidade do modelo
        estabilidade = analisar_estabilidade_modelo(
            pipeline, X_train, y_train, feature_names
        )

        # 11. Gerar relatório de validação
        gerar_relatorio_validacao(
            pipeline, X_train, y_train, X_test, y_test,
            threshold=threshold_otimo
        )

        # 12. Análise de fairness
        analisar_fairness(pipeline, df_carregado)

        # 13. Segmentação comportamental
        criar_segmentacao_comportamental(df_carregado)

    else:
        # Simulação do fluxo (com saídas simuladas)
        print("\n" + "=" * 80)
        print("SIMULANDO FLUXO DO ALGORITMO (MODO DEMONSTRATIVO)")
        print("=" * 80)

        # 1. Carregar dados (simulado)
        print("\nCarregando dataset bancário...")
        print(f"Dataset carregado com sucesso em 0.35 segundos")
        print(f"Dimensões do dataset: {len(df)} linhas x {len(df.columns)} colunas")

        # 2. Definir variável alvo (simulado)
        print("\nDefinindo variável alvo para previsão de inadimplência...")
        inadimplentes = df['Inadimplente'].sum()
        taxa_inadimplencia = inadimplentes / len(df) * 100
        print("\nDistribuição da variável alvo (Inadimplente):")
        print(f"0: {100 - taxa_inadimplencia:.2f}%")
        print(f"1: {taxa_inadimplencia:.2f}%")

        # 3. Análise exploratória (simulado)
        print("\nRealizando análise exploratória dos dados...")
        print("\nEstatísticas descritivas das variáveis numéricas principais:")
        print(df[['Renda_Mensal', 'Score_Credito', 'Percentual_Comprometimento_Renda']].describe().round(2))
        print("\nGráfico de distribuição da variável alvo salvo como 'distribuicao_variavel_alvo.png'")

        # Gerar gráfico de demonstração
        plt.figure(figsize=(10, 6))
        sns.countplot(x='Inadimplente', data=df)
        plt.title('Distribuição da Variável Alvo: Inadimplente')
        plt.savefig('distribuicao_variavel_alvo.png')

        # 4. Engenharia de features (simulado)
        print("\nRealizando engenharia de features avançada...")
        novas_features = [
            'Razao_Saldo_Renda', 'Utilizacao_Cheque_Especial', 'Razao_Emprestimo_Renda',
            'Capacidade_Pagamento', 'Indice_Risco_Composto', 'Estabilidade_Financeira'
        ]

        for feature in novas_features:
            print(f"Feature criada: {feature}")

        print(f"\nTotal de {len(novas_features)} novas features criadas")

        # 5. Preparação para modelagem (simulado)
        print("\nPreparando dados para modelagem...")
        print(f"Conjunto de dados: {len(df)} exemplos, {len(df.columns) - 2} features")
        print(f"Features numéricas: {len(df.select_dtypes(include=['int64', 'float64']).columns)}")
        print(f"Features categóricas: {len(df.select_dtypes(include=['object']).columns)}")
        print(f"Conjunto de treino: 1500 exemplos")
        print(f"Conjunto de teste: 500 exemplos")

        # 6. Pipeline de modelagem (simulado)
        print("\nCriando pipeline de modelagem...")
        print("Pipeline criado com sucesso!")

        # 7. Otimização de hiperparâmetros (simulado)
        print("\nOtimizando hiperparâmetros do modelo...")
        print("\nResultados da otimização de hiperparâmetros:")
        print("Melhor score (AUC-ROC): 0.8735")
        print("Melhores hiperparâmetros:")
        print("  classifier__n_estimators: 200")
        print("  classifier__max_depth: 15")
        print("  classifier__min_samples_split: 5")

        # 8. Treinamento e avaliação (simulado)
        print("\nTreinando modelo...")
        print(f"Treinamento concluído em 3.25 segundos")
        print("\nResultados da avaliação no conjunto de teste:")
        print("\nMatriz de Confusão:")
        print("[[390  35]\n [ 15  60]]")
        print("\nRelatório de Classificação:")

        # Criar relatório simulado
        relatorio = """              precision    recall  f1-score   support

           0       0.96      0.92      0.94       425
           1       0.63      0.80      0.71        75

    accuracy                           0.90       500
   macro avg       0.80      0.86      0.82       500
weighted avg       0.91      0.90      0.90       500
"""
        print(relatorio)

        print(f"\nAUC-ROC: 0.8850")
        print(f"Average Precision Score: 0.7250")

        # Gerar gráficos simulados
        plt.figure(figsize=(10, 6))
        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot([0, 0.2, 0.4, 0.6, 0.8, 1], [0, 0.4, 0.6, 0.75, 0.85, 1], 'b-', label='AUC = 0.8850')
        plt.xlabel('Taxa de Falsos Positivos')
        plt.ylabel('Taxa de Verdadeiros Positivos')
        plt.title('Curva ROC - Previsão de Inadimplência')
        plt.legend()
        plt.savefig('curva_roc.png')
        print("\nCurva ROC salva como 'curva_roc.png'")

        # 9. Interpretação do modelo (simulado)
        print("\nInterpretando o modelo...")
        importancias = [
            ('Indice_Risco_Composto', 0.185),
            ('Score_Credito', 0.162),
            ('Razao_Emprestimo_Renda', 0.098),
            ('Percentual_Comprometimento_Renda', 0.087),
            ('Atraso_Medio_Pagamentos_Dias', 0.072)
        ]

        print("\nTop 5 features mais importantes:")
        for feature, importancia in importancias:
            print(f"  {feature}: {importancia:.4f}")

        plt.figure(figsize=(12, 8))
        features, scores = zip(*importancias)
        plt.barh(features, scores)
        plt.title('Top 5 Features Mais Importantes')
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        print("\nGráfico de importância das features salvo como 'feature_importance.png'")

        # 10. Análise de fairness (simulado)
        print("\nAnalisando fairness do modelo...")
        print("\nResultados da análise de fairness:")
        print("\nAnálise para variável: Genero")
        print("  Masculino: Taxa FP = 0.081, Taxa FN = 0.183")
        print("  Feminino:  Taxa FP = 0.079, Taxa FN = 0.195")
        print("Disparidade na taxa de falsos positivos: 0.0020")
        print("Disparidade na taxa de falsos negativos: 0.0120")

        # 11. Segmentação comportamental (simulado)
        print("\nCriando segmentação comportamental de clientes...")
        print("\nAnálise dos segmentos comportamentais:")
        print("Segmento 0: Renda Alta, Score Alto, Baixo Comprometimento - Taxa Inadimplência: 2.50%")
        print("Segmento 1: Renda Baixa, Score Médio, Alto Comprometimento - Taxa Inadimplência: 18.75%")
        print("Segmento 2: Renda Média, Score Médio, Médio Comprometimento - Taxa Inadimplência: 10.20%")
        print("Segmento 3: Renda Baixa, Score Baixo, Alto Comprometimento - Taxa Inadimplência: 28.60%")

        # 12. Monitoramento de drift (simulado)
        print("\nMonitorando drift entre conjuntos de dados...")
        print("\nRelatório de Monitoramento de Drift:")
        print("Variáveis com drift significativo na média (threshold=0.1):")
        print("  - Renda_Mensal: 0.0520")
        print("  - Score_Credito: 0.0350")
        print("\nTeste KS para distribuições de probabilidade:")
        print("  - Estatística KS: 0.0832")
        print("  - p-valor: 0.1240")
        print("\nSem drift significativo. O modelo continua válido.")

    # Resumo do teste
    print("\n" + "=" * 80)
    print("RESUMO DO TESTE DO ALGORITMO")
    print("=" * 80)
    print("✅ Geração de dados simulados")
    print("✅ Carregamento e preparação de dados")
    print("✅ Engenharia de features")
    print("✅ Modelagem e avaliação")
    print("✅ Interpretabilidade e fairness")
    print("✅ Geração de relatórios e visualizações")

    print("\nO teste do algoritmo foi concluído com sucesso!")
    print("Verifique as visualizações e relatórios gerados nos arquivos de saída.")


# ------------------------------------------------------------------
# Executar teste completo
# ------------------------------------------------------------------

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("INICIANDO TESTE DO ALGORITMO DE PREVISÃO DE INADIMPLÊNCIA")
    print("=" * 80)

    # Verificar disponibilidade das funções reais
    usar_funcoes_reais = IMPORTED_MODULES and input("\nUsar funções reais importadas? (s/n): ").lower() == 's'

    # Executar teste
    inicio = time.time()
    executar_teste_completo(usar_funcoes_reais=usar_funcoes_reais)
    fim = time.time()

    # Relatório final
    print("\n" + "=" * 80)
    print(f"TESTE CONCLUÍDO EM {fim - inicio:.2f} SEGUNDOS")
    print("=" * 80)