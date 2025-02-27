# ------------------------------------------------------------------
# Algoritmo Interim de Previsão de Risco de Inadimplência Bancária
# Objetivo: Classificar clientes bancários quanto ao risco de inadimplência
# Versão: 0.1 (Interim)
# ------------------------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score, precision_recall_curve
from sklearn.inspection import permutation_importance
import time
import warnings

warnings.filterwarnings('ignore')


# ------------------------------------------------------------------
# 1. Carregamento e exploração inicial dos dados
# ------------------------------------------------------------------

def carregar_dados(caminho_arquivo):
    """
    Carrega o dataset bancário e exibe informações básicas

    Args:
        caminho_arquivo: Caminho para o arquivo CSV do dataset

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

        return df

    except Exception as e:
        print(f"Erro ao carregar o dataset: {e}")
        return None


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
        print("\nAVISO: Classes muito desbalanceadas! Considerar técnicas de balanceamento.")

    return df


# ------------------------------------------------------------------
# 3. Análise exploratória e engenharia de features
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

    return df


def engenharia_features(df):
    """
    Realiza engenharia de features para melhorar o modelo

    Args:
        df: DataFrame com os dados bancários

    Returns:
        DataFrame com novas features adicionadas
    """
    print("\nRealizando engenharia de features...")

    # Cópia para não modificar o original
    df_features = df.copy()

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

    # Exibir informações sobre as novas features
    novas_features = list(set(df_features.columns) - set(df.columns))
    print(f"\nTotal de {len(novas_features)} novas features criadas")

    return df_features


# ------------------------------------------------------------------
# 4. Preparação para modelagem
# ------------------------------------------------------------------

def preparar_dados_modelagem(df, var_alvo='Inadimplente', test_size=0.25, random_state=42):
    """
    Prepara os dados para modelagem, separando features e alvo

    Args:
        df: DataFrame com os dados preparados
        var_alvo: Nome da variável alvo
        test_size: Proporção do conjunto de teste
        random_state: Semente para reprodutibilidade

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
        'Data_Abertura_Conta', 'CEP', 'Status_Emprestimo'  # Status_Emprestimo é usado para criar o alvo
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

def criar_pipeline_modelagem(X_train):
    """
    Cria um pipeline de pré-processamento e modelagem

    Args:
        X_train: Conjunto de dados de treino (para identificar tipos de colunas)

    Returns:
        Pipeline de modelagem, preprocessador
    """
    print("\nCriando pipeline de modelagem...")

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

    # Pipeline completo com modelo
    # Na versão interim, usamos Random Forest como baseline
    # Poderíamos experimentar outros modelos em versões futuras
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=10,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        ))
    ])

    print("Pipeline criado com sucesso!")
    return pipeline, preprocessor


# ------------------------------------------------------------------
# 6. Treinamento e avaliação do modelo
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

    # Validação cruzada (mais robusta para avaliar o modelo)
    print("\nRealizando validação cruzada...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
    print(f"AUC-ROC média (validação cruzada 5-fold): {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

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
    print("\nCurva ROC salva como 'curva_roc.png'")

    # Curva de Precisão-Recall (importante para classes desbalanceadas)
    plt.figure(figsize=(10, 6))
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    plt.plot(recall, precision)
    plt.xlabel('Recall')
    plt.ylabel('Precisão')
    plt.title('Curva Precisão-Recall - Previsão de Inadimplência')
    plt.savefig('curva_precisao_recall.png')
    print("Curva Precisão-Recall salva como 'curva_precisao_recall.png'")

    return pipeline


# ------------------------------------------------------------------
# 7. Interpretação do modelo (features importantes)
# ------------------------------------------------------------------

def interpretar_modelo(pipeline, X_train, y_train, feature_names):
    """
    Interpreta o modelo, identificando as features mais importantes

    Args:
        pipeline: Pipeline de modelagem treinado
        X_train: Dados de treino
        y_train: Alvo de treino
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

        # Tentativa de obter nomes de features transformadas (pode variar dependendo do modelo)
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
# 8. Função principal para executar o algoritmo completo
# ------------------------------------------------------------------

def executar_algoritmo_completo(caminho_arquivo):
    """
    Executa o algoritmo completo de previsão de inadimplência

    Args:
        caminho_arquivo: Caminho para o arquivo CSV do dataset

    Returns:
        Modelo treinado e métricas de avaliação
    """
    print("=" * 80)
    print("ALGORITMO INTERIM DE PREVISÃO DE INADIMPLÊNCIA BANCÁRIA")
    print("=" * 80)

    # 1. Carregar dados
    df = carregar_dados(caminho_arquivo)
    if df is None:
        return None

    # 2. Definir variável alvo
    df = definir_variavel_alvo(df)

    # 3. Análise exploratória
    df = analise_exploratoria(df)

    # 4. Engenharia de features
    df = engenharia_features(df)

    # 5. Preparar dados para modelagem
    X_train, X_test, y_train, y_test, feature_names = preparar_dados_modelagem(df)
    if X_train is None:
        return None

    # 6. Criar pipeline de modelagem
    pipeline, preprocessor = criar_pipeline_modelagem(X_train)

    # 7. Treinar e avaliar modelo
    modelo = treinar_e_avaliar_modelo(pipeline, X_train, X_test, y_train, y_test, feature_names)

    # 8. Interpretar modelo
    importancias = interpretar_modelo(pipeline, X_train, y_train, feature_names)

    # 9. Salvar modelo e resultados
    print("\nSalvando modelo e resultados...")
    import joblib
    joblib.dump(pipeline, 'modelo_inadimplencia_bancaria.pkl')
    print("Modelo salvo como 'modelo_inadimplencia_bancaria.pkl'")

    # Salvar importância das features
    if importancias is not None:
        importancias.to_csv('importancia_features.csv', index=False)
        print("Importância das features salva como 'importancia_features.csv'")

    print("\nResumo do algoritmo:")
    print(f"- Dataset: {df.shape[0]} clientes, {df.shape[1]} atributos")
    print(f"- Conjunto de treino: {X_train.shape[0]} exemplos")
    print(f"- Conjunto de teste: {X_test.shape[0]} exemplos")
    print(f"- Distribuição da variável alvo: {df['Inadimplente'].value_counts().to_dict()}")

    # Exibir recomendações para próximos passos
    print("\nRecomendações para próximas etapas:")
    print("1. Experimentar diferentes algoritmos de classificação (Gradient Boosting, XGBoost)")
    print("2. Otimizar hiperparâmetros via grid search ou bayesian optimization")
    print("3. Implementar técnicas de balanceamento para lidar com classes desbalanceadas")
    print("4. Criar features adicionais baseadas em conhecimento do domínio bancário")
    print("5. Adicionar validação de negócio e threshold de classificação otimizado")

    print("\nAlgoritmo interim concluído com sucesso!")
    return modelo, pipeline, importancias


# ------------------------------------------------------------------
# 9. Função para fazer previsões em novos dados
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

        print(f"\nResumo das previsões (threshold={threshold}):")
        print(f"- Total de clientes analisados: {len(dados_previsao)}")
        print(f"- Clientes classificados como inadimplentes: {classificacoes.sum()}")
        print(f"- Percentual de inadimplência previsto: {classificacoes.mean() * 100:.2f}%")

        # Distribuição das categorias de risco
        print("\nDistribuição das categorias de risco:")
        print(dados_previsao['Categoria_Risco'].value_counts(normalize=True).sort_index() * 100)

        return dados_previsao

    except Exception as e:
        print(f"Erro ao realizar previsões: {e}")
        return None


# ------------------------------------------------------------------
# 10. Função para gerar perfis de risco
# ------------------------------------------------------------------

def gerar_perfis_risco(dados_previsao, top_n=5):
    """
    Gera perfis de risco baseados nas previsões

    Args:
        dados_previsao: DataFrame com os dados e previsões
        top_n: Número de perfis de alto risco para mostrar

    Returns:
        DataFrames com perfis de alto e baixo risco
    """
    if 'Probabilidade_Inadimplencia' not in dados_previsao.columns:
        print("ERRO: Dados de previsão não contêm probabilidades de inadimplência")
        return None, None

    print("\nGerando perfis de risco...")

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

    return perfis_alto_risco, perfis_baixo_risco


# ------------------------------------------------------------------
# Execução do algoritmo quando o script é executado diretamente
# ------------------------------------------------------------------

if __name__ == "__main__":
    # ALTERE ESTA SEÇÃO - Remova o uso de argparse e defina o caminho diretamente

    # Remova ou comente todo o bloco de código abaixo:
    # import argparse
    # parser = argparse.ArgumentParser(description='Algoritmo de Previsão de Inadimplência Bancária')
    # parser.add_argument('--arquivo', type=str, required=True, help='Caminho para o arquivo CSV do dataset')
    # ...
    # args = parser.parse_args()

    # Substitua pelo caminho fixo do seu arquivo:
    arquivo_dataset = "../raw/dataset_bancario.csv"  # COLOQUE SEU CAMINHO AQUI

    # Defina o modo e outros parâmetros diretamente:
    modo = "treinar"  # ou "prever"
    modelo_path = "modelo_inadimplencia_bancaria.pkl"
    threshold = 0.5

    # Executar conforme o modo selecionado
    if modo == 'treinar':
        # Treinar novo modelo
        modelo, pipeline, importancias = executar_algoritmo_completo(arquivo_dataset)

    else:  # modo == 'prever'
        # Carregar modelo existente e fazer previsões
        try:
            print(f"Carregando modelo de {modelo_path}...")
            import joblib

            modelo = joblib.load(modelo_path)

            # Carregar dados para previsão
            print(f"Carregando dados para previsão de {arquivo_dataset}...")
            novos_dados = pd.read_csv(arquivo_dataset)

            # Fazer previsões
            dados_previsao = prever_inadimplencia(modelo, novos_dados, threshold=threshold)

            # Gerar perfis de risco
            if dados_previsao is not None:
                perfis_alto_risco, perfis_baixo_risco = gerar_perfis_risco(dados_previsao)

                # Salvar resultados
                dados_previsao.to_csv('resultados_previsao.csv', index=False)
                print("Resultados da previsão salvos como 'resultados_previsao.csv'")

        except Exception as e:
            print(f"Erro ao carregar modelo ou fazer previsões: {e}")
