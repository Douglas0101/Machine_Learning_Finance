from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
import matplotlib.pyplot as plt
import seaborn as sns
import time


def processar_dados_inadimplencia(df, estrategia_divisao='padrao', estrategia_escala='standard',
                                  estrategia_balanceamento=None, val_size=0.1, random_state=42):
    """
    Implementação avançada de processamento e divisão de dados para modelos de inadimplência

    Args:
        df: DataFrame com os dados bancários
        estrategia_divisao: 'padrao', 'temporal', 'stratified_kfold'
        estrategia_escala: 'standard', 'robust', 'minmax'
        estrategia_balanceamento: None, 'smote', 'adasyn', 'undersample', 'smoteenn'
        val_size: Tamanho do conjunto de validação
        random_state: Semente para reprodutibilidade

    Returns:
        Conjuntos de dados de treino, validação e teste, preprocessador
    """
    print("\n" + "=" * 80)
    print("PROCESSAMENTO AVANÇADO E DIVISÃO DE DADOS")
    print("=" * 80)

    inicio = time.time()

    print(f"\nParâmetros de configuração:")
    print(f"- Estratégia de divisão: {estrategia_divisao}")
    print(f"- Estratégia de escala: {estrategia_escala}")
    print(f"- Estratégia de balanceamento: {estrategia_balanceamento}")
    print(f"- Tamanho do conjunto de validação: {val_size}")

    # 1. Verificar se a variável alvo está presente
    if 'Inadimplente' not in df.columns:
        raise ValueError("A coluna 'Inadimplente' não foi encontrada no DataFrame.")

    # Estatísticas iniciais do dataset
    print(f"\nEstatísticas iniciais do dataset:")
    print(f"- Total de registros: {df.shape[0]}")
    print(f"- Total de features: {df.shape[1]}")
    print(f"- Distribuição da variável alvo:")
    for valor, contagem in df['Inadimplente'].value_counts().items():
        percentual = contagem / len(df) * 100
        print(f"  - Classe {valor}: {contagem} ({percentual:.2f}%)")

    # 2. Identificar tipos de variáveis
    var_alvo = 'Inadimplente'
    y = df[var_alvo].copy()

    # Colunas a excluir (ID, alvo, etc.)
    colunas_excluir = [
        var_alvo, 'ID_Cliente', 'Nome_Completo', 'Data_Nascimento',
        'Data_Abertura_Conta', 'CEP', 'Status_Emprestimo', 'Mes_Ano'
    ]

    # Filtrar apenas colunas existentes
    colunas_excluir = [col for col in colunas_excluir if col in df.columns]
    X = df.drop(columns=colunas_excluir).copy()

    # Identificar tipos de features
    colunas_numericas = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    colunas_categoricas = X.select_dtypes(include=['object', 'category']).columns.tolist()
    colunas_data = [col for col in X.columns if col.lower().startswith('data_')]

    # Para divisão temporal, precisamos remover as colunas de data do conjunto X
    if estrategia_divisao == 'temporal' and colunas_data:
        coluna_temporal = colunas_data[0]
        X_temporal = X[coluna_temporal].copy()
        X = X.drop(columns=colunas_data)
        colunas_numericas = [col for col in colunas_numericas if col not in colunas_data]
    else:
        coluna_temporal = None

    print(f"\nTipos de features identificadas:")
    print(f"- Features numéricas: {len(colunas_numericas)}")
    print(f"- Features categóricas: {len(colunas_categoricas)}")
    if coluna_temporal:
        print(f"- Coluna temporal para divisão: {coluna_temporal}")

    # 3. Divisão em treino/validação/teste
    if estrategia_divisao == 'temporal' and coluna_temporal:
        # Divisão temporal dos dados
        print("\nRealizando divisão temporal dos dados...")

        # Ordenar por data
        indices_ordenados = np.argsort(X_temporal.values)
        X_ordenado = X.iloc[indices_ordenados].reset_index(drop=True)
        y_ordenado = y.iloc[indices_ordenados].reset_index(drop=True)

        # Calcular pontos de corte
        n_amostras = len(X_ordenado)
        test_cutoff = int(n_amostras * 0.8)  # 20% finais para teste
        val_cutoff = int(n_amostras * (0.8 - val_size))  # % anterior para validação

        # Divisão dos conjuntos
        X_train = X_ordenado.iloc[:val_cutoff]
        y_train = y_ordenado.iloc[:val_cutoff]

        X_val = X_ordenado.iloc[val_cutoff:test_cutoff]
        y_val = y_ordenado.iloc[val_cutoff:test_cutoff]

        X_test = X_ordenado.iloc[test_cutoff:]
        y_test = y_ordenado.iloc[test_cutoff:]

        print(f"Divisão temporal concluída:")
        print(f"- Train: {len(X_train)} amostras (primeiros {val_cutoff / (n_amostras) * 100:.1f}%)")
        print(f"- Validation: {len(X_val)} amostras (próximos {val_size * 100:.1f}%)")
        print(f"- Test: {len(X_test)} amostras (últimos {(1 - 0.8) * 100:.1f}%)")

        # Verificar se há vazamento de dados pelo período
        if coluna_temporal in df.columns:
            data_min_train = df.iloc[X_train.index][coluna_temporal].min()
            data_max_train = df.iloc[X_train.index][coluna_temporal].max()
            data_min_val = df.iloc[X_val.index][coluna_temporal].min()
            data_max_val = df.iloc[X_val.index][coluna_temporal].max()
            data_min_test = df.iloc[X_test.index][coluna_temporal].min()
            data_max_test = df.iloc[X_test.index][coluna_temporal].max()

            print(f"\nVerificação de vazamento temporal:")
            print(f"- Train: {data_min_train} a {data_max_train}")
            print(f"- Validation: {data_min_val} a {data_max_val}")
            print(f"- Test: {data_min_test} a {data_max_test}")

    elif estrategia_divisao == 'stratified_kfold':
        # Configuração para validação cruzada estratificada
        print("\nConfigurando dados para validação cruzada estratificada...")

        # Primeiro, separar teste
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=0.2, random_state=random_state, stratify=y
        )

        # Depois, separar validação
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size / (0.8), random_state=random_state, stratify=y_temp
        )

        # Configurar k-fold estratificado para uso posterior
        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

        print(f"Divisão para validação cruzada estratificada concluída:")
        print(f"- Train: {len(X_train)} amostras")
        print(f"- Validation: {len(X_val)} amostras")
        print(f"- Test: {len(X_test)} amostras")
        print(f"- Configurado StratifiedKFold com 5 divisões para validação cruzada")

    else:
        # Divisão padrão estratificada
        print("\nRealizando divisão estratificada padrão...")

        # Primeiro, separar teste
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=0.2, random_state=random_state, stratify=y
        )

        # Depois, separar validação
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size / (0.8), random_state=random_state, stratify=y_temp
        )

        print(f"Divisão estratificada padrão concluída:")
        print(f"- Train: {len(X_train)} amostras")
        print(f"- Validation: {len(X_val)} amostras")
        print(f"- Test: {len(X_test)} amostras")

    # 4. Análise de valores ausentes
    print("\nAnálise de valores ausentes:")
    valores_ausentes = X_train.isnull().mean() * 100
    valores_ausentes = valores_ausentes[valores_ausentes > 0].sort_values(ascending=False)

    if not valores_ausentes.empty:
        print("Features com valores ausentes:")
        for feature, pct in valores_ausentes.items():
            print(f"  - {feature}: {pct:.2f}%")

        # Visualizar padrão de valores ausentes
        plt.figure(figsize=(10, 6))
        sns.heatmap(X_train.isnull(), yticklabels=False, cbar=False, cmap='viridis')
        plt.title('Padrão de Valores Ausentes')
        plt.tight_layout()
        plt.savefig('valores_ausentes.png')
        print("Padrão de valores ausentes salvo em 'valores_ausentes.png'")
    else:
        print("Não há valores ausentes no conjunto de dados.")

    # 5. Verificar distribuição do target nos conjuntos
    print("\nDistribuição da variável alvo nos conjuntos:")
    print(f"- Train: {dict(y_train.value_counts())}")
    print(f"- Validation: {dict(y_val.value_counts())}")
    print(f"- Test: {dict(y_test.value_counts())}")

    # 6. Configurar preprocessador
    # Escolher tipo de scaler
    if estrategia_escala == 'robust':
        scaler = RobustScaler()
        print("\nUsando RobustScaler para melhor tratamento de outliers.")
    elif estrategia_escala == 'minmax':
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        print("\nUsando MinMaxScaler para escala entre 0 e 1.")
    else:  # default: standard
        scaler = StandardScaler()
        print("\nUsando StandardScaler para padronização z-score.")

    # Escolher estratégia de imputação
    usar_knn_imputer = sum(valores_ausentes) > 10  # Usar KNNImputer se muitos valores ausentes

    if usar_knn_imputer:
        numeric_transformer = Pipeline(steps=[
            ('imputer', KNNImputer(n_neighbors=5)),
            ('scaler', scaler)
        ])
        print("Usando KNNImputer para imputação de valores numéricos ausentes.")
    else:
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', scaler)
        ])
        print("Usando SimpleImputer (median) para imputação de valores numéricos ausentes.")

    # Transformer para variáveis categóricas
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Combinar preprocessadores
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, colunas_numericas),
            ('cat', categorical_transformer, colunas_categoricas)
        ]
    )

    # Aplicar preprocessamento
    print("\nAplicando preprocessamento aos conjuntos de dados...")
    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test)

    print(f"Dimensões após preprocessamento:")
    print(f"- X_train: {X_train_processed.shape}")
    print(f"- X_val: {X_val_processed.shape}")
    print(f"- X_test: {X_test_processed.shape}")

    # 7. Balanceamento de classes (apenas no conjunto de treino)
    if estrategia_balanceamento:
        print(f"\nAplicando estratégia de balanceamento: {estrategia_balanceamento}")

        # Contagem original
        print(f"Distribuição original: {dict(y_train.value_counts())}")

        # Aplicar técnica de balanceamento
        if estrategia_balanceamento == 'smote':
            sampler = SMOTE(random_state=random_state)
        elif estrategia_balanceamento == 'adasyn':
            sampler = ADASYN(random_state=random_state)
        elif estrategia_balanceamento == 'undersample':
            sampler = RandomUnderSampler(random_state=random_state)
        elif estrategia_balanceamento == 'smoteenn':
            sampler = SMOTEENN(random_state=random_state)
        else:
            raise ValueError(f"Estratégia de balanceamento não reconhecida: {estrategia_balanceamento}")

        X_train_processed, y_train = sampler.fit_resample(X_train_processed, y_train)

        print(f"Distribuição após balanceamento: {dict(pd.Series(y_train).value_counts())}")
        print(f"Novas dimensões de X_train: {X_train_processed.shape}")

    # 8. Verificação final de qualidade
    print("\nVerificação final de qualidade dos dados:")

    # Verificar valores finais
    valores_nulos = np.isnan(X_train_processed).sum()
    infinitos = np.isinf(X_train_processed).sum()

    if valores_nulos > 0:
        print(f"ALERTA: Ainda existem {valores_nulos} valores nulos após processamento!")
    else:
        print("✅ Não há valores nulos nos dados processados.")

    if infinitos > 0:
        print(f"ALERTA: Existem {infinitos} valores infinitos nos dados processados!")
    else:
        print("✅ Não há valores infinitos nos dados processados.")

    # Tempo total
    fim = time.time()
    print(f"\nProcessamento de dados concluído em {fim - inicio:.2f} segundos.")

    # Retornar dados processados e preprocessador
    return X_train_processed, X_val_processed, X_test_processed, y_train, y_val, y_test, preprocessor


def analisar_features_processadas(X_train_original, X_train_processed, y_train, preprocessor):
    """
    Analisa as features após processamento para entender o impacto das transformações

    Args:
        X_train_original: DataFrame com dados originais de treino
        X_train_processed: Array com dados processados de treino
        y_train: Array com a variável alvo
        preprocessor: ColumnTransformer usado para processar os dados

    Returns:
        None (gera visualizações)
    """
    print("\n" + "=" * 80)
    print("ANÁLISE DE FEATURES APÓS PROCESSAMENTO")
    print("=" * 80)

    # 1. Obter nomes das features após one-hot encoding
    try:
        # Extrair nomes das features do preprocessador
        features_categoricas = []
        onehot_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
        categorical_features = preprocessor.transformers_[1][2]  # Índice das colunas categóricas

        for i, categoria in enumerate(categorical_features):
            valores = onehot_encoder.categories_[i]
            for valor in valores:
                features_categoricas.append(f"{categoria}_{valor}")

        # Nomes das features numéricas
        numerical_features = preprocessor.transformers_[0][2]  # Índice das colunas numéricas

        # Combinar todos os nomes de features
        nomes_features = list(numerical_features) + features_categoricas

        # Limitar para o tamanho real das features processadas
        nomes_features = nomes_features[:X_train_processed.shape[1]]

        print(f"Total de features após processamento: {len(nomes_features)}")
        print(f"Primeiras 10 features: {nomes_features[:10]}")
    except Exception as e:
        print(f"Não foi possível extrair nomes das features: {e}")
        # Usar nomes genéricos
        nomes_features = [f"feature_{i}" for i in range(X_train_processed.shape[1])]

    # 2. Análise de correlação com a variável alvo (para features numéricas)
    try:
        # Converter para DataFrame para facilitar a análise
        X_processed_df = pd.DataFrame(X_train_processed, columns=nomes_features)
        X_processed_df['target'] = y_train

        # Calcular correlação entre features e target
        correlacoes = X_processed_df.corr()['target'].sort_values(ascending=False)

        # Excluir a própria correlação do target
        correlacoes = correlacoes.drop('target')

        # Mostrar top features positivas e negativas
        print("\nTop 10 features mais correlacionadas positivamente com o target:")
        print(correlacoes.head(10))

        print("\nTop 10 features mais correlacionadas negativamente com o target:")
        print(correlacoes.tail(10))

        # Visualizar top correlações
        plt.figure(figsize=(12, 8))
        top_corr = pd.concat([correlacoes.head(10), correlacoes.tail(10)])
        sns.barplot(x=top_corr.values, y=top_corr.index)
        plt.title('Top Features Correlacionadas com Inadimplência')
        plt.axvline(x=0, color='black', linestyle='-')
        plt.tight_layout()
        plt.savefig('correlacoes_top_features.png')
        print("\nGráfico de correlações salvo como 'correlacoes_top_features.png'")
    except Exception as e:
        print(f"Erro na análise de correlação: {e}")

    # 3. Análise de distribuição de features processadas
    try:
        # Selecionar subset de features para visualização
        n_features_show = min(5, X_train_processed.shape[1])

        # Para cada feature, mostrar distribuição por classe
        plt.figure(figsize=(15, n_features_show * 3))

        for i in range(n_features_show):
            feature_name = nomes_features[i]

            # Separar valores por classe
            valores_0 = X_processed_df[X_processed_df['target'] == 0][feature_name]
            valores_1 = X_processed_df[X_processed_df['target'] == 1][feature_name]

            # Criar subplot
            plt.subplot(n_features_show, 1, i + 1)

            # Plotar KDE para cada classe
            sns.kdeplot(valores_0, label='Não Inadimplente (0)', fill=True, alpha=0.3)
            sns.kdeplot(valores_1, label='Inadimplente (1)', fill=True, alpha=0.3)

            plt.title(f'Distribuição de {feature_name} por Classe')
            plt.legend()

        plt.tight_layout()
        plt.savefig('distribuicao_features_processadas.png')
        print("\nGráfico de distribuição de features processadas salvo como 'distribuicao_features_processadas.png'")
    except Exception as e:
        print(f"Erro na análise de distribuição: {e}")

    # 4. Análise de colinearidade (features correlacionadas entre si)
    try:
        # Calcular matriz de correlação
        matriz_corr = X_processed_df.drop('target', axis=1).corr()

        # Identificar pares de features altamente correlacionadas
        threshold = 0.8
        features_correlacionadas = []

        for i in range(len(matriz_corr.columns)):
            for j in range(i + 1, len(matriz_corr.columns)):
                if abs(matriz_corr.iloc[i, j]) > threshold:
                    features_correlacionadas.append(
                        (matriz_corr.columns[i], matriz_corr.columns[j], matriz_corr.iloc[i, j])
                    )

        if features_correlacionadas:
            print("\nFeatures altamente correlacionadas entre si (|corr| > 0.8):")
            for f1, f2, corr in features_correlacionadas[:10]:  # Mostrar primeiros 10 pares
                print(f"  - {f1} e {f2}: {corr:.4f}")

            if len(features_correlacionadas) > 10:
                print(f"  ... e mais {len(features_correlacionadas) - 10} pares.")

            # Visualizar mapa de calor para correlações
            plt.figure(figsize=(12, 10))
            mask = np.triu(np.ones_like(matriz_corr, dtype=bool))
            sns.heatmap(matriz_corr, mask=mask, cmap='coolwarm', center=0, annot=False,
                        square=True, linewidths=.5, cbar_kws={"shrink": .5})
            plt.title('Matriz de Correlação entre Features')
            plt.tight_layout()
            plt.savefig('matriz_correlacao.png')
            print("\nMatriz de correlação salva como 'matriz_correlacao.png'")
        else:
            print("\nNão foram encontradas features altamente correlacionadas entre si.")
    except Exception as e:
        print(f"Erro na análise de colinearidade: {e}")

    # 5. Verificação de separabilidade entre classes
    try:
        # Usar PCA para reduzir dimensionalidade e visualizar separabilidade
        from sklearn.decomposition import PCA

        # Aplicar PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_train_processed)

        # Criar DataFrame com resultados
        pca_df = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])
        pca_df['target'] = y_train

        # Visualizar
        plt.figure(figsize=(10, 8))
        sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='target', alpha=0.7)
        plt.title('Visualização PCA das Classes')
        plt.tight_layout()
        plt.savefig('visualizacao_pca.png')
        print("\nVisualização PCA salva como 'visualizacao_pca.png'")

        # Variação explicada
        variancia_explicada = pca.explained_variance_ratio_
        print(f"\nVariância explicada pelos primeiros 2 componentes PCA: {sum(variancia_explicada) * 100:.2f}%")
        print(f"- PC1: {variancia_explicada[0] * 100:.2f}%")
        print(f"- PC2: {variancia_explicada[1] * 100:.2f}%")
    except Exception as e:
        print(f"Erro na análise PCA: {e}")

    print("\nAnálise de features processadas concluída.")


def gerar_dataset_validacao_temporal(X, y, n_passos_historicos=3, passo_previsao=1, test_size=0.2):
    """
    Gera conjuntos de dados para validação temporal com features de histórico

    Args:
        X: DataFrame com features originais
        y: Series com a variável alvo
        n_passos_historicos: Número de passos históricos a incluir
        passo_previsao: Quantos passos à frente prever
        test_size: Proporção do conjunto de teste

    Returns:
        X_treino, X_teste, y_treino, y_teste formatados para séries temporais
    """
    print("\n" + "=" * 80)
    print("GERAÇÃO DE DATASET PARA VALIDAÇÃO TEMPORAL")
    print("=" * 80)

    # Verificar se há coluna temporal
    colunas_temporais = [col for col in X.columns if 'data' in col.lower()]

    if not colunas_temporais:
        raise ValueError("Não foi encontrada coluna temporal para gerar dataset de validação temporal.")

    coluna_temporal = colunas_temporais[0]
    print(f"Usando coluna temporal: {coluna_temporal}")

    # Ordenar dados por tempo
    df_completo = X.copy()
    df_completo['target'] = y
    df_completo = df_completo.sort_values(coluna_temporal).reset_index(drop=True)

    # Excluir colunas temporais dos features
    X_temp = df_completo.drop(columns=colunas_temporais + ['target'])
    feature_names = X_temp.columns.tolist()

    # Preparar arrays para armazenar os dados formatados
    X_historico = []
    y_alvo = []

    # Preparar dados com histórico
    for i in range(n_passos_historicos, len(df_completo) - passo_previsao):
        # Sequência de histórico
        historico = df_completo.iloc[i - n_passos_historicos:i][feature_names].values.flatten()

        # Valor alvo (1 ou mais passos à frente)
        alvo = df_completo.iloc[i + passo_previsao - 1]['target']

        X_historico.append(historico)
        y_alvo.append(alvo)

    # Converter para arrays
    X_historico = np.array(X_historico)
    y_alvo = np.array(y_alvo)

    # Divisão treino/teste preservando a ordem temporal
    cutoff = int(len(X_historico) * (1 - test_size))

    X_treino = X_historico[:cutoff]
    y_treino = y_alvo[:cutoff]

    X_teste = X_historico[cutoff:]
    y_teste = y_alvo[cutoff:]

    print(f"\nDataset temporal gerado com sucesso:")
    print(f"- Cada exemplo contém {n_passos_historicos} passos de histórico")
    print(f"- Previsão de inadimplência {passo_previsao} passo(s) à frente")
    print(
        f"- Dimensão dos features: {len(feature_names)} features × {n_passos_historicos} passos = {len(feature_names) * n_passos_historicos} features por exemplo")
    print(f"- Conjunto de treino: {X_treino.shape}")
    print(f"- Conjunto de teste: {X_teste.shape}")

    return X_treino, X_teste, y_treino, y_teste


def exportar_dados_processados(X_train, X_val, X_test, y_train, y_val, y_test, preprocessor,
                               diretorio_saida='dados_processados'):
    """
    Exporta os dados processados e o preprocessador para uso posterior

    Args:
        X_train, X_val, X_test: Arrays com dados processados
        y_train, y_val, y_test: Arrays com variáveis alvo
        preprocessor: ColumnTransformer usado para processar os dados
        diretorio_saida: Diretório onde os dados serão salvos

    Returns:
        None
    """
    import os
    import joblib
    from datetime import datetime

    print("\n" + "=" * 80)
    print("EXPORTAÇÃO DE DADOS PROCESSADOS")
    print("=" * 80)

    # Criar diretório se não existir
    if not os.path.exists(diretorio_saida):
        os.makedirs(diretorio_saida)
        print(f"Diretório '{diretorio_saida}' criado.")

    # Timestamp para versionamento
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Salvar arrays de dados
    dataset_files = {
        'X_train': X_train,
        'X_val': X_val,
        'X_test': X_test,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test
    }

    for nome, dados in dataset_files.items():
        arquivo = f"{diretorio_saida}/{nome}_{timestamp}.joblib"
        joblib.dump(dados, arquivo)
        print(f"✅ {nome} salvo como {arquivo}")

    # Salvar preprocessador
    preprocessor_file = f"{diretorio_saida}/preprocessor_{timestamp}.joblib"
    joblib.dump(preprocessor, preprocessor_file)
    print(f"✅ Preprocessador salvo como {preprocessor_file}")

    # Criar arquivo de metadata
    metadata = {
        'timestamp': timestamp,
        'shapes': {
            'X_train': X_train.shape,
            'X_val': X_val.shape,
            'X_test': X_test.shape,
            'y_train': y_train.shape if hasattr(y_train, 'shape') else (len(y_train),),
            'y_val': y_val.shape if hasattr(y_val, 'shape') else (len(y_val),),
            'y_test': y_test.shape if hasattr(y_test, 'shape') else (len(y_test),)
        },
        'distribuicao_classes': {
            'y_train': dict(pd.Series(y_train).value_counts()),
            'y_val': dict(pd.Series(y_val).value_counts()),
            'y_test': dict(pd.Series(y_test).value_counts())
        }
    }

    metadata_file = f"{diretorio_saida}/metadata_{timestamp}.joblib"
    joblib.dump(metadata, metadata_file)
    print(f"✅ Metadata salvo como {metadata_file}")

    # Criar arquivo README com instruções de carregamento
    readme_content = f"""# Dados Processados para Modelo de Inadimplência

Gerado em: {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}
Versão: {timestamp}

## Arquivos
- X_train_{timestamp}.joblib: Features de treinamento processadas
- y_train_{timestamp}.joblib: Variável alvo de treinamento
- X_val_{timestamp}.joblib: Features de validação processadas
- y_val_{timestamp}.joblib: Variável alvo de validação
- X_test_{timestamp}.joblib: Features de teste processadas
- y_test_{timestamp}.joblib: Variável alvo de teste
- preprocessor_{timestamp}.joblib: ColumnTransformer para processar novos dados
- metadata_{timestamp}.joblib: Informações sobre os conjuntos de dados

## Como carregar os dados

```python
import joblib

# Carregar dados de treinamento
X_train = joblib.load('dados_processados/X_train_{timestamp}.joblib')
y_train = joblib.load('dados_processados/y_train_{timestamp}.joblib')

# Carregar dados de validação
X_val = joblib.load('dados_processados/X_val_{timestamp}.joblib')
y_val = joblib.load('dados_processados/y_val_{timestamp}.joblib')

# Carregar dados de teste
X_test = joblib.load('dados_processados/X_test_{timestamp}.joblib')
y_test = joblib.load('dados_processados/y_test_{timestamp}.joblib')

# Carregar preprocessador
preprocessor = joblib.load('dados_processados/preprocessor_{timestamp}.joblib')

# Para processar novos dados
X_novo_processado = preprocessor.transform(X_novo)
```

## Estatísticas dos Dados
- X_train: {X_train.shape}
- X_val: {X_val.shape}
- X_test: {X_test.shape}
- Distribuição y_train: {dict(pd.Series(y_train).value_counts())}
- Distribuição y_val: {dict(pd.Series(y_val).value_counts())}
- Distribuição y_test: {dict(pd.Series(y_test).value_counts())}
"""

    readme_file = f"{diretorio_saida}/README.md"
    with open(readme_file, 'w') as f:
        f.write(readme_content)
    print(f"✅ Arquivo README gerado em {readme_file}")

    print(f"\nDados processados salvos com sucesso no diretório '{diretorio_saida}'.")
    print(f"Use a versão {timestamp} para carregar os dados de forma consistente.")


# =======================================================================
# Exemplo de uso das funções
# =======================================================================

if __name__ == "__main__":
    # Configuração do exemplo
    import pandas as pd
    import numpy as np


    # Criar dataset sintético para teste
    # Correção da função criar_dataset_sintetico

    def criar_dataset_sintetico(n_amostras=1000):
        """
        Gera um dataset sintético para teste do algoritmo de inadimplência
        """
        # Gerar dados aleatórios
        np.random.seed(42)

        # Datas para os últimos 3 anos
        datas_base = pd.date_range(end=pd.Timestamp.now(), periods=1095, freq='D')

        # Selecionar datas aleatórias
        indices = np.random.choice(len(datas_base), size=n_amostras, replace=True)
        datas = datas_base[indices]

        # Features numéricas
        renda = np.random.normal(5000, 2000, size=n_amostras)
        idade = np.random.randint(18, 80, size=n_amostras)
        score_credito = np.random.randint(300, 900, size=n_amostras)
        tempo_cliente = np.random.randint(1, 240, size=n_amostras)
        valor_emprestimo = np.random.normal(20000, 15000, size=n_amostras)
        comprometimento = np.random.uniform(0.1, 0.8, size=n_amostras)

        # Features categóricas
        estado_civil = np.random.choice(['Solteiro', 'Casado', 'Divorciado', 'Viúvo'], size=n_amostras)
        genero = np.random.choice(['Masculino', 'Feminino'], size=n_amostras)
        escolaridade = np.random.choice(['Fundamental', 'Médio', 'Superior', 'Pós-graduação'], size=n_amostras)
        tipo_conta = np.random.choice(['Corrente', 'Poupança', 'Salário', 'Premium'], size=n_amostras)

        # CORREÇÃO #1: Garantir que não há valores NaN nas variáveis usadas para calcular a probabilidade
        # Substituir possíveis NaNs por valores médios
        renda_clean = np.nan_to_num(renda, nan=5000)
        score_credito_clean = np.nan_to_num(score_credito, nan=600)
        comprometimento_clean = np.nan_to_num(comprometimento, nan=0.5)
        idade_clean = np.nan_to_num(idade, nan=40)
        valor_emprestimo_clean = np.nan_to_num(valor_emprestimo, nan=20000)

        # Calcular probabilidade de inadimplência baseada nas features
        logit = (0.005 * score_credito_clean +
                 0.0002 * renda_clean -
                 3 * comprometimento_clean +
                 0.02 * idade_clean -
                 0.0001 * valor_emprestimo_clean)

        prob_inadimplencia = 1 / (1 + np.exp(logit))

        # Introduzir dependência temporal
        mes = pd.DatetimeIndex(datas).month

        # CORREÇÃO #2: Garantir que a probabilidade não exceda 1 após a multiplicação
        # Ajustar o fator para meses do primeiro trimestre
        fator_aumento = 1.5
        prob_inadimplencia_temp = prob_inadimplencia.copy()
        prob_inadimplencia_temp[mes <= 3] *= fator_aumento

        # CORREÇÃO #3: Limitar as probabilidades ao intervalo [0, 1]
        prob_inadimplencia_final = np.clip(prob_inadimplencia_temp, 0, 1)

        # Verificar se ainda existem valores inválidos
        if np.isnan(prob_inadimplencia_final).any() or (prob_inadimplencia_final < 0).any() or (
                prob_inadimplencia_final > 1).any():
            raise ValueError("Ainda existem probabilidades inválidas após correções!")

        # Gerar target com base na probabilidade
        inadimplencia = np.random.binomial(1, prob_inadimplencia_final)

        # Criar DataFrame
        df = pd.DataFrame({
            'ID_Cliente': [f'CLIENTE_{i:06d}' for i in range(n_amostras)],
            'Data_Referencia': datas,
            'Renda_Mensal': renda,
            'Idade': idade,
            'Score_Credito': score_credito,
            'Tempo_Cliente_Meses': tempo_cliente,
            'Valor_Emprestimo': valor_emprestimo,
            'Percentual_Comprometimento_Renda': comprometimento * 100,
            'Estado_Civil': estado_civil,
            'Genero': genero,
            'Escolaridade': escolaridade,
            'Tipo_Conta': tipo_conta,
            'Inadimplente': inadimplencia
        })

        # Adicionar valores ausentes aleatórios (5%)
        for col in df.columns:
            if col not in ['ID_Cliente', 'Data_Referencia', 'Inadimplente']:
                mask = np.random.random(size=len(df)) < 0.05
                df.loc[mask, col] = np.nan

        # Imprimir estatísticas sobre a inadimplência no dataset
        taxa_inadimplencia = df['Inadimplente'].mean() * 100
        print(f"Taxa de inadimplência no dataset sintético: {taxa_inadimplencia:.2f}%")

        # Verificar se há correlação esperada entre as variáveis
        print("\nCorrelação com inadimplência:")
        correlacoes = df[['Score_Credito', 'Renda_Mensal', 'Percentual_Comprometimento_Renda',
                          'Idade', 'Valor_Emprestimo', 'Inadimplente']].corr()['Inadimplente'].sort_values()
        print(correlacoes)

        return df


    # Criar dataset sintético
    print("Criando dataset sintético para exemplo...")
    df = criar_dataset_sintetico(n_amostras=5000)
    print(f"Dataset criado: {df.shape[0]} registros, {df.shape[1]} colunas")

    # Exemplo 1: Processamento padrão
    X_train, X_val, X_test, y_train, y_val, y_test, preprocessor = processar_dados_inadimplencia(
        df,
        estrategia_divisao='padrao',
        estrategia_escala='standard',
        estrategia_balanceamento='smote',
        val_size=0.1,
        random_state=42
    )

    # Exemplo 2: Análise das features processadas
    analisar_features_processadas(
        df.drop(columns=['ID_Cliente', 'Inadimplente']),
        X_train,
        y_train,
        preprocessor
    )

    # Exemplo 3: Exportar dados processados
    exportar_dados_processados(
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        preprocessor,
        diretorio_saida='dados_processados'
    )

    # Exemplo 4: Dataset para validação temporal (opcional)
    try:
        X_temporal = df.drop(columns=['Inadimplente'])
        y_temporal = df['Inadimplente']

        X_treino_temporal, X_teste_temporal, y_treino_temporal, y_teste_temporal = gerar_dataset_validacao_temporal(
            X_temporal, y_temporal, n_passos_historicos=3, passo_previsao=1
        )

        print(f"\nDimensões do dataset temporal:")
        print(f"- X_treino_temporal: {X_treino_temporal.shape}")
        print(f"- X_teste_temporal: {X_teste_temporal.shape}")
    except Exception as e:
        print(f"\nExemplo de validação temporal não executado: {e}")