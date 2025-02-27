# Dados Processados para Modelo de Inadimplência

Gerado em: 27/02/2025 14:30:25
Versão: 20250227_143025

## Arquivos
- X_train_20250227_143025.joblib: Features de treinamento processadas
- y_train_20250227_143025.joblib: Variável alvo de treinamento
- X_val_20250227_143025.joblib: Features de validação processadas
- y_val_20250227_143025.joblib: Variável alvo de validação
- X_test_20250227_143025.joblib: Features de teste processadas
- y_test_20250227_143025.joblib: Variável alvo de teste
- preprocessor_20250227_143025.joblib: ColumnTransformer para processar novos dados
- metadata_20250227_143025.joblib: Informações sobre os conjuntos de dados

## Como carregar os dados

```python
import joblib

# Carregar dados de treinamento
X_train = joblib.load('dados_processados/X_train_20250227_143025.joblib')
y_train = joblib.load('dados_processados/y_train_20250227_143025.joblib')

# Carregar dados de validação
X_val = joblib.load('dados_processados/X_val_20250227_143025.joblib')
y_val = joblib.load('dados_processados/y_val_20250227_143025.joblib')

# Carregar dados de teste
X_test = joblib.load('dados_processados/X_test_20250227_143025.joblib')
y_test = joblib.load('dados_processados/y_test_20250227_143025.joblib')

# Carregar preprocessador
preprocessor = joblib.load('dados_processados/preprocessor_20250227_143025.joblib')

# Para processar novos dados
X_novo_processado = preprocessor.transform(X_novo)
```

## Estatísticas dos Dados
- X_train: (4958, 20)
- X_val: (500, 20)
- X_test: (1000, 20)
- Distribuição y_train: {1: 2479, 0: 2479}
- Distribuição y_val: {0: 354, 1: 146}
- Distribuição y_test: {0: 708, 1: 292}
