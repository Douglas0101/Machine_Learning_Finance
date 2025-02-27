import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import datetime, timedelta

# Configuração inicial
np.random.seed(42)
random.seed(42)
fake = Faker('pt_BR')
Faker.seed(42)

# Número de registros
n_registros = 50000


# Funções auxiliares
def gerar_id_unico(n):
    return list(range(1, n + 1))


def gerar_nomes(n):
    return [fake.name() for _ in range(n)]


def gerar_datas_nascimento(n):
    # Idade entre 18 e 85 anos, com maior concentração entre 25-55
    hoje = datetime.now()
    idades = np.random.triangular(18, 35, 85, n).astype(int)
    return [(hoje - timedelta(days=int(365.25 * idade))).date() for idade in idades]


def calcular_idade(data_nascimento):
    hoje = datetime.now().date()
    idade = hoje.year - data_nascimento.year
    if hoje.month < data_nascimento.month or (hoje.month == data_nascimento.month and hoje.day < data_nascimento.day):
        idade -= 1
    return idade


# Geração de dados demográficos
id_cliente = gerar_id_unico(n_registros)
nome_completo = gerar_nomes(n_registros)
data_nascimento = gerar_datas_nascimento(n_registros)
idade = [calcular_idade(d) for d in data_nascimento]

# Gênero (com distribuição realista)
genero = np.random.choice(['Masculino', 'Feminino', 'Outro'], size=n_registros, p=[0.49, 0.49, 0.02])

# Estado civil (considerando correlação com idade)
estado_civil = []
for i in range(n_registros):
    if idade[i] < 25:
        estado_civil.append(np.random.choice(['Solteiro', 'Casado', 'Divorciado', 'Viúvo'], p=[0.85, 0.14, 0.01, 0.0]))
    elif idade[i] < 40:
        estado_civil.append(np.random.choice(['Solteiro', 'Casado', 'Divorciado', 'Viúvo'], p=[0.40, 0.50, 0.09, 0.01]))
    elif idade[i] < 60:
        estado_civil.append(np.random.choice(['Solteiro', 'Casado', 'Divorciado', 'Viúvo'], p=[0.20, 0.60, 0.15, 0.05]))
    else:
        estado_civil.append(np.random.choice(['Solteiro', 'Casado', 'Divorciado', 'Viúvo'], p=[0.15, 0.50, 0.15, 0.20]))

# Nacionalidade (maior concentração brasileira)
nacionalidade = np.random.choice(
    ['Brasileira', 'Portuguesa', 'Argentina', 'Italiana', 'Alemã', 'Outras'],
    size=n_registros,
    p=[0.93, 0.02, 0.02, 0.01, 0.01, 0.01]
)

# Nível educacional
nivel_educacional = []
for i in range(n_registros):
    if idade[i] < 25:
        nivel_educacional.append(np.random.choice(
            ['Ensino Fundamental', 'Ensino Médio', 'Superior Incompleto', 'Superior Completo', 'Pós-Graduação'],
            p=[0.05, 0.40, 0.40, 0.14, 0.01]
        ))
    elif idade[i] < 40:
        nivel_educacional.append(np.random.choice(
            ['Ensino Fundamental', 'Ensino Médio', 'Superior Incompleto', 'Superior Completo', 'Pós-Graduação'],
            p=[0.10, 0.30, 0.20, 0.30, 0.10]
        ))
    else:
        nivel_educacional.append(np.random.choice(
            ['Ensino Fundamental', 'Ensino Médio', 'Superior Incompleto', 'Superior Completo', 'Pós-Graduação'],
            p=[0.20, 0.35, 0.15, 0.20, 0.10]
        ))

# Profissão
profissoes = [
    'Professor', 'Médico', 'Engenheiro', 'Advogado', 'Contador', 'Empresário',
    'Vendedor', 'Motorista', 'Enfermeiro', 'Técnico', 'Analista', 'Gerente',
    'Recepcionista', 'Estudante', 'Aposentado', 'Autônomo', 'Comerciante'
]

profissao = []
for i in range(n_registros):
    if idade[i] >= 65:
        profissao.append(
            np.random.choice(['Aposentado', *profissoes], p=[0.7, *[0.3 / (len(profissoes))] * len(profissoes)]))
    elif idade[i] <= 23 and nivel_educacional[i] in ['Superior Incompleto', 'Superior Completo']:
        profissao.append(
            np.random.choice(['Estudante', *profissoes], p=[0.6, *[0.4 / (len(profissoes))] * len(profissoes)]))
    else:
        profissao.append(np.random.choice(profissoes))

# Renda mensal (correlação com nível educacional e idade)
renda_mensal = []
for i in range(n_registros):
    base_renda = 0
    if nivel_educacional[i] == 'Ensino Fundamental':
        base_renda = np.random.lognormal(mean=8.0, sigma=0.3)  # ~3000
    elif nivel_educacional[i] == 'Ensino Médio':
        base_renda = np.random.lognormal(mean=8.3, sigma=0.3)  # ~4000
    elif nivel_educacional[i] == 'Superior Incompleto':
        base_renda = np.random.lognormal(mean=8.5, sigma=0.3)  # ~5000
    elif nivel_educacional[i] == 'Superior Completo':
        base_renda = np.random.lognormal(mean=8.9, sigma=0.4)  # ~7500
    elif nivel_educacional[i] == 'Pós-Graduação':
        base_renda = np.random.lognormal(mean=9.2, sigma=0.5)  # ~10000

    # Ajuste por idade (experiência)
    if idade[i] < 25:
        base_renda *= 0.8
    elif idade[i] >= 25 and idade[i] < 35:
        base_renda *= 1.1
    elif idade[i] >= 35 and idade[i] < 50:
        base_renda *= 1.3
    elif idade[i] >= 50 and idade[i] < 65:
        base_renda *= 1.2
    else:
        base_renda *= 0.9

    # Ajuste especial para aposentados
    if profissao[i] == 'Aposentado':
        base_renda = base_renda * 0.7

    # Ajuste para empresários (maior variância)
    if profissao[i] == 'Empresário':
        base_renda = base_renda * np.random.lognormal(mean=0.5, sigma=0.7)

    renda_mensal.append(round(base_renda, 2))

# Número de dependentes (correlação com estado civil e idade)
num_dependentes = []
for i in range(n_registros):
    if estado_civil[i] == 'Solteiro' or idade[i] > 65:
        num_dependentes.append(np.random.choice([0, 1, 2], p=[0.8, 0.15, 0.05]))
    elif estado_civil[i] == 'Casado' and idade[i] < 45:
        num_dependentes.append(np.random.choice([0, 1, 2, 3, 4], p=[0.15, 0.3, 0.35, 0.15, 0.05]))
    elif estado_civil[i] == 'Casado':
        num_dependentes.append(np.random.choice([0, 1, 2, 3], p=[0.3, 0.3, 0.3, 0.1]))
    elif estado_civil[i] == 'Divorciado':
        num_dependentes.append(np.random.choice([0, 1, 2], p=[0.4, 0.4, 0.2]))
    else:  # Viúvo
        num_dependentes.append(np.random.choice([0, 1], p=[0.8, 0.2]))

# Residência própria (correlação com idade e renda)
residencia_propria = []
for i in range(n_registros):
    prob_propria = 0.3  # Base

    # Ajuste por idade
    if idade[i] >= 30:
        prob_propria += min((idade[i] - 30) * 0.01, 0.3)  # Até +30%

    # Ajuste por renda
    renda_base = 5000  # Referência
    prob_propria += min((renda_mensal[i] - renda_base) / renda_base * 0.2, 0.3)  # Até +30%

    # Limitar entre 0.1 e 0.9
    prob_propria = max(0.1, min(prob_propria, 0.9))

    residencia_propria.append('Sim' if random.random() < prob_propria else 'Não')

# Tempo de residência (correlação com idade e residência própria)
tempo_residencia_anos = []
for i in range(n_registros):
    max_tempo = max(1, idade[i] - 18)  # Máximo possível desde 18 anos

    if residencia_propria[i] == 'Sim':
        # Proprietários tendem a ficar mais tempo
        media_tempo = max_tempo * 0.4
    else:
        # Não proprietários tendem a mudar mais
        media_tempo = max_tempo * 0.15

    tempo = round(min(random.gammavariate(2.0, media_tempo / 2.0), max_tempo), 1)
    tempo_residencia_anos.append(tempo)

# Dados de conta bancária
id_conta = gerar_id_unico(n_registros)

# Tipo de conta (correlação com idade e renda)
tipo_conta = []
for i in range(n_registros):
    if idade[i] < 25 and renda_mensal[i] < 3000:
        tipo_conta.append(np.random.choice(['Corrente', 'Poupança', 'Salário', 'Digital'], p=[0.3, 0.3, 0.3, 0.1]))
    elif renda_mensal[i] >= 8000:
        tipo_conta.append(
            np.random.choice(['Corrente', 'Poupança', 'Salário', 'Digital', 'Premium'], p=[0.4, 0.2, 0.1, 0.1, 0.2]))
    else:
        tipo_conta.append(np.random.choice(['Corrente', 'Poupança', 'Salário', 'Digital'], p=[0.5, 0.2, 0.2, 0.1]))

# Data de abertura da conta (correlação com idade)
data_abertura_conta = []
hoje = datetime.now().date()
for i in range(n_registros):
    # Verificar se a pessoa tem pelo menos 18 anos
    if idade[i] >= 18:
        # Idade adulta (quando completou 18 anos)
        idade_adulta = hoje - timedelta(days=int((idade[i] - 18) * 365.25))
        # Tempo máximo desde idade adulta (não pode ser negativo)
        max_dias = max(0, (hoje - idade_adulta).days)
    else:
        # Para pessoas com menos de 18 anos, considerar que a conta foi aberta recentemente
        # (alguns bancos permitem contas para menores com autorização dos pais)
        max_dias = min(365, int(idade[i] * 30))  # No máximo idade em meses

    # Gera uma data aleatória entre idade adulta (ou recente) e hoje
    dias_aleatorios = random.randint(0, max_dias) if max_dias > 0 else 0
    data_abertura = hoje - timedelta(days=dias_aleatorios)
    data_abertura_conta.append(data_abertura)

# Tempo de relacionamento em anos
tempo_relacionamento = [(hoje - data).days / 365.25 for data in data_abertura_conta]

# Saldo atual (correlação com renda, tipo de conta e tempo)
saldo_atual = []
for i in range(n_registros):
    base_saldo = renda_mensal[i] * 2  # Base: ~2 meses de renda

    # Ajuste por tipo de conta
    if tipo_conta[i] == 'Poupança':
        base_saldo *= 3  # Poupança tende a ter mais saldo
    elif tipo_conta[i] == 'Premium':
        base_saldo *= 4  # Premium tende a ter muito mais saldo

    # Ajuste por tempo de relacionamento
    base_saldo *= (1 + tempo_relacionamento[i] * 0.05)  # +5% por ano

    # Distribuição log-normal para criar variabilidade realista
    variabilidade = np.random.lognormal(mean=0, sigma=0.8)

    # Possibilidade de saldo negativo para contas correntes (cheque especial)
    if tipo_conta[i] == 'Corrente' and random.random() < 0.1:  # 10% de chance
        saldo_atual.append(round(-base_saldo * 0.2 * random.random(), 2))
    else:
        saldo_atual.append(round(base_saldo * variabilidade, 2))

# Limite de cheque especial (correlação com renda e tipo de conta)
limite_cheque_especial = []
for i in range(n_registros):
    if tipo_conta[i] in ['Corrente', 'Premium']:
        if tipo_conta[i] == 'Premium':
            base_limite = min(renda_mensal[i] * 2.5, 50000)
        else:
            base_limite = min(renda_mensal[i] * 1.2, 20000)

        # Adicionar variação
        limite = base_limite * np.random.uniform(0.7, 1.3)
        limite_cheque_especial.append(round(limite, 2))
    else:
        limite_cheque_especial.append(0.0)  # Sem cheque especial

# Dados de transações
numero_transacoes_mes_media = []
for i in range(n_registros):
    if tipo_conta[i] == 'Premium':
        base_transacoes = np.random.normal(40, 10)
    elif tipo_conta[i] == 'Corrente':
        base_transacoes = np.random.normal(30, 8)
    elif tipo_conta[i] == 'Digital':
        base_transacoes = np.random.normal(25, 7)
    elif tipo_conta[i] == 'Salário':
        base_transacoes = np.random.normal(15, 5)
    else:  # Poupança
        base_transacoes = np.random.normal(5, 2)

    # Ajuste por renda
    multiplicador_renda = min(3, max(0.5, renda_mensal[i] / 5000))
    numero_transacoes_mes_media.append(max(1, round(base_transacoes * np.sqrt(multiplicador_renda))))

# Valor médio de transação (correlação com renda e tipo de conta)
valor_transacao_media = []
for i in range(n_registros):
    base_valor = renda_mensal[i] * 0.1  # Base: 10% da renda

    # Ajuste por tipo de conta
    if tipo_conta[i] == 'Premium':
        base_valor *= 2
    elif tipo_conta[i] == 'Poupança':
        base_valor *= 3

    # Distribuição log-normal para mais realismo
    valor = base_valor * np.random.lognormal(mean=0, sigma=0.5)
    valor_transacao_media.append(round(valor, 2))

# Tipo de transação mais comum
tipo_transacao_mais_comum = []
for i in range(n_registros):
    if tipo_conta[i] == 'Poupança':
        tipo_transacao_mais_comum.append(np.random.choice(
            ['Depósito', 'Saque', 'Transferência'],
            p=[0.6, 0.3, 0.1]
        ))
    elif tipo_conta[i] == 'Salário':
        tipo_transacao_mais_comum.append(np.random.choice(
            ['Depósito', 'Saque', 'Transferência', 'Pagamento de Boleto', 'Compra Débito', 'Compra Crédito'],
            p=[0.1, 0.2, 0.2, 0.3, 0.1, 0.1]
        ))
    else:
        tipo_transacao_mais_comum.append(np.random.choice(
            ['Depósito', 'Saque', 'Transferência', 'Pagamento de Boleto', 'Compra Débito', 'Compra Crédito'],
            p=[0.1, 0.1, 0.2, 0.2, 0.2, 0.2]
        ))

# Frequência de transações online (correlação com idade e tipo de conta)
frequencia_transacoes_online = []
for i in range(n_registros):
    # Base de probabilidade de uso frequente
    prob_alta = 0.3

    # Ajuste por idade
    if idade[i] < 30:
        prob_alta += 0.4
    elif idade[i] < 50:
        prob_alta += 0.2

    # Ajuste por tipo de conta
    if tipo_conta[i] == 'Digital':
        prob_alta += 0.3
    elif tipo_conta[i] == 'Premium':
        prob_alta += 0.2

    # Limitar entre 0.1 e 0.9
    prob_alta = min(0.9, max(0.1, prob_alta))

    # Calcular restante disponível para outras probabilidades
    restante = 1.0 - prob_alta

    # Distribuir o restante entre média e baixa, garantindo valores não-negativos
    prob_media = min(restante * 0.7, 0.9)
    prob_baixa = max(0.0, restante - prob_media)  # Garante que não seja negativo

    # Normalizar para garantir que a soma seja exatamente 1.0
    soma = prob_alta + prob_media + prob_baixa
    if soma > 0:  # Evitar divisão por zero
        prob_alta /= soma
        prob_media /= soma
        prob_baixa /= soma
    else:
        # Fallback para valores seguros se algo der errado
        prob_alta, prob_media, prob_baixa = 0.5, 0.3, 0.2

    frequencia_transacoes_online.append(np.random.choice(
        ['Alta', 'Média', 'Baixa'],
        p=[prob_alta, prob_media, prob_baixa]
    ))

# Dados de empréstimos
tem_emprestimo_ativo = []
for i in range(n_registros):
    # Probabilidade base
    prob_emprestimo = 0.2

    # Ajustar por idade
    if 30 <= idade[i] <= 55:
        prob_emprestimo += 0.1

    # Ajustar por tipo de conta
    if tipo_conta[i] in ['Premium', 'Corrente']:
        prob_emprestimo += 0.1

    # Ajustar por renda
    if renda_mensal[i] > 5000:
        prob_emprestimo += 0.1

    # Limitar entre 0.05 e 0.6
    prob_emprestimo = min(0.6, max(0.05, prob_emprestimo))

    tem_emprestimo_ativo.append('Sim' if random.random() < prob_emprestimo else 'Não')

# Tipo de empréstimo, valor, taxa, prazo e status
tipo_emprestimo = []
valor_emprestimo = []
taxa_juros_emprestimo = []
prazo_emprestimo_meses = []
status_emprestimo = []

for i in range(n_registros):
    if tem_emprestimo_ativo[i] == 'Sim':
        # Tipo de empréstimo
        if idade[i] >= 30 and renda_mensal[i] >= 5000:
            tipo = np.random.choice(
                ['Pessoal', 'Imobiliário', 'Veículo', 'Consignado'],
                p=[0.3, 0.3, 0.3, 0.1]
            )
        else:
            tipo = np.random.choice(
                ['Pessoal', 'Veículo', 'Consignado'],
                p=[0.6, 0.3, 0.1]
            )
        tipo_emprestimo.append(tipo)

        # Valor do empréstimo
        if tipo == 'Imobiliário':
            valor = renda_mensal[i] * np.random.uniform(20, 40) * 12
        elif tipo == 'Veículo':
            valor = renda_mensal[i] * np.random.uniform(5, 15)
        elif tipo == 'Pessoal':
            valor = renda_mensal[i] * np.random.uniform(2, 10)
        else:  # Consignado
            valor = renda_mensal[i] * np.random.uniform(3, 12)
        valor_emprestimo.append(round(valor, 2))

        # Taxa de juros
        if tipo == 'Imobiliário':
            taxa = np.random.uniform(0.6, 1.2)
        elif tipo == 'Veículo':
            taxa = np.random.uniform(1.0, 2.0)
        elif tipo == 'Pessoal':
            taxa = np.random.uniform(2.0, 4.0)
        else:  # Consignado
            taxa = np.random.uniform(1.2, 2.5)
        taxa_juros_emprestimo.append(round(taxa, 2))

        # Prazo do empréstimo
        if tipo == 'Imobiliário':
            prazo = np.random.choice([180, 240, 300, 360])
        elif tipo == 'Veículo':
            prazo = np.random.choice([36, 48, 60, 72])
        elif tipo == 'Pessoal':
            prazo = np.random.choice([12, 24, 36, 48])
        else:  # Consignado
            prazo = np.random.choice([24, 36, 48, 60])
        prazo_emprestimo_meses.append(prazo)

        # Status do empréstimo
        status_prob = np.random.random()
        if status_prob < 0.85:
            status = 'Em Dia'
        elif status_prob < 0.95:
            status = 'Em Atraso'
        else:
            status = 'Inadimplente'
        status_emprestimo.append(status)
    else:
        tipo_emprestimo.append(None)
        valor_emprestimo.append(None)
        taxa_juros_emprestimo.append(None)
        prazo_emprestimo_meses.append(None)
        status_emprestimo.append(None)

# Dados de interação com o banco
canais_utilizados = []
for i in range(n_registros):
    if idade[i] < 30:
        canais_utilizados.append(np.random.choice(
            ['Agência Física', 'Internet Banking', 'Mobile Banking', 'Telefone'],
            p=[0.1, 0.3, 0.55, 0.05]
        ))
    elif idade[i] < 50:
        canais_utilizados.append(np.random.choice(
            ['Agência Física', 'Internet Banking', 'Mobile Banking', 'Telefone'],
            p=[0.25, 0.3, 0.35, 0.1]
        ))
    else:
        canais_utilizados.append(np.random.choice(
            ['Agência Física', 'Internet Banking', 'Mobile Banking', 'Telefone'],
            p=[0.45, 0.25, 0.2, 0.1]
        ))

# Número de reclamações no último ano
numero_reclamacoes = []
for i in range(n_registros):
    prob_reclamar = 0.2  # Base: 20% de chance

    # Clientes Premium tendem a reclamar mais (expectativas mais altas)
    if tipo_conta[i] == 'Premium':
        prob_reclamar += 0.1

    # Clientes com empréstimos em atraso ou inadimplentes tendem a ter mais reclamações
    if tem_emprestimo_ativo[i] == 'Sim' and status_emprestimo[i] in ['Em Atraso', 'Inadimplente']:
        prob_reclamar += 0.3

    if random.random() < prob_reclamar:
        # Número de reclamações segue distribuição Poisson
        numero_reclamacoes.append(np.random.poisson(1.5))
    else:
        numero_reclamacoes.append(0)

# Satisfação do cliente (correlação com número de reclamações e status de empréstimo)
satisfacao_cliente = []
for i in range(n_registros):
    base_satisfacao = 3.0  # Base: Neutro (escala 1-5)

    # Ajuste por reclamações
    if numero_reclamacoes[i] > 0:
        base_satisfacao -= numero_reclamacoes[i] * 0.8

    # Ajuste por status de empréstimo
    if tem_emprestimo_ativo[i] == 'Sim':
        if status_emprestimo[i] == 'Em Dia':
            base_satisfacao += 0.5
        elif status_emprestimo[i] == 'Em Atraso':
            base_satisfacao -= 1.0
        elif status_emprestimo[i] == 'Inadimplente':
            base_satisfacao -= 1.5

    # Limitar entre 1 e 5
    base_satisfacao = min(5, max(1, base_satisfacao))

    # Adicionar ruído aleatório
    satisfacao = round(base_satisfacao + np.random.normal(0, 0.5))
    satisfacao = min(5, max(1, satisfacao))

    categorias = {
        1: 'Muito Insatisfeito',
        2: 'Insatisfeito',
        3: 'Neutro',
        4: 'Satisfeito',
        5: 'Muito Satisfeito'
    }
    satisfacao_cliente.append(categorias[satisfacao])

# CEP (tornando dados mais localizados)
cep = [f"{random.randint(10000, 99999)}-{random.randint(100, 999)}" for _ in range(n_registros)]

# Score de crédito
score_credito = []
for i in range(n_registros):
    base_score = 650  # Base média

    # Ajustes positivos
    if tem_emprestimo_ativo[i] == 'Sim' and status_emprestimo[i] == 'Em Dia':
        base_score += 50
    if residencia_propria[i] == 'Sim':
        base_score += 30
    if renda_mensal[i] > 5000:
        base_score += min(100, int(renda_mensal[i] / 1000) * 10)
    if tempo_relacionamento[i] > 5:
        base_score += min(50, int(tempo_relacionamento[i]) * 5)

    # Ajustes negativos
    if tem_emprestimo_ativo[i] == 'Sim' and status_emprestimo[i] == 'Em Atraso':
        base_score -= 80
    if tem_emprestimo_ativo[i] == 'Sim' and status_emprestimo[i] == 'Inadimplente':
        base_score -= 150
    if numero_reclamacoes[i] > 2:
        base_score -= 30

    # Adicionar variação
    base_score += np.random.normal(0, 25)

    # Limitar entre 100 e 1000
    score_credito.append(min(1000, max(100, int(base_score))))

# ---------- DADOS ADICIONAIS SENSÍVEIS ----------

# Histórico de saldo (saldo anterior de 3 meses atrás)
saldo_anterior = []
variacao_saldo_percentual = []
for i in range(n_registros):
    # Gerar uma variação plausível com base no tipo de conta e perfil
    if tipo_conta[i] == 'Poupança':
        # Poupança tende a crescer gradualmente
        variacao = np.random.normal(0.03, 0.02)  # Media 3% crescimento
    elif tipo_conta[i] == 'Corrente':
        # Contas correntes tendem a flutuar mais
        variacao = np.random.normal(0.0, 0.15)  # Média 0%, maior volatilidade
    elif tipo_conta[i] == 'Premium':
        # Contas premium tendem a crescer mais
        variacao = np.random.normal(0.05, 0.08)  # Média 5% crescimento
    else:
        variacao = np.random.normal(0.01, 0.05)  # Variação padrão

    # Aplicar a variação inversa para calcular o saldo anterior
    saldo_ant = saldo_atual[i] / (1 + variacao) if variacao > -0.99 else saldo_atual[i] * 0.1

    saldo_anterior.append(round(saldo_ant, 2))
    variacao_saldo_percentual.append(round(variacao * 100, 2))

# Limite de crédito (cartão)
limite_cartao_credito = []
for i in range(n_registros):
    if tipo_conta[i] == 'Premium':
        # Limite mais alto para contas premium
        base_limite = min(renda_mensal[i] * np.random.uniform(3, 6), 100000)
    else:
        # Limite padrão correlacionado com renda
        base_limite = min(renda_mensal[i] * np.random.uniform(1, 3), 50000)

    # Ajustes com base no score de crédito
    if score_credito[i] > 800:
        multiplicador = np.random.uniform(1.2, 1.5)
    elif score_credito[i] > 700:
        multiplicador = np.random.uniform(1.0, 1.2)
    elif score_credito[i] > 600:
        multiplicador = np.random.uniform(0.8, 1.0)
    elif score_credito[i] > 400:
        multiplicador = np.random.uniform(0.5, 0.8)
    else:
        multiplicador = np.random.uniform(0.1, 0.5)

    # Aplicar o multiplicador e arredondar
    limite = round(base_limite * multiplicador, 2)
    limite_cartao_credito.append(limite)

# Utilização do limite de crédito
utilizacao_credito_percentual = []
for i in range(n_registros):
    # Correlação com o score de crédito - scores mais altos tendem a usar menos do limite disponível
    if score_credito[i] > 800:
        utilizacao_base = np.random.beta(2, 5) * 100  # Concentrado em valores mais baixos
    elif score_credito[i] > 650:
        utilizacao_base = np.random.beta(2, 3) * 100  # Distribuição mais equilibrada
    else:
        utilizacao_base = np.random.beta(5, 2) * 100  # Concentrado em valores mais altos

    utilizacao_credito_percentual.append(round(utilizacao_base, 2))

# Histórico de empréstimos mais detalhado
numero_emprestimos_anteriores = []
valor_total_emprestimos_anteriores = []
taxa_media_emprestimos_anteriores = []
for i in range(n_registros):
    # Número de empréstimos anteriores correlacionado com idade e tempo de relacionamento
    base_num = 0
    if idade[i] > 30:
        base_num += int((idade[i] - 30) / 10)  # +1 a cada 10 anos acima de 30
    if tempo_relacionamento[i] > 3:
        base_num += int(tempo_relacionamento[i] / 5)  # +1 a cada 5 anos de relacionamento

    # Adicionar aleatoriedade
    num_emp = max(0, np.random.poisson(base_num))
    numero_emprestimos_anteriores.append(num_emp)

    # Valor total de empréstimos anteriores
    if num_emp > 0:
        # Base é renda anual multiplicada por fator que depende do número de empréstimos
        base_valor = renda_mensal[i] * 12 * (num_emp * 0.5)
        # Adicionar variabilidade
        variabilidade = np.random.lognormal(0, 0.6)
        valor_total = round(base_valor * variabilidade, 2)
        valor_total_emprestimos_anteriores.append(valor_total)

        # Taxa média de juros dos empréstimos anteriores
        if score_credito[i] > 800:
            taxa_base = np.random.uniform(0.8, 1.5)
        elif score_credito[i] > 650:
            taxa_base = np.random.uniform(1.5, 2.8)
        else:
            taxa_base = np.random.uniform(2.8, 4.5)

        taxa_media_emprestimos_anteriores.append(round(taxa_base, 2))
    else:
        valor_total_emprestimos_anteriores.append(0)
        taxa_media_emprestimos_anteriores.append(None)

# Perfil bancário (comportamento e segmento)
perfil_investidor = []
segmento_banco = []
for i in range(n_registros):
    # Perfil de investidor baseado em renda, saldo e idade
    if renda_mensal[i] > 15000 or saldo_atual[i] > 100000:
        perfil_base = np.random.choice(['Arrojado', 'Moderado', 'Conservador'], p=[0.5, 0.3, 0.2])
    elif renda_mensal[i] > 5000 or saldo_atual[i] > 30000:
        perfil_base = np.random.choice(['Arrojado', 'Moderado', 'Conservador'], p=[0.3, 0.5, 0.2])
    else:
        perfil_base = np.random.choice(['Arrojado', 'Moderado', 'Conservador'], p=[0.1, 0.3, 0.6])

    # Ajustar por idade - pessoas mais velhas tendem a ser mais conservadoras
    if idade[i] > 60 and perfil_base == 'Arrojado':
        perfil_base = np.random.choice(['Moderado', 'Conservador'], p=[0.7, 0.3])
    elif idade[i] < 30 and perfil_base == 'Conservador':
        perfil_base = np.random.choice(['Arrojado', 'Moderado'], p=[0.4, 0.6])

    perfil_investidor.append(perfil_base)

    # Segmento do banco (classificação interna)
    if tipo_conta[i] == 'Premium' or renda_mensal[i] > 15000:
        segmento_banco.append('Private')
    elif renda_mensal[i] > 8000 or saldo_atual[i] > 50000:
        segmento_banco.append('Alta Renda')
    elif renda_mensal[i] > 4000 or saldo_atual[i] > 20000:
        segmento_banco.append('Média Renda')
    else:
        segmento_banco.append('Varejo')

# Produtos contratados
possui_cartao_credito = []
possui_seguro_vida = []
possui_previdencia = []
possui_investimentos = []
for i in range(n_registros):
    # Cartão de crédito - base alta de penetração
    prob_cartao = 0.7
    if renda_mensal[i] > 3000:
        prob_cartao += 0.2
    if idade[i] > 25 and idade[i] < 65:
        prob_cartao += 0.1
    possui_cartao_credito.append('Sim' if random.random() < min(0.95, prob_cartao) else 'Não')

    # Seguro de vida - menor penetração, mais comum em certos perfis
    prob_seguro = 0.2
    if idade[i] > 35:
        prob_seguro += 0.1
    if renda_mensal[i] > 5000:
        prob_seguro += 0.15
    if num_dependentes[i] > 0:
        prob_seguro += 0.15
    possui_seguro_vida.append('Sim' if random.random() < min(0.8, prob_seguro) else 'Não')

    # Previdência privada
    prob_previdencia = 0.1
    if idade[i] > 30:
        prob_previdencia += 0.05
    if renda_mensal[i] > 7000:
        prob_previdencia += 0.2
    if perfil_investidor[i] in ['Moderado', 'Conservador']:
        prob_previdencia += 0.1
    possui_previdencia.append('Sim' if random.random() < min(0.7, prob_previdencia) else 'Não')

    # Investimentos além da poupança
    prob_invest = 0.1
    if renda_mensal[i] > 5000:
        prob_invest += 0.2
    if idade[i] > 25 and idade[i] < 60:
        prob_invest += 0.1
    if nivel_educacional[i] in ['Superior Completo', 'Pós-Graduação']:
        prob_invest += 0.15
    if perfil_investidor[i] == 'Arrojado':
        prob_invest += 0.2
    possui_investimentos.append('Sim' if random.random() < min(0.8, prob_invest) else 'Não')

# Comportamento de pagamentos
percentual_pagamento_fatura = []
atraso_medio_pagamentos_dias = []
for i in range(n_registros):
    if possui_cartao_credito[i] == 'Sim':
        # Percentual médio de pagamento da fatura
        if score_credito[i] > 750:
            perc_pagamento = np.random.beta(8, 2) * 100  # Concentrado próximo a 100%
        elif score_credito[i] > 600:
            perc_pagamento = np.random.beta(5, 2) * 100  # Média-alta
        else:
            perc_pagamento = np.random.beta(2, 2) * 100  # Distribuído mais uniformemente
        percentual_pagamento_fatura.append(round(perc_pagamento, 2))

        # Atraso médio em pagamentos
        if score_credito[i] > 750:
            atraso = np.random.exponential(0.5)  # Raramente atrasa
        elif score_credito[i] > 600:
            atraso = np.random.exponential(2.0)  # Ocasionalmente atrasa
        else:
            atraso = np.random.exponential(5.0)  # Frequentemente atrasa
        atraso_medio_pagamentos_dias.append(round(atraso, 1))
    else:
        percentual_pagamento_fatura.append(None)
        atraso_medio_pagamentos_dias.append(None)

# Dados de vulnerabilidade financeira
percentual_comprometimento_renda = []
risco_inadimplencia = []
for i in range(n_registros):
    # Calcular comprometimento de renda com dívidas
    # Base: proporção da renda mensal gasta com pagamentos de dívidas
    comprometimento_base = 0.2  # 20% em média

    # Adicionar comprometimento se tiver empréstimo ativo
    if tem_emprestimo_ativo[i] == 'Sim':
        # Estimativa simplificada de prestação mensal
        prestacao_est = valor_emprestimo[i] / prazo_emprestimo_meses[i] * (1 + taxa_juros_emprestimo[i] / 100)
        comprometimento_base += prestacao_est / renda_mensal[i]

    # Adicionar comprometimento com cartão de crédito
    if possui_cartao_credito[i] == 'Sim':
        # Estimar valor médio de fatura com base no limite e utilização
        fatura_media = limite_cartao_credito[i] * (utilizacao_credito_percentual[i] / 100)
        # Se paga apenas o mínimo (15%), vai acumulando dívida cara
        if percentual_pagamento_fatura[i] < 30:
            comprometimento_base += (fatura_media * 0.15) / renda_mensal[i]
        else:
            comprometimento_base += (fatura_media * (percentual_pagamento_fatura[i] / 100)) / renda_mensal[i]

    # Adicionar variação aleatória
    comprometimento = comprometimento_base * np.random.uniform(0.8, 1.2)

    # Limitar a valores plausíveis (até 100% em casos extremos)
    comprometimento = min(1.0, max(0.05, comprometimento))
    percentual_comprometimento_renda.append(round(comprometimento * 100, 2))

    # Calcular risco de inadimplência (score interno do banco)
    # Base: inversamente proporcional ao score de crédito
    risco_base = 100 - (score_credito[i] / 10)

    # Ajustar por comprometimento de renda
    risco_base += comprometimento * 50  # Alto comprometimento aumenta o risco

    # Ajustar por histórico de pagamentos
    if possui_cartao_credito[i] == 'Sim' and atraso_medio_pagamentos_dias[i] > 0:
        risco_base += min(30, atraso_medio_pagamentos_dias[i] * 3)

    # Ajustar por histórico de empréstimos
    if tem_emprestimo_ativo[i] == 'Sim' and status_emprestimo[i] in ['Em Atraso', 'Inadimplente']:
        risco_base += 40

    # Normalizar para escala 0-100
    risco = min(100, max(0, risco_base))
    risco_inadimplencia.append(round(risco, 1))

# Atualizar o DataFrame com as novas colunas
df = pd.DataFrame({
    # Colunas originais
    'ID_Cliente': id_cliente,
    'Nome_Completo': nome_completo,
    'Data_Nascimento': data_nascimento,
    'Idade': idade,
    'Genero': genero,
    'Estado_Civil': estado_civil,
    'Nacionalidade': nacionalidade,
    'Nivel_Educacional': nivel_educacional,
    'Profissao': profissao,
    'Renda_Mensal': renda_mensal,
    'Numero_Dependentes': num_dependentes,
    'Residencia_Propria': residencia_propria,
    'Tempo_Residencia_Anos': tempo_residencia_anos,
    'CEP': cep,
    'Score_Credito': score_credito,
    'ID_Conta': id_conta,
    'Tipo_Conta': tipo_conta,
    'Data_Abertura_Conta': data_abertura_conta,
    'Tempo_Relacionamento_Anos': tempo_relacionamento,
    'Saldo_Atual': saldo_atual,
    'Limite_Cheque_Especial': limite_cheque_especial,
    'Numero_Transacoes_Mes_Media': numero_transacoes_mes_media,
    'Valor_Transacao_Media': valor_transacao_media,
    'Tipo_Transacao_Mais_Comum': tipo_transacao_mais_comum,
    'Frequencia_Transacoes_Online': frequencia_transacoes_online,
    'Tem_Emprestimo_Ativo': tem_emprestimo_ativo,
    'Tipo_Emprestimo': tipo_emprestimo,
    'Valor_Emprestimo': valor_emprestimo,
    'Taxa_Juros_Emprestimo': taxa_juros_emprestimo,
    'Prazo_Emprestimo_Meses': prazo_emprestimo_meses,
    'Status_Emprestimo': status_emprestimo,
    'Canais_Utilizados': canais_utilizados,
    'Numero_Reclamacoes_Ultimo_Ano': numero_reclamacoes,
    'Satisfacao_Cliente': satisfacao_cliente,

    # Novas colunas de dados sensíveis
    'Saldo_Anterior_3M': saldo_anterior,
    'Variacao_Saldo_Percentual_3M': variacao_saldo_percentual,
    'Limite_Cartao_Credito': limite_cartao_credito,
    'Utilizacao_Credito_Percentual': utilizacao_credito_percentual,
    'Numero_Emprestimos_Anteriores': numero_emprestimos_anteriores,
    'Valor_Total_Emprestimos_Anteriores': valor_total_emprestimos_anteriores,
    'Taxa_Media_Emprestimos_Anteriores': taxa_media_emprestimos_anteriores,
    'Perfil_Investidor': perfil_investidor,
    'Segmento_Banco': segmento_banco,
    'Possui_Cartao_Credito': possui_cartao_credito,
    'Possui_Seguro_Vida': possui_seguro_vida,
    'Possui_Previdencia': possui_previdencia,
    'Possui_Investimentos': possui_investimentos,
    'Percentual_Pagamento_Fatura': percentual_pagamento_fatura,
    'Atraso_Medio_Pagamentos_Dias': atraso_medio_pagamentos_dias,
    'Percentual_Comprometimento_Renda': percentual_comprometimento_renda,
    'Risco_Inadimplencia': risco_inadimplencia
})

# Inserir Missing Data de forma realista
# ------------------------------------------

# MCAR (Missing Completely At Random) - menor proporção
colunas_mcar = ['CEP', 'Tempo_Residencia_Anos', 'Numero_Reclamacoes_Ultimo_Ano']
for coluna in colunas_mcar:
    mcar_mask = np.random.random(n_registros) < 0.03  # 3% de missing completamente aleatório
    df.loc[mcar_mask, coluna] = np.nan

# MAR (Missing At Random) - baseado em outras variáveis observadas
# Renda tende a faltar mais para certos grupos
mar_renda_mask = ((df['Nivel_Educacional'] == 'Ensino Fundamental') & (np.random.random(n_registros) < 0.15)) | \
                 ((df['Profissao'] == 'Autônomo') & (np.random.random(n_registros) < 0.12)) | \
                 ((df['Idade'] > 65) & (np.random.random(n_registros) < 0.10))
df.loc[mar_renda_mask, 'Renda_Mensal'] = np.nan

# Nível educacional tende a faltar mais para clientes mais antigos
mar_educacao_mask = (df['Tempo_Relacionamento_Anos'] > 10) & (np.random.random(n_registros) < 0.12)
df.loc[mar_educacao_mask, 'Nivel_Educacional'] = np.nan

# Profissão tende a faltar mais para certos grupos
mar_profissao_mask = ((df['Estado_Civil'] == 'Aposentado') & (np.random.random(n_registros) < 0.15)) | \
                     ((df['Idade'] < 25) & (np.random.random(n_registros) < 0.10))
df.loc[mar_profissao_mask, 'Profissao'] = np.nan

# MNAR (Missing Not At Random) - probabilidade depende do próprio valor
# Renda muito alta tende a não ser declarada
mnar_renda_alta_mask = (df['Renda_Mensal'] > df['Renda_Mensal'].quantile(0.9)) & (np.random.random(n_registros) < 0.20)
df.loc[mnar_renda_alta_mask, 'Renda_Mensal'] = np.nan

# Clientes com baixa satisfação tendem a não responder pesquisas
mnar_satisfacao_mask = (df['Satisfacao_Cliente'].isin(['Muito Insatisfeito', 'Insatisfeito'])) & (np.random.random(n_registros) < 0.25)
df.loc[mnar_satisfacao_mask, 'Satisfacao_Cliente'] = np.nan

# Clientes inadimplentes tendem a ter dados de contato desatualizados
mnar_inadimplentes_mask = (df['Tem_Emprestimo_Ativo'] == 'Sim') & \
                          (df['Status_Emprestimo'] == 'Inadimplente') & \
                          (np.random.random(n_registros) < 0.30)
df.loc[mnar_inadimplentes_mask, 'CEP'] = np.nan

# Missing para dados sobre empréstimos
# Para quem não tem empréstimo ativo, os campos relacionados são naturalmente null
# df.loc[df['Tem_Emprestimo_Ativo'] == 'Não', ['Tipo_Emprestimo', 'Valor_Emprestimo', 'Taxa_Juros_Emprestimo',
#                                             'Prazo_Emprestimo_Meses', 'Status_Emprestimo']] = None

# Alguns campos específicos com proporções de missing
campos_com_missing = {
    'Nacionalidade': 0.05,
    'Tempo_Residencia_Anos': 0.08,
    'Canais_Utilizados': 0.07,
    'Frequencia_Transacoes_Online': 0.06,
    'Numero_Dependentes': 0.04,
    'Valor_Transacao_Media': 0.09
}

for campo, proporcao in campos_com_missing.items():
    mask = np.random.random(n_registros) < proporcao
    df.loc[mask, campo] = np.nan

# Exportar para CSV
df.to_csv('dataset_bancario.csv', index=False)

# Verificar estatísticas do dataset
print(f"Dimensões do dataset: {df.shape}")
print("\nPrimeiras linhas:")
print(df.head())
print("\nInformações gerais:")
print(df.info())
print("\nEstatísticas descritivas:")
print(df.describe())
print("\nValores ausentes por coluna:")
print(df.isnull().sum())
print(f"\nPercentual total de valores ausentes: {df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100:.2f}%")