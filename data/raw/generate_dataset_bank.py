import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from faker import Faker

# ==============================
# 1. CONFIGURAÇÕES E PARÂMETROS
# ==============================
NUM_TRANSACOES = 50_000
NUM_USUARIOS = 5_000

# Sementes para reprodutibilidade
random.seed(42)
np.random.seed(42)

# Instanciando Faker
fake = Faker('pt_BR')  # Pode trocar para 'en_US', etc.


# ==============================
# 2. FUNÇÕES AUXILIARES
# ==============================
def random_date(start_dt, end_dt):
    """
    Retorna um datetime aleatório entre 'start_dt' e 'end_dt'.
    """
    delta = end_dt - start_dt
    int_delta = delta.days * 24 * 60 * 60
    random_second = random.randrange(int_delta)
    return start_dt + timedelta(seconds=random_second)


def maybe_missing(value, prob_missing):
    """
    Retorna None (ou np.nan) com probabilidade 'prob_missing',
    caso contrário retorna 'value'.
    """
    if random.random() < prob_missing:
        return None
    else:
        return value


# ==============================
# 3. LISTAS DE REFERÊNCIA
# ==============================
TIPOS_CONTA = ['corrente', 'poupanca', 'investimento']
VALOR_FAIXAS = {
    'corrente': (10, 3000),
    'poupanca': (10, 5000),
    'investimento': (100, 50000)
}
TIPOS_TRANSACAO = ['deposito', 'saque', 'transferencia', 'pagamento', 'compra']
CANAIS = ['online', 'offline', 'agência']
DISPOSITIVOS = ['web', 'mobile', 'atm', 'terminal_pos']

STATUS_TRANSACAO = ['aprovada', 'pendente', 'rejeitada']
MOTIVOS_RECUZA = [
    'Saldo Insuficiente', 'Fraude Suspeita',
    'Conta Inativa', 'Limite Excedido', 'Dados Incorretos'
]

CATEGORIAS_MERCHANT = [
    'restaurante', 'supermercado', 'gasolina', 'farmacia', 'padaria',
    'eletronicos', 'academia', 'hotel', 'cafeteria', 'outros'
]

DIAS_SEMANA = ['Seg', 'Ter', 'Qua', 'Qui', 'Sex', 'Sab', 'Dom']

START_DATE = datetime(2023, 1, 1)
END_DATE = datetime(2025, 1, 1)

# ==============================
# 4. GERAÇÃO DE USUÁRIOS
# ==============================
usuarios_dict = {}
for user_id in range(1, NUM_USUARIOS + 1):
    tipo = random.choice(TIPOS_CONTA)

    # Saldo inicial em faixa ampla
    saldo_inicial = random.uniform(100, 50000)

    # Localização principal
    localizacao_comum = f"{fake.city()}, {fake.current_country()}"

    # Canal/dispositivo preferencial
    canal_preferencial = random.choice(CANAIS)
    dispositivo_preferencial = random.choice(['web', 'mobile'])

    # Renda mensal na faixa 1.000 a 55.000 e renda anual
    renda_mensal = round(random.uniform(1000, 55000), 2)
    renda_anual = round(renda_mensal * 12, 2)

    # Profissão
    profissao = fake.job()

    usuarios_dict[user_id] = {
        "tipo_conta": tipo,
        "saldo": saldo_inicial,
        "localizacao_comum": localizacao_comum,
        "canal_pref": canal_preferencial,
        "disp_pref": dispositivo_preferencial,
        "renda_mensal": renda_mensal,
        "renda_anual": renda_anual,
        "profissao": profissao
    }

# ==============================
# 5. SIMULAÇÃO DE TRANSAÇÕES
# ==============================
lista_registros = []

for i in range(1, NUM_TRANSACOES + 1):
    id_transacao = i

    # Escolher um usuário aleatório
    id_usuario = random.randint(1, NUM_USUARIOS)
    user_info = usuarios_dict[id_usuario]

    tipo_conta = user_info["tipo_conta"]
    saldo_atual = user_info["saldo"]

    # Gerar data/hora
    data_transacao = random_date(START_DATE, END_DATE)
    horario_transacao = data_transacao.hour
    dia_semana_transacao = DIAS_SEMANA[data_transacao.weekday()]

    # Escolher tipo de transação
    tipo_transacao = random.choice(TIPOS_TRANSACAO)

    # Geração do valor da transacao com base no tipo_conta
    vmin, vmax = VALOR_FAIXAS[tipo_conta]
    valor_transacao = random.uniform(vmin, vmax)

    # Merchant e categoria
    merchant_name = fake.company()
    categoria_merchant = random.choice(CATEGORIAS_MERCHANT)

    # Localização da transação
    if random.random() < 0.8:
        localizacao_transacao = user_info["localizacao_comum"]
    else:
        localizacao_transacao = f"{fake.city()}, {fake.country()}"

    # Canal e dispositivo
    if random.random() < 0.7:
        canal_transacao = user_info["canal_pref"]
    else:
        canal_transacao = random.choice(CANAIS)

    if random.random() < 0.7:
        dispositivo_transacao = user_info["disp_pref"]
    else:
        dispositivo_transacao = random.choice(DISPOSITIVOS)

    # Saldo antes
    saldo_conta_antes = saldo_atual

    # Cálculo do saldo depois
    if tipo_transacao == 'deposito':
        saldo_conta_depois = saldo_conta_antes + valor_transacao
    else:
        saldo_conta_depois = saldo_conta_antes - valor_transacao

    # Atualizar saldo
    usuarios_dict[id_usuario]["saldo"] = saldo_conta_depois

    # Status inicial
    status_transacao = 'aprovada'
    motivo_recusa = None

    # IP e user-agent
    endereco_ip = fake.ipv4_public()
    navegador_usuario = fake.user_agent()

    # Lógica de fraude
    flag_fraude = False
    score_fraude = 0.0

    # Regra 1: valor muito alto
    limiar_valor = vmax * 0.95
    if valor_transacao > limiar_valor:
        score_fraude += 0.4

    # Regra 2: localização incomum
    if localizacao_transacao != user_info["localizacao_comum"]:
        score_fraude += 0.3

    # Regra 3: dispositivo/canal "suspeitos"
    if dispositivo_transacao == 'atm' and canal_transacao == 'online':
        score_fraude += 0.3
    if dispositivo_transacao == 'terminal_pos' and canal_transacao == 'online':
        score_fraude += 0.3

    # Se ambos diferentes do preferencial
    if (dispositivo_transacao != user_info["disp_pref"]) and (canal_transacao != user_info["canal_pref"]):
        score_fraude += 0.2

    # Regra 4: horário/dia suspeito
    if horario_transacao < 6 or dia_semana_transacao == 'Dom':
        score_fraude += 0.2

    # Saldo negativo => rejeita
    if saldo_conta_depois < 0:
        status_transacao = 'rejeitada'
        motivo_recusa = 'Saldo Insuficiente'

    # Verifica pontuação de fraude
    if score_fraude > 0.6:
        flag_fraude = True
        if status_transacao != 'rejeitada':  # caso ainda não esteja rejeitada
            if score_fraude < 0.8:
                status_transacao = 'pendente'
            else:
                status_transacao = 'rejeitada'
                motivo_recusa = 'Fraude Suspeita'

    # Log de evento
    log_evento = (
        f"Transação {id_transacao} | Usuário {id_usuario} | "
        f"Tipo de conta: {tipo_conta} | {tipo_transacao.upper()} de {valor_transacao:.2f} | "
        f"Localização: {localizacao_transacao} | Dispositivo: {dispositivo_transacao}, "
        f"Canal: {canal_transacao} | Status: {status_transacao} | "
        f"{'Fraude suspeita' if flag_fraude else 'Sem alerta de fraude'} | "
        f"Score de fraude: {score_fraude:.2f}"
    )

    # Montar registro
    registro = {
        "id_transacao": id_transacao,
        "id_usuario": id_usuario,
        "tipo_conta": tipo_conta,
        "data_transacao": data_transacao,
        "valor_transacao": valor_transacao,
        "tipo_transacao": tipo_transacao,
        "merchant": merchant_name,
        "categoria_merchant": categoria_merchant,
        "localizacao_transacao": localizacao_transacao,
        "dispositivo_transacao": dispositivo_transacao,
        "saldo_conta_antes": saldo_conta_antes,
        "saldo_conta_depois": saldo_conta_depois,
        "canal_transacao": canal_transacao,
        "status_transacao": status_transacao,
        "motivo_recusa": motivo_recusa,
        "endereco_ip": endereco_ip,
        "navegador_usuario": navegador_usuario,
        "flag_fraude": flag_fraude,
        "score_fraude": score_fraude,
        "log_evento": log_evento,
        "horario_transacao": horario_transacao,
        "dia_semana_transacao": dia_semana_transacao,
        # Colunas de perfil do usuário
        "renda_mensal": user_info["renda_mensal"],
        "renda_anual": user_info["renda_anual"],
        "profissao": user_info["profissao"]
    }

    lista_registros.append(registro)

# ==============================
# 6. CRIAÇÃO DO DATAFRAME
# ==============================
df = pd.DataFrame(lista_registros)

# ==============================
# 7. SIMULAÇÃO DE MISSING DATA
# ==============================
p_merchant_missing = 0.10
p_categoria_missing = 0.15
p_local_missing = 0.10
p_ip_missing = 0.05
p_navegador_missing = 0.05

df['merchant'] = df['merchant'].apply(lambda x: maybe_missing(x, p_merchant_missing))
df['categoria_merchant'] = df['categoria_merchant'].apply(lambda x: maybe_missing(x, p_categoria_missing))
df['localizacao_transacao'] = df['localizacao_transacao'].apply(lambda x: maybe_missing(x, p_local_missing))
df['endereco_ip'] = df['endereco_ip'].apply(lambda x: maybe_missing(x, p_ip_missing))
df['navegador_usuario'] = df['navegador_usuario'].apply(lambda x: maybe_missing(x, p_navegador_missing))

# Exemplo de verificação
print(df.head(10))
print("Tamanho final do dataset:", len(df))

# Se quiser salvar em CSV
df.to_csv("dataset_financeiro_simulado.csv", index=False, encoding='utf-8')
