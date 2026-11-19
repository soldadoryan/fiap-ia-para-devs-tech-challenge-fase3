import html
import os
import re
import unicodedata
import warnings
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal, TypedDict

warnings.filterwarnings("ignore", message=r".*langchain-community.*is being sunset", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=r"Normalizing L2 is not applicable", category=UserWarning,
                        module=r"langchain_community\.vectorstores\.faiss")

from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings
from pydantic import BaseModel

from dadosMock import DATAS_DISPONIVEIS, EXAMES_MARCADOS, PACIENTES, TIPOS_EXAME

RAIZ = Path(__file__).resolve().parent
DIR_RAW = RAIZ / "data" / "raw"
DIR_PROCESSED = RAIZ / "data" / "processed"
DIR_FAISS = Path(os.getenv("FAISS_DIR", RAIZ / "data" / "faiss"))
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
MODELO_LLM = "llama3.1:8b"
MODELO_EMBEDDING = "bge-m3"
NUM_CTX = 8192
TAMANHO_CHUNK, OVERLAP_CHUNK = 1200, 150
TAMANHO_MINIMO_RESPOSTA = 60
TAMANHO_LOTE = 64
K_BUSCA, K_FINAL = 10, 4
DISTANCIA_MAXIMA = 0.38
DISTANCIA_CONFIANCA = 0.30
MAX_REFORMULACOES = 2
PASTAS_SEM_RESPOSTA = {"10_MPlus_ADAM_QA", "11_MPlusDrugs_QA", "12_MPlusHerbsSupplements_QA"}
MAX_TENTATIVAS_ESCOLHA = 3
MAX_DATAS_OFERECIDAS = 6
DIAS_SEMANA = ["segunda", "terça", "quarta", "quinta", "sexta", "sábado", "domingo"]



def acumularCaminho(atual: list[str] | None, novo: list[str] | None) -> list[str]:
    atual, novo = atual or [], novo or []
    if atual and novo[:len(atual)] == atual:
        return novo
    return atual + novo

class EstadoAgente(TypedDict, total=False):
    pergunta: str
    forcar_indexacao: bool
    limite: int | None
    pubmedqa_completo: bool
    ler_pastas_vazias: bool
    interativo: bool
    base_pronta: bool
    estatisticas: dict
    categoria: Literal["medica", "prontuario", "exames", "emergencia", "fora_escopo"]
    intencao_exame: Literal["consultar", "marcar", "remarcar_ou_cancelar", "resultado"] | None
    foco_prontuario: Literal["resumo", "alergias", "medicamentos", "finalidade_medicamentos", "consultas",
                             "outro"] | None
    paciente_nome_informado: str | None
    paciente_id_informado: int | None
    paciente_cpf_informado: str | None
    consulta_busca: str
    consultas_tentadas: list[str]
    pergunta_reformulada: str
    documentos: list[Document]
    pontuacoes: list[float]
    contexto: str
    avaliacao: Literal["bom", "fraco", "insuficiente"]
    motivo_avaliacao: str
    tentativas: int
    situacao_paciente: Literal["encontrado", "nao_encontrado", "ambiguo", "nao_informado"]
    paciente_id: int
    paciente_nome: str
    prontuario: str
    tipo_exame: str | None
    exames_marcados: list[dict]
    pode_marcar: bool
    datas_disponiveis: list[dict]
    data_escolhida: dict
    situacao_escolha: Literal["escolhida", "cancelada", "nova_pergunta", "esgotada"]
    resposta: str
    fontes: list[dict]
    valida: bool
    problemas_validacao: list[str]
    caminho: Annotated[list[str], acumularCaminho]


class EstadoIndexacao(TypedDict, total=False):
    limite: int | None
    pubmedqa_completo: bool
    ler_pastas_vazias: bool
    base_pronta: bool
    documentos_brutos: list[Document]
    documentos_tratados: list[Document]
    estatisticas: dict
    caminho: Annotated[list[str], acumularCaminho]


class SaidaIndexacao(TypedDict, total=False):
    base_pronta: bool
    estatisticas: dict
    caminho: Annotated[list[str], acumularCaminho]


class Classificacao(BaseModel):
    categoria: Literal["medica", "prontuario", "exames", "emergencia", "fora_escopo"]
    intencao_exame: Literal["consultar", "marcar", "remarcar_ou_cancelar", "resultado"] | None = None
    foco_prontuario: Literal["resumo", "alergias", "medicamentos", "finalidade_medicamentos", "consultas",
                             "outro"] | None = None
    paciente_nome: str | None = None
    paciente_id: int | None = None
    paciente_cpf: str | None = None
    tipo_exame: str | None = None
    consulta_en: str = ""


class InterpretacaoEscolha(BaseModel):
    acao: Literal["escolher", "cancelar", "nova_pergunta"]
    opcao: int | None = None


class AvaliacaoContexto(BaseModel):
    suficiente: bool
    motivo: str


class Reformulacao(BaseModel):
    consulta_en: str


class Validacao(BaseModel):
    em_portugues: bool
    fundamentada: bool
    problemas: list[str]


@lru_cache
def obterEmbeddings() -> OllamaEmbeddings:
    return OllamaEmbeddings(model=MODELO_EMBEDDING, base_url=OLLAMA_URL)


@lru_cache
def obterVectorStore() -> FAISS:
    return FAISS.load_local(str(DIR_FAISS), obterEmbeddings(), allow_dangerous_deserialization=True,
                            distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT, normalize_L2=True)


def invalidarVectorStore() -> None:
    obterVectorStore.cache_clear()


def distanciaDoScore(score: float) -> float:
    return 1.0 - float(score)


@lru_cache
def obterLLM(temperatura: float = 0.0) -> ChatOllama:
    return ChatOllama(model=MODELO_LLM, base_url=OLLAMA_URL, temperature=temperatura,
                      num_ctx=NUM_CTX, keep_alive="30m")


def obterLLMEstruturado(schema: type[BaseModel]):
    return obterLLM(0.0).with_structured_output(schema, method="json_schema")


def contarVetores() -> int:
    if not (DIR_FAISS / "index.faiss").is_file():
        return 0
    return obterVectorStore().index.ntotal


def limparTexto(texto: str) -> str:
    texto = html.unescape(texto or "")
    texto = re.sub(r"<[^>]+>", " ", texto)
    texto = unicodedata.normalize("NFC", texto)
    texto = "".join(c for c in texto if c == "\n" or unicodedata.category(c)[0] != "C" or c in "\t\r")
    texto = re.sub(r"\s{2,}-\s+", "\n- ", texto)
    texto = re.sub(r"[^\S\n]*\n\s*", "\n", texto)
    texto = re.sub(r"[^\S\n]+", " ", texto)
    return texto.strip()


def removerPerguntaInicial(resposta: str) -> str:
    return re.sub(r"^[^.?!]{5,150}\?\s+", "", resposta, count=1)


def removerAcentos(texto: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", texto) if unicodedata.category(c) != "Mn")


STOPWORDS_EN = {"the", "of", "and", "is", "are", "to", "in", "a", "that", "for", "with", "as", "on",
                "by", "or", "can", "be", "this", "it", "from", "at", "an", "which", "may", "was",
                "were", "has", "have", "not", "their", "these", "also", "you", "your"}


def normalizarConsulta(texto: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", removerAcentos(texto.lower())))


def pareceIngles(texto: str) -> bool:
    palavras = re.findall(r"[a-zA-Z']+", texto.lower())
    if not palavras:
        return False
    return sum(p in STOPWORDS_EN for p in palavras) / len(palavras) > 0.30


def formatarContexto(documentos: list[Document]) -> str:
    blocos = []
    for n, doc in enumerate(documentos, 1):
        m = doc.metadata
        texto = doc.page_content
        if m.get("dataset") == "pubmedqa":
            origem = "PubMedQA, resultado de um estudo, não é recomendação clínica"
            texto = texto.replace("\nDecisão final:", "\nDecisão final do estudo:")
        else:
            origem = f"MedQuAD/{m.get('fonte', '')}"
        blocos.append(f"[{n}] ({origem} - {m.get('titulo', '')})\n{texto}")
    return "\n\n".join(blocos)


def formatarFontes(fontes: list[dict]) -> str:
    linhas = [f"[{f['n']}] {f['titulo']} ({f['dataset']}) {f['url']}".rstrip() for f in fontes]
    return "Fontes:\n" + "\n".join(linhas)


def registrarPasso(no: str, detalhe: str = "") -> list[str]:
    print(f"[{no}] {detalhe}", flush=True)
    return [no]


def logarAcao(no: str, mensagem: str) -> None:
    print(f"[{no}] {mensagem}", flush=True)


SYSTEM_PROMPT = """Você é o Assistente Virtual de Informação em Saúde de um hospital. Seu público são pacientes, acompanhantes e profissionais que fazem perguntas em português.

REGRAS OBRIGATÓRIAS:
1. Responda SEMPRE em português do Brasil, mesmo que o contexto esteja em inglês. Traduza os termos técnicos e, quando útil, mantenha o termo original entre parênteses.
2. Use SOMENTE as informações do CONTEXTO fornecido. Não use conhecimento próprio. Se o contexto não responder à pergunta, diga claramente: "Não encontrei essa informação na base de conhecimento consultada."
3. Cite as fontes com o número entre colchetes, como [1] ou [2], logo após cada afirmação que vier delas.
4. Não faça diagnóstico, não prescreva medicamentos e não indique doses.
5. Se a pergunta descrever sinais de urgência (dor no peito, falta de ar, desmaio, sangramento intenso, sinais de AVC, pensamentos suicidas), oriente a procurar imediatamente o pronto-socorro ou ligar para o SAMU (192).
6. Seja claro e objetivo: no máximo 3 parágrafos curtos ou uma lista de tópicos.
7. Termine sempre com: "Esta informação é educativa e não substitui a avaliação de um profissional de saúde.\""""

PROMPT_RESPOSTA = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", """CONTEXTO (trechos recuperados, possivelmente em inglês):
{contexto}

SITUAÇÃO DO CONTEXTO: {situacao_contexto}

PERGUNTA DO USUÁRIO:
{pergunta}

Escreva a resposta em português do Brasil seguindo todas as regras. Use apenas o CONTEXTO e cite as fontes como [n].
Se usar um trecho do PubMedQA, deixe claro que é o resultado de um estudo e informe a "Decisão final do estudo" (sim, não ou talvez) que aparece no trecho."""),
])

PROMPT_CLASSIFICACAO = ChatPromptTemplate.from_messages([
    ("system", """Você é o roteador de um assistente hospitalar. Leia a mensagem do usuário e preencha os campos pedidos.

CATEGORIA ("categoria"), em ordem de prioridade:
1. "emergencia": sinais de risco imediato descritos como algo que está acontecendo (com a pessoa ou com alguém perto): dor no peito, falta de ar, desmaio, convulsão, sangramento intenso, sinais de AVC, overdose ou ingestão exagerada de remédio, envenenamento, ideação suicida. Vale mesmo que a mensagem cite um paciente ou um exame.
2. "prontuario": pede dados do prontuário de um paciente específico (resumo, alergias, medicamentos em uso, para que servem os remédios dele, últimas consultas, tipo sanguíneo, idade, doenças).
3. "exames": trata dos exames agendados de um paciente específico (ver, marcar, remarcar, cancelar ou saber o resultado).
4. "medica": dúvida geral de saúde sem paciente específico: doenças, sintomas, exames em geral (o que é, para que serve, preparo), tratamentos, medicamentos, pesquisas. "O que é hemograma?" é "medica".
5. "fora_escopo": qualquer outro assunto (geografia, piadas, esportes, programação, conversa casual).
Se a mensagem cita um paciente específico (nome, número ou CPF) e não é emergência, a categoria é "prontuario" ou "exames", nunca "medica".

INTENÇÃO DO EXAME ("intencao_exame"): obrigatória quando a categoria for "exames" (nunca null); nas outras categorias, null:
- "consultar": ver quais exames o paciente tem marcados, quando e onde.
- "marcar": marcar ou agendar um novo exame.
- "remarcar_ou_cancelar": remarcar, adiar, antecipar, desmarcar ou cancelar um exame.
- "resultado": saber o resultado de um exame feito.

FOCO DO PRONTUÁRIO ("foco_prontuario"), só quando a categoria for "prontuario"; nas outras, null:
- "resumo": prontuário completo ou visão geral do paciente.
- "alergias": alergias.
- "medicamentos": quais medicamentos o paciente usa (sem perguntar para que servem).
- "finalidade_medicamentos": para que servem, como agem ou quais os efeitos dos remédios do paciente.
- "consultas": consultas passadas ou marcadas.
- "outro": qualquer outro dado (tipo sanguíneo, idade, CPF, doenças crônicas).

PACIENTE (preencha sempre que a mensagem citar uma pessoa, mesmo nas categorias "prontuario" e "exames" com pedido curto; se não citar, null):
- "paciente_nome": o nome, ou parte dele, do jeito que aparece na mensagem (ex.: "Kobayashi", "Joao"). Nomes de exames, doenças e remédios não são nome de paciente.
- "paciente_id": o número do paciente, em algarismos ("paciente numero dois" -> 2).
- "paciente_cpf": o CPF, se aparecer.

TIPO DO EXAME ("tipo_exame"), só quando a categoria for "exames" e a mensagem citar um exame: devolva EXATAMENTE um destes nomes ou null se não citar ou se o exame não estiver na lista:
{tipos_exame}

CONSULTA EM INGLÊS ("consulta_en"): só para a categoria "medica", a pergunta traduzida como consulta de busca objetiva em inglês (ex.: "symptoms of type 2 diabetes"). Nas outras categorias, deixe vazio.

EXEMPLOS (mensagem -> classificação):
"Quais são os sintomas do diabetes?" -> categoria=medica, consulta_en="symptoms of diabetes"
"O que é hemograma?" -> categoria=medica, consulta_en="what is a complete blood count"
"Estou com dor no peito e falta de ar agora" -> categoria=emergencia
"Tomei 30 comprimidos de uma vez" -> categoria=emergencia
"Qual a capital da França?" -> categoria=fora_escopo
"Qual o prontuário da Ana Beatriz Kobayashi?" -> categoria=prontuario, foco_prontuario=resumo, paciente_nome="Ana Beatriz Kobayashi"
"Quais remédios o paciente 5 usa e para que servem?" -> categoria=prontuario, foco_prontuario=finalidade_medicamentos, paciente_id=5
"Prontuário do paciente numero dois" -> categoria=prontuario, foco_prontuario=resumo, paciente_id=2
"O paciente 3 tem consulta marcada?" -> categoria=prontuario, foco_prontuario=consultas, paciente_id=3
"Quais exames do paciente 5 estão marcados?" -> categoria=exames, intencao_exame=consultar, paciente_id=5
"Exames da Kobayashi" -> categoria=exames, intencao_exame=consultar, paciente_nome="Kobayashi"
"Marcar hemograma para o paciente 2" -> categoria=exames, intencao_exame=marcar, paciente_id=2, tipo_exame="Hemograma completo"
"Quero marcar um exame para a Maria Fernanda" -> categoria=exames, intencao_exame=marcar, paciente_nome="Maria Fernanda", tipo_exame=null
"Remarcar o exame da Maria Aparecida" -> categoria=exames, intencao_exame=remarcar_ou_cancelar, paciente_nome="Maria Aparecida"
"Quero ver os exames do Joao" -> categoria=exames, intencao_exame=consultar, paciente_nome="Joao"
"Cancelar o exame de raio x do paciente 5" -> categoria=exames, intencao_exame=remarcar_ou_cancelar, paciente_id=5, tipo_exame="Raio X de tórax"
"Qual o resultado do hemograma do paciente 1?" -> categoria=exames, intencao_exame=resultado, paciente_id=1, tipo_exame="Hemograma completo\""""),
    ("human", "Mensagem: {pergunta}"),
])

PROMPT_ESCOLHA = ChatPromptTemplate.from_messages([
    ("system", """O usuário está escolhendo uma data para marcar um exame. As opções numeradas são:
{opcoes}

Interprete o texto digitado e preencha:
- "acao": "escolher" se ele indica uma das opções (pelo número, por ordinal como "a segunda" ou "o primeiro horário", ou pela data/hora); "cancelar" se desiste ou não quer mais marcar; "nova_pergunta" se faz uma pergunta ou pedido sobre outro assunto, sem relação com a escolha.
- "opcao": o número da opção escolhida quando a ação for "escolher"; senão, null. Se o texto não indicar nenhuma opção com clareza, use "escolher" com opcao null.
Ordinais (primeiro/primeira = 1, segundo/segunda = 2, terceiro/terceira = 3, e assim por diante) indicam a POSIÇÃO na lista, mesmo que um dia da semana tenha o mesmo nome (a "segunda" opção é a 2, não a segunda-feira). "nova_pergunta" só vale para uma frase clara sobre outro assunto; texto sem sentido ou vago é "escolher" com opcao null.

Exemplos:
"a segunda" -> acao=escolher, opcao=2
"quero a opção 3" -> acao=escolher, opcao=3
"cancela" -> acao=cancelar, opcao=null
"não quero mais" -> acao=cancelar, opcao=null
"quais exames da Maria?" -> acao=nova_pergunta, opcao=null
"a segunda" -> acao=escolher, opcao=2
"asdf" -> acao=escolher, opcao=null
"talvez" -> acao=escolher, opcao=null"""),
    ("human", "Texto digitado: {escolha}"),
])

PROMPT_PRONTUARIO = ChatPromptTemplate.from_messages([
    ("system", """Você é o assistente de prontuários de um hospital e responde, em português do Brasil, perguntas sobre o prontuário de um paciente.

REGRAS OBRIGATÓRIAS:
1. Use SOMENTE os dados do PRONTUÁRIO fornecido. Não invente alergias, doenças, medicamentos, doses, datas ou resultados. Não use conhecimento próprio para completar nada.
Escreva cada medicamento como está no prontuário (nome, dose e frequência), sem explicar para que serve, como age ou os efeitos, porque isso não está no prontuário.
2. O prontuário tem estas seções: dados básicos (CPF, nascimento, idade, tipo sanguíneo), Alergias, Condições crônicas, Medicamentos em uso e Últimas consultas. Perguntas sobre essas seções SEMPRE têm resposta: leia a seção e responda com o que está nela.
3. Quando uma seção estiver como "nenhuma registrada", responda que não há registro dessa informação no prontuário (por exemplo, "Não há alergias registradas para {nome}."), e não que o paciente com certeza não tem aquilo.
Se a seção tiver itens, mas nenhum do tipo perguntado (por exemplo, a pergunta é sobre alergia a medicamentos e só constam látex e frutos do mar), cite os itens que constam e diga que nenhum deles é do tipo perguntado.
Somente se a pergunta pedir algo que não existe em nenhuma seção (por exemplo, resultado de um exame, endereço ou plano de saúde), diga: "Essa informação não consta no prontuário de {nome}."
4. Se a pergunta for específica (por exemplo, alergias, medicamentos, tipo sanguíneo ou a última consulta), responda só o que foi perguntado, em uma ou duas frases.
5. Se a pergunta for genérica (por exemplo, "mostre o prontuário" ou "como está o paciente"), faça um resumo em tópicos: dados básicos, alergias, condições crônicas, medicamentos em uso e últimas consultas.
6. Não faça diagnóstico, não sugira tratamento e não altere doses.
7. Termine sempre com: "Esta informação é educativa e não substitui a avaliação de um profissional de saúde.\""""),
    ("human", """PRONTUÁRIO:
{prontuario}

PERGUNTA:
{pergunta}

Responda em português do Brasil seguindo todas as regras."""),
])

PROMPT_AVALIACAO = ChatPromptTemplate.from_messages([
    ("system", """Você avalia se trechos recuperados de uma base médica são úteis para responder a uma pergunta (em português). Os trechos costumam estar em inglês; isso é normal e não é motivo para recusar.
- "suficiente" = true quando AO MENOS UM trecho responde à pergunta ou contém a informação pedida, mesmo que seja um único trecho, mesmo que esteja em inglês e mesmo que não responda a tudo.
- "suficiente" = false somente quando nenhum trecho tem relação útil com o assunto da pergunta (por exemplo, falam de outra doença, outro medicamento ou outro tema).
Trechos marcados como PubMedQA são resumos de estudos que já trazem a conclusão (contexto, resultados e "Decisão final do estudo"). Se o tema do estudo é o mesmo da pergunta, o trecho é suficiente.
Em "motivo", explique em uma frase curta em português, citando o número do trecho útil quando houver."""),
    ("human", """Pergunta: {pergunta}

Trechos:
{contexto}"""),
])

PROMPT_REFORMULACAO = ChatPromptTemplate.from_messages([
    ("system", """Você reescreve consultas de busca para uma base de conhecimento médica em inglês (MedQuAD e resumos do PubMed).
As buscas anteriores não trouxeram trechos relevantes. Gere UMA nova consulta em inglês, curta (até 12 palavras), DIFERENTE de todas as consultas já tentadas: use sinônimos, o nome técnico da condição, um termo mais geral ou um termo mais específico. Não repita nem apenas reordene uma consulta já tentada."""),
    ("human", """Pergunta original (português): {pergunta}
Consultas já tentadas (não repita):
{consultas_tentadas}
Motivo da falha: {motivo}"""),
])

PROMPT_VALIDACAO = ChatPromptTemplate.from_messages([
    ("system", """Você é um revisor de segurança de respostas de um assistente de saúde.
Verifique a RESPOSTA em relação ao CONTEXTO:
- "em_portugues": true se a resposta estiver escrita em português (termos técnicos em inglês entre parênteses são aceitos).
- "fundamentada": true se todas as afirmações médicas da resposta estiverem apoiadas no CONTEXTO; false se houver informação que não aparece no CONTEXTO.
- "problemas": lista curta, em português, dos problemas encontrados (vazia se nenhum)."""),
    ("human", """CONTEXTO:
{contexto}

RESPOSTA:
{resposta}"""),
])

PROMPT_TRADUCAO = ChatPromptTemplate.from_messages([
    ("human", """Traduza o texto abaixo para português do Brasil, mantendo exatamente o conteúdo, a formatação e as citações [n]. Responda apenas com o texto traduzido.

{resposta}"""),
])

MENSAGEM_EMERGENCIA = """⚠ O que você descreveu pode ser uma emergência médica.
Procure AGORA o pronto-socorro mais próximo ou ligue para o SAMU (192). Em caso de acidente ou risco à vida, os Bombeiros atendem no 193.
Se houver pensamentos de se ferir, ligue para o CVV (188, 24 horas).
Não espere a melhora dos sintomas e, se possível, não vá sozinho(a).
Este assistente não realiza atendimento de urgência."""

MENSAGEM_FORA_ESCOPO = ("Sou o assistente de saúde do hospital. Posso responder dúvidas sobre doenças, sintomas, "
                        "exames e tratamentos, consultar o prontuário de pacientes cadastrados e consultar ou marcar "
                        "exames. Por favor, reformule sua pergunta dentro desses temas.")

MENSAGEM_PACIENTE_NAO_ENCONTRADO = ("Não encontrei esse paciente no cadastro. Informe o nome, o número em "
                                    "algarismos (ex.: paciente 3) ou o CPF de um destes pacientes:\n{lista}")

MENSAGEM_PACIENTE_AMBIGUO = ("Encontrei mais de um paciente com esse nome. Diga o nome completo ou o número "
                             "do paciente:\n{lista}")

MENSAGEM_PACIENTE_NAO_INFORMADO = ("Não identifiquei de qual paciente você está falando. Informe o nome ou o número "
                                   "do paciente (ex.: paciente 3).")

MENSAGEM_FINALIDADE_REMEDIO = ("O prontuário não informa para que serve cada medicamento. Para essa explicação, "
                               "consulte o médico ou o farmacêutico.")

MENSAGEM_SEM_EXAMES = "{nome} (paciente {id}) não tem exames marcados."

MENSAGEM_EXAMES_MARCADOS = "Exames marcados para {nome} (paciente {id}):\n{lista}"

MENSAGEM_TIPO_EXAME_AUSENTE = "Para marcar, diga qual exame. Tipos aceitos:\n{lista}"

MENSAGEM_EXAME_JA_MARCADO = ("{nome} já tem {tipo} marcado para {quando}. Para remarcar, procure a central "
                             "de agendamento.")

MENSAGEM_REMARCACAO = ("Remarcar, adiar ou cancelar um exame não é feito por este assistente. Procure a central "
                       "de agendamento.")

MENSAGEM_SEM_RESULTADOS = ("Resultados de exames não ficam disponíveis neste assistente. Consulte o médico que "
                           "pediu o exame.")

MENSAGEM_SEM_DATAS = "Não há horários livres para {tipo} nas próximas semanas. Procure a central de agendamento."

MENSAGEM_ESCOLHA_DATA = 'Datas disponíveis para {tipo} ({nome}). Diga qual opção você quer ou que deseja cancelar:'

MENSAGEM_ESCOLHA_INVALIDA = ('A opção "{escolha}" não está na lista. Escolha uma opção de 1 a {total}. '
                             'Tentativas restantes: {restantes}.')

MENSAGEM_EXAME_MARCADO = ("Exame marcado.\nPaciente: {nome} (paciente {id})\nExame: {tipo}\nData: {quando}\n"
                          "Local: {local}\nPreparo: {preparo}")

MENSAGEM_MARCACAO_CANCELADA = "Marcação cancelada. Nenhum exame foi marcado para {nome}."

MENSAGEM_NOVA_PERGUNTA = "Você digitou uma nova pergunta em vez do número da data. Ela será respondida a seguir."

MENSAGEM_TENTATIVAS_ESGOTADAS = "Nenhuma opção válida depois de {maximo} tentativas. A marcação foi cancelada."

MENSAGEM_HORARIO_OCUPADO = "Esse horário acabou de ser ocupado. Peça a marcação de novo para ver as datas livres."

MENSAGEM_DATAS_SEM_ESCOLHA = ("Datas disponíveis para {tipo} ({nome}):\n{lista}\n\nA marcação precisa da sua "
                              "escolha. Rode python main.py (modo interativo) e repita o pedido para escolher a data. "
                              "Nenhum exame foi marcado.")

MENSAGEM_SEM_CONTEXTO = ("Não encontrei informação confiável sobre isso na base de conhecimento consultada "
                         "(MedQuAD e PubMedQA). Recomendo conversar com um médico ou farmacêutico.")

AVISO_LEGAL = "Esta informação é educativa e não substitui a avaliação de um profissional de saúde."


def somenteDigitos(texto: str) -> str:
    return "".join(c for c in texto or "" if c.isdigit())


def obterPaciente(paciente_id: int) -> dict | None:
    return next((p for p in PACIENTES if p["id"] == paciente_id), None)


def listarPacientes(pacientes: list[dict]) -> str:
    return "\n".join(f"- Paciente {p['id']}: {p['nome']}" for p in pacientes)


def calcularIdade(nascimento: str) -> int:
    n, hoje = date.fromisoformat(nascimento), date.today()
    return hoje.year - n.year - ((hoje.month, hoje.day) < (n.month, n.day))


def formatarData(data: str) -> str:
    return f"{date.fromisoformat(data):%d/%m/%Y}"


def formatarHorario(data: str, hora: str) -> str:
    return f"{DIAS_SEMANA[date.fromisoformat(data).weekday()]}, {formatarData(data)} às {hora}"


def _listarOuNenhuma(itens: list[str]) -> str:
    return "; ".join(itens) if itens else "nenhuma registrada"


def formatarProntuario(p: dict) -> str:
    linhas = [f"Paciente {p['id']} - {p['nome']}",
              f"CPF: {p['cpf']} | Nascimento: {formatarData(p['data_nascimento'])} "
              f"({calcularIdade(p['data_nascimento'])} anos) | Tipo sanguíneo: {p['tipo_sanguineo']}",
              f"Alergias: {_listarOuNenhuma(p['alergias'])}",
              f"Condições crônicas: {_listarOuNenhuma(p['condicoes_cronicas'])}",
              f"Medicamentos em uso: {_listarOuNenhuma(p['medicamentos'])}",
              "Últimas consultas:"]
    linhas += [f"- {formatarData(c['data'])}, {c['especialidade']} ({c['medico']}): {c['resumo']}"
               for c in sorted(p["consultas"], key=lambda c: c["data"], reverse=True)] or ["- nenhuma registrada"]
    return "\n".join(linhas)


def examesDoPaciente(paciente_id: int) -> list[dict]:
    return sorted(EXAMES_MARCADOS.get(paciente_id, []), key=lambda e: (e["data"], e["hora"]))


def horariosLivres(paciente_id: int) -> list[dict]:
    ocupados = {(e["data"], e["hora"]) for e in EXAMES_MARCADOS.get(paciente_id, [])}
    return [h for h in DATAS_DISPONIVEIS
            if h["livre"] and (h["data"], h["hora"]) not in ocupados][:MAX_DATAS_OFERECIDAS]


def registrarExame(paciente_id: int, tipo: str, horario: dict) -> dict | None:
    vaga = next((h for h in DATAS_DISPONIVEIS
                 if h["data"] == horario.get("data") and h["hora"] == horario.get("hora") and h["livre"]), None)
    if vaga is None:
        return None
    vaga["livre"] = False
    exame = {"tipo": tipo, "data": vaga["data"], "hora": vaga["hora"], "local": TIPOS_EXAME[tipo]["local"]}
    EXAMES_MARCADOS.setdefault(paciente_id, []).append(exame)
    return exame


def textoNormalizado(texto: str | None) -> str:
    return " ".join(removerAcentos(texto or "").lower().split())


def buscarPacientes(paciente_id: int | None, cpf: str | None, nome: str | None) -> tuple[list[dict], str]:
    if paciente_id is not None:
        achados = [p for p in PACIENTES if p["id"] == paciente_id]
        if achados:
            return achados, "id"
    if somenteDigitos(cpf):
        achados = [p for p in PACIENTES if somenteDigitos(p["cpf"]) == somenteDigitos(cpf)]
        if achados:
            return achados, "cpf"
    termos = textoNormalizado(nome).split()
    if termos:
        achados = [p for p in PACIENTES if all(t in textoNormalizado(p["nome"]) for t in termos)]
        if achados:
            return achados, "nome"
    return [], ""


def identificarPaciente(estado: EstadoAgente) -> dict:
    paciente_id, cpf, nome = (estado.get("paciente_id_informado"), estado.get("paciente_cpf_informado"),
                              estado.get("paciente_nome_informado"))
    informado = f'id={paciente_id} cpf="{cpf or ""}" nome="{nome or ""}"'
    logarAcao("identificarPaciente", "Buscando no cadastro em memória com os dados extraídos pelo classificador...")
    if paciente_id is None and not somenteDigitos(cpf) and not textoNormalizado(nome):
        return {"situacao_paciente": "nao_informado", "resposta": MENSAGEM_PACIENTE_NAO_INFORMADO,
                "caminho": registrarPasso("identificarPaciente", "nenhum dado de paciente informado")}
    achados, origem = buscarPacientes(paciente_id, cpf, nome)
    if len(achados) == 1:
        p = achados[0]
        return {"situacao_paciente": "encontrado", "paciente_id": p["id"], "paciente_nome": p["nome"],
                "caminho": registrarPasso("identificarPaciente",
                                          f"paciente {p['id']} ({p['nome']}) por {origem} | {informado}")}
    if achados:
        return {"situacao_paciente": "ambiguo",
                "resposta": MENSAGEM_PACIENTE_AMBIGUO.format(lista=listarPacientes(achados)),
                "caminho": registrarPasso("identificarPaciente",
                                          f"ambiguo, {len(achados)} pacientes por {origem} | {informado}")}
    return {"situacao_paciente": "nao_encontrado",
            "resposta": MENSAGEM_PACIENTE_NAO_ENCONTRADO.format(lista=listarPacientes(PACIENTES)),
            "caminho": registrarPasso("identificarPaciente", f"nao encontrado | {informado}")}
