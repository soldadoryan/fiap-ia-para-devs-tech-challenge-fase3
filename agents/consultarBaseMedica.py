import re

from langgraph.graph import END, START, StateGraph

from funcoesPadrao import (AVISO_LEGAL, DISTANCIA_CONFIANCA, DISTANCIA_MAXIMA, K_BUSCA, K_FINAL, MAX_REFORMULACOES,
                           MENSAGEM_SEM_CONTEXTO, PROMPT_AVALIACAO, PROMPT_REFORMULACAO, PROMPT_RESPOSTA,
                           PROMPT_TRADUCAO, PROMPT_VALIDACAO, STOPWORDS_EN, AvaliacaoContexto, EstadoAgente,
                           Reformulacao, Validacao, distanciaDoScore, formatarContexto, formatarFontes, logarAcao,
                           normalizarConsulta, obterLLM, obterLLMEstruturado, obterVectorStore, pareceIngles,
                           registrarPasso)


def buscarContexto(estado: EstadoAgente) -> dict:
    logarAcao("buscarContexto", f'Buscando os {K_BUSCA} trechos mais próximos para "{estado["consulta_busca"]}"...')
    brutos = obterVectorStore().similarity_search_with_score(estado["consulta_busca"], k=K_BUSCA)
    resultados = sorted(((doc, distanciaDoScore(score)) for doc, score in brutos), key=lambda r: r[1])
    docs, notas, docs_vistos, temas_vistos = [], [], set(), set()
    for doc, distancia in resultados:
        m = doc.metadata
        tema = (m["titulo"], m["qtype"])
        if distancia > DISTANCIA_MAXIMA or m["doc_id"] in docs_vistos or tema in temas_vistos:
            continue
        docs_vistos.add(m["doc_id"])
        temas_vistos.add(tema)
        docs.append(doc)
        notas.append(round(float(distancia), 3))
        if len(docs) == K_FINAL:
            break

    fontes = [{"n": n, "titulo": d.metadata["titulo"], "url": d.metadata["url"],
               "dataset": d.metadata["dataset"]} for n, d in enumerate(docs, 1)]
    brutas = [round(float(d), 3) for _, d in resultados]
    ids = [d.metadata.get("id", "?") for d in docs]
    tentadas = estado.get("consultas_tentadas", []) + [estado["consulta_busca"]]
    return {"documentos": docs, "pontuacoes": notas, "contexto": formatarContexto(docs), "fontes": fontes,
            "consultas_tentadas": tentadas,
            "caminho": registrarPasso("buscarContexto",
                                      f"{len(docs)} docs | dist={notas} | ids={ids}"
                                      f" | corte={DISTANCIA_MAXIMA} | brutas={brutas}")}


def avaliarContexto(estado: EstadoAgente) -> dict:
    pontuacoes = estado.get("pontuacoes") or []
    if not estado["documentos"]:
        avaliacao, motivo = "fraco", "nenhum trecho abaixo da distância máxima"
    elif min(pontuacoes) <= DISTANCIA_CONFIANCA:
        avaliacao = "bom"
        motivo = f"bom por distância (melhor={min(pontuacoes)} <= {DISTANCIA_CONFIANCA})"
    else:
        logarAcao("avaliarContexto", f"Avaliando se os {len(estado['documentos'])} trechos respondem a pergunta...")
        try:
            r = (PROMPT_AVALIACAO | obterLLMEstruturado(AvaliacaoContexto)).invoke(
                {"pergunta": estado["pergunta"], "contexto": estado["contexto"]})
            avaliacao, motivo = ("bom" if r.suficiente else "fraco"), r.motivo
        except Exception as erro:
            avaliacao, motivo = "fraco", f"avaliação por LLM falhou ({erro})"
    if avaliacao == "fraco" and estado.get("tentativas", 0) >= MAX_REFORMULACOES:
        avaliacao = "insuficiente"
    return {"avaliacao": avaliacao, "motivo_avaliacao": motivo,
            "caminho": registrarPasso("avaliarContexto", f"{avaliacao} - {motivo}")}


PALAVRAS_PERGUNTA = {"what", "how", "why", "when", "who", "does", "do", "i", "there", "any", "other",
                     "potential", "possible", "about", "should", "could", "together", "take", "taking"}


def _variacaoDeterministica(tentadas: list[str]) -> str:
    chaves = [p for p in normalizarConsulta(tentadas[0]).split()
              if p not in STOPWORDS_EN and p not in PALAVRAS_PERGUNTA]
    vistas = {normalizarConsulta(c) for c in tentadas}
    for candidata in (" ".join(chaves[:3]), f"{chaves[0]} overview" if chaves else "", chaves[0] if chaves else ""):
        if candidata and normalizarConsulta(candidata) not in vistas:
            return candidata
    return f"{tentadas[0]} medical information"


def reformularPergunta(estado: EstadoAgente) -> dict:
    tentadas = estado.get("consultas_tentadas") or [estado["consulta_busca"]]
    logarAcao("reformularPergunta",
              f"Reformulando a consulta (tentativa {estado.get('tentativas', 0) + 1} de {MAX_REFORMULACOES})...")
    try:
        r = (PROMPT_REFORMULACAO | obterLLMEstruturado(Reformulacao)).invoke(
            {"pergunta": estado["pergunta"], "consultas_tentadas": "\n".join(f"- {c}" for c in tentadas),
             "motivo": estado.get("motivo_avaliacao", "")})
        nova, nota = r.consulta_en.strip(), ""
    except Exception as erro:
        nova, nota = "", f" (LLM falhou: {erro})"
    if not nova or normalizarConsulta(nova) in {normalizarConsulta(c) for c in tentadas}:
        repetida = nova
        nova = _variacaoDeterministica(tentadas)
        nota += f' (LLM repetiu "{repetida}", variação determinística aplicada)' if repetida else \
            " (variação determinística aplicada)"
    tentativas = estado.get("tentativas", 0) + 1
    return {"consulta_busca": nova, "pergunta_reformulada": nova, "tentativas": tentativas,
            "caminho": registrarPasso("reformularPergunta", f'tentativa {tentativas}: "{nova}"{nota}')}


def montarResposta(estado: EstadoAgente) -> dict:
    if not estado.get("documentos") or estado.get("avaliacao") == "insuficiente":
        logarAcao("montarResposta", "Sem contexto útil, usando mensagem fixa...")
    else:
        logarAcao("montarResposta", f"Montando resposta com {len(estado['documentos'])} trechos...")
    if not estado.get("documentos"):
        return {"resposta": MENSAGEM_SEM_CONTEXTO,
                "caminho": registrarPasso("montarResposta", "sem contexto, mensagem fixa")}
    if estado.get("avaliacao") == "insuficiente":
        return {"resposta": MENSAGEM_SEM_CONTEXTO,
                "caminho": registrarPasso("montarResposta", "contexto insuficiente, mensagem fixa")}
    resposta = (PROMPT_RESPOSTA | obterLLM(0.2)).invoke(
        {"contexto": estado["contexto"], "situacao_contexto": "adequado", "pergunta": estado["pergunta"]}
    ).content.strip()
    return {"resposta": resposta,
            "caminho": registrarPasso("montarResposta", f"{len(resposta)} caracteres")}


AVISO_FUNDAMENTACAO = "⚠ Parte desta resposta pode não estar sustentada pelas fontes consultadas."


def _fontesCitadas(resposta: str, fontes: list[dict]) -> list[dict]:
    citados = {int(n) for grupo in re.findall(r"\[(\d+(?:\s*,\s*\d+)*)\]", resposta)
               for n in re.findall(r"\d+", grupo)}
    return [f for f in fontes if f["n"] in citados]


def validarResposta(estado: EstadoAgente) -> dict:
    resposta = estado["resposta"]
    if resposta == MENSAGEM_SEM_CONTEXTO:
        logarAcao("validarResposta", "Resposta fixa sem contexto, pulando a validação...")
        return {"resposta": resposta, "valida": True, "problemas_validacao": [],
                "caminho": registrarPasso("validarResposta", "valida=True (sem contexto)")}

    logarAcao("validarResposta", "Validando idioma, citações e fundamentação...")
    try:
        v = (PROMPT_VALIDACAO | obterLLMEstruturado(Validacao)).invoke(
            {"contexto": estado["contexto"], "resposta": resposta})
        em_portugues, fundamentada, problemas = v.em_portugues, v.fundamentada, list(v.problemas)
    except Exception as erro:
        em_portugues, fundamentada = True, False
        problemas = [f"validação por LLM falhou: {erro}"]

    if not em_portugues or pareceIngles(resposta):
        logarAcao("validarResposta", "Traduzindo a resposta para português...")
        try:
            resposta = (PROMPT_TRADUCAO | obterLLM(0.0)).invoke({"resposta": resposta}).content.strip()
            problemas.append("resposta traduzida para português")
        except Exception as erro:
            problemas.append(f"tradução falhou: {erro}")
    logarAcao("validarResposta", "Anexando fontes e aviso legal...")
    if not fundamentada:
        resposta = f"{AVISO_FUNDAMENTACAO}\n\n{resposta}"
    if "não substitui" not in resposta:
        resposta += f"\n\n{AVISO_LEGAL}"
    citadas = _fontesCitadas(resposta, estado["fontes"])
    if citadas:
        resposta += f"\n\n{formatarFontes(citadas)}"
    else:
        problemas.append("resposta sem citações [n]; nenhuma fonte listada")

    valida = em_portugues and fundamentada
    return {"resposta": resposta, "valida": valida, "problemas_validacao": problemas,
            "caminho": registrarPasso("validarResposta",
                                      f"valida={valida} | fontes citadas={[f['n'] for f in citadas]}")}


def rotearAvaliacao(estado: EstadoAgente) -> str:
    return "reformularPergunta" if estado["avaliacao"] == "fraco" else "montarResposta"


def construirConsultarBaseMedica():
    g = StateGraph(EstadoAgente)
    for nome, fn in [("buscarContexto", buscarContexto), ("avaliarContexto", avaliarContexto),
                     ("reformularPergunta", reformularPergunta), ("montarResposta", montarResposta),
                     ("validarResposta", validarResposta)]:
        g.add_node(nome, fn)
    g.add_edge(START, "buscarContexto")
    g.add_edge("buscarContexto", "avaliarContexto")
    g.add_conditional_edges("avaliarContexto", rotearAvaliacao, ["reformularPergunta", "montarResposta"])
    g.add_edge("reformularPergunta", "buscarContexto")
    g.add_edge("montarResposta", "validarResposta")
    g.add_edge("validarResposta", END)
    return g.compile()
