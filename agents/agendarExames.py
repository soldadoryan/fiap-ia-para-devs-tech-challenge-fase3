from functools import lru_cache

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from funcoesPadrao import (MAX_TENTATIVAS_ESCOLHA, MENSAGEM_DATAS_SEM_ESCOLHA, MENSAGEM_ESCOLHA_DATA,
                           MENSAGEM_ESCOLHA_INVALIDA, MENSAGEM_EXAME_JA_MARCADO, MENSAGEM_EXAME_MARCADO,
                           MENSAGEM_EXAMES_MARCADOS, MENSAGEM_HORARIO_OCUPADO, MENSAGEM_MARCACAO_CANCELADA,
                           MENSAGEM_NOVA_PERGUNTA, MENSAGEM_REMARCACAO, MENSAGEM_SEM_DATAS, MENSAGEM_SEM_EXAMES,
                           MENSAGEM_SEM_RESULTADOS, MENSAGEM_TENTATIVAS_ESGOTADAS, MENSAGEM_TIPO_EXAME_AUSENTE,
                           PROMPT_ESCOLHA, TIPOS_EXAME, EstadoAgente, InterpretacaoEscolha, examesDoPaciente,
                           formatarHorario, horariosLivres, identificarPaciente, logarAcao, obterLLMEstruturado,
                           registrarExame, registrarPasso)


def opcoesDeData(horarios: list[dict]) -> list[str]:
    return [f"{n}. {formatarHorario(h['data'], h['hora'])}" for n, h in enumerate(horarios, 1)]


def verificarExamesMarcados(estado: EstadoAgente) -> dict:
    pid, nome = estado["paciente_id"], estado["paciente_nome"]
    logarAcao("verificarExamesMarcados", f"Buscando exames marcados para {nome}...")
    exames = examesDoPaciente(pid)
    lista = "\n".join(f"- {e['tipo']}: {formatarHorario(e['data'], e['hora'])} - {e['local']}" for e in exames)
    marcados = (MENSAGEM_EXAMES_MARCADOS.format(nome=nome, id=pid, lista=lista) if exames
                else MENSAGEM_SEM_EXAMES.format(nome=nome, id=pid))
    intencao = estado.get("intencao_exame")
    if intencao == "remarcar_ou_cancelar":
        return {"exames_marcados": exames, "pode_marcar": False, "resposta": f"{MENSAGEM_REMARCACAO}\n\n{marcados}",
                "caminho": registrarPasso("verificarExamesMarcados",
                                          f"remarcar ou cancelar não suportado | {len(exames)} exames marcados")}
    if intencao == "resultado":
        return {"exames_marcados": exames, "pode_marcar": False,
                "resposta": f"{MENSAGEM_SEM_RESULTADOS}\n\n{marcados}",
                "caminho": registrarPasso("verificarExamesMarcados",
                                          f"resultado não disponível | {len(exames)} exames marcados")}
    if intencao != "marcar":
        return {"exames_marcados": exames, "pode_marcar": False, "resposta": marcados,
                "caminho": registrarPasso("verificarExamesMarcados", f"consultar | {len(exames)} exames marcados")}

    logarAcao("verificarExamesMarcados", "Conferindo o tipo de exame pedido...")
    tipo = estado.get("tipo_exame")
    repetido = next((e for e in exames if e["tipo"] == tipo), None)
    if not tipo:
        resposta = MENSAGEM_TIPO_EXAME_AUSENTE.format(lista="\n".join(f"- {t}" for t in TIPOS_EXAME))
        detalhe = "marcar | tipo de exame não identificado"
    elif repetido:
        resposta = MENSAGEM_EXAME_JA_MARCADO.format(nome=nome, tipo=tipo,
                                                    quando=formatarHorario(repetido["data"], repetido["hora"]))
        detalhe = f"marcar | {tipo} já marcado em {repetido['data']} {repetido['hora']}"
    else:
        resposta = ""
        detalhe = f"marcar | tipo={tipo} | {len(exames)} exames já marcados"
    return {"exames_marcados": exames, "pode_marcar": not resposta, "resposta": resposta,
            "caminho": registrarPasso("verificarExamesMarcados", detalhe)}


def listarDatasDisponiveis(estado: EstadoAgente) -> dict:
    logarAcao("listarDatasDisponiveis", f"Buscando horários livres para {estado['tipo_exame']}...")
    horarios = horariosLivres(estado["paciente_id"])
    if not horarios:
        return {"datas_disponiveis": [], "resposta": MENSAGEM_SEM_DATAS.format(tipo=estado["tipo_exame"]),
                "caminho": registrarPasso("listarDatasDisponiveis", "nenhum horário livre")}
    detalhe = f"{len(horarios)} horários de {horarios[0]['data']} a {horarios[-1]['data']}"
    if estado.get("interativo"):
        return {"datas_disponiveis": horarios, "caminho": registrarPasso("listarDatasDisponiveis", detalhe)}
    logarAcao("listarDatasDisponiveis", "Modo --pergunta, só listando as datas sem marcar...")
    resposta = MENSAGEM_DATAS_SEM_ESCOLHA.format(tipo=estado["tipo_exame"], nome=estado["paciente_nome"],
                                                 lista="\n".join(opcoesDeData(horarios)))
    return {"datas_disponiveis": horarios, "resposta": resposta,
            "caminho": registrarPasso("listarDatasDisponiveis", f"{detalhe} | modo --pergunta, sem escolha")}


@lru_cache(maxsize=64)
def interpretarEscolha(texto: str, opcoes: tuple[str, ...]) -> tuple[str, int | None]:
    r = (PROMPT_ESCOLHA | obterLLMEstruturado(InterpretacaoEscolha)).invoke(
        {"opcoes": "\n".join(opcoes), "escolha": texto})
    return r.acao, r.opcao


def escolherData(estado: EstadoAgente) -> dict:
    horarios = estado["datas_disponiveis"]
    opcoes = opcoesDeData(horarios)
    pausa = {"no": "escolherData",
             "titulo": MENSAGEM_ESCOLHA_DATA.format(tipo=estado["tipo_exame"], nome=estado["paciente_nome"]),
             "opcoes": opcoes, "erro": ""}
    digitadas, escolhido, situacao = [], {}, "esgotada"
    for tentativa in range(1, MAX_TENTATIVAS_ESCOLHA + 1):
        escolha = str(interrupt(pausa)).strip()
        digitadas.append(escolha)
        if escolha.isdigit():
            acao, opcao = "escolher", int(escolha)
        elif escolha:
            logarAcao("escolherData", f'Interpretando "{escolha}" com o LLM...')
            acao, opcao = interpretarEscolha(escolha, tuple(opcoes))
        else:
            acao, opcao = "escolher", None
        if acao == "cancelar":
            situacao = "cancelada"
            break
        if acao == "nova_pergunta":
            situacao = "nova_pergunta"
            break
        if opcao is not None and 1 <= opcao <= len(horarios):
            escolhido, situacao = horarios[opcao - 1], "escolhida"
            break
        pausa = {**pausa, "erro": MENSAGEM_ESCOLHA_INVALIDA.format(
            escolha=escolha, total=len(horarios), restantes=MAX_TENTATIVAS_ESCOLHA - tentativa)}
    logarAcao("escolherData", f"Conferindo as opções digitadas ({', '.join(digitadas)})...")
    detalhe = f"{situacao} | tentativas={len(digitadas)}"
    if escolhido:
        detalhe += f" | {escolhido['data']} {escolhido['hora']}"
    return {"data_escolhida": escolhido, "situacao_escolha": situacao,
            "caminho": registrarPasso("escolherData", detalhe)}


def marcarExame(estado: EstadoAgente) -> dict:
    pid, nome, tipo = estado["paciente_id"], estado["paciente_nome"], estado["tipo_exame"]
    situacao = estado.get("situacao_escolha")
    if situacao in ("cancelada", "nova_pergunta"):
        logarAcao("marcarExame", "Marcação cancelada pelo usuário, nada a gravar...")
        resposta = MENSAGEM_MARCACAO_CANCELADA.format(nome=nome)
        if situacao == "nova_pergunta":
            resposta += f"\n{MENSAGEM_NOVA_PERGUNTA}"
        return {"resposta": resposta,
                "caminho": registrarPasso("marcarExame", f"cancelada pelo usuário ({situacao})")}
    if situacao != "escolhida":
        logarAcao("marcarExame", "Tentativas esgotadas, nada a gravar...")
        return {"resposta": MENSAGEM_TENTATIVAS_ESGOTADAS.format(maximo=MAX_TENTATIVAS_ESCOLHA),
                "caminho": registrarPasso("marcarExame", "cancelada, tentativas esgotadas")}
    horario = estado["data_escolhida"]
    logarAcao("marcarExame", f"Gravando {tipo} para {nome} em {horario['data']} {horario['hora']}...")
    exame = registrarExame(pid, tipo, horario)
    if exame is None:
        return {"resposta": MENSAGEM_HORARIO_OCUPADO, "caminho": registrarPasso("marcarExame", "horário ocupado")}
    resposta = MENSAGEM_EXAME_MARCADO.format(nome=nome, id=pid, tipo=tipo,
                                             quando=formatarHorario(exame["data"], exame["hora"]),
                                             local=exame["local"], preparo=TIPOS_EXAME[tipo]["preparo"])
    return {"resposta": resposta,
            "caminho": registrarPasso("marcarExame", f"{tipo} em {exame['data']} {exame['hora']} | "
                                                     f"{len(examesDoPaciente(pid))} exames do paciente")}


def rotearIdentificacao(estado: EstadoAgente) -> str:
    return "verificarExamesMarcados" if estado.get("situacao_paciente") == "encontrado" else END


def rotearMarcacao(estado: EstadoAgente) -> str:
    return "listarDatasDisponiveis" if estado.get("pode_marcar") else END


def rotearDatas(estado: EstadoAgente) -> str:
    return "escolherData" if estado.get("datas_disponiveis") and estado.get("interativo") else END


def construirAgendarExames():
    g = StateGraph(EstadoAgente)
    g.add_node("identificarPaciente", identificarPaciente)
    g.add_node("verificarExamesMarcados", verificarExamesMarcados)
    g.add_node("listarDatasDisponiveis", listarDatasDisponiveis)
    g.add_node("escolherData", escolherData)
    g.add_node("marcarExame", marcarExame)
    g.add_edge(START, "identificarPaciente")
    g.add_conditional_edges("identificarPaciente", rotearIdentificacao, ["verificarExamesMarcados", END])
    g.add_conditional_edges("verificarExamesMarcados", rotearMarcacao, ["listarDatasDisponiveis", END])
    g.add_conditional_edges("listarDatasDisponiveis", rotearDatas, ["escolherData", END])
    g.add_edge("escolherData", "marcarExame")
    g.add_edge("marcarExame", END)
    return g.compile()
