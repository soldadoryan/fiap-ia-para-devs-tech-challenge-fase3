from langgraph.graph import END, START, StateGraph

from funcoesPadrao import (AVISO_LEGAL, MENSAGEM_FINALIDADE_REMEDIO, PROMPT_PRONTUARIO, EstadoAgente,
                           formatarProntuario, identificarPaciente, logarAcao, obterLLM, obterPaciente,
                           registrarPasso)


def buscarProntuario(estado: EstadoAgente) -> dict:
    logarAcao("buscarProntuario", f"Lendo o prontuário do paciente {estado['paciente_id']} no cadastro em memória...")
    p = obterPaciente(estado["paciente_id"])
    detalhe = (f"{p['nome']} | {len(p['alergias'])} alergias | {len(p['medicamentos'])} medicamentos"
               f" | {len(p['consultas'])} consultas")
    return {"prontuario": formatarProntuario(p), "caminho": registrarPasso("buscarProntuario", detalhe)}


def responderProntuario(estado: EstadoAgente) -> dict:
    cabecalho = f"{estado['paciente_nome']} (paciente {estado['paciente_id']}):"
    if estado.get("foco_prontuario") == "finalidade_medicamentos":
        logarAcao("responderProntuario", "Pergunta sobre para que servem os remédios, usando só o prontuário...")
        medicamentos = obterPaciente(estado["paciente_id"])["medicamentos"]
        lista = "\n".join(f"- {m}" for m in medicamentos)
        texto = f"Medicamentos em uso:\n{lista}" if medicamentos else "Não há medicamentos em uso registrados."
        resposta = f"{cabecalho}\n{texto}\n\n{MENSAGEM_FINALIDADE_REMEDIO}\n\n{AVISO_LEGAL}"
        return {"resposta": resposta, "caminho": registrarPasso(
            "responderProntuario", f"finalidade dos remédios, resposta fixa sem LLM | {len(medicamentos)} medicamentos")}
    logarAcao("responderProntuario", "Respondendo a pergunta com base no prontuário...")
    try:
        texto = (PROMPT_PRONTUARIO | obterLLM(0.0)).invoke(
            {"prontuario": estado["prontuario"], "pergunta": estado["pergunta"],
             "nome": estado["paciente_nome"]}).content.strip()
        nota = ""
    except Exception as erro:
        texto, nota = estado["prontuario"], f" (LLM falhou, prontuário completo devolvido: {erro})"
    resposta = f"{cabecalho}\n{texto}"
    if "não substitui" not in resposta:
        resposta += f"\n\n{AVISO_LEGAL}"
    return {"resposta": resposta, "caminho": registrarPasso("responderProntuario", f"{len(resposta)} caracteres{nota}")}


def rotearIdentificacao(estado: EstadoAgente) -> str:
    return "buscarProntuario" if estado.get("situacao_paciente") == "encontrado" else END


def construirConsultarProntuario():
    g = StateGraph(EstadoAgente)
    g.add_node("identificarPaciente", identificarPaciente)
    g.add_node("buscarProntuario", buscarProntuario)
    g.add_node("responderProntuario", responderProntuario)
    g.add_edge(START, "identificarPaciente")
    g.add_conditional_edges("identificarPaciente", rotearIdentificacao, ["buscarProntuario", END])
    g.add_edge("buscarProntuario", "responderProntuario")
    g.add_edge("responderProntuario", END)
    return g.compile()
