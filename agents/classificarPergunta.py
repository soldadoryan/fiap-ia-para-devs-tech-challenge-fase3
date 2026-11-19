from funcoesPadrao import (PROMPT_CLASSIFICACAO, TIPOS_EXAME, Classificacao, EstadoAgente, logarAcao,
                           obterLLMEstruturado, registrarPasso, textoNormalizado)


def listarTiposAceitos() -> str:
    return "\n".join(f"- {tipo} (também dito: {', '.join(info['apelidos'])})" for tipo, info in TIPOS_EXAME.items())


def validarTipoExame(tipo: str | None) -> str | None:
    por_nome = {textoNormalizado(t): t for t in TIPOS_EXAME}
    return por_nome.get(textoNormalizado(tipo))


def limparCampo(valor: str | None) -> str | None:
    valor = (valor or "").strip()
    return None if valor.lower() in ("", "null", "none") else valor


def interpretarPergunta(pergunta: str) -> Classificacao:
    return (PROMPT_CLASSIFICACAO | obterLLMEstruturado(Classificacao)).invoke(
        {"pergunta": pergunta, "tipos_exame": listarTiposAceitos()})


def classificarPergunta(estado: EstadoAgente) -> dict:
    pergunta = estado["pergunta"]
    logarAcao("classificarPergunta", "Classificando a pergunta com o LLM (categoria, intenção, paciente e exame)...")
    r = interpretarPergunta(pergunta)
    categoria = r.categoria
    intencao = r.intencao_exame if categoria == "exames" else None
    foco = r.foco_prontuario if categoria == "prontuario" else None
    tipo = validarTipoExame(r.tipo_exame) if categoria == "exames" else None
    consulta = r.consulta_en.strip() or pergunta
    nome, cpf = limparCampo(r.paciente_nome), limparCampo(r.paciente_cpf)
    detalhe = f"categoria={categoria}"
    if categoria == "medica":
        detalhe += f' | consulta_en="{consulta}"'
    if categoria in ("prontuario", "exames"):
        detalhe += f' | intencao={intencao} | foco={foco} | paciente(id={r.paciente_id}, nome="{nome or ""}", ' \
                   f'cpf="{cpf or ""}") | tipo_exame={tipo}'
    return {"categoria": categoria, "intencao_exame": intencao, "foco_prontuario": foco,
            "paciente_nome_informado": nome, "paciente_id_informado": r.paciente_id,
            "paciente_cpf_informado": cpf, "tipo_exame": tipo, "consulta_busca": consulta,
            "caminho": registrarPasso("classificarPergunta", detalhe)}
