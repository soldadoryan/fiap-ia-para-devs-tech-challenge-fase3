from funcoesPadrao import MENSAGEM_EMERGENCIA, MENSAGEM_FORA_ESCOPO, EstadoAgente, logarAcao, registrarPasso


def responderForaEscopo(estado: EstadoAgente) -> dict:
    emergencia = estado["categoria"] == "emergencia"
    logarAcao("responderForaEscopo",
              f"Montando mensagem fixa de {'emergência' if emergencia else 'fora de escopo'}...")
    return {"resposta": MENSAGEM_EMERGENCIA if emergencia else MENSAGEM_FORA_ESCOPO,
            "caminho": registrarPasso("responderForaEscopo", estado["categoria"])}
