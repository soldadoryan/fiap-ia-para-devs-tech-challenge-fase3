from funcoesPadrao import EstadoAgente, contarVetores, logarAcao, registrarPasso


def verificarBase(estado: EstadoAgente) -> dict:
    if estado.get("forcar_indexacao"):
        logarAcao("verificarBase", "Reindexação solicitada, ignorando o índice FAISS existente...")
    else:
        logarAcao("verificarBase", "Verificando se o índice FAISS existe em data/faiss...")
    total = 0 if estado.get("forcar_indexacao") else contarVetores()
    pronta = total > 0
    if pronta:
        detalhe = f"base pronta ({total} vetores)"
    elif estado.get("forcar_indexacao"):
        detalhe = "reindexacao solicitada"
    else:
        detalhe = "base vazia, sera indexada"
    return {"base_pronta": pronta, "caminho": registrarPasso("verificarBase", detalhe)}
