import argparse
import json
import sys
import time
import urllib.request
import uuid

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from agents.agendarExames import construirAgendarExames
from agents.classificarPergunta import classificarPergunta
from agents.consultarBaseMedica import construirConsultarBaseMedica
from agents.consultarProntuario import construirConsultarProntuario
from agents.indexarBase import construirIndexarBase
from agents.responderForaEscopo import responderForaEscopo
from agents.verificarBase import verificarBase
from funcoesPadrao import MODELO_EMBEDDING, MODELO_LLM, OLLAMA_URL, EstadoAgente, logarAcao

ROTAS_CATEGORIA = {"medica": "consultarBaseMedica", "prontuario": "consultarProntuario",
                   "exames": "agendarExames"}
PALAVRAS_SAIR = {"sair", "exit", "quit"}


def rotearBase(estado: EstadoAgente) -> str:
    if estado.get("base_pronta"):
        return "classificarPergunta" if estado.get("pergunta") else END
    return "indexarBase"


def rotearAposIndexacao(estado: EstadoAgente) -> str:
    return "classificarPergunta" if estado.get("pergunta") else END


def rotearCategoria(estado: EstadoAgente) -> str:
    return ROTAS_CATEGORIA.get(estado["categoria"], "responderForaEscopo")


def construirGrafo(para_desenho: bool = False):
    g = StateGraph(EstadoAgente)
    g.add_node("verificarBase", verificarBase)
    g.add_node("indexarBase", construirIndexarBase(para_desenho))
    g.add_node("classificarPergunta", classificarPergunta)
    g.add_node("consultarBaseMedica", construirConsultarBaseMedica())
    g.add_node("consultarProntuario", construirConsultarProntuario())
    g.add_node("agendarExames", construirAgendarExames())
    g.add_node("responderForaEscopo", responderForaEscopo)

    g.add_edge(START, "verificarBase")
    g.add_conditional_edges("verificarBase", rotearBase, ["indexarBase", "classificarPergunta", END])
    g.add_conditional_edges("indexarBase", rotearAposIndexacao, ["classificarPergunta", END])
    g.add_conditional_edges("classificarPergunta", rotearCategoria,
                            ["consultarBaseMedica", "consultarProntuario", "agendarExames", "responderForaEscopo"])
    for nome in ("consultarBaseMedica", "consultarProntuario", "agendarExames", "responderForaEscopo"):
        g.add_edge(nome, END)
    return g.compile(checkpointer=MemorySaver())


def testarOllama() -> bool:
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=5) as r:
            instalados = {m["name"] for m in json.load(r)["models"]}
    except Exception:
        print("Ollama indisponível. Inicie o Ollama (ollama serve).")
        return False
    faltando = [m for m in (MODELO_LLM, MODELO_EMBEDDING)
                if not any(n == m or n.startswith(f"{m}:") for n in instalados)]
    for modelo in faltando:
        print(f"Modelo ausente. Execute: ollama pull {modelo}")
    return not faltando


def mostrarOpcoes(pausa: dict) -> None:
    logarAcao(pausa["no"], "Aguardando a escolha da data...")
    if pausa.get("erro"):
        print(f"\n{pausa['erro']}")
    print(f"\n{pausa['titulo']}")
    print("\n".join(pausa["opcoes"]))


def executar(grafo, pergunta: str, args, forcar_indexacao: bool, interativo: bool) -> tuple[bool, str]:
    estado = {"pergunta": pergunta, "forcar_indexacao": forcar_indexacao, "limite": args.limite,
              "pubmedqa_completo": args.pubmedqa_completo, "ler_pastas_vazias": args.ler_pastas_vazias,
              "interativo": interativo, "tentativas": 0, "intencao_exame": None, "foco_prontuario": None,
              "paciente_nome_informado": None, "paciente_id_informado": None, "paciente_cpf_informado": None,
              "tipo_exame": None, "caminho": []}
    config = {"configurable": {"thread_id": uuid.uuid4().hex}, "recursion_limit": 30}
    inicio, escolha = time.time(), ""
    try:
        final = grafo.invoke(estado, config=config)
        while "__interrupt__" in final:
            mostrarOpcoes(final["__interrupt__"][0].value)
            espera = time.time()
            try:
                escolha = input('\nSua escolha ("sair" encerra): ').strip()
            except EOFError:
                escolha = "sair"
            inicio += time.time() - espera
            if escolha.lower() in PALAVRAS_SAIR:
                print("\nMarcação interrompida. Nenhum exame foi marcado.")
                return True, "sair"
            final = grafo.invoke(Command(resume=escolha), config=config)
    except ConnectionError:
        print("\nO Ollama parou de responder durante a execução. Inicie o Ollama (ollama serve) e tente de novo.")
        return False, ""
    except Exception as erro:
        print(f"\nErro ao processar a pergunta: {type(erro).__name__}: {erro}")
        return False, ""
    if final.get("resposta"):
        print(f"\n{final['resposta']}\n")
    print(f"Caminho: {' -> '.join(final['caminho'])}")
    print(f"Tempo: {time.time() - inicio:.1f}s")
    return True, escolha if final.get("situacao_escolha") == "nova_pergunta" else ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Assistente médico com RAG (LangChain + LangGraph + Ollama)")
    parser.add_argument("--pergunta", help="pergunta única; sem ela, abre o loop interativo")
    parser.add_argument("--indexar", action="store_true", help="força a reindexação da base")
    parser.add_argument("--limite", type=int, help="indexa só N documentos (teste rápido)")
    parser.add_argument("--pubmedqa-completo", action="store_true", help="inclui ori_pqau/ori_pqaa se existirem")
    parser.add_argument("--ler-pastas-vazias", action="store_true", help="lê também as pastas 10 a 12 do MedQuAD")
    parser.add_argument("--mostrar-grafo", nargs="?", const="ascii", choices=["ascii", "mermaid"],
                        help="desenha o grafo no terminal (ascii, padrão) ou imprime o código Mermaid")
    args = parser.parse_args()

    grafo = construirGrafo(para_desenho=bool(args.mostrar_grafo))
    if args.mostrar_grafo == "mermaid":
        print(grafo.get_graph(xray=True).draw_mermaid())
        return
    if args.mostrar_grafo:
        print(grafo.get_graph().draw_ascii())
        for nome, subgrafo in grafo.get_subgraphs():
            print(f"\nSubgrafo {nome}:")
            print(subgrafo.get_graph().draw_ascii())
        return
    if not testarOllama():
        sys.exit(1)

    if args.pergunta or args.indexar:
        ok, _ = executar(grafo, args.pergunta or "", args, args.indexar, False)
        if not ok:
            sys.exit(1)
    else:
        print('Assistente médico. Digite sua pergunta ("sair" encerra).')
        pendente = ""
        while True:
            try:
                pergunta = pendente or input("\nPergunta: ").strip()
            except EOFError:
                break
            if pergunta.lower() in PALAVRAS_SAIR:
                break
            if pendente:
                print(f"\nPergunta: {pendente}")
            pendente = executar(grafo, pergunta, args, False, True)[1] if pergunta else ""


if __name__ == "__main__":
    main()
