import hashlib
import json
import random
import re
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterator

from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START, StateGraph
from tqdm import tqdm

from funcoesPadrao import (DIR_FAISS, DIR_PROCESSED, DIR_RAW, OVERLAP_CHUNK, PASTAS_SEM_RESPOSTA, TAMANHO_CHUNK,
                           TAMANHO_LOTE, TAMANHO_MINIMO_RESPOSTA, EstadoIndexacao, SaidaIndexacao,
                           invalidarVectorStore, limparTexto, logarAcao, obterEmbeddings, registrarPasso,
                           removerPerguntaInicial)

DIR_PUBMEDQA = DIR_RAW / "pubmedqa" / "data"
DIR_MEDQUAD = DIR_RAW / "MedQuAD"
ARQUIVOS_PUBMEDQA_EXTRAS = ["ori_pqau.json", "ori_pqaa.json"]


class CarregadorPubMedQA(BaseLoader):

    def __init__(self, caminhos: list[Path]):
        self.caminhos = caminhos

    def lazy_load(self) -> Iterator[Document]:
        for caminho in self.caminhos:
            with open(caminho, encoding="utf-8") as f:
                registros = json.load(f)
            for pmid, r in registros.items():
                yield Document(
                    page_content=r.get("LONG_ANSWER", ""),
                    metadata={
                        "dataset": "pubmedqa",
                        "pmid": str(pmid),
                        "pergunta": r.get("QUESTION", ""),
                        "contextos": r.get("CONTEXTS", []),
                        "rotulos": r.get("LABELS", []),
                        "meshes": "; ".join(r.get("MESHES", [])),
                        "ano": str(r.get("YEAR", "")),
                        "final_decision": r.get("final_decision", ""),
                    },
                )


class CarregadorMedQuAD(BaseLoader):

    def __init__(self, raiz: Path, ignorar: set[str] = PASTAS_SEM_RESPOSTA):
        self.raiz = raiz
        self.ignorar = ignorar

    def lazy_load(self) -> Iterator[Document]:
        pastas = sorted((p for p in self.raiz.iterdir() if p.is_dir() and re.match(r"^\d+_", p.name)),
                        key=lambda p: int(p.name.split("_")[0]))
        for pasta in pastas:
            if pasta.name in self.ignorar:
                continue
            for xml in sorted(pasta.glob("*.xml")):
                yield from self._ler_xml(xml, pasta.name)

    def _ler_xml(self, xml: Path, pasta: str) -> Iterator[Document]:
        try:
            raiz = ET.parse(xml).getroot()
        except ET.ParseError:
            return
        focus = (raiz.findtext("Focus") or "").strip()
        for n, par in enumerate(raiz.iter("QAPair"), 1):
            pergunta = par.find("Question")
            resposta = (par.findtext("Answer") or "").strip()
            if pergunta is None:
                continue
            yield Document(
                page_content=resposta,
                metadata={
                    "dataset": "medquad",
                    "pasta": pasta,
                    "fonte": raiz.get("source", ""),
                    "url": raiz.get("url", ""),
                    "focus": focus,
                    "qtype": pergunta.get("qtype", ""),
                    "doc_id": f"{pasta}/{xml.stem}-{par.get('pid') or n}",
                    "pergunta": (pergunta.text or "").strip(),
                },
            )


def extrairDadosDataset(estado: EstadoIndexacao) -> dict:
    arquivos = [DIR_PUBMEDQA / "ori_pqal.json"]
    if estado.get("pubmedqa_completo"):
        arquivos += [DIR_PUBMEDQA / a for a in ARQUIVOS_PUBMEDQA_EXTRAS if (DIR_PUBMEDQA / a).exists()]
    ignorar = set() if estado.get("ler_pastas_vazias") else PASTAS_SEM_RESPOSTA

    logarAcao("extrairDadosDataset", f"Lendo PubMedQA ({', '.join(a.name for a in arquivos)})...")
    pubmed = list(CarregadorPubMedQA(arquivos).lazy_load())
    logarAcao("extrairDadosDataset", "Lendo MedQuAD (pastas 1 a 9)..." if ignorar
              else "Lendo MedQuAD (pastas 1 a 12)...")
    medquad = list(CarregadorMedQuAD(DIR_MEDQUAD, ignorar).lazy_load())
    vazios = sum(1 for d in medquad if not d.page_content)

    estatisticas = {
        "pubmedqa_bruto": len(pubmed),
        "medquad_bruto": len(medquad),
        "medquad_sem_resposta": vazios,
        "pastas_ignoradas": sorted(ignorar),
    }
    detalhe = f"{len(pubmed)} PubMedQA + {len(medquad)} pares MedQuAD ({vazios} sem resposta)"
    if ignorar:
        detalhe += f"; pastas sem resposta puladas (copyright MedlinePlus): {', '.join(sorted(ignorar))}"
    return {"documentos_brutos": pubmed + medquad, "estatisticas": estatisticas,
            "caminho": registrarPasso("extrairDadosDataset", detalhe)}


DECISAO_PT = {"yes": "sim", "no": "não", "maybe": "talvez"}


def _descartarResposta(resposta: str) -> bool:
    return (len(resposta) < TAMANHO_MINIMO_RESPOSTA
            or (resposta.endswith("?") and len(resposta) < 150)
            or (resposta.startswith("See the ") and len(resposta) < 200))


def _focoDaPergunta(pergunta: str) -> str:
    return re.sub(r"^What (is|are)( \(are\))?\s*", "", pergunta).rstrip(" ?")


def _tratarMedQuAD(brutos: list[Document], estatisticas: dict) -> list[Document]:
    divisor = RecursiveCharacterTextSplitter(chunk_size=TAMANHO_CHUNK, chunk_overlap=OVERLAP_CHUNK,
                                             separators=["\n\n", "\n", ". ", " ", ""])
    vistos: set[str] = set()
    chunks: list[Document] = []
    respondidos = descartados = duplicados = 0
    for bruto in brutos:
        resposta = limparTexto(bruto.page_content)
        if not resposta:
            continue
        respondidos += 1
        resposta = removerPerguntaInicial(resposta).strip()
        if _descartarResposta(resposta):
            descartados += 1
            continue
        chave = hashlib.sha1(resposta.lower().encode("utf-8")).hexdigest()
        if chave in vistos:
            duplicados += 1
            continue
        vistos.add(chave)

        m = bruto.metadata
        pergunta = limparTexto(m["pergunta"])
        foco = limparTexto(m["focus"]) or _focoDaPergunta(pergunta)
        for i, trecho in enumerate(divisor.split_text(resposta)):
            texto = (f"Fonte: MedQuAD / {m['fonte']} | Tema: {foco} | Tipo: {m['qtype']}\n"
                     f"Pergunta: {pergunta}\nResposta: {trecho}")
            chunks.append(Document(page_content=texto, metadata={
                "dataset": "medquad", "fonte": m["fonte"], "url": m["url"], "titulo": foco,
                "qtype": m["qtype"], "doc_id": m["doc_id"], "pmid": "", "final_decision": "",
                "ano": "", "chunk": i, "id": f"medquad:{m['doc_id']}:{i}",
            }))
    estatisticas.update(medquad_respondidos=respondidos, medquad_descartados_qualidade=descartados,
                        medquad_duplicados_removidos=duplicados,
                        medquad_pares_finais=respondidos - descartados - duplicados,
                        medquad_chunks=len(chunks))
    return chunks


def _tratarPubMedQA(brutos: list[Document], estatisticas: dict) -> list[Document]:
    docs: dict[str, Document] = {}
    for bruto in brutos:
        m = bruto.metadata
        pmid = m["pmid"]
        linhas = [f"Fonte: PubMedQA (PMID {pmid}, {m['ano']})",
                  f"Pergunta de pesquisa: {limparTexto(m['pergunta'])}"]
        rotulos = m["rotulos"] or ["CONTEXTO"] * len(m["contextos"])
        linhas += [f"{rotulo}: {limparTexto(ctx)}" for rotulo, ctx in zip(rotulos, m["contextos"])]
        linhas.append(f"Conclusão: {limparTexto(bruto.page_content)}")
        decisao = m["final_decision"]
        if decisao:
            linhas.append(f"Decisão final: {DECISAO_PT.get(decisao, decisao)}")
        docs[pmid] = Document(page_content="\n".join(linhas), metadata={
            "dataset": "pubmedqa", "fonte": "PubMed", "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            "titulo": limparTexto(m["pergunta"]), "qtype": "", "doc_id": pmid, "pmid": pmid,
            "final_decision": decisao, "ano": m["ano"], "chunk": 0, "id": f"pubmedqa:{pmid}",
            "meshes": m["meshes"],
        })
    estatisticas["pubmedqa_documentos"] = len(docs)
    return list(docs.values())


def tratarDadosDataset(estado: EstadoIndexacao) -> dict:
    brutos = estado["documentos_brutos"]
    logarAcao("tratarDadosDataset", "Limpando textos e descartando respostas vazias ou curtas...")
    logarAcao("tratarDadosDataset", "Removendo duplicados...")
    logarAcao("tratarDadosDataset", f"Cortando respostas em chunks de {TAMANHO_CHUNK} caracteres...")
    estatisticas = dict(estado.get("estatisticas", {}))
    medquad = _tratarMedQuAD([d for d in brutos if d.metadata["dataset"] == "medquad"], estatisticas)
    pubmed = _tratarPubMedQA([d for d in brutos if d.metadata["dataset"] == "pubmedqa"], estatisticas)
    documentos = medquad + pubmed

    limite = estado.get("limite")
    if limite and limite < len(documentos):
        logarAcao("tratarDadosDataset", f"Amostrando {limite} documentos (limite)...")
        documentos = random.Random(42).sample(documentos, limite)
    estatisticas["total_antes_limite"] = len(medquad) + len(pubmed)
    estatisticas["total_indexar"] = len(documentos)

    logarAcao("tratarDadosDataset", "Gravando data/processed...")
    DIR_PROCESSED.mkdir(parents=True, exist_ok=True)
    with open(DIR_PROCESSED / "documentos.jsonl", "w", encoding="utf-8") as f:
        for d in documentos:
            f.write(json.dumps({"page_content": d.page_content, "metadata": d.metadata},
                               ensure_ascii=False) + "\n")
    with open(DIR_PROCESSED / "estatisticas.json", "w", encoding="utf-8") as f:
        json.dump(estatisticas, f, ensure_ascii=False, indent=2)

    detalhe = (f"{len(medquad)} chunks MedQuAD + {len(pubmed)} PubMedQA"
               f" -> {len(documentos)} para indexar")
    return {"documentos_tratados": documentos, "estatisticas": estatisticas,
            "caminho": registrarPasso("tratarDadosDataset", detalhe)}


def indexarDocumentos(estado: EstadoIndexacao) -> dict:
    unicos = {}
    for doc in estado["documentos_tratados"]:
        unicos.setdefault(doc.metadata["id"], doc)
    documentos = list(unicos.values())
    descartados = len(estado["documentos_tratados"]) - len(documentos)

    logarAcao("indexarDocumentos", "Apagando índice antigo...")
    shutil.rmtree(DIR_FAISS, ignore_errors=True)
    invalidarVectorStore()
    indice = None
    lotes = -(-len(documentos) // TAMANHO_LOTE)
    logarAcao("indexarDocumentos", f"Gerando embeddings e gravando {len(documentos)} vetores em {lotes} lotes...")
    for i in tqdm(range(0, len(documentos), TAMANHO_LOTE), desc="Indexando", unit="lote"):
        lote = documentos[i:i + TAMANHO_LOTE]
        ids = [d.metadata["id"] for d in lote]
        if indice is None:
            indice = FAISS.from_documents(lote, obterEmbeddings(), ids=ids,
                                          distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
                                          normalize_L2=True)
        else:
            indice.add_documents(lote, ids=ids)
    total = 0
    if indice is not None:
        logarAcao("indexarDocumentos", f"Salvando índice em {DIR_FAISS.name if DIR_FAISS.name != 'faiss' else 'data/faiss'}...")
        DIR_FAISS.mkdir(parents=True, exist_ok=True)
        indice.save_local(str(DIR_FAISS))
        total = indice.index.ntotal
    invalidarVectorStore()
    detalhe = f"{total} vetores indexados"
    if descartados:
        detalhe += f" ({descartados} documentos com ID repetido descartados)"
    return {"base_pronta": True, "documentos_brutos": [], "documentos_tratados": [],
            "caminho": registrarPasso("indexarDocumentos", detalhe)}


def construirIndexarBase(para_desenho: bool = False):
    g = StateGraph(EstadoIndexacao, output_schema=SaidaIndexacao)
    g.add_node("extrairDadosDataset", extrairDadosDataset)
    g.add_node("tratarDadosDataset", tratarDadosDataset)
    g.add_node("indexarDocumentos", indexarDocumentos)
    g.add_edge(START, "extrairDadosDataset")
    g.add_edge("extrairDadosDataset", "tratarDadosDataset")
    g.add_edge("tratarDadosDataset", "indexarDocumentos")
    g.add_edge("indexarDocumentos", END)
    return g.compile(checkpointer=None if para_desenho else False)
