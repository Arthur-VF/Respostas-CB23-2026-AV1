"""Leitura do log SCADA da estação elevatória e preparação dos dados.

Este módulo é a porta de entrada da aplicação: transforma o texto do log em registros
estruturados e em séries numéricas. As funções marcadas com `NotImplementedError` são as
issues abertas do Marco 1; as docstrings descrevem o contrato que cada uma deve cumprir.
Não mude nomes nem assinaturas públicas: outros módulos e os testes dependem deles.
"""
from __future__ import annotations

import array
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Callable

import numpy as np

from elevatoria.serie import SerieTemporal
from fornecido.cronometro import cronometrar

# ---------------------------------------------------------------------------------------
# Issue #1: leitor do log
# ---------------------------------------------------------------------------------------

#: Padrão de uma linha válida do log, compilado com `re.compile(..., re.VERBOSE)` e com um
#: comentário em cada parte. Grupos nomeados: `data` (AAAA-MM-DD), `hora` (HH:MM:SS),
#: `nivel` (INFO, WARN ou ALARME), `tag` (1 ou 2 letras maiúsculas seguidas de 1 a 3
#: dígitos) e `resto` (o restante da linha, começando por um caractere que não é espaço).
#: As partes são separadas por um ou mais espaços. A linha inteira deve casar: o padrão é
#: aplicado com `LINHA.fullmatch(linha)`.
LINHA: re.Pattern = re.compile(r"""
    (?P<data>\d{4}-\d{2}-\d{2})\s+
      (?P<hora>\d{2}:\d{2}:\d{2})\s+
        (?P<nivel>INFO|WARN|ALARME)\s+
          (?P<tag>[A-Z]{1,2}\d{1,3})\s+
            (?P<resto>.*)""", re.VERBOSE)  # type: ignore[assignment]  # TODO issue #1


@dataclass(frozen=True)
class Registro:
    """Uma linha válida do log: instante, nível, tag e os pares chave=valor convertidos."""

    instante: datetime
    nivel: str
    tag: str
    valores: dict[str, float | str]


def valida_tag(s: str) -> bool:
    padrao = r"[A-Z]{1,2}\d{1,3}"
    return re.fullmatch(padrao, s) is not None


def ler_log(texto: str) -> tuple[list[Registro], list[str]]:


    if texto == "":
        return [], []

    invalidas = []
    registros = []

    for linha in texto.splitlines():
        if linha.strip() == "":
            continue

        resultado = LINHA.fullmatch(linha)
        if resultado is None:
            invalidas.append(linha)
            continue

        data = resultado.group("data")
        horas = resultado.group("hora")

        try:
            instante = datetime.strptime(
                f"{data} {horas}", "%Y-%m-%d %H:%M:%S"
            )
        except ValueError:
            invalidas.append(linha)
            continue

        resto = resultado.group("resto")
        pares = re.findall(r"(\w+)=(\S+)", resto)

        if not pares:
            invalidas.append(linha)
            continue

        valores = {}
        for restos in pares:
            valor = restos[1]
            chave = restos[0]

            try:
                valor = float(valor)
            except ValueError:
                pass

            valores[chave] = valor

        nivel = resultado.group("nivel")
        tag = resultado.group("tag")

        registro = Registro(instante=instante,nivel=nivel,tag=tag,valores=valores)
        registros.append(registro)

    return registros, invalidas



def por_tag(registros: list[Registro], tag: str) -> list[Registro]:
    """Registros da `tag`, na ordem original."""
    return [r for r in registros if r.tag == tag]


def contagem_por_tag(registros: list[Registro]) -> dict[str, int]:
    tags = {r.tag for r in registros}
    contagem = {tag: sum(1 for r in registros if r.tag==tag) for tag in tags}
    return contagem

def serie(registros: list[Registro], tag: str, chave: str) -> SerieTemporal:
    valor = []
    for i in registros:
        if i.tag == tag:
            for chaves in i.valores:
                if chaves == chave:
                    valor.append(i.valores[chaves])
    return SerieTemporal.de_lista(valor)                




# ---------------------------------------------------------------------------------------
# Issue #2: conversores por tag
# ---------------------------------------------------------------------------------------
def criar_conversores(escalas: dict[str, float]) -> dict[str, Callable[[float], float]]:
    conversores = {}
    for tag, fator in escalas.items():
        conversores[tag] = lambda c, fator=fator: fator * c 
    return conversores    


# ---------------------------------------------------------------------------------------
# Issue #3: memória e tempo
# ---------------------------------------------------------------------------------------
def medir_memoria(contagens: np.ndarray) -> dict[str, int]:
    from array import array

    if isinstance(contagens, np.ndarray) is False:
        raise ValueError("contagens deve ser um ndarray")
    
    if np.issubdtype(contagens.dtype, np.integer) is False or contagens.ndim != 1:
        raise ValueError("Os números não são inteiros e/ou o array não possui 1D")

    dicionario = {}

    lista = contagens.tolist()
    j = sys.getsizeof(lista)
    for i in lista:
        j += sys.getsizeof(i)
    dicionario["list"] = j

    dados_array = array("H", contagens.tolist())

    k = dados_array.itemsize * len(contagens)
    dicionario["array('H')"] = k

    uint = contagens.astype(np.uint16).nbytes
    dicionario["uint16"] = uint
    int32 = contagens.astype(np.int32).nbytes
    dicionario["int32"] = int32
    int64 = contagens.astype(np.int64).nbytes
    dicionario["int64"] = int64
    float32 = contagens.astype(np.float32).nbytes
    dicionario["float32"] = float32
    float64 = contagens.astype(np.float64).nbytes
    dicionario["float64"] = float64

    return dicionario


def converter_laco(contagens: list[int]) -> list[float]:
    """Converte contagens em kPa (`c * 600 / 4095`) com `for` e `append`."""
    resultado = []
    for i in contagens:
        resultado.append(i * 600 / 4095)
    return resultado    


def converter_compreensao(contagens: list[int]) -> list[float]:
    return [i * 600 / 4095 for i in contagens]

def converter_map(contagens: list[int]) -> list[float]:
    return list(map(lambda x: x * 600 / 4095, contagens))


def converter_vetorizado(contagens: np.ndarray) -> np.ndarray:
    lista = np.array(contagens)
    resultado = lista * (600 / 4095)
    return resultado


def medir_tempos(contagens: np.ndarray, k: int = 10) -> dict[str, float]:
    """Tempo médio (s) de cada versão da conversão, em `k` repetições.

    Use `cronometrar` (de `fornecido.cronometro`). Chaves: `"laço"`, `"compreensão"` e
    `"map+lambda"` (que recebem a lista `contagens.tolist()`, criada uma única vez, fora da
    medição) e `"vetorizado"` (que recebe o próprio `ndarray`).
    """
    lista = contagens.tolist()
    tempo_laco = cronometrar(converter_laco, lista, k=k)
    tempo_compreensão = cronometrar(converter_compreensao, lista, k=k)
    tempo_maplambda = cronometrar(converter_map, lista, k=k)
    tempo_vetorizado = cronometrar(converter_vetorizado, contagens, k=k)
    dicionario_tempo = {}
    dicionario_tempo["laço"] = tempo_laco
    dicionario_tempo["compreensão"] = tempo_compreensão
    dicionario_tempo["map+lambda"] = tempo_maplambda
    dicionario_tempo["vetorizado"] = tempo_vetorizado
    return dicionario_tempo