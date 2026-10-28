"""Seus testes do Marco 1 (veja a seção 6 do ENUNCIADO_MARCO1.md).

Regras: pelo menos 8 testes seus (o exemplo abaixo não conta), uma docstring em cada teste
dizendo qual caso ele cobre, e floats comparados com assertAlmostEqual ou np.allclose.
"""
import unittest

from elevatoria.dados import por_tag
from elevatoria.serie import SerieTemporal
import numpy as np

class TestMediaMovelRegressao(unittest.TestCase):

    def test_media_movel(self):
        lista = SerieTemporal.de_lista(list(range(1000)))
        resultado = lista.media_movel(40)

        self.assertEqual(len(resultado), 961)

class TestReamostras(unittest.TestCase):

    def test_descarta_sobra(self):
        lista = SerieTemporal.de_lista(list(range(11)))
        resultado = lista.reamostrar(2)

        self.assertEqual(len(resultado), 5)
        self.assertTrue(np.allclose(resultado, [0.5, 2.5, 4.5, 6.5, 8.5]))

class TestPorTagExemplo(unittest.TestCase):
    """Exemplo do formato. `por_tag` é código existente da base."""

    def test_lista_vazia(self):
        """Sem registros, por_tag devolve uma lista vazia."""
        self.assertEqual(por_tag([], "PT101"), [])


# Acrescente abaixo as suas classes de teste, começando pelos testes de regressão da
# issue #4 (escreva-os ANTES de corrigir o defeito e confirme que falham).


if __name__ == "__main__":
    unittest.main()
