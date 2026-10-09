# Soluções do instrutor

Nível 1: acurácia 0,7; precisão 0,8; revocação 2/3; especificidade 0,75; F1 8/11.
O baseline de 90% deixa passar todos os positivos.

Nível 2: VN/(VN+FP); sem negativos, denominador zero, portanto `None`.
Em limiares 0,3/0,5/0,7, as matrizes são [[1,1],[0,2]], [[2,0],[0,2]], [[2,0],[1,1]].

Nível 3: custos na matriz inicial 42 e 24. Nos limiares manuais: primeiro custo
2/0/20; segundo 20/0/2. Esses quatro pontos não demonstram generalização nem
validade dos custos. Fixar a política na validação e avaliar uma vez no teste.
