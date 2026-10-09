# Soluções do instrutor

Nível 1: 8+4+4+1=17; 68 bytes float32, 136 float64. JSON e runtime adicionam custo.
Nível 2: load não pode chamar fit; média/escala são persistidas, não recalculadas.
Carga inclui leitura/parse/reconstrução; pipeline mede transformações/forward/política.
Nível 3: exigir ambiente, repetições e unidade. Uma redução de precisão pode trocar
classe e abstenção. Medir métricas/latência antes e depois. Local mantém entrada
no computador, mas logs e permissões ainda importam; remoto envolve envio e rede.
