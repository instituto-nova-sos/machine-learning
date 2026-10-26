import time
import pandas as pd
from geopy.distance import geodesic
from geopy.geocoders import Nominatim

# 1. Carregar o CSV com os endereços do Rio de Janeiro
df = pd.read_csv('enderecos_rj.csv')

# 2. Configurar o Geolocalizador
geolocalizador = Nominatim(user_agent="projeto_otimizador_rj")

# 3. Pegar o endereço digitado pelo usuário
lugar_input = input('Digite o seu endereço completo: ')
print(f"\nBuscando coordenadas para: {lugar_input}...")

user_loc = geolocalizador.geocode(lugar_input)

if not user_loc:
    print("Não foi possível encontrar este endereço. Tente ser mais específico.")
    exit()

user_coords = (user_loc.latitude, user_loc.longitude)
print(f"Sua coordenada: {user_coords}\n")

print("Calculando distâncias para os endereços do CSV...")

menor_distancia = float('inf')
endereco_mais_proximo = None

# 4. Percorrer o CSV calculando a distância de cada endereço
for index, row in df.iterrows():
    endereco_completo = row['Endereco_Completo']

    try:
        # Geocodifica o endereço do CSV
        loc = geolocalizador.geocode(endereco_completo)

        if loc:
            target_coords = (loc.latitude, loc.longitude)

            # Calcula a distância em linha reta (pode usar OSRM se quiser rota de carro)
            distancia = geodesic(user_coords, target_coords).kilometers

            print(
                f"- {endereco_completo} fica a {distancia:.2f} km de distância."
            )

            # Verifica se é o menor caminho até agora
            if distancia < menor_distancia:
                menor_distancia = distancia
                endereco_mais_proximo = endereco_completo

        # Pausa obrigatória de 1 segundo para respeitar a política do Nominatim
        time.sleep(1)

    except Exception as e:
        print(f"Erro ao processar o endereço {endereco_completo}: {e}")

# 5. Resultado final
if endereco_mais_proximo:
    print("\n" + "=" * 50)
    print("🎯 CAMINHO MAIS ÓTIMO ENCONTRADO:")
    print(f"O destino mais próximo é: {endereco_mais_proximo}")
    print(f"Distância aproximada: {menor_distancia:.2f} km")
    print("=" * 50)
else:
    print("Não foi possível calcular a rota para os endereços da lista.")