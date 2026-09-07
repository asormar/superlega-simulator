import re
import pandas as pd

def parse_flashscore_data(raw_text, season_year):
    # Expresión regular para capturar los equipos y el marcador
    # Busca: Nombre Equipo1 \n Nombre Equipo2 \n Marcador (empieza por 30, 31, 32, 23, 13, 03)
    pattern = r"([a-zA-Z\s]+)\n([a-zA-Z\s]+)\n(\d{2})"
    
    matches = re.findall(pattern, raw_text)
    
    data = []
    for m in matches:
        home_team = m[0].strip()
        away_team = m[1].strip()
        score = m[2] # Los primeros dos dígitos son los sets
        
        home_sets = int(score[0])
        away_sets = int(score[1])
        
        data.append({
            "season": season_year,
            "home_club": home_team,
            "away_club": away_team,
            "home_sets": home_sets,
            "away_sets": away_sets
        })
    
    return pd.DataFrame(data)

# Tu texto copiado (puedes pegarlo todo aquí)
texto_flashscore = """
Jornada 14
10.02.2021
MonzaMonza
VeronaVerona
30251625232523
Jornada 16
10.02.2021
RavennaRavenna
MilanoMilano
131425222525211825
Jornada 22
07.02.2021
PiacenzaPiacenza
MilanoMilano
3225221825251518251511
07.02.2021
PadovaPadova
VeronaVerona
3223252523252213251510
06.02.2021
TrentinoTrentino
RavennaRavenna
312519232525232519
06.02.2021
PerugiaPerugia
MonzaMonza
03122518252025
06.02.2021
ModenaModena
CisternaCisterna
312624192525132518
06.02.2021
Vibo ValentiaVibo Valentia
Lube CivitanovaLube Civitanova
3222252125252225211510
Jornada 21
03.02.2021
RavennaRavenna
PadovaPadova
312725251719252522
03.02.2021
MilanoMilano
TrentinoTrentino
30252225232522
03.02.2021
VeronaVerona
ModenaModena
132125251523252125
03.02.2021
Lube CivitanovaLube Civitanova
PerugiaPerugia
2325212325252320251416
03.02.2021
MonzaMonza
PiacenzaPiacenza
03202530322225
02.02.2021
CisternaCisterna
Vibo ValentiaVibo Valentia
3215252725212525211512
Jornada 20
24.01.2021
RavennaRavenna
MonzaMonza
132522212515251525
24.01.2021
PiacenzaPiacenza
VeronaVerona
312517251923252520
24.01.2021
ModenaModena
PerugiaPerugia
132523212518252125
24.01.2021
CisternaCisterna
Lube CivitanovaLube Civitanova
03182518252125
24.01.2021
Vibo ValentiaVibo Valentia
TrentinoTrentino
03132516252025
24.01.2021
MilanoMilano
PadovaPadova
312628252225172520
Jornada 15
21.01.2021
MilanoMilano
ModenaModena
3220253028251821252018
21.01.2021
Lube CivitanovaLube Civitanova
MonzaMonza
312520251922252523
Jornada 19
19.01.2021
MonzaMonza
Vibo ValentiaVibo Valentia
312125252125182521
17.01.2021
VeronaVerona
RavennaRavenna
03232523252225
17.01.2021
PerugiaPerugia
MilanoMilano
312518232525162521
17.01.2021
PadovaPadova
PiacenzaPiacenza
2326242517262816251115
16.01.2021
Lube CivitanovaLube Civitanova
ModenaModena
3225202830251916251510
16.01.2021
TrentinoTrentino
CisternaCisterna
312426262425202518
Jornada 14
14.01.2021
Vibo ValentiaVibo Valentia
MilanoMilano
312521182525222512
Jornada 12
13.01.2021
TrentinoTrentino
PadovaPadova
30251325223533
Jornada 17
12.01.2021
PiacenzaPiacenza
ModenaModena
03333515252225
Jornada 18
10.01.2021
RavennaRavenna
PerugiaPerugia
03212521252225
10.01.2021
MilanoMilano
Lube CivitanovaLube Civitanova
2321253230252322251015
10.01.2021
TrentinoTrentino
MonzaMonza
3225162025251720251513
10.01.2021
CisternaCisterna
VeronaVerona
03232522252325
09.01.2021
PiacenzaPiacenza
Vibo ValentiaVibo Valentia
312225272525202522
Jornada 9
07.01.2021
MonzaMonza
RavennaRavenna
312520251622252522
Jornada 10
06.01.2021
TrentinoTrentino
MilanoMilano
30251825172515
Jornada 16
05.01.2021
Lube CivitanovaLube Civitanova
PiacenzaPiacenza
312519222525202519
Jornada 17
03.01.2021
VeronaVerona
MilanoMilano
312125252225222517
03.01.2021
PerugiaPerugia
TrentinoTrentino
132025252319252125
03.01.2021
Vibo ValentiaVibo Valentia
RavennaRavenna
30252125212522
03.01.2021
MonzaMonza
CisternaCisterna
312522232525232522
02.01.2021
PadovaPadova
Lube CivitanovaLube Civitanova
03232522251825
Jornada 16
30.12.2020
MonzaMonza
PadovaPadova
3223252523212525111513
Jornada 11
30.12.2020
RavennaRavenna
TrentinoTrentino
132025251813251925
Jornada 16
28.12.2020
VeronaVerona
Vibo ValentiaVibo Valentia
312125251925142522
27.12.2020
CisternaCisterna
PerugiaPerugia
03142519252125
27.12.2020
ModenaModena
TrentinoTrentino
131825272925232225
Jornada 10
23.12.2020
PerugiaPerugia
Lube CivitanovaLube Civitanova
2325222830182525181215
23.12.2020
ModenaModena
VeronaVerona
312519283025232519
Jornada 15
20.12.2020
TrentinoTrentino
PiacenzaPiacenza
30251625212517
20.12.2020
CisternaCisterna
RavennaRavenna
132225242625231925
20.12.2020
Vibo ValentiaVibo Valentia
PadovaPadova
30262425222725
19.12.2020
PerugiaPerugia
VeronaVerona
312520252226282520
Jornada 10
17.12.2020
PadovaPadova
RavennaRavenna
132426252725222225
Jornada 9
16.12.2020
VeronaVerona
PiacenzaPiacenza
132325192525212426
Jornada 10
16.12.2020
Vibo ValentiaVibo Valentia
CisternaCisterna
30251725162518
Jornada 14
14.12.2020
PadovaPadova
PerugiaPerugia
03242623251525
14.12.2020
TrentinoTrentino
Lube CivitanovaLube Civitanova
30252225232522
13.12.2020
PiacenzaPiacenza
CisternaCisterna
30252125192517
12.12.2020
ModenaModena
RavennaRavenna
30252125152512
Jornada 5
09.12.2020
TrentinoTrentino
ModenaModena
30262425222520
Jornada 8
09.12.2020
RavennaRavenna
VeronaVerona
03192520253335
Jornada 13
06.12.2020
ModenaModena
Vibo ValentiaVibo Valentia
03212516251825
06.12.2020
PadovaPadova
CisternaCisterna
312518202525162725
06.12.2020
VeronaVerona
TrentinoTrentino
2325232624222520251015
06.12.2020
PerugiaPerugia
PiacenzaPiacenza
30252025192521
05.12.2020
RavennaRavenna
Lube CivitanovaLube Civitanova
232521252321252025815
04.12.2020
MilanoMilano
MonzaMonza
2320252426251325221115
Jornada 9
03.12.2020
PerugiaPerugia
ModenaModena
30252025232624
02.12.2020
Lube CivitanovaLube Civitanova
CisternaCisterna
30252325152517
Jornada 12
29.11.2020
Vibo ValentiaVibo Valentia
PerugiaPerugia
132325292717252225
29.11.2020
CisternaCisterna
MilanoMilano
03212520251825
29.11.2020
MonzaMonza
ModenaModena
312514212525132522
28.11.2020
PiacenzaPiacenza
RavennaRavenna
311925252125212522
28.11.2020
Lube CivitanovaLube Civitanova
VeronaVerona
312517251219252513
Jornada 11
25.11.2020
CisternaCisterna
ModenaModena
231925251927252225715
Jornada 9
25.11.2020
TrentinoTrentino
Vibo ValentiaVibo Valentia
131825252223252325
Jornada 8
25.11.2020
MilanoMilano
PerugiaPerugia
132521202521252225
Jornada 11
22.11.2020
VeronaVerona
PadovaPadova
132125272523253335
22.11.2020
MonzaMonza
PerugiaPerugia
30252025212522
21.11.2020
MilanoMilano
PiacenzaPiacenza
132523162522251925
15.11.2020
Lube CivitanovaLube Civitanova
Vibo ValentiaVibo Valentia
132325251820252325
Jornada 10
14.11.2020
PiacenzaPiacenza
MonzaMonza
132225251626281725
Jornada 18
14.11.2020
ModenaModena
PadovaPadova
30252325173028
Jornada 9
07.11.2020
PadovaPadova
MilanoMilano
2316252521152525181115
Jornada 8
01.11.2020
CisternaCisterna
TrentinoTrentino
131725202525222426
01.11.2020
PiacenzaPiacenza
PadovaPadova
311625252025192522
01.11.2020
Vibo ValentiaVibo Valentia
MonzaMonza
312519182525222522
31.10.2020
ModenaModena
Lube CivitanovaLube Civitanova
03242622252325
Jornada 7
25.10.2020
Lube CivitanovaLube Civitanova
MilanoMilano
30252026242522
25.10.2020
MonzaMonza
TrentinoTrentino
3225212125192525161513
25.10.2020
PadovaPadova
ModenaModena
131725232525211825
25.10.2020
PerugiaPerugia
RavennaRavenna
30251825202624
25.10.2020
VeronaVerona
CisternaCisterna
3225192628252324261511
24.10.2020
Vibo ValentiaVibo Valentia
PiacenzaPiacenza
131925202525231925
Jornada 6
18.10.2020
CisternaCisterna
MonzaMonza
312523252321252523
18.10.2020
Lube CivitanovaLube Civitanova
PadovaPadova
30251125202514
18.10.2020
MilanoMilano
VeronaVerona
311525251927252522
18.10.2020
ModenaModena
PiacenzaPiacenza
30251625172624
18.10.2020
RavennaRavenna
Vibo ValentiaVibo Valentia
2325232325212525191517
18.10.2020
TrentinoTrentino
PerugiaPerugia
132517222517252527
Jornada 5
14.10.2020
PerugiaPerugia
CisternaCisterna
30252325212520
14.10.2020
MilanoMilano
RavennaRavenna
3225223032242625221512
14.10.2020
PadovaPadova
MonzaMonza
2328262515122524261416
14.10.2020
PiacenzaPiacenza
Lube CivitanovaLube Civitanova
132931212525202325
14.10.2020
Vibo ValentiaVibo Valentia
VeronaVerona
3225222325202525171511
Jornada 4
11.10.2020
ModenaModena
MilanoMilano
132426192525202426
11.10.2020
MonzaMonza
Lube CivitanovaLube Civitanova
03212523252025
11.10.2020
PadovaPadova
Vibo ValentiaVibo Valentia
132325242625232225
11.10.2020
RavennaRavenna
CisternaCisterna
30252225212521
11.10.2020
VeronaVerona
PerugiaPerugia
03222522252325
11.10.2020
PiacenzaPiacenza
TrentinoTrentino
03222513252125
Jornada 3
08.10.2020
RavennaRavenna
ModenaModena
132624222520251925
07.10.2020
CisternaCisterna
PiacenzaPiacenza
132522212522252125
07.10.2020
Lube CivitanovaLube Civitanova
TrentinoTrentino
30251526242518
07.10.2020
MilanoMilano
Vibo ValentiaVibo Valentia
131925252325272025
07.10.2020
PerugiaPerugia
PadovaPadova
30251825142521
07.10.2020
VeronaVerona
MonzaMonza
312521222525162523
Jornada 2
04.10.2020
CisternaCisterna
PadovaPadova
03222516251825
04.10.2020
Lube CivitanovaLube Civitanova
RavennaRavenna
322325252225142325159
04.10.2020
PiacenzaPiacenza
PerugiaPerugia
132225172525192025
04.10.2020
TrentinoTrentino
VeronaVerona
03262823252125
04.10.2020
Vibo ValentiaVibo Valentia
ModenaModena
03212526282025
03.10.2020
MonzaMonza
MilanoMilano
132225252119252025
Jornada 1
30.09.2020
PerugiaPerugia
Vibo ValentiaVibo Valentia
30353325202624
28.09.2020
VeronaVerona
Lube CivitanovaLube Civitanova
03232520252125
27.09.2020
MilanoMilano
CisternaCisterna
30252026242520
27.09.2020
ModenaModena
MonzaMonza
132025252323252225
27.09.2020
PadovaPadova
TrentinoTrentino
03252715252426
27.09.2020
RavennaRavenna
PiacenzaPiacenza
132025252023251925
"""

# Procesar
df_partidos = parse_flashscore_data(texto_flashscore, "2025/2026")

# Guardar
df_partidos.to_csv("DB/enfrentamientos_directos/enfrentamientos_directos.csv", index=False)
print(f"¡Hecho! Se han procesado {len(df_partidos)} partidos.")
print(df_partidos.head())