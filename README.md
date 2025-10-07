# Avinor - Prediksjon av avvik i flytrafikken 

## Bakgrunn 
Avinor er en av verdens ledende aktører innen fjernstyrte tårn (remote towers), der lufttrafikktjenester leveres for én eller flere flyplasser fra et senter ved hjelp av kameraer og sensorer i stedet for et tradisjonelt tårn på stedet. Remote Towers betyr at flygeinformasjon blir gitt fra et annet sted enn selve flyplassen.

AFIS-fullmektig (Aerodrome Flight Information Service) har ansvar for å overvåke luftrommet rundt en flyplass, gi trafikkinformasjon til piloter og bidra til sikkerhet – uten å gi direkte instruksjoner slik som flygeledere gjør. På flyplasser med lite trafikk er det mulig å ha en AFIS-fullmektig med ansvar for tre flyplasser parallelt. Da har AFIS-fullmektig ansvar for en flyplassgruppe. Det skal nå etableres seks slike grupper som hver kontrollerer tre flyplasser samtidig. Dette kalles multippel-drift.

Formålet med dette prosjektet er å analysere når AFIS-fullmektig opplever størst arbeidsbelastning. Belastningen påvirkes særlig av to forhold:
1. Antall fly det kommuniseres med samtidig
2. Krevende operative forhold, værforhold som fører til hyppigere og mer detaljert dialog med pilotene.

## Problembeskrivelse 
Formålet med dette prosjektet er å analysere når AFIS-fullmektig opplever høy arbeidsbelastning. AFIS-fullmektig vil oppleve høy arbeidsbelastning når flere fly krever informasjon på samme tid. Arbeidsbelastningen øker også når det er krevende værforhold. Vi ønsker at deltagerne utvikler en modell som predikerer sannsynlighet for samtidighet (dvs. at AFIS-fullmektig må kommunisere med flere fly samtidig). Dersom fly fulgte ruteplanen hadde det vært enkelt å vite dette, men fly påvirkes av vær og andre operasjonelle forhold som gjør at det oppstår forsinkelser2. I tillegg er det en del uregelmessig trafikk som følge av ambulansefly eller annen trafikk på flyplassen. Det er også viktig for AVINOR å ha en forklarbar modell av regulatoriske hensyn (AVINOR må kunne dokumentere for luftfartstilsynet at vi har rett bemanning under gitte forhold).

### Prosjekt leveranse 
Prosjektet har som mål å utvikle samtidighetsmodell – Sannsynlighet for samtidig aktiv dialog

[Bakgrunn for prosjekt.pdf](https://github.com/user-attachments/files/22747900/Bakgrunn.for.prosjekt.pdf)

## Metodevalg og tilnærming 

I dette prosjektet har vi utviklet en maskinlæringsbasert løsning for å predikere samtidige hendelser i flytrafikken basert på historiske flydata fra Avinor. Løsningen bygger på en modulær pipeline for datarensing, feature engineering og modellering, med XGBoost som hovedmodell. Modellen oppnådde en ROC-AUC på 0.79 på testdatasettet, noe som viser en tydelig forbedring sammenlignet med en tilfeldig baseline. Systemet er designet for å være skalerbart og kan videreutvikles med flere datakilder, som værdata og sanntidsinformasjon, for å gi enda mer presise prediksjoner. 

### Metodikken besto av: 
1. Datagrunnlag: Historiske flydata med planlagte og faktiske tider, samt metadata som flyplassgrupper.Prosjektet hadde opprinnelig som plan å bruke værdata fra Meteorologisk institutt, men dette lot seg ikke gjennomføre på grunn av problemer med API-kallene 
2. Preprosessering: Preprosesseringsfunksjonen rydder og strukturerer flydata ved å fjerne kansellerte flyvninger, kombinere avganger og ankomster, beregne samtidige flyvninger (concurrency) og legge til avledede variabler som dato, time og årstid, aggregert per flyplassgruppe. pipeline_preprocessor funksjonen gjør så datasettet klart for maskinlæring ved å skalere de numeriske variablene og one-hot-enkode de kategoriske, slik at modellen kan bruke begge typer informasjon effektivt. 
3. Modellering: I prosjektet er det brukt en maskinlæringsmodell basert på XGBoost, som kombineres med en preprosesseringspipeline for å håndtere både numeriske og kategoriske data. Hyperparametere ble justert med GridSearchCV og evaluert med AUC-ROC via kryssvalidering, slik at den beste modellen ble valgt ut. 
4. Evaluering: For å evaluere modellen ble det brukt AUC-ROC som ytelsesmål. På testdatasettet oppnådde modellen en AUC-ROC-score på rundt 0.79 noe som viser at den har en grei evne til å skille mellom perioder med og uten samtidige flyvninger. I tillegg ble en ROC-kurve generert, der modellens ytelse sammenlignes med en tilfeldig baseline. Figuren viser at modellen ligger tydelig over baselinen, noe som bekrefter at den gir bedre prediksjoner enn ren gjetting. 

Metoden er egnet fordi den tar hensyn til både kategoriske og numeriske input, og kan beregne sannsynligheter (prediksjoner mellom 0 og 1), noe som er påkrevd i konkurranseformatet. 

## Systemarkitektur og arkitektur 

Løsningen er bygget opp i en modulær arkitektur med klare komponenter samlet i src/-mappen. Hver modul har et tydelig ansvar, og systemet kan derfor enkelt videreutvikles eller skaleres: 

- Datahåndtering (data_preprocessing.py): Leser inn flydata, håndterer rensing og transformerer rådata til et treningsklart format. Kategoriske variabler som airport_group gjøres numeriske via one-hot encoding. 
- Feature engineering: Innebygd i datahåndteringen er logikk for å beregne samtidige flyvninger og legge til informasjon som kan fange opp sesongvariasjoner. 
- Modell (model.py): Definerer de maskinlæringsmodellene som benyttes. Vi har testet både Logistic Regression og Random Forest, og vi har også strukturert koden slik at mer avanserte modeller som XGBoost kan integreres i samme pipeline. 
- Trening (train.py): Kjører hele treningsprosessen: preprosessering, splitting av data i trenings- og testsett, trening av modell, og lagring av resultatene. Treningen utvides med hyperparameter-tuning ved bruk av GridSearchCV. 
- Evaluering (evaluate.py): Evaluerer den trente modellen med sentrale metrikker. Fokus har vært på ROC-AUC, som er konkurransens offisielle evalueringsmetode, men vi genererer også confusion matrix, classification report og ROC-kurver for mer detaljert analyse. 
- Prediksjon (main.py): Benyttes til å kjøre modellen på nye datasett (f.eks. preds_mal.csv) og lagrer resultatene i CSV-format med sannsynligheter (pred-kolonnen). Dette sikrer at innleveringen følger kravene i konkurransen.

### Oversikt over system arkitektur 

<img width="481" height="584" alt="Avinor_Predictions_Architecture" src="https://github.com/user-attachments/assets/718aa884-6a00-48b3-9bc7-417de6d05dab" />

### Evaluering av modeller 

<img width="1478" height="830" alt="image" src="https://github.com/user-attachments/assets/b887e1df-dc2a-4e70-8f7a-17a3a6243778" />




## Modellen for vår prediksjon - XGBoost 

### Modell: 

- XGBoost Classifier integrert i en Scikit-learn Pipeline med preprocessing. 
- Grunnen til at XGBoost er valgt er modellens styrke på strukturerte data og evne til å håndtere både lineære og ikke-lineære sammenhenger. 

### Hyperparametere: 

- learning_rate: [0.01, 0.1, 0.2] 
- max_depth: [3, 6, 9] 
- n_estimators: [100, 200] 
- Optimal kombinasjon velges automatisk gjennom GridSearchCV. 

### Evalueringsmetodikk: 

Delt datasettet i trenings- og testsett basert på en tidsbasert cutoff (1. juli 2024). 
Treningssett brukes for kryssvalidering (3-fold), mens testsett brukes for sluttvurdering. 
Evalueringsmål: ROC-AUC, som er mer robust enn ren nøyaktighet i denne konteksten. 

## Kildekode 
### Struktur: 
- src/data_preprocessing.py – Rensing, feature engineering, splitting, pipeline-definisjon. 
- src/model.py – Opprettelse av XGBoost-modellen og parameterrutenett. 
- src/train.py – Trening med GridSearchCV, samt prediksjonsfunksjon. 
- src/evaluate.py – Evaluering og visualisering (ROC-kurve). 
- notebooks/ – Utforskende dataanalyse (EDA). 
- data/ – Trenings- og testdata. 

<img width="268" height="829" alt="Skjermbilde 2025-10-07 132842" src="https://github.com/user-attachments/assets/109070ff-0a8a-4c4b-ad33-75cd7fc370af" />

### Hvordan kjøre løsningen: 

- Installer avhengigheter med: poetry install 
- Kjør evalueringen: python -m src.evaluate - Dette trener modellen og lagrer ROC-kurven og confusion matrisen som png og klassifikasjons rapporten printes. 
- Kjør treningen of prediksjonen: python main.py - om du ønsker å predikere for annen data må du endre filbanen i main.py til csv-filen du ønsker å predikere for. Filen kjører treningen og predikerer for datasettet for oktober (oppdatert). 
- Resultatet lagres som data/tootoonchi_minaipour.csv - Dette kan endres i funksjonen model_predict som ligger i train.py.

### Videreutvikling og skalering: 
Kan utvides med flere features (værdata, flytype, trafikkvolum). En EDA for værdata har blitt gjennomført og dette anbefales å bygge på. I tillegg kan man hente inn data fra flere kilder, for eksempel sanntidsdata fra flyplassystemer eller eksterne API-er, for å gjøre prediksjonene mer presise. På modellnivå kan det testes mer avanserte algoritmer, som dyp læring eller eventuelle ensemble-metoder, og sammenlignes mot XGBoost for å finne den beste tilnærmingen. 
