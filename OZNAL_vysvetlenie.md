# OZNAL Projekt — Kompletné vysvetlenie

> Tento dokument vysvetľuje čo, prečo a ako sa v projekte robilo. Je štruktúrovaný tak, aby ho bolo možné rozširovať o ďalšie scenáre. Obsahuje metodologické vysvetlenia, poučky k teórii a popis netriviálnych príkazov.

---

## Obsah

1. [Projekt a dataset](#1-projekt-a-dataset)
2. [Exploratory Data Analysis (EDA)](#2-exploratory-data-analysis-eda)
3. [Preprocessing a príprava dát](#3-preprocessing-a-príprava-dát)
4. [Scenario 2: Parametrické modely](#4-scenario-2-parametrické-modely)
5. [Vyhodnocovacie metriky](#5-vyhodnocovacie-metriky)
6. [Výsledky a záver — Scenario 2](#6-výsledky-a-záver--scenario-2)
7. [Slovník kľúčových príkazov](#7-slovník-kľúčových-príkazov)

---

## 1. Projekt a dataset

### Zdroj a obsah

Dataset pochádza z [fueleconomy.gov](https://fueleconomy.gov) — oficiálna databáza EPA (Environmental Protection Agency, USA), ktorá testuje spotrebu paliva každého vozidla predávaného na americkom trhu od roku 1984.

| Vlastnosť | Hodnota |
|-----------|---------|
| Riadky | 49 846 |
| Stĺpce | 84 |
| Roky | 1984 – 2026 |
| Cieľová premenná | `comb08` |

### Cieľová premenná: `comb08`

`comb08` vyjadruje **kombinovanú spotrebu paliva v MPG** (Miles Per Gallon — počet míľ na galón paliva). EPA ho počíta ako vážený priemer mestskej jazdy (55%) a diaľničnej jazdy (45%).

- **Vyšší MPG = efektívnejšie vozidlo** (menej paliva na rovnakú vzdialenosť)
- Bežné hodnoty: 15–30 MPG pre klasické autá, 40–60 MPG pre hybridy
- 1 MPG ≈ 235 l/100 km (pre prepočet: l/100 km = 235.21 / MPG)

### Dôležité rozlíšenie: MPG vs. MPGe

Pre elektrické vozidlá EPA uvádza **MPGe** (Miles Per Gallon *equivalent*) — umelý prepočet kde EPA stanovila, že 33,7 kWh = 1 galón. Napr. Tesla Model 3 má ~130 MPGe — čo **neznamená**, že spotrebuje galón každých 130 míľ, ale že jej spotreba elektrickej energie zodpovedá tejto ekvivalencii.

**MPG a MPGe sú rôzne fyzikálne veličiny, ktoré zdieľajú rovnaký stĺpec `comb08`.** Toto je kľúčový dôvod, prečo sme čisto elektrické vozidlá (BEV) vyradili z modelovania — viac v sekcii Preprocessing.

---

## 2. Exploratory Data Analysis (EDA)

### Čo je EDA a prečo sa robí

EDA (Exploratory Data Analysis) je prvá, neformálna fáza každého projektu strojového učenia. Jej cieľom **nie** je stavať modely, ale **porozumieť dátam** predtým, ako sa s nimi niečo robí.

Konkrétne hľadáme:
- Aké premenné máme, aké typy a aké hodnoty nadobúdajú
- Kde sú chýbajúce hodnoty a koľko ich je
- Ako vyzerajú distribúcie — sú symetrické, skewed, bimodálne?
- Aké sú korelácie medzi prediktormi a s cieľovou premennou
- Či existujú outliere alebo podozrivé hodnoty
- Čo treba spraviť v preprocessingu (imputovať, transformovať, vyradiť)

> **📚 Poučka: Prečo EDA záleží**
>
> "Garbage in, garbage out" — model je len taký dobrý, ako sú dáta. EDA odhalí problémy skôr, než strávíš hodiny trénovaním modelu na zlých dátach. Veľa dátových vedcov odhaduje, že EDA + preprocessing zaberie 70–80% času projektu.

### Základná štruktúra

```r
dim(data)     # rozmery: 49846 riadkov, 84 stĺpcov
glimpse(data) # rýchly prehľad typov a prvých hodnôt
```

**`glimpse()`** je funkcia z balíka `dplyr`. Vypíše každý stĺpec na jeden riadok spolu s jeho dátovým typom (`<chr>`, `<dbl>`, `<int>`, `<lgl>`) a prvými hodnotami. Je to čitateľnejší alternatív k základnej funkcii `str()`.

Typy stĺpcov v R:
| Typ | Čo je | Príklad |
|-----|-------|---------|
| `<dbl>` | desatinné číslo (double) | 3.14, 20.5 |
| `<int>` | celé číslo (integer) | 4, 8, 12 |
| `<chr>` | reťazec (character/text) | "Toyota", "Diesel" |
| `<lgl>` | logická hodnota | TRUE, FALSE |

### Chýbajúce hodnoty

```r
data %>%
  summarise(across(everything(), ~mean(is.na(.)))) %>%
  pivot_longer(everything(), names_to = "column", values_to = "missing_pct") %>%
  filter(missing_pct > 0.6) %>%
  arrange(desc(missing_pct))
```

- **`across(everything(), ~mean(is.na(.)))`** — pre každý stĺpec spočíta podiel NA hodnôt (mean z TRUE/FALSE = podiel TRUEčiek)
- **`pivot_longer()`** — transformuje wide → long formát (vysvetlené v sekcii 7)

**Zistenie:** Stĺpce `evMotor`, `rangeA`, `fuelType2`, `c240bDscr`, `c240Dscr`, `sCharger`, `guzzler`, `atvType`, `tCharger`, `trans_dscr`, `startStop`, `mfrCode` majú >60% chýbajúcich hodnôt → vyradené.

> **📚 Poučka: Kedy vyradiť stĺpec vs. imputovať**
>
> Neexistuje jedno presné pravidlo, ale praktická orientácia:
> - **< 5% NA** → imputácia je bezpečná
> - **5–20% NA** → imputácia s opatrnosťou; závisí od toho, prečo hodnoty chýbajú
> - **20–60% NA** → zvážiť vyradenie; imputácia vnáša veľa šumu
> - **> 60% NA** → spravidla vyradiť (ako sme spravili my)
>
> Dôležitejšia otázka ako "koľko NA" je: **prečo hodnoty chýbajú?**
> - Ak chýbajú náhodne (MCAR — Missing Completely At Random) → imputácia je bezpečná
> - Ak chýbajú systematicky (napr. `displ` chýba u elektromobilov) → chýbanie samo o sebe nesie informáciu

### Distribúcia cieľovej premennej

```r
ggplot(data, aes(x = comb08)) +
  geom_histogram(bins = 50) +
  theme_minimal()
```

Distribúcia `comb08` je **right-skewed** (pravostranná asymetria):
- Väčšina vozidiel: 15–30 MPG
- Dlhý chvost vpravo: hybridné a elektrické vozidlá s >50 MPGe

Po vyradení BEV je rozsah 7–74 MPG, distribúcia je výrazne homogénnejšia.

### Korelácie

```r
ggcorrplot(cor_matrix, method = "circle", type = "lower", lab = TRUE)
```

**`ggcorrplot()`** zobrazí korelačnú maticu ako farebný heatmap. `type = "lower"` zobrazí len dolný trojuholník (symetria).

Kľúčové korelácie s `comb08`:
| Premenná | Korelácia | Interpretácia |
|----------|-----------|---------------|
| `cylinders` | ~ −0.77 | Viac valcov = vyššia spotreba |
| `displ` | ~ −0.75 | Väčší objem = vyššia spotreba |
| `year` | ~ +0.37 | Novšie autá sú efektívnejšie |

`cylinders` a `displ` sú navzájom korelované ~0.91 — to je **multikolinearita**.

> **📚 Poučka: Multikolinearita**
>
> Multikolinearita nastáva keď sú dva alebo viac prediktorov silne navzájom korelované. Pre OLS to znamená problém: model nevie rozlíšiť "kto z nich" vlastne spôsobuje zmenu v `comb08`. Koeficienty môžu byť nestabilné — malá zmena v dátach by zmenila koeficient výrazne.
>
> **Riešenie:** Ridge regresia — penalizuje veľké koeficienty a tým "rozdelí váhu" medzi korelované prediktory férovejšie.
>
> **Detekcia:** Korelácia >0.8 medzi dvoma prediktormi je varovný signál. Formálnejšia metóda je VIF (Variance Inflation Factor) — VIF > 10 signalizuje problém.

### Kategorické premenné

```r
data %>%
  select(fuelType, drive, VClass, trany) %>%
  pivot_longer(everything(), names_to = "variable", values_to = "value") %>%
  ggplot(aes(y = value)) +
  geom_bar() +
  facet_wrap(~variable, scales = "free")
```

- **`fuelType`**: silne nevyvážená — "Regular" a "Premium" dominujú, exotické palivá (Hydrogen, CNG) sú vzácne → grouping do `fuel_group`
- **`drive`**: FWD a RWD dominujú → ostatné typy 4WD zlúčené
- **`VClass`**: bežné triedy dobre zastúpené
- **`trany`**: 200+ unikátnych hodnôt ako reťazce → feature engineering do `transmission_type`, `is_cvt`, `n_gears`

---

## 3. Preprocessing a príprava dát

### Čo je preprocessing a prečo

Surové dáta sú zriedka vo formáte, v akom ich model dokáže použiť. Preprocessing zahŕňa:
- Opravu/vyradenie problematických hodnôt
- Feature engineering (tvorbu nových, informatívnejších premenných)
- Transformácie (normalizácia, dummy encoding)
- Rozdelenie na trénovaciu a testovaciu sadu

Kľúčové pravidlo: **všetky štatistiky (priemery, mediány, škály) sa učia iba z trénovacej sady** a potom sa aplikujú na testovaciu. Porušenie tohto pravidla sa volá *data leakage*.

> **📚 Poučka: Data Leakage (únik informácie)**
>
> Data leakage nastáva keď informácia z budúcnosti alebo z testovacieho setu "unikne" do trénovania. Príklad: ak normalizuješ celý dataset naraz (pred splitom), priemer a štandardná odchýlka zahŕňajú testovací set. Model sa "dozvie" niečo o teste, čo by v reálnom nasadení nevedel — metriky sú potom príliš optimistické.
>
> **Riešenie:** Recipe v tidymodels je navrhnuté presne na tento problém — `prep()` sa volá len s trénovacími dátami a `bake()` aplikuje uložené štatistiky.

### 3.1 Feature Engineering

#### Identifikácia elektrických vozidiel

```r
data_fe <- data_fe %>%
  mutate(
    is_electric = (fuelType == "Electricity") | (fuelType1 == "Electricity"),
    is_electric = replace_na(is_electric, FALSE),
    has_combustion_engine = !is_electric,
    displ     = if_else(is_electric, 0, displ),
    cylinders = if_else(is_electric, 0, cylinders)
  )
```

- **`mutate()`** — pridáva nové alebo upravuje existujúce stĺpce; všetky riadky naraz
- **`replace_na(is_electric, FALSE)`** — keď sú oba `fuelType` a `fuelType1` NA, výsledok `|` je tiež NA → nahradíme FALSE (vozidlo nie je elektrické, len má chýbajúce hodnoty)
- **`if_else(is_electric, 0, displ)`** — vektorizovaná podmienka: pre každý riadok, ak `is_electric == TRUE`, vráti 0, inak pôvodnú hodnotu `displ`. BEV nemajú spaľovací motor, takže `displ` a `cylinders` nastavíme na 0 pred ich neskorším vyradením.

#### Klasifikácia paliva (fuel_group)

```r
fuel_group = case_when(
  fuelType %in% c("Regular", "Premium", "Midgrade")  ~ "Gasoline",
  fuelType == "Diesel"                                ~ "Diesel",
  fuelType == "Electricity"                           ~ "Electricity",
  fuelType %in% c(
    "Premium and Electricity",
    "Regular Gas and Electricity", ...
  )                                                   ~ "Hybrid/Electric mix",
  fuelType %in% c("Gasoline or E85", ...)             ~ "Flex-fuel/E85",
  TRUE                                                ~ "Other/Alternative"
)
```

**`case_when()`** je vektorizovaná `if-elseif-else` podmienka z `dplyr`. Vyhodnocuje podmienky zhora nadol — prvá pravdivá "vyhrá". `TRUE ~ "Other"` je catch-all pre všetky ostatné prípady (ekvivalent `else`).

**Prečo grouping namiesto pôvodného `fuelType`?**
Pôvodný `fuelType` má ~20 kategórií, z ktorých mnohé sú extrémne vzácne (napr. "Hydrogen" — pár vozidiel). Po dummy encoding by to vytvorilo ~20 binárnych stĺpcov, väčšina s takmer nulovou variabilitou. `fuel_group` redukuje na 6 zmysluplných kategórií.

#### Spracovanie prevodovky

```r
transmission_type = case_when(
  str_detect(trany, regex("^Automatic", ignore_case = TRUE)) ~ "Automatic",
  str_detect(trany, regex("^Manual",    ignore_case = TRUE)) ~ "Manual",
  TRUE                                                       ~ "Other"
),
is_cvt  = str_detect(trany, regex("variable gear ratios", ignore_case = TRUE)),
is_cvt  = replace_na(is_cvt, FALSE),
n_gears = as.integer(str_match(trany, "(\\d+)-spd")[, 2]),
has_discrete_gears = !is.na(n_gears),
n_gears = replace_na(n_gears, 0)
```

- **`str_detect(text, pattern)`** — vráti TRUE/FALSE, či reťazec obsahuje daný vzor (regulárny výraz)
- **`regex("^Automatic", ignore_case = TRUE)`** — `^` znamená "začína na". `ignore_case` — nezáleží na veľkosti písmen.
- **`str_match(trany, "(\\d+)-spd")[, 2]`** — hľadá vzor "jedno alebo viac číslic"-spd. `(\\d+)` je zachytávacia skupina. Výsledok je matica kde stĺpec 1 = celý match, stĺpec 2 = prvá zachytávaná skupina. `[, 2]` vyberie druhý stĺpec.

Príklad krok za krokom pre `"Automatic 6-spd S6"`:
```
str_match("Automatic 6-spd S6", "(\\d+)-spd")
→ matica: [1,1] = "6-spd"   [1,2] = "6"
[, 2] → "6"
as.integer("6") → 6
```

Výsledky feature engineering:
- `transmission_type`: Automatic (36 548), Manual (13 287), Other (11)
- `is_cvt`: 1 201 vozidiel s CVT
- `n_gears`: hodnoty 1–10, kde 0 = "not applicable" (CVT alebo prevodovky bez diskrétneho počtu stupňov)
- `has_discrete_gears`: TRUE = počet stupňov bol vyčítaný z `trany`; FALSE = CVT alebo špeciálna prevodovka

**Prečo `n_gears = NA` nie je náhodná chýbajúca hodnota:**
CVT (continuously variable transmission) a niektoré automatické prevodovky nemajú diskrétny počet rýchlostných stupňov — vzorec `"\\d+-spd"` jednoducho nie je v ich popise. To, že `str_match()` nič nenájde, nie je náhoda ani chyba záznamu. Ide o štrukturálnu vlastnosť prevodovky. Preto `n_gears` nenahradzujeme mediánom — to by zavádzalo falošnú informáciu, akoby CVT malo napr. 5 stupňov. Namiesto toho `has_discrete_gears` explicitne zachytáva túto skutočnosť a `n_gears = 0` slúži ako technický kód pre "not applicable", ktorý sa má čítať spolu s `has_discrete_gears`.

### 3.2 Vylúčenie elektrických vozidiel

```r
n_ev <- sum(data_fe$is_electric)

vehicles_model <- data_fe %>%
  filter(!is_electric) %>%
  select(-any_of(cols_to_remove)) %>%
  select(-any_of(c("trany", "fuelType", "fuelType1", "is_electric", "has_combustion_engine")))
```

- **`filter(!is_electric)`** — ponechá len riadky kde `is_electric == FALSE`
- **`select(-any_of(...))`** — vyradí stĺpce zo zoznamu; `any_of()` je bezpečná verzia (nevyhodí chybu ak stĺpec neexistuje)

**Výsledok:** Vyradených **1 425 BEV** (2.9% datasetu), zostalo **48 421 vozidiel**.

**Prečo toto rozhodnutie malo obrovský efekt:**
BEV mali `comb08` hodnoty 100–146 (MPGe), pričom model predikoval na základe parametrov spaľovacieho motora. Pre BEV sú tieto prediktory nulové alebo nezmyselné → model sa systematicky mýlil o 50–100+ MPG pre každý BEV. Po vylúčení:
- RMSE: 8.11 → **3.36 MPG** (−59%)
- MAE: 5.15 → **2.19 MPG** (−57%)

### 3.3 Train/Test Split

```r
set.seed(123)
data_split <- initial_split(vehicles_model, prop = 0.8, strata = comb08)
train_data <- training(data_split)
test_data  <- testing(data_split)
# Výsledok: Train 38 735, Test 9 686
```

- **`set.seed(123)`** — nastaví seed generátora náhodných čísel. Zabezpečí reprodukovateľnosť — každé spustenie dá rovnaké rozdelenie. Číslo 123 je ľubovoľné.
- **`initial_split(prop = 0.8)`** — 80% trénovací set, 20% testovací
- **`strata = comb08`** — **stratifikované** rozdelenie: dataset sa pred rozdelením rozdelí do "binov" podľa hodnôt `comb08` a z každého binu sa odoberie 80%. Zabezpečí, že distribúcia MPG je v oboch setoch rovnaká.

> **📚 Poučka: Prečo 80/20 a stratifikácia**
>
> **80/20:** Zlatý priemer — trénovací set musí byť dostatočne veľký na naučenie modelu, testovací dostatočne veľký na spoľahlivé vyhodnotenie. Pre veľké datasety (>10 000) môžeš ísť aj na 90/10. Pre malé datasety (<1 000) sa uvažuje o CV bez pevného test setu.
>
> **Stratifikácia:** Bez nej by náhodné rozdelenie mohlo náhodou dať testovacím dátam viac "extrémnych" vozidiel (SUV alebo hybridy). Stratifikácia garantuje, že obe sady reprezentujú rovnakú distribúciu `comb08`.
>
> **Zlaté pravidlo:** Testovací set **nikdy nevidí model** počas trénovania ani tuningu. Ak ho použiješ na rozhodnutia (napr. "vyberiem model, ktorý má lepšie test RMSE"), stáva sa de facto trénovacím setom a tvoja záverečná metrika je optimisticky zaujatá.

### 3.4 Cross-Validation

```r
folds <- vfold_cv(train_data, v = 5, strata = comb08)
# Výsledok: 5 foldov, každý ~30 987 / 7 748
```

**`vfold_cv(v = 5)`** rozdelí trénovacie dáta na 5 rovnako veľkých foldov. Model sa trénuje 5-krát — vždy na 4 foldoch a evaluuje na 1. Výsledné metriky sa spriemerujú.

```
Fold 1: [Train: 2,3,4,5] [Validate: 1]
Fold 2: [Train: 1,3,4,5] [Validate: 2]
Fold 3: [Train: 1,2,4,5] [Validate: 3]
Fold 4: [Train: 1,2,3,5] [Validate: 4]
Fold 5: [Train: 1,2,3,4] [Validate: 5]
→ Výsledok: 5 hodnôt RMSE → priemer a štandardná chyba
```

> **📚 Poučka: Cross-Validation — prečo a kedy**
>
> **Prečo nie len train/test split?**
> Split dáva jeden odhad metriky. Pri menšom datasete môže byť tento odhad nestabilný. CV dáva 5 odhadov a ich priemer je oveľa spoľahlivejší.
>
> **Hlavný účel v našom projekte:** Hyperparameter tuning. Penalty pre Ridge/LASSO/Elastic Net ladíme na CV foldoch — testovací set sa pri tomto procese **vôbec nepoužíva**. Až keď máme finálny model s najlepšou penalty, raz ho evaluujeme na test sete.
>
> **Prečo 5 foldov?** Je to štandard. Viac foldov (napr. 10) dáva presnejší odhad, ale trvá dlhšie. Pri >10 000 vzorkách je 5-fold dostatočné.

#### Ako presne prebieha ladenie hyperparametra `penalty`

V našom projekte sa `penalty` neladí podľa test setu ani podľa jedného "najlepšieho" foldu. Pre každú kandidátnu hodnotu `penalty` sa spustí celá 5-fold cross-validácia:

1. Model sa natrénuje na 4 foldoch a vyhodnotí na 1 validačnom folde.
2. Toto sa zopakuje 5-krát, aby každý fold bol raz validačný.
3. Z piatich validačných výsledkov sa vypočíta priemerné RMSE.
4. Rovnaký proces sa zopakuje pre všetky hodnoty `penalty` v gride.
5. Vyberie sa tá `penalty`, ktorá má najnižšie priemerné CV RMSE.

Schématicky:

```text
penalty = 0.0001:
  fold 1 validate RMSE
  fold 2 validate RMSE
  fold 3 validate RMSE
  fold 4 validate RMSE
  fold 5 validate RMSE
  -> priemer RMSE

penalty = 0.001:
  fold 1 validate RMSE
  fold 2 validate RMSE
  fold 3 validate RMSE
  fold 4 validate RMSE
  fold 5 validate RMSE
  -> priemer RMSE

...

vyberie sa penalty s najnižším priemerným RMSE
```

Dôležitý detail: finálny model sa potom **netrénuje na jednom najlepšom folde**. Po výbere najlepšej `penalty` sa model natrénuje nanovo na **celom `train_data`**:

```r
best_ridge <- ridge_cv %>% select_best(metric = "rmse")
ridge_final_wf <- ridge_wf %>% finalize_workflow(best_ridge)
ridge_fit <- ridge_final_wf %>% fit(data = train_data)
```

Test set (`test_data`) sa použije až po tomto kroku — na finálne vyhodnotenie už natrénovaného modelu.

### 3.5 Preprocessing Recipe

Recipe definuje sériu transformačných krokov, ktoré sa majú aplikovať na dáta pred modelovaním. Je to "recept" na preprocessing: najprv povie, čo sa má spraviť, a až pri `prep()` sa z trénovacích dát naučia konkrétne hodnoty ako mediány, módy, dummy úrovne alebo priemery a smerodajné odchýlky.

Kľúčová vlastnosť: **všetky naučené parametre preprocessingu sa počítajú iba z trénovacej sady**. Potom sa tie isté uložené hodnoty aplikujú na train aj test. Takto sa zabraňuje data leakage, pretože testovací set neovplyvní imputáciu, normalizáciu ani výber dummy úrovní.

```r
model_recipe <- recipe(comb08 ~ ., data = train_data) %>%
  step_unknown(all_nominal_predictors()) %>%
  step_other(all_nominal_predictors(), threshold = 0.01) %>%
  step_impute_median(displ, cylinders) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_zv(all_predictors()) %>%
  step_nzv(all_predictors()) %>%
  step_normalize(all_numeric_predictors())
```

`recipe(comb08 ~ ., data = train_data)` — definuje vzorec: `comb08` je výstup, `.` znamená "všetky ostatné stĺpce sú prediktory". `data = train_data` sa používa len na určenie názvov a typov stĺpcov — dáta sa ešte nespracúvajú.

#### Odkiaľ sú tieto `step_*` kroky?

Tieto funkcie sú z balíka **`recipes`**, ktorý je súčasťou ekosystému **tidymodels**. V našom `vehicles_EDA.Rmd` sa balík načíta cez:

```r
library(recipes)
```

V `sources/Tutorials` je spomenutý všeobecný koncept: balík `recipes` umožňuje skladať preprocessing ako postupnosť pipeovateľných feature-engineering krokov a `workflows` potom spája recipe s modelom. Konkrétna kombinácia `step_unknown()`, `step_other()`, `step_impute_*()`, `step_dummy()`, `step_zv()`, `step_nzv()` a `step_normalize()` však v tutorialoch nie je nadiktovaná ako hotový blok. Je to štandardná tidymodels preprocessing pipeline zvolená podľa toho, čo naše dáta a modely potrebujú:

- máme kategorické premenné (`make`, `VClass`, `drive`, `fuel_group`, `transmission_type`),
- máme zriedkavé kategórie,
- máme chýbajúce hodnoty,
- používame lineárne modely, ktoré potrebujú numerické vstupy,
- používame Ridge/LASSO/Elastic Net, pri ktorých je dôležitá spoločná mierka numerických prediktorov.

Inými slovami: AI pri generovaní kódu pravdepodobne neprebralo tento presný blok z jedného tutorialu, ale poskladalo bežné kroky z balíka `recipes` podľa typických pravidiel pre tidymodels a podľa problémov viditeľných v EDA.

#### Čo robí každý krok

**`step_unknown(all_nominal_predictors())`**

Nominal predictors sú kategorické prediktory, teda premenné ako značka auta, trieda vozidla, typ pohonu alebo typ paliva. Tento krok rieši chýbajúce hodnoty v kategóriách tak, že im vytvorí samostatnú úroveň `"unknown"`.

Prečo je to dôležité: model potom nestratí riadky len preto, že niektorá kategória chýba. Pri autách môže chýbať napríklad informácia o pohone alebo prevodovke. Namiesto vyhodenia pozorovania model dostane signál: "táto kategória nebola známa".

**`step_other(all_nominal_predictors(), threshold = 0.01)`**

Tento krok spojí veľmi zriedkavé kategórie do jednej kategórie `"other"`. Hodnota `threshold = 0.01` znamená, že úrovne s výskytom pod približne 1 % sa zlúčia.

Prečo je to dôležité: ak by sme nechali všetky vzácne značky alebo triedy áut samostatne, po dummy encodingu by vzniklo veľa stĺpcov s veľmi málo jednotkami. Takéto premenné sú nestabilné, môžu pridávať šum a zbytočne komplikujú model. Zlúčenie do `"other"` znižuje dimenziu a robí model robustnejší.

**`step_impute_median(displ, cylinders)`**

Tento krok dopĺňa chýbajúce hodnoty v `displ` a `cylinders` mediánom vypočítaným z trénovacej sady. Po vyfiltrovaní BEV ostáva malé množstvo reálne chýbajúcich hodnôt v týchto prediktoroch pre niektoré non-BEV riadky (napr. záznamy bez nameraného objemu motora).

`n_gears` sa mediánom nenahrádza — jeho chýbajúce hodnoty nie sú náhodné, ale signalizujú CVT alebo prevodovky bez diskrétneho počtu stupňov. Táto informácia je explicitne zachytená v `has_discrete_gears` a `n_gears` je nastavené na 0 vo feature engineeringu. Imputovať ho mediánom by zavádzalo falošnú informáciu.

Prečo medián: medián je odolnejší voči extrémnym hodnotám než priemer. Pri automobilových dátach môžu mať niektoré numerické premenné šikmé rozdelenie alebo extrémy, takže medián je konzervatívnejšia voľba.

**Prečo nie `step_impute_mode(all_nominal_predictors())`**

`step_impute_mode()` sme z recipe odstránili z troch dôvodov:

1. `step_unknown()` už rieši chýbajúce nominálne hodnoty — vytvorí pre ne samostatnú kategóriu `"unknown"`, čo je informatívnejšie ako falošná imputácia.
2. Veľa kategorických features má vlastný fallback vo feature engineeringu: `drive` → `"Unknown"`, `fuel_group` → `"Other"`, `transmission_type` → `"Other"`, `VClass` → `"Other"`.
3. Pre technické kategorické vlastnosti je lepšie zachovať `"Unknown"` než imputovať najčastejšiu kategóriu — imputácia módom by zavádzala falošnú informáciu, akoby vozidlo patrilo do najčastejšej triedy pohonu alebo paliva.

**`step_dummy(all_nominal_predictors())`**

Tento krok zmení kategorické premenné na binárne 0/1 stĺpce. Napríklad `transmission_type = Automatic/Manual/Other` sa prevedie na dummy premenné reprezentujúce jednotlivé kategórie.

Prečo je to nutné: lineárna regresia, Ridge, LASSO aj Elastic Net pracujú s číselnou maticou prediktorov. Textové kategórie ako `"Manual"` alebo `"Compact Cars"` model priamo nevie použiť.

**`step_zv(all_predictors())`**

`zv` znamená zero variance. Tento krok odstráni prediktory, ktoré majú vo všetkých riadkoch rovnakú hodnotu.

Prečo je to dôležité: konštantný stĺpec nemôže vysvetľovať rozdiely v `comb08`, lebo sa nemení. Taký prediktor neprináša informáciu a môže zhoršovať numerickú stabilitu modelu.

**`step_nzv(all_predictors())`**

`nzv` znamená near-zero variance. Tento krok odstráni prediktory, ktoré síce nie sú úplne konštantné, ale takmer všetky hodnoty sú rovnaké.

Príklad: dummy premenná, ktorá je `0` v 99.5 % riadkov a `1` len v pár prípadoch. Taký stĺpec často vzniká zo vzácnych kategórií a modelu dáva veľmi slabý alebo nestabilný signál. Tento krok je v súlade s EDA, kde sa riešili nízkovariačné premenné.

**`step_normalize(all_numeric_predictors())`**

Tento krok štandardizuje numerické prediktory: odčíta priemer a vydelí smerodajnou odchýlkou. Po normalizácii majú numerické prediktory približne priemer 0 a smerodajnú odchýlku 1.

Prečo je to dôležité: Ridge, LASSO a Elastic Net penalizujú veľkosť koeficientov. Ak by jeden prediktor mal rozsah 1984-2026 (`year`) a iný 4-16 (`cylinders`), penalizácia by nebola férová. Normalizácia zabezpečí, že koeficienty sú penalizované porovnateľne.

> **📚 Poučka: Prečo je normalizácia nevyhnutná pre regularizované modely**
>
> Ridge a LASSO penalizujú veľkosť koeficientov β. Koeficient závisí od **mierky** premennej:
> - `year` (rozsah 1984–2026): zmena o 1 rok má malý koeficient (napr. β = 0.3)
> - `cylinders` (rozsah 4–16): zmena o 1 valec má väčší koeficient (napr. β = −2.5)
>
> Bez normalizácie by penalizácia nespravodlivo uprednostňovala prediktory s malou mierkou. Po normalizácii (priemer 0, SD 1) sú všetky prediktory na rovnakej škále a penalizácia je férová.
>
> **OLS normalizáciu nepotrebuje** — jeho výsledok je numericky identický bez ohľadu na škálu (koeficienty sa automaticky prispôsobia). Ale pre konzistenciu a porovnateľnosť koeficientov ju používame pre všetky modely.

```r
prep_recipe <- prep(model_recipe, training = train_data)
train_processed <- bake(prep_recipe, new_data = NULL)
test_processed  <- bake(prep_recipe, new_data = test_data)
```

- **`prep()`** — "naučí" recipe: vypočíta mediány, mody, škály z `train_data`
- **`bake(new_data = NULL)`** — aplikuje naučenú recipe na trénovacie dáta (NULL = použij dáta z prep)
- **`bake(new_data = test_data)`** — aplikuje **rovnaké** uložené štatistiky na testovací set

Výsledok: 21 prediktorov po dummy encoding, 0 chýbajúcich hodnôt.

---

## 4. Scenario 2: Parametrické modely

### Čo sú parametrické modely

Parametrický model predpokladá, že vzťah medzi prediktormi a výstupom má **konkrétnu matematickú formu** s **konečným počtom parametrov**. Pre lineárnu regresiu:

$$\hat{y} = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \ldots + \beta_p x_p$$

Model sa "naučí" optimálne hodnoty koeficientov β (to je trénovanie). Počet parametrov = počet prediktorov + 1 (intercept).

**Výhody:**
- **Interpretovateľné** — každý koeficient β hovorí "o koľko MPG sa zmení predikcia pri zmene xⱼ o 1 (štandardizovanú) jednotku, ceteris paribus"
- **Rýchle** — málo parametrov = rýchle trénovanie aj na veľkých datasetoch
- **Dobre generalizujú** keď je vzťah skutočne lineárny
- **Stabilné** — malé zmeny v dátach nemenia koeficienty dramaticky

**Nevýhody:**
- **Predpokladajú linearitu** — ak je skutočný vzťah nelineárny (napr. exponenciálny alebo s interakciami), model bude systematicky chybovať
- **Obmedzená flexibilita** — nevedia zachytiť komplexné vzory bez explicitného feature engineering

> **📚 Poučka: Parametrické vs. Neparametrické modely**
>
> **Parametrický model:** Funkčná forma je fixná vopred, model sa učí len koeficienty. Príklady: lineárna regresia, logistická regresia, Naive Bayes.
>
> **Neparametrický model:** Funkčná forma nie je predpísaná, model si ju učí priamo z dát. Príklady: k-Nearest Neighbors, Random Forest, Support Vector Machine, neurónové siete.
>
> Analógia: Parametrický model je ako vyplnenie formulára — šablóna je daná, vypĺňaš len hodnoty. Neparametrický je ako voľná esej — tvar si určuje sám z obsahu.
>
> **Kedy ktorý?** Parametrické modely sú lepšie keď máš málo dát, chceš interpretovateľnosť, alebo vieš že vzťah je lineárny. Neparametrické sú lepšie keď máš veľa dát, vzťahy sú komplexné, a interpretovateľnosť je sekundárna.

### Workflow v tidymodels

Pre každý model definujeme workflow — objekt, ktorý zabalí recipe + model specification:

```r
model_wf <- workflow() %>%
  add_recipe(model_recipe) %>%
  add_model(model_spec)
```

**Prečo workflow?**
Bez workflow by si musel manuálne transformovať dáta pred každou predikciou. S workflow stačí zavolať `predict(workflow_fit, new_data = test_data)` — recipe sa aplikuje automaticky. Predovšetkým pri cross-validácii to zabraňuje data leakage: recipe sa aplikuje znovu pre každý fold zvlášť.

### 4.1 Lineárna Regresia (OLS)

#### Teória

OLS (Ordinary Least Squares = metóda najmenších štvorcov) hľadá koeficienty β, ktoré **minimalizujú sumu štvorcov reziduálov**:

$$\min_\beta \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 = \min_\beta \| y - X\beta \|^2$$

Pre tento problém existuje analytické (presné) riešenie:

$$\hat{\beta} = (X^T X)^{-1} X^T y$$

Toto je uzavretý vzorec — na rozdiel od iných modelov (napr. neurónových sietí) nie je potrebná iteratívna optimalizácia.

#### Kedy použiť OLS

- Ako **baseline** — referenčný bod, voči ktorému meráme zlepšenie regularizovaných modelov
- Keď je vzťah skutočne lineárny
- Keď počet prediktorov p << počet vzoriek n (čím väčší pomer n/p, tým stabilnejšie OLS)
- Keď **neexistuje** silná multikolinearita

#### Čo je špeciálne

- **Žiadna regularizácia** — koeficienty nie sú nijako obmedzované → maximálna flexibilita, ale aj maximálna variabilita
- **Gauss-Markov theorem:** Za predpokladu splnenia podmienok (linearita, homoskedasticity, nezávislosť, normálne reziduály) je OLS **BLUE** (Best Linear Unbiased Estimator) — najlepší nestranný lineárny odhad
- Výpočtovo najrýchlejší zo štyroch modelov

#### Kód

```r
lm_spec <- linear_reg() %>%
  set_engine("lm") %>%
  set_mode("regression")

lm_wf <- workflow() %>%
  add_recipe(model_recipe) %>%
  add_model(lm_spec)
```

```r
set.seed(123)
lm_cv <- lm_wf %>%
  fit_resamples(
    resamples = folds,
    metrics   = eval_metrics,
    control   = control_resamples(save_pred = TRUE)
  )
```

- **`set_engine("lm")`** — použije základnú R funkciu `lm()`. Pre glmnet (Ridge/LASSO/EN) by sme použili `"glmnet"`.
- **`fit_resamples()`** — spustí trénovanie + evaluáciu na každom z 5 foldov
- **`control_resamples(save_pred = TRUE)`** — uloží predikcie z každého validation foldu. Bez toho sú dostupné len agregované metriky.

Na záver trénujeme na **celom** trénovacom sete (nie len na folde):

```r
lm_fit <- lm_wf %>% fit(data = train_data)
```

A extrahujeme koeficienty:

```r
lm_fit %>%
  extract_fit_parsnip() %>%   # vytiahne natrénovaný model z workflow
  tidy() %>%                  # prevedie na čistý tibble (term, estimate, ...)
  filter(term != "(Intercept)") %>%
  slice_max(abs(estimate), n = 20)  # top 20 podľa absolútnej hodnoty
```

#### Výsledky

| Metrika | CV | Test |
|---------|-----|------|
| RMSE | 3.366 ± 0.029 | 3.36 |
| MAE | 2.212 | 2.19 |
| R² | 0.659 | 0.661 |

### 4.2 Ridge Regresia

#### Teória

Ridge (tiež L2 regularizácia) pridáva k OLS strate **penalizačný člen**:

$$\min_\beta \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 + \lambda \sum_{j=1}^{p} \beta_j^2$$

Druhý člen $\lambda \sum \beta_j^2$ penalizuje veľké koeficienty. Parameter **λ (penalty)** riadi silu penalizácie:
- λ = 0 → identické s OLS
- λ → ∞ → všetky koeficienty → 0 (model predikuje len priemer)
- Optimálne λ niekde medzi — nájdeme ho cross-validáciou

**Kľúčová vlastnosť Ridge:** Koeficienty sú **stlačené smerom k nule, ale nikdy nie sú presne nula**. Všetky prediktory zostávajú v modeli.

#### Kedy použiť Ridge

- Keď existuje **multikolinearita** — Ridge rozdelí váhu rovnomernejšie medzi korelované prediktory
- Keď máš veľa prediktorov s podobnou dôležitosťou
- Keď **nechceš** feature selection (všetky prediktory si chceš zachovať)
- Ako stabilizácia keď n/p ratio je nízke

> **📚 Poučka: Bias-Variance Tradeoff**
>
> Každý model čelí dileme: čím je model flexibilnejší (menej biased), tým má väčšiu varianciu (citlivosť na konkrétne trénovacie dáta).
>
> - **OLS:** nízky bias (neobmedzuje koeficienty), vyššia variancia
> - **Ridge s veľkým λ:** väčší bias (koeficienty sú odtlačené od OLS riešenia), nižšia variancia
>
> Regularizácia je vedome zavedená zaujatosť (bias), ktorá výmenou za to znižuje varianciu. Funguje vždy, keď zníženie variancie prevýši zvýšenie biasu — čo je typické pri multikolinearite alebo pri n << p.

#### Kód

```r
ridge_spec <- linear_reg(penalty = tune(), mixture = 0) %>%
  set_engine("glmnet") %>%
  set_mode("regression")
```

- **`mixture = 0`** — čistá L2 penalizácia (Ridge). Parameter `mixture` α ∈ [0, 1]: 0 = Ridge, 1 = LASSO.
- **`penalty = tune()`** — hodnotu λ hľadáme cez cross-validáciu
- **`set_engine("glmnet")`** — knižnica `glmnet` efektívne vypočíta koeficienty pre celý regularizačný path naraz

```r
ridge_grid <- grid_regular(penalty(range = c(-4, 2)), levels = 50)
```

Tu sú definované konkrétne hodnoty `penalty`, ktoré sa budú skúšať. `penalty = tune()` v modeli len hovorí "toto je hyperparameter, treba ho nájsť"; samotné kandidátne hodnoty dodáva až `ridge_grid`.

- **`grid_regular()`** — vytvorí rovnomerne rozmiestnené hodnoty **na log-škále**
- `range = c(-4, 2)` → $10^{-4}$ až $10^2$ = 0.0001 až 100
- `levels = 50` → 50 hodnôt λ na otestovanie
- `glmnet` ich vyhodnotí takmer rovnako rýchlo ako 1 hodnotu (počíta celú regularizačnú cestu)

```r
set.seed(123)
ridge_cv <- ridge_wf %>%
  tune_grid(
    resamples = folds,
    grid      = ridge_grid,
    metrics   = eval_metrics
  )

best_ridge <- ridge_cv %>% select_best(metric = "rmse")
ridge_final_wf <- ridge_wf %>% finalize_workflow(best_ridge)
ridge_fit      <- ridge_final_wf %>% fit(data = train_data)
```

- **`tune_grid()`** — pre každú kombináciu hyperparametrov a každý CV fold spustí trénovanie a evaluáciu. Výsledok: tabuľka metrík pre každú hodnotu penalty.
- **`select_best(metric = "rmse")`** — nájde riadok s najlepšou (najnižšou) priemernou CV RMSE
- **`finalize_workflow(best_ridge)`** — dosadí konkrétnu hodnotu penalty do workflow (namiesto `tune()`)
- **`fit(data = train_data)`** — natrénuje finálny model na celom trénovacom sete

Pre Ridge teda prebehne približne **50 penalty hodnôt × 5 foldov = 250 validačných behov**. Nevyberá sa najlepší jeden fold, ale najlepšia `penalty` podľa priemeru RMSE cez všetkých 5 validačných foldov.

#### Výsledky

- Best penalty: **0.0001** (minimum gridu)
- CV RMSE: **3.376 ± 0.029**

Best penalty na minime gridu signalizuje, že regularizácia takmer nepomáha — model by preferoval ešte menšiu penalizáciu (= blíže k OLS).

### 4.3 LASSO Regresia

#### Teória

LASSO (Least Absolute Shrinkage and Selection Operator) používa **L1 penalizáciu**:

$$\min_\beta \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 + \lambda \sum_{j=1}^{p} |\beta_j|$$

**Kľúčová vlastnosť LASSO:** L1 penalizácia má geometrickú vlastnosť, že optimum sa nachádza v "rohoch" L1 gule (diamant tvar v 2D) — kde mnohé koeficienty sú **presne nula**. LASSO teda robí **feature selection** súčasne s regresiou.

> **📚 Poučka: Prečo L1 nuluje koeficienty a L2 nie**
>
> Geometricky si predstav 2 prediktory (β₁, β₂):
> - L2 obmedzenie tvorí **kruh** — ∑β² ≤ t
> - L1 obmedzenie tvorí **diamant** — ∑|β| ≤ t
>
> Optimalizujeme stratu (ktorej izolinky sú elipsy) s obmedzením. Optimum = miesto kde sa elipsa dotkne obmedzujúceho tvaru:
> - **Kruh:** dotyčnica kdekoľvek na povrchu → β₁ ≠ 0, β₂ ≠ 0
> - **Diamant:** dotyčnica sa veľmi pravdepodobne stane v rohu → napr. β₁ = 0, β₂ ≠ 0
>
> Vo vysokých dimenziách (veľa prediktorov) má diamant obrovské množstvo rohov a hrán → LASSO typicky nuluje väčšinu koeficientov.

#### Kedy použiť LASSO

- Keď predpokladáš, že len **niektoré prediktory sú skutočne dôležité** (sparse model)
- Keď chceš **automatickú feature selection** bez manuálneho výberu
- Keď je dôležitá **interpretovateľnosť** — menej prediktorov = jednoduchší model

#### Kód

```r
lasso_spec <- linear_reg(penalty = tune(), mixture = 1) %>%
  set_engine("glmnet") %>%
  set_mode("regression")

lasso_grid <- grid_regular(penalty(range = c(-4, 0)), levels = 50)
```

- **`mixture = 1`** — čistá L1 penalizácia (LASSO)
- `lasso_grid` definuje konkrétne hodnoty `penalty`, ktoré sa skúšajú pri tuningu.
- `range = c(-4, 0)` znamená $10^{-4}$ až $10^0$, teda 0.0001 až 1.
- `levels = 50` znamená 50 hodnôt `penalty`.
- Tento grid je užší ako pri Ridge (`0.0001` až `100`), pretože LASSO typicky dosahuje dostatočné zmršťovanie koeficientov už pri menších hodnotách λ.

Aj tu platí rovnaký mechanizmus: pre každú z 50 hodnôt `penalty` sa spraví 5-fold CV, vypočíta sa priemerné RMSE a `select_best(metric = "rmse")` vyberie hodnotu s najnižším priemerným CV RMSE. Až potom sa finálny LASSO model natrénuje na celom `train_data`.

#### Výsledky

- Best penalty: **0.0001** (minimum gridu)
- Ponechaných prediktorov: **21 z 21** (žiadny nulovaný)
- CV RMSE: **3.366 ± 0.029**

**Prečo LASSO nič nevyradilo?** Pri `penalty = 1e-4` a 38 735 vzorkách je každý prediktor prínosom. Selekcia by nastala pri vyšších hodnotách λ, ale za cenu horšej prediktívnej presnosti. Toto je pozitívna informácia — naša sada prediktorov je "čistá" (preprocessing bol efektívny).

### 4.4 Elastic Net

#### Teória

Elastic Net kombinuje L1 aj L2 penalizáciu:

$$\min_\beta \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 + \lambda \left[ \alpha \sum_{j=1}^{p} |\beta_j| + (1-\alpha) \sum_{j=1}^{p} \beta_j^2 \right]$$

Parameter **mixture (α)** určuje mix (v tidymodels = `mixture`):
- α = 0 → čistý Ridge
- α = 1 → čistý LASSO
- α = 0.5 → rovnaký mix oboch

Elastic Net teda **tuninguje dva hyperparametre**: `penalty` (λ) a `mixture` (α).

#### Kedy použiť Elastic Net

- Keď chceš výhody LASSO (feature selection), ale máš **skupiny korelovaných prediktorov**
- LASSO pri korelovaných prediktoroch náhodne vyberie jeden a ostatné nuluje → nestabilné riešenie
- Ridge naopak nerobí výber → veľa "malých" koeficientov
- Elastic Net: vyberie skupinu korelovaných prediktorov a ich váhu rozdelí (Ridge správanie v rámci skupiny), ale skupiny nepotrebné nuluje (LASSO správanie medzi skupinami)
- Prakticky: **bezpečná "all-around" voľba** — ak dáta preferujú Ridge alebo LASSO, Elastic Net sa k nim priblíži

#### Kód

```r
en_spec <- linear_reg(penalty = tune(), mixture = tune()) %>%
  set_engine("glmnet") %>%
  set_mode("regression")

en_grid <- grid_regular(
  penalty(range = c(-4, 0)),
  mixture(range = c(0, 1)),
  levels = 10
)
```

`grid_regular()` s dvoma parametrami vytvorí **10 × 10 = 100 kombinácií**. `glmnet` ich vyhodnotí efektívne — pre každú hodnotu `mixture` vypočíta celý regularizačný path (všetky `penalty`) naraz.

Konkrétne:

- `penalty(range = c(-4, 0))` → 10 hodnôt od 0.0001 po 1
- `mixture(range = c(0, 1))` → 10 hodnôt od 0 po 1
- spolu sa skúša 100 dvojíc `(penalty, mixture)`

Pre každú dvojicu `(penalty, mixture)` sa znovu spraví 5-fold CV. Elastic Net teda porovnáva priemerné CV RMSE pre 100 konfigurácií a vyberie tú najlepšiu. V našom výsledku to bola kombinácia `penalty = 0.0001` a `mixture = 0.222`.

#### Výsledky

- Best penalty: **0.0001**, Best mixture: **0.222** (bližšie k Ridge ako k LASSO)
- CV RMSE: **3.366 ± 0.029**

Mixture = 0.222 hovorí, že dáta mierne preferujú Ridge-like správanie — čo dáva zmysel pri multikolinearite (`cylinders`–`displ`).

### Prehľad všetkých štyroch modelov

| Model | Penalizácia | Nuluje koeficienty? | Tunable params | Hlavná výhoda |
|-------|-------------|---------------------|----------------|---------------|
| OLS | Žiadna | Nie | 0 | Interpretovateľný baseline, BLUE |
| Ridge | L2 (∑β²) | Nie | λ | Stabilita pri multikolinearite |
| LASSO | L1 (∑\|β\|) | **Áno** | λ | Feature selection, sparse model |
| Elastic Net | L1 + L2 | Áno | λ, α | Kombinácia výhod oboch |

---

## 5. Vyhodnocovacie metriky

```r
eval_metrics <- metric_set(rmse, rsq, mae)
```

**`metric_set()`** vytvorí skupinu metrík, ktoré sa počítajú naraz pri každom CV folde a pri finálnom vyhodnotení.

### 5.1 MSE — Mean Squared Error

$$\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

- Priemerná **kvadratická** chyba
- Jednotky: MPG² (kvadrát jednotiek target premennej) → ťažko interpretovateľné
- Väčšie chyby penalizované silnejšie (kvadraticky)
- V projekte **nepoužívame priamo** — nahradila ju RMSE (čitateľnejšia)

### 5.2 RMSE — Root Mean Squared Error

$$\text{RMSE} = \sqrt{\text{MSE}} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$$

- **Jednotky:** rovnaké ako target (MPG) → interpretovateľné
- Odmocnina "vracia" jednotky späť do pôvodnej škály
- Väčšie chyby stále penalizované silnejšie ako malé (kvadratická strata pred odmocninou)
- **Nižší = lepší**

**Interpretácia hodnôt:**
| RMSE | Interpretácia pri mediáne 20 MPG |
|------|----------------------------------|
| 8.11 MPG | ~41% relatívna chyba — zlé (obsahoval BEV outliere) |
| **3.36 MPG** | ~17% relatívna chyba — rozumné pre lineárny model |
| 1.0 MPG | ~5% — výborné |
| 0.5 MPG | ~2.5% — excelentné |

> **📚 Poučka: RMSE vs MAE — kedy ktorú použiť**
>
> Obe metriky sú v jednotkách target premennej, obe hovorí o "priemernej chybe". Rozdiel je v tom, ako reagujú na veľké chyby:
>
> - **RMSE:** Väčšie chyby sú penalizované kvadraticky. Ak model raz veľmi zmýli (napr. predikuje 10 MPG pre auto s 50 MPG), RMSE to "pocíti" výrazne. Preto keď sú v dátach outliere, RMSE je vždy > MAE.
>
> - **MAE:** Všetky chyby sú penalizované lineárne — rovnomerne. Robustnejší voči outlierom.
>
> **Pravidlo:** Ak je RMSE výrazne väčší ako MAE (napr. RMSE = 8, MAE = 5 — ako predtým s BEV), signalizuje to prítomnosť veľkých chýb (outlierov). Po vyradení BEV: RMSE = 3.36, MAE = 2.19 — pomer ≈ 1.53, čo je normálne.
>
> **Ktorú optimalizovať?** RMSE je štandard v regresii, lebo je matematicky jednoduchšia (kvadratická strata = hladká funkcia = dobre sa derivuje). MAE je lepšia keď sú outliere žiaduce ignorovať.

### 5.3 MAE — Mean Absolute Error

$$\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$

- **Jednotky:** rovnaké ako target (MPG)
- Priama interpretácia: "priemerne sa mýlime o X MPG"
- Robustný voči outlierom (lineárna penalizácia)
- **Nižší = lepší**

**Naša hodnota: MAE = 2.19 MPG** → model sa priemerne mýli o 2.19 MPG. Pre denné použitie (napr. porovnávanie efektivity áut) je to prijateľná presnosť.

### 5.4 R² — Koeficient determinácie

$$R^2 = 1 - \frac{\text{SS}_{res}}{\text{SS}_{tot}} = 1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}$$

kde $\text{SS}_{res}$ = suma štvorcov reziduálov (nevysvetlená variabilita) a $\text{SS}_{tot}$ = celková variabilita.

- **Rozsah:** typicky 0 až 1; môže byť záporný pre extrémne zlé modely
- **Interpretácia:** podiel variancie target premennej, ktorý model vysvetľuje
- **Vyšší = lepší** (max = 1.0)

**Naša hodnota: R² = 0.661** → model vysvetľuje 66.1% variancie v `comb08`. Zvyšných 33.9% je nevysvetlená variabilita (nelinearita, chýbajúce prediktory, šum).

> **📚 Poučka: Interpretácia R² v kontexte**
>
> R² sám o sebe nevypovedá o "dobrote" modelu — záleží od domény:
> - Fyzika (napr. meranie dĺžky): R² > 0.99 je normálne
> - Technické predikcie (spotreba auta): R² = 0.65–0.80 je dobrý výsledok
> - Spoločenské vedy (predikcia správania ľudí): R² = 0.2–0.4 môže byť výborný výsledok
>
> **Prečo R² vzrástol len o 3% po vylúčení EVs, keď RMSE klesla o 59%?**
>
> R² je normalizovaná metrika. Porovnáva $SS_{res}$ (chyby modelu) s $SS_{tot}$ (celková variabilita dát). Po vylúčení BEV:
> - $SS_{res}$ klesla výrazne (menej extrémnych chýb)
> - Ale $SS_{tot}$ tiež klesla (rozsah 7–146 → 7–74 MPG = menšia celková variabilita)
>
> Oba menovatelia klesli súčasne, takže ich podiel (a teda R²) zostal podobný. RMSE je absolútna metrika v MPG — preto vidí výrazný pokles.
>
> **Záver:** Na posúdenie skutočného zlepšenia je RMSE informatívnejšia. R² hovorí viac o tom, ako dobre model zachytáva *štruktúru* dát, nie o absolútnej presnosti.

### 5.5 CV vs. Test gap

```r
cv_rmse_comparison
```

| Model | CV RMSE | Test RMSE | Gap |
|-------|---------|-----------|-----|
| Linear Regression | 3.366 | 3.36 | 0.003 |
| Elastic Net | 3.366 | 3.36 | 0.002 |
| LASSO | 3.366 | 3.36 | 0.002 |
| Ridge | 3.376 | 3.37 | 0.005 |

**Gap** je absolútny rozdiel medzi CV RMSE (odhad počas trénovania) a Test RMSE (reálny výkon na nových dátach).

> **📚 Poučka: Čo hovorí CV–Test gap**
>
> - **Malý gap (< 0.1 RMSE):** CV bol spoľahlivý odhad. Model dobre generalizuje. Nie je overfit.
> - **Veľký gap (CV RMSE << Test RMSE):** Model overfit-ol trénovacie dáta. CV nestačil odhaliť overfitting — príčina môže byť data leakage alebo príliš flexibilný model.
> - **Záporný gap (CV RMSE > Test RMSE):** Neobvyklé, ale možné pri malom testovacom sete. Testovací set mohol byť "ľahší" ako priemer CV foldov.
>
> **Naše hodnoty: gap < 0.005** — prakticky nulový. Cross-validácia bola veľmi spoľahlivý odhad skutočného výkonu.

---

## 6. Výsledky a záver — Scenario 2

### Vplyv vylúčenia elektrických vozidiel

Vylúčenie 1 425 BEV (2.9% datasetu) malo dramatický efekt:

| Metrika | S BEV (pred) | Bez BEV (po) | Zmena |
|---------|--------------|--------------|-------|
| RMSE | 8.11 MPG | 3.36 MPG | **−59%** |
| MAE | 5.15 MPG | 2.19 MPG | **−57%** |
| R² | 0.642 | 0.661 | +3% |
| Max comb08 | 146 MPGe | 74 MPG | −49% |

Toto potvrdzuje, že BEV segment bol dominantným zdrojom chyby — nie kvôli slabosti modelu, ale kvôli nekompatibilite MPGe s MPG v rovnakej regresii.

### Porovnanie modelov

Všetky štyri modely dosiahli takmer identické výsledky na test sete. Najlepší výkon mal OLS a LASSO (ex aequo):

| Model | Test RMSE | Test MAE | Test R² |
|-------|-----------|----------|---------|
| Linear Regression | **3.36** | 2.19 | **0.661** |
| LASSO | **3.36** | 2.19 | **0.661** |
| Elastic Net | **3.36** | 2.19 | **0.661** |
| Ridge | 3.37 | **2.17** | 0.660 |

### Prečo regularizácia nepomohla

Tri regularizované modely zvolili `penalty = 1e-4` (minimum gridu), LASSO nuloval žiadny prediktor. Dôvody:

1. **Pomer n/p je obrovský:** 38 735 vzoriek, 21 prediktorov → OLS má extrémne nízku varianciu, nie je čo stabilizovať
2. **Preprocessing bol efektívny:** Vyradenie near-zero variance stĺpcov, grouping vzácnych kategórií → zostala čistá, informatívna sada prediktorov bez redundancie
3. **Multikolinearita nie je kritická:** `cylinders`–`displ` korelácia ~0.91 je reálna, ale OLS s 38k vzorkami je napriek nej stabilný

Toto je **pozitívny výsledok**, nie chyba. Potvrdzuje, že dataset a preprocessing sú kvalitné.

### Reziduálna neistota

R² = 0.661 → 33.9% variancie zostáva nevysvetlené. Zdroje:

- **Nelinearita:** Efekt `cylinders` nie je konštantný (pridanie valca ku 4-cylindrovému motoru ≠ ku V8)
- **PHEV a mild hybridy:** Čiastočne elektrické vozidlá majú charakteristiku medzi ICE a BEV
- **Výrobcovské špecifiká:** Turbodúchadlá, variabilné časovanie ventilov, cylinder deactivation — všetko zachytené len ako priemer v `make` dummies
- **Chýbajúce prediktory:** Hmotnosť vozidla, aerodynamický koeficient (Cd) — výrazné determinanty MPG, ale nie sú v datasete

---

## 7. Slovník kľúčových príkazov

### `collect_metrics()`, `show_best()`, `select_best()`

```r
# Priemerné metriky cez všetky foldy (pre OLS bez tuning)
collect_metrics(lm_cv)

# Najlepšie konfigurácie zoradené podľa metriky (tu: top 1)
show_best(ridge_cv, metric = "rmse", n = 1)

# Len riadok s najlepšou konfiguráciou (na použitie vo finalize_workflow)
best_ridge <- select_best(ridge_cv, metric = "rmse")
```

Rozdiel: `show_best()` vráti celú tabuľku najlepších konfigurácií s metrikami. `select_best()` vráti len riadok s hyperparametrami (bez stĺpcov metrík) — presne to, čo potrebuje `finalize_workflow()`.

### `finalize_workflow()` a `fit()`

```r
ridge_final_wf <- ridge_wf %>% finalize_workflow(best_ridge)
ridge_fit       <- ridge_final_wf %>% fit(data = train_data)
```

- **`finalize_workflow(best_ridge)`** — dosadí konkrétne číslo za `tune()` placeholder. Predtým bol `penalty = tune()` — teraz je napr. `penalty = 0.0001`.
- **`fit(data = train_data)`** — natrénuje finálny model na **celom** trénovacom sete. Dôležité: po tuningu vždy trénujeme na celom train (nie len na foldoch) — viac dát = lepší model.

### `extract_fit_parsnip()` a `tidy()`

```r
lm_fit %>%
  extract_fit_parsnip() %>%
  tidy()
```

- **`extract_fit_parsnip()`** — workflow objekt obsahuje recipe + model + natrénované parametre. Táto funkcia vytiahne len samotný natrénovaný model (parsnip objekt).
- **`tidy()`** (balík `broom`) — konvertuje výstup modelu do štandardného "tidy" formátu: tibble s riadkami pre každý koeficient a stĺpcami `term`, `estimate`, `std.error`, `statistic`, `p.value`. Funguje pre desiatky typov modelov s rovnakým rozhraním.

### `pivot_longer()`

```r
test_comparison %>%
  pivot_longer(
    cols      = c(RMSE, MAE, Rsq),
    names_to  = "metric",
    values_to = "value"
  )
```

Transformuje "wide" formát → "long" formát. ggplot2 typicky vyžaduje long formát (jeden riadok = jeden bod v grafe).

```
# WIDE (pred):
model   RMSE  MAE   Rsq
OLS     3.36  2.19  0.661
Ridge   3.37  2.17  0.660

# LONG (po):
model   metric  value
OLS     RMSE    3.36
OLS     MAE     2.19
OLS     Rsq     0.661
Ridge   RMSE    3.37
Ridge   MAE     2.17
Ridge   Rsq     0.660
```

`facet_wrap(~metric)` potom vytvorí samostatný panel pre každú metriku.

### `slice_max()` a `slice_min()`

```r
slice_max(abs(estimate), n = 20)  # top 20 najväčších (absolútna hodnota)
slice_min(estimate, n = 5)         # bottom 5 najmenších
```

Vyberie n riadkov s najväčšou/najmenšou hodnotou zadaného výrazu. Ekvivalent `arrange() %>% head()`, ale čitateľnejší.

### `control_resamples()`

```r
control_resamples(save_pred = TRUE)
```

Riadi správanie `fit_resamples()`. Bez `save_pred = TRUE` sú dostupné len agregované metriky — neuložia sa samotné predikcie. S `TRUE` môžeš neskôr urobiť actual vs. predicted graf pre CV predikcie, analyzovať reziduály po foldoch, atď.

### `grid_regular()`

```r
grid_regular(penalty(range = c(-4, 2)), levels = 50)
```

Vytvorí rovnomerne rozmiestnené hodnoty na log-škále (pre `penalty`). Ekvivalentne pre oba hyperparametre:

```r
grid_regular(
  penalty(range = c(-4, 0)),
  mixture(range = c(0, 1)),
  levels = 10
)
# → 10 × 10 = 100 kombinácií
```

Alternatíva: `grid_random(n = 200)` — náhodné kombinácie (lepšie pre veľa hyperparametrov).

### `str_detect()` a `str_match()`

```r
# Vráti TRUE/FALSE — obsahuje reťazec vzor?
str_detect("Automatic 6-spd", regex("^Automatic", ignore_case = TRUE))
# → TRUE

# Vráti maticu: [celý match, skupina 1, skupina 2, ...]
str_match("Automatic 6-spd", "(\\d+)-spd")
# → matrix: [1,1] = "6-spd", [1,2] = "6"
```

Regulárne výrazy (regex) — stručná referencia:
| Vzor | Čo znamená |
|------|------------|
| `^` | Začiatok reťazca |
| `$` | Koniec reťazca |
| `\\d` | Ľubovoľná číslica (0-9) |
| `\\d+` | Jedna alebo viac číslic |
| `\\w` | Ľubovoľný alfanumerický znak |
| `(...)` | Zachytávacia skupina |
| `.` | Ľubovoľný znak |
| `*` | 0 alebo viac opakovaní predchádzajúceho |
| `+` | 1 alebo viac opakovaní |

---

*Dokument bude rozšírený o ďalšie scenáre.*
