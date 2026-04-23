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
8. [Scenario 5: Lineárna diskriminačná analýza (LDA)](#8-scenario-5-lineárna-diskriminačná-analýza-lda)

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

> **📚 Poučka: Prečo sa testovací set nesmie použiť viackrát**
>
> Predstav si, že máš 10 rôznych modelov a pre každý zmeriaš test RMSE. Potom vybereš model s najlepším test RMSE. Čo si vlastne urobil? Vybral si model, ktorý *náhodou* sedí najlepšie na konkrétnych 9 686 testovacích riadkoch. Ak by si dostal iný náhodný test set, možno by vyhral iný model.
>
> Toto je **implicit multiple testing** — čím viac modelov porovnáš na test sete, tým väčšia šanca, že "víťaz" je víťaz len náhodou. Štatistici to nazývajú **overfitting na testovací set**.
>
> Správny postup: všetky rozhodnutia (výber modelu, ladenie hyperparametrov) robiť na cross-validation foldoch z trénovacej sady. Test set použiť **raz** na záverečné číslo, ktoré ide do reportu.

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

> **📚 Poučka: Prečo CV a nie tretí "validačný" set**
>
> Alternatívou ku CV by bol **train / validation / test** triple split: trénovaciu sadu rozdeliť ešte raz na tréning (napr. 64%) a validáciu (16%), a validation sadu použiť na hyperparameter tuning.
>
> Problém: každé rozdelenie zahodí dáta. Pri 48 000 riadkoch to nie je kritické, ale pri 5 000 riadkoch by bol model trénovaný len na 3 200 príkladoch — príliš málo. CV riešenie: každý riadok je raz validačný, 4× trénovací. Nič sa nezahadzuje, každý riadok prispieva k odhadu výkonu.
>
> Navyše CV dáva **stabilnejší odhad** — priemer 5 validačných výsledkov je menej citlivý na konkrétne rozloženie dát ako jeden validation set.
>
> **Kedy CV nestačí:** Ak máš časové rady (napr. ceny akcií), bežná CV "miešaním" porušuje časové poradie. Tam sa používa *time-series CV* (training window posúvaná vpred). Pre naše vozidlá je náhodné miešanie správne — poradie záznamu nemá žiadny vplyv.

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

### 3.6 Prečo `recipe` a nie manuálny preprocessing

Preprocessing sa dá spraviť aj bez `recipe` — ručnými transformáciami priamo na dataframe. Porovnanie:

**Manuálny prístup:**
```r
# Vypočítaj štatistiky z trénovacej sady
train_means <- colMeans(train_data[numerics], na.rm = TRUE)
train_sds   <- apply(train_data[numerics], 2, sd, na.rm = TRUE)

# Normalizuj train
train_scaled <- sweep(train_data[numerics], 2, train_means, "-")
train_scaled <- sweep(train_scaled,         2, train_sds,   "/")

# Aplikuj TIE ISTÉ parametre na test
test_scaled  <- sweep(test_data[numerics],  2, train_means, "-")
test_scaled  <- sweep(test_scaled,          2, train_sds,   "/")
```

Problémy:
1. **Náchylné na chyby** — ľahko zabudneš použiť `train_means` na test set a omylom vypočítaš nové z `test_data`
2. **Nefunguje v CV** — pri cross-validácii musíš tieto výpočty zopakovať ručne pre každý fold zvlášť, inak dôjde k leakage
3. **Ťažko rozšíriteľné** — pridanie nového kroku (napr. `step_other`) znamená zmenu na viacerých miestach

**Recipe prístup:**
```r
recipe <- recipe(comb08 ~ ., data = train_data) %>%
  step_normalize(all_numeric_predictors())

# V cross-validácii:
# tidymodels automaticky pre každý fold:
#   prep(recipe, training = fold_train)   ← len z fold_train
#   bake(recipe, new_data = fold_valid)   ← aplikuje fold_train štatistiky
```

Recipe je **deklaratívny** — opisuješ čo chceš, nie ako to urobiť. `prep()` + `bake()` sa starajú o správny tok dát. Vo workflow sa recipe aplikuje samostatne pre každý CV fold, čo je jediný správny postup.

> **📚 Poučka: Data leakage cez normalizáciu — konkrétny príklad**
>
> Predstav si dataset: train_data má `displ` s priemerom 3.2 a SD 1.1. Test_data má `displ` s priemerom 3.0 a SD 1.05 (mierne iné kvôli náhode).
>
> **Správne:** normalizuješ test pomocou train štatistík (3.2, 1.1). Test hodnota 5.0 sa stane (5.0 − 3.2) / 1.1 = 1.636.
>
> **Chybne:** normalizuješ test pomocou jeho vlastných štatistík (3.0, 1.05). Test hodnota 5.0 sa stane (5.0 − 3.0) / 1.05 = 1.905.
>
> Model bol trénovaný s "jazykom" train normalizácie. Chybná normalizácia dá modelu vstup s iným "jazykom" — predikcie sú systematicky posunuté. Pri veľkých datasetoch je rozdiel malý, ale pri malých datasetoch alebo pri outlieroch môže byť zásadný.

### 3.7 Prečo `workflow()` — recipe + model ako celok

```r
lm_wf <- workflow() %>%
  add_recipe(model_recipe) %>%
  add_model(lm_spec)
```

`workflow()` z balíka `workflows` (tidymodels) zbalí recipe a model do jedného objektu. Bez workflow by si musel pred každou predikciou manuálne bake-núť dáta:

```r
# Bez workflow — ručne:
test_baked <- bake(prep_recipe, new_data = test_data)
predictions <- predict(lm_fit, new_data = test_baked)

# S workflow — automaticky:
predictions <- predict(lm_wf_fit, new_data = test_data)  # recipe sa aplikuje interne
```

Kľúčová výhoda je v **cross-validácii**: `fit_resamples(lm_wf, resamples = folds)` pre každý fold:
1. Rozdelí fold na train/validation časti
2. Zavolá `prep(recipe, training = fold_train)` — naučí recipe len z fold train
3. Zavolá `bake(recipe, new_data = fold_valid)` — aplikuje fold_train štatistiky na validation
4. Natrénuje model na fold_train
5. Evaluuje na fold_valid

Toto sa deje automaticky. Bez workflow by si to musel naprogramovať ručne v slučke — a väčšina ľudí by urobila chybu v kroku 2 (použila by celý train namiesto len fold_train).

> **📚 Poučka: Závislosť workflow od testovacieho setu**
>
> Zaujímavá vlastnosť: keď zavoláš `predict(fitted_workflow, new_data = test_data)`, workflow použije **štatistiky naučené z celého `train_data`** (nie z foldov, nie z test_data). To je správne — finálny model bol trénovaný na celom `train_data`, takže aj normalizácia musí pochádzať z celého `train_data`.
>
> Toto je dôvod, prečo `fit(lm_wf, data = train_data)` robí dve veci naraz: `prep(recipe, training = train_data)` a `fit(model, data = prepped_train_data)`. Oba kroky sa naučia z `train_data` a uložia sa dovnútra workflow objektu.

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

---

## 8. Scenario 5: Lineárna diskriminačná analýza (LDA)

### Čo je LDA a kedy sa používa

LDA (Linear Discriminant Analysis) je metóda, ktorá stojí na pomedzí dvoch úloh: **redukcie dimenzionality** a **klasifikácie**. Vstupom sú číselné prediktory a kategoriálna cieľová premenná. Výstupom sú **diskriminačné osi** — nové súradnice, v ktorých sú triedy čo najlepšie od seba oddelené.

Formálne: LDA hľadá lineárne kombinácie prediktorov (váhové vektory w), ktoré **maximalizujú pomer rozptylu medzi triedami k rozptylu vnútri tried**:

$$\max_w \frac{w^T S_B w}{w^T S_W w}$$

kde:
- $S_B$ = "between-class scatter matrix" — miera toho, ako ďaleko sú stredy tried od seba
- $S_W$ = "within-class scatter matrix" — miera toho, ako "rozptýlené" sú body vnútri každej triedy

Riešením tejto optimalizácie sú **vlastné vektory (eigenvektory)** matice $S_W^{-1} S_B$. Každý vlastný vektor definuje jednu diskriminačnú os.

> **📚 Poučka: LDA vs. PCA — supervised vs. unsupervised**
>
> Obe metódy robia redukciu dimenzionality — hľadajú "najlepší" nízkorozmerný podpriestor.
>
> | | PCA | LDA |
> |--|-----|-----|
> | Typ | Unsupervised | Supervised |
> | Čo maximalizuje | Celkový rozptyl dát | Separáciu tried |
> | Potrebuje labels? | Nie | Áno |
> | Počet osí | min(n−1, p) | min(K−1, p) kde K = počet tried |
>
> PCA by pre naše dáta mohlo nájsť os, ktorá vysvetľuje veľa variancie, ale nemusí separovať Low/Medium/High triedy. LDA explicitne hľadá os, ktorá triedy separuje — preto je pre klasifikáciu relevantnejšia.
>
> **Analógia:** PCA hľadá smer, z ktorého vidíš dáta "najroztiahnutejšie". LDA hľadá smer, z ktorého vidíš *skupiny* "najoddeliteľnejšie".

> **📚 Poučka: Predpoklady LDA**
>
> LDA je parametrický model s troma hlavnými predpokladmi:
> 1. **Multivariátna normalita** — prediktory majú v každej triede normálne rozdelenie
> 2. **Homogenita kovariancie** — kovariancie matice sú rovnaké pre všetky triedy (homokedasticity)
> 3. **Linearita** — hranice medzi triedami sú lineárne (roviny v p-dimenzionálnom priestore)
>
> V praxi je LDA pomerne robustné voči miernym porušeniam týchto predpokladov, najmä pri veľkých datasetoch. Ak predpoklad homogénnej kovariancie výrazne neplatí, QDA (Quadratic Discriminant Analysis) je lepšia voľba — umožňuje každej triede mať vlastnú kovarianciu (za cenu väčšieho počtu parametrov).

**Kedy použiť LDA:**
- Keď chceš **klasifikovať** pozorovania do tried a zároveň pochopiť, ktoré prediktory triedy separujú
- Keď chceš **vizualizovať** dáta v redukovanom priestore (2D projekcia LD1 × LD2)
- Ako **baseline klasifikátor** pred použitím zložitejších metód
- Keď triedy sú **lineárne separovateľné** — t.j. existuje lineárna hranica, ktorá ich oddeľuje

---

### 8.1 Diskretizácia cieľovej premennej

LDA je **klasifikačná** metóda — vyžaduje kategoriálnu cieľovú premennú. Naša pôvodná premenná `comb08` je spojitá (MPG). Musíme ju teda previesť na triedy.

```r
efficiency_cuts <- quantile(vehicles_model$comb08, probs = c(0, 1/3, 2/3, 1))

vehicles_lda <- vehicles_model %>%
  mutate(
    efficiency_class = cut(
      comb08,
      breaks         = efficiency_cuts,
      labels         = c("Low", "Medium", "High"),
      include.lowest = TRUE
    )
  ) %>%
  select(-comb08)
```

#### `quantile()`

```r
quantile(x, probs = c(0, 1/3, 2/3, 1))
```

`quantile()` vypočíta **kvantily** — hodnoty, pod ktorými leží daný podiel pozorovania. `probs = c(0, 1/3, 2/3, 1)` vráti štyri hodnoty:
- 0. percentil = minimum
- 33.3. percentil = **1. tertil** (pod ním leží 1/3 dát)
- 66.7. percentil = **2. tertil** (pod ním leží 2/3 dát)
- 100. percentil = maximum

Výsledok pre naše dáta:
```
  0%  33%  67% 100%
   7   18   22   74
```
Teda: Low = 7–18 MPG, Medium = 18–22 MPG, High = 22–74 MPG.

#### `cut()`

```r
cut(x, breaks = c(7, 18, 22, 74), labels = c("Low", "Medium", "High"), include.lowest = TRUE)
```

`cut()` rozdelí spojitý vektor čísel do intervalov (binov). Parametre:
- `breaks` — hranice intervalov. Pre n tried potrebuješ n+1 hraničných bodov.
- `labels` — názvy tried (musí byť o 1 menej ako `breaks`)
- `include.lowest = TRUE` — zahrnie aj minimum do prvého intervalu. Bez toho by hodnota presne rovná minimu ostala `NA`.

Výsledok je **factor** (kategoriálna premenná) s úrovňami Low < Medium < High.

> **📚 Poučka: Prečo tertily a nie fixné hranice?**
>
> Mohli by sme zvoliť intuitívne hranice: napr. Low < 20 MPG, 20–30 MPG = Medium, High > 30 MPG. Problém:
> - Tieto hranice by dali veľmi **nevyvážené triedy** — väčšina áut je v rozsahu 15–25 MPG
> - LDA predpokladá "rozumne" vyvážené triedy pre stabilné odhady kovariančných matíc
>
> Tertily garantujú, že každá trieda má **≈ 1/3 dát** bez ohľadu na tvar distribúcie. Nevýhoda: hranice 18 MPG a 22 MPG sú úzky koridok (Medium trieda má len 4 MPG šírku), čo sťaží klasifikáciu vozidiel blízkych hraniciam.
>
> **Záver:** Tertily sú technicky správne, ale interpretačne kompromis. V praxi by sa diskutovalo, či 18–22 MPG skutočne tvorí zmysluplnú "strednú efektívnosť" alebo je to len artefakt rovnomerného delenia.

**Výsledné rozdelenie:**
```
   Low  Medium    High
 18023   15512   14886

Podiely: Low 37.2%, Medium 32.0%, High 30.7%
```

Triedy nie sú dokonale vyvážené — Low dominuje, lebo distribúcia MPG má ľahký pravostranný skok (veľa bežných áut s 15–18 MPG). Pre LDA je toto stále prijateľné.

---

### 8.2 Train/Test Split a Cross-Validácia pre klasifikáciu

```r
set.seed(123)
lda_split <- initial_split(vehicles_lda, prop = 0.8, strata = efficiency_class)
lda_train  <- training(lda_split)
lda_test   <- testing(lda_split)
lda_folds  <- vfold_cv(lda_train, v = 5, strata = efficiency_class)
```

Postup je identický s Scenárom 2 — rovnaká funkcia `initial_split()` a `vfold_cv()`, len `strata` teraz odkazuje na kategorickú premennú `efficiency_class` namiesto spojitého `comb08`.

**Prečo nové splity?** `vehicles_lda` je iný objekt ako `vehicles_model` — má `efficiency_class` namiesto `comb08`. Staré `train_data`/`test_data` by tiež funčne fungovali (majú rovnaké riadky), ale nový split je čistejší — explicitne ukazuje, že pre každý scenár sa dáta pripravujú od začiatku s príslušnou cieľovou premennou.

**Výsledok:** Training: 38 735 riadkov, Test: 9 686 riadkov — rovnaké rozmery ako v Scenári 2 (rovnaký dataset, rovnaký seed, len iná cieľová premenná).

---

### 8.3 Recipe a Model Specification pre LDA

```r
lda_recipe <- recipe(efficiency_class ~ ., data = lda_train) %>%
  step_unknown(all_nominal_predictors()) %>%
  step_other(all_nominal_predictors(), threshold = 0.01) %>%
  step_impute_median(displ, cylinders) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_zv(all_predictors()) %>%
  step_nzv(all_predictors()) %>%
  step_normalize(all_numeric_predictors())
```

Recipe je **identický** s `model_recipe` zo Scenára 2 — rovnaké kroky, len zmenená cieľová premenná (`efficiency_class` namiesto `comb08`). Toto je zámerné — zabezpečuje porovnateľnosť predspracovania naprieč scenármi.

> **📚 Poučka: Prečo je normalizácia dôležitá pre LDA?**
>
> LDA počíta kovariancie matice a hľadá diskriminačné smery v priestore prediktorov. Ak má jeden prediktor rozsah 1984–2026 (`year`) a iný 4–16 (`cylinders`), vzdialenosti v priestore prediktorov sú dominované premennou s veľkým rozsahom — `year` bude "vyzerať" dôležitejšie len kvôli škále.
>
> Normalizácia (priemer = 0, SD = 1) zabezpečí, že každý prediktor prispieva rovnomerne k výpočtu kovariancií. Výsledné záťaže (loadings) diskriminačných osí sú potom **porovnateľné** — môžeme priamo povedať "prediktor A má väčší vplyv ako B".

```r
lda_spec <- discrim_linear() %>%
  set_engine("MASS") %>%
  set_mode("classification")
```

#### `discrim_linear()`

`discrim_linear()` je funkcia z balíka **`discrim`** — rozšírenie tidymodels ekosystému pre diskriminačnú analýzu. Je analógom `linear_reg()` pre regresiu.

- **`set_engine("MASS")`** — použije funkciu `MASS::lda()` z balíka MASS (Modern Applied Statistics with S). MASS je klasický štatistický balík v R, `lda()` je jeho najznámejšia funkcia.
- **`set_mode("classification")`** — explicitne povie, že ide o klasifikačnú úlohu (nie regresiu)

> **📚 Poučka: Prečo discrim + MASS engine namiesto priameho `MASS::lda()`?**
>
> Oba prístupy vedú k rovnakému výsledku. Rozdiel je v ekosystéme:
>
> **`MASS::lda()` priamo (base R):**
> ```r
> lda_fit_basic <- MASS::lda(efficiency_class ~ ., data = train_baked)
> lda_pred      <- predict(lda_fit_basic, newdata = test_baked)
> lda_pred$class      # predikované triedy
> lda_pred$x          # LD súradnice
> lda_pred$posterior  # posteriórne pravdepodobnosti
> ```
> Výhoda: priamočiarejší prístup k internému stavu modelu (`$x`, `$posterior`, `$scaling`). Nevýhoda: musíš ručne riadiť preprocessing — hlavne normalizáciu treba počítať z train a aplikovať na test manuálne (inak data leakage).
>
> **`discrim_linear()` v tidymodels:**
> ```r
> lda_wf <- workflow() %>% add_recipe(lda_recipe) %>% add_model(lda_spec)
> lda_fit <- lda_wf %>% fit(data = lda_train)
> ```
> Výhoda: recipe sa automaticky aplikuje správne (train štatistiky na test), konzistentné s ostatnými modelmi v projekte. Nevýhoda: interné objekty LDA sú "zabalené" a treba ich extrahovať.
>
> Pre jednorázové LDA bez tuningu je base R prístup jednoduchší. Pre konzistenciu s Scenárom 2 sme zvolili tidymodels.

---

### 8.4 Krížová validácia pre LDA

```r
lda_cv <- fit_resamples(
  lda_wf,
  resamples = lda_folds,
  metrics   = metric_set(accuracy),
  control   = control_resamples(save_pred = FALSE)
)
```

LDA je **uzavreté riešenie** — neexistuje hyperparameter ako `penalty`, ktorý by sme ladili. Preto `tune_grid()` nepotrebujeme, stačí `fit_resamples()`.

Krížová validácia tu slúži jedinému účelu: **nestranný odhad presnosti** pred tréningom na celom trénovacom sete. Testovací set sa stále nepoužíva.

**Výsledok:**
```
CV Accuracy: 0.7754 ± 0.0009
```

Štandardná chyba 0.0009 je extrémne malá — pri 38 735 vzorkách je CV odhad veľmi stabilný.

---

### 8.5 Trénovanie modelu a extrakcia LDA objektu

```r
lda_fit <- lda_wf %>% fit(data = lda_train)

mass_lda <- lda_fit %>%
  extract_fit_parsnip() %>%
  pluck("fit")
```

#### `extract_fit_parsnip()` a `pluck("fit")`

`lda_fit` je workflow objekt — "obal", ktorý obsahuje recipe aj model. Na prístup k internému stavu MASS lda objektu potrebujeme ho "vybaliť":

1. **`extract_fit_parsnip(lda_fit)`** — vyberie z workflow parsnip model objekt (wrapper okolo MASS lda)
2. **`pluck("fit")`** — z parsnip objektu vytiahne samotný `MASS::lda` objekt uložený pod kľúčom `"fit"`

`mass_lda` je teraz štandardný výstup funkcie `MASS::lda()` s týmito položkami:

| Položka | Typ | Čo obsahuje |
|---------|-----|-------------|
| `mass_lda$prior` | named vector | Apriorné pravdepodobnosti tried |
| `mass_lda$means` | matrix | Stredné hodnoty prediktorov pre každú triedu |
| `mass_lda$scaling` | matrix | Záťaže (coefficients) diskriminačných funkcií |
| `mass_lda$svd` | vector | Singulárne hodnoty (pre výpočet proportion of trace) |

**Apriorné pravdepodobnosti:**
```
   Low  Medium    High
0.3722  0.3204  0.3074
```
Tieto hodnoty odrážajú rozdelenie tried v trénovacej sade. LDA ich používa pri Bayesovskej klasifikácii — prior pravdepodobnosť sa kombinuje s likelihood z dát.

> **📚 Poučka: Bayesovský základ LDA**
>
> LDA klasifikuje nové pozorovanie do triedy s najväčšou **posteriórnou pravdepodobnosťou**. Podľa Bayesovho teorému:
>
> $$P(\text{trieda} = k \mid x) \propto P(x \mid \text{trieda} = k) \cdot P(\text{trieda} = k)$$
>
> kde:
> - $P(x \mid \text{trieda} = k)$ = likelihood — ako pravdepodobné je pozorovanie x v triede k (Gaussovská distribúcia)
> - $P(\text{trieda} = k)$ = prior — apriorná pravdepodobnosť triedy (z dát: 37%, 32%, 31%)
>
> Preto ak sú triedy nevyvážené (napr. Low má 37%), model bude mierne uprednostňovať Low aj pri neistých prípadoch — akceptuje väčší "prior dôkaz" v prospech Low.

---

### 8.6 Diskriminačné osi — redukcia dimenzionality

```r
prop_trace <- mass_lda$svd^2 / sum(mass_lda$svd^2)
names(prop_trace) <- paste0("LD", seq_along(prop_trace))
print(round(prop_trace, 4))
```

**Výsledok:**
```
   LD1    LD2
0.9821 0.0179
```

#### Čo je `proportion of trace`?

Pre K tried LDA nájde K−1 diskriminačných osí (pre 3 triedy = 2 osi: LD1 a LD2). Každá os má priradené **singulárne číslo (singular value)**, ktoré meria, ako veľkú separáciu medzi triedami daná os zachytáva.

`proportion of trace` = podiel kvadrátu singulárneho čísla na celkovom súčte kvadrátov:

$$\text{prop}_{LD1} = \frac{\text{svd}_1^2}{\text{svd}_1^2 + \text{svd}_2^2} = \frac{0.9821 \cdot \text{total}}{\text{total}} = 0.9821$$

**Interpretácia:** LD1 zachytáva **98.2%** celkovej separácie medzi triedami. LD2 pridáva len 1.8%. V praxi to znamená, že na vizualizáciu tried stačí **1D projekcia** na LD1 — LD2 nepridáva takmer nič.

> **📚 Poučka: Analogia s PCA — explained variance vs. proportion of trace**
>
> V PCA hovoríme o "proportion of explained variance" — podiel celkovej variancie, ktorú daná os zachytáva. V LDA hovoríme o "proportion of trace" — podiel celkovej *separácie tried*.
>
> Obe metriky slúžia rovnakej otázke: **koľko osí potrebujeme?** Ak prvá os vysvetlí >90%, zvyšné osi sú zanedbateľné. Pre naše dáta: LD1 = 98.2% → triedy sú takmer dokonale separovateľné jednou lineárnou kombináciou prediktorov.
>
> Toto je silný výsledok! Hovorí, že "Low vs. High efektívnosť" je v zásade **jednorozmerný problém** — existuje jedna lineárna kombinácia engineových parametrov, ktorá takmer dokonale oddeľuje neefektívne od efektívnych vozidiel.

#### Projekcia trénovacích dát

```r
prepped     <- prep(lda_recipe, training = lda_train)
train_baked <- bake(prepped, new_data = lda_train)

lda_proj <- predict(mass_lda, newdata = select(train_baked, -efficiency_class))

lda_train_proj <- tibble(
  LD1              = lda_proj$x[, 1],
  LD2              = lda_proj$x[, 2],
  efficiency_class = lda_train$efficiency_class
)
```

#### `prep()` a `bake()` — manuálna aplikácia recipe

V Scenári 2 nám stačilo `predict(lda_fit, new_data = ...)` — workflow sa postaral o preprocessing automaticky. Tu ale potrebujeme prístup k "surovým" prediktorom pre `predict(mass_lda, ...)` (MASS lda objekt), nie cez workflow.

- **`prep(lda_recipe, training = lda_train)`** — "naučí" recipe z trénovacích dát (vypočíta mediány, škály, dummy úrovne). Vracia *prepped recipe* objekt.
- **`bake(prepped, new_data = lda_train)`** — aplikuje naučené transformácie na dáta. `new_data = lda_train` aplikuje na train; `new_data = lda_test` by aplikovalo *rovnaké škály* na testovací set.

> **📚 Poučka: Prečo `prep()` musí vidieť len train dáta**
>
> `prep(lda_recipe, training = lda_train)` vypočíta napr. mediány pre imputáciu a priemery/SD pre normalizáciu **len z trénovacích dát**. Keď potom zavoláš `bake(prepped, new_data = lda_test)`, tieto uložené hodnoty sa aplikujú na testovací set.
>
> Keby si zavolal `prep(lda_recipe, training = bind_rows(lda_train, lda_test))`, štatistiky by boli vypočítané z oboch setov — to je **data leakage**. Testovací set by "pomáhal nastaviť" preprocessing, čo v reálnom nasadení nie je možné (test data v čase trénovania neexistujú).

#### `predict(mass_lda, newdata = ...)$x`

```r
lda_proj <- predict(mass_lda, newdata = select(train_baked, -efficiency_class))
```

Funkcia `predict()` pre MASS lda objekt vracia **list** (nie tibble ako tidymodels):
- `$class` — faktor predikovaných tried
- `$posterior` — matica posteriórnych pravdepodobností (riadky = pozorovania, stĺpce = triedy)
- `$x` — **matica LD súradníc** (riadky = pozorovania, stĺpce = LD1, LD2, ...)

`lda_proj$x[, 1]` extrahuje prvý stĺpec = LD1 súradnice pre každé pozorovanie.

#### Vizualizácia s `stat_ellipse()`

```r
ggplot(lda_train_proj, aes(x = LD1, y = LD2, color = efficiency_class)) +
  geom_point(alpha = 0.25, size = 0.6) +
  stat_ellipse(level = 0.95, linewidth = 1.2)
```

**`stat_ellipse(level = 0.95)`** nakreslí **95% elipsu spoľahlivosti** pre každú triedu. Elipsa je vytvorená z kovariančnej matice bodov v každej triede — reprezentuje oblasť, v ktorej by sme čakali 95% nových pozorovaní z tej istej triedy (za predpokladu normálneho rozdelenia).

> **📚 Poučka: Čo elipsy spoľahlivosti hovoria o separácii tried**
>
> Ak sa elipsy **neprekrývajú**: triedy sú dobre separované v danom priestore.
> Ak sa elipsy **výrazne prekrývajú**: triedy nie sú lineárne oddeliteľné a model bude mať vysokú chybovosť v oblastiach prekryvu.
>
> Pre naše dáta: Low a High elipsy by mali byť dobre oddelené pozdĺž LD1. Medium elipsa by mala sedieť medzi nimi — čo vysvetlí, prečo Medium má najnižšiu per-class accuracy (68.8%).

**LD axis ranges (trénovacie dáta):**
```
LD1: -6.386 až 4.729
LD2: -3.422 až 3.743
```

Rozsah LD1 je väčší ako LD2 — konzistentné s tým, že LD1 zachytáva viac variability.

---

### 8.7 Záťaže diskriminačných osí (Loadings)

```r
lda_loadings <- as_tibble(mass_lda$scaling, rownames = "predictor")
```

**`mass_lda$scaling`** je matica rozmerov p × (K−1), kde p = počet prediktorov, K−1 = počet diskriminačných osí. Každý stĺpec (LD1, LD2) obsahuje koeficienty lineárnej kombinácie:

$$LD1 = w_1 \cdot x_1 + w_2 \cdot x_2 + \ldots + w_p \cdot x_p$$

kde $w_j$ je záťaž j-teho prediktora.

**Top 5 prediktorov pre LD1:**
```
predictor                  LD1
displ                   -0.765
cylinders               -0.618
has_discrete_gears      -0.615
drive_Front.Wheel.Drive  0.504
n_gears                  0.350
```

**Interpretácia záťaží:**
- **Záporná záťaž** (displ, cylinders, has_discrete_gears): väčšia hodnota týchto prediktorov posúva vozidlo smerom k Low triede (nižšia efektívnosť = vyššia spotreba)
- **Kladná záťaž** (drive_Front.Wheel.Drive, n_gears): FWD pohon a viac prevodových stupňov sú asociované s vyššou efektívnosťou

**Fyzikálna interpretácia:**
- `displ` (−0.765): Väčší objem motora = viac paliva = nižšia efektívnosť. Najsilnejší diskriminátor.
- `cylinders` (−0.618): Viac valcov = väčší motor = nižšia efektívnosť
- `has_discrete_gears` (−0.615): Vozidlá s diskrétnymi stupňami (nie CVT) sú priemerne menej efektívne — CVT optimalizuje prevod kontinuálne
- `drive_Front.Wheel.Drive` (+0.504): FWD je bežné u efektívnych mestských a kompaktných áut
- `n_gears` (+0.350): Viac prevodových stupňov → motor môže pracovať v optimálnych otáčkach pri rôznych rýchlostiach → lepšia efektívnosť

> **📚 Poučka: Záťaže vs. koeficienty regresie**
>
> Záťaže LDA sú **štandardizované** (prediktory boli normalizované) — môžeme ich priamo porovnávať. Väčšia absolútna hodnota = silnejší diskriminátor.
>
> Toto je analogické Ridge koeficientom zo Scenára 2. Tam najväčšie (absolútne) koeficienty mali tiež `cylinders` a `displ` — konzistentnosť medzi scenármi potvrdzuje, že tieto prediktory sú skutočne najdôležitejšie pre efektívnosť vozidiel.

---

### 8.8 Klasifikačná výkonnosť

```r
lda_test_pred <- predict(lda_fit, new_data = lda_test) %>%
  bind_cols(lda_test %>% select(efficiency_class)) %>%
  rename(pred = .pred_class, truth = efficiency_class)
```

`predict(lda_fit, new_data = lda_test)` cez tidymodels workflow vracia tibble so stĺpcom `.pred_class` — predikovaná trieda pre každé pozorovanie. Potom pripojíme skutočné triedy z `lda_test` a premenujeme stĺpce pre čitateľnosť.

#### Klasifikačné metriky

Na rozdiel od regresie (RMSE, MAE, R²) klasifikácia používa iné metriky:

```r
lda_acc  <- accuracy(lda_test_pred,  truth = truth, estimate = pred)
lda_prec <- precision(lda_test_pred, truth = truth, estimate = pred, estimator = "macro")
lda_rec  <- recall(lda_test_pred,    truth = truth, estimate = pred, estimator = "macro")
```

**Výsledky:**
```
Accuracy (celková):   0.7791
Precision (macro):    0.7790
Recall    (macro):    0.7779
```

#### Accuracy

$$\text{Accuracy} = \frac{\text{počet správne klasifikovaných}}{\text{celkový počet}}$$

Najjednoduchšia metrika: aký podiel všetkých pozorovaní bol správne zaradený.

**Naša hodnota: 77.9%** — model správne klasifikuje takmer 8 z 10 vozidiel. Pre 3-triednu úlohu (náhodný odhad = 33%) je to solídny výsledok.

> **📚 Poučka: Kedy Accuracy nestačí**
>
> Accuracy je zavádzajúca pri **nevyvážených triedach**. Predstav si: Low = 90% dát, Medium = 5%, High = 5%. Model, ktorý vždy predikuje Low, dosiahne 90% Accuracy — ale je to zbytočný model.
>
> Pre naše dáta sú triedy relatívne vyvážené (37%, 32%, 31%), takže Accuracy je tu vhodná metrika.

#### Precision a Recall

Pre binárnu klasifikáciu (dve triedy: Positive/Negative):
$$\text{Precision} = \frac{TP}{TP + FP} \qquad \text{Recall} = \frac{TP}{TP + FN}$$

Pre **multitriednu** klasifikáciu treba rozšíriť definíciu. Jeden prístup: vypočítaj precision/recall pre každú triedu zvlášť (one-vs-rest) a spriemeruj.

| Trieda | TP | FP | FN | Precision | Recall |
|--------|----|----|-----|-----------|--------|
| Low | správne Low | iné → Low | Low → iné | 2935/(2935+629+33) | 2935/(2935+336+8) |
| Medium | správne Med | iné → Med | Med → iné | 2133/(2133+336+632) | 2133/(2133+629+499) |
| High | správne High | iné → High | High → iné | 2471/(2471+8+499) | 2471/(2471+33+632) |

**`estimator = "macro"`** — aritmetický priemer precision/recall cez všetky triedy. Každá trieda má rovnakú váhu bez ohľadu na počet pozorovaní.

> **📚 Poučka: Macro vs. Weighted averaging**
>
> Pri `estimator = "macro"`: každá trieda má rovnakú váhu → Low, Medium, High sú rovnako dôležité
> Pri `estimator = "macro_weighted"`: váha = počet pozorovaní v triede → väčšia trieda má väčší vplyv
>
> Pre akademické porovnanie modelov je "macro" štandard — nechceme, aby výsledok závisel od náhodnej nevyváženosti tried.

#### Konfúzna matica

```
          Truth
Prediction  Low Medium High
    Low    2935    336    8
    Medium  629   2133  499
    High     33    632 2471
```

Konfúzna matica (confusion matrix) ukazuje pre každú kombináciu (predikovaná trieda × skutočná trieda) počet pozorovaní. **Diagonála** = správne klasifikácie. Mimo diagonály = chyby.

**Čo vidíme:**
- Väčšina chýb je medzi **susednými triedami** (Low↔Medium, Medium↔High)
- Len 8 vozidiel: predikované Low, skutočné High (a 33 opačne) — extrémne chyby sú vzácne
- Medium trieda má najviac chýb (629 + 499 = 1128 chybne klasifikovaných z 3103)

> **📚 Poučka: Čo konfúzna matica hovorí o type chýb**
>
> Nie všetky chyby sú rovnako závažné. Pre MPG klasifikáciu:
> - Low predikovaný ako High (8 prípadov) = hovorím, že auto je efektívne, ale v skutočnosti je neefektívne → **veľká chyba** (napr. zákazník by bol sklamaný)
> - Low predikovaný ako Medium (336 prípadov) = menší omyl, triedy sú susedné
>
> V iných doménach (napr. lekárska diagnostika) by zámerné asymetrické penalizovanie chýb viedlo k použitiu **cost-sensitive** klasifikátorov. Pre naše akademické účely postačuje macro accuracy.

**Per-class accuracy:**

| Trieda | n | Správne | Accuracy |
|--------|---|---------|----------|
| Low | 3 605 | 2 935 | 81.4% |
| Medium | 3 103 | 2 133 | 68.8% |
| High | 2 978 | 2 471 | 83.0% |

Medium je najťažšia trieda — 4 MPG šírka koridoru (18–22 MPG) spôsobuje, že vozidlá pri hraniciach sú takmer nerozlíšiteľné.

---

### 8.9 Výsledky a záver — Scenario 5

#### Redukcia dimenzionality: výsledok

LD1 zachytáva 98.2% celkovej separácie medzi triedami. Dáta sú takmer dokonale separovateľné **v jednej dimenzii** — lineárna kombinácia motorových parametrov (displ, cylinders, has_discrete_gears, FWD, n_gears) dokáže takmer sama o sebe zaradiť vozidlo do správnej efektívnostnej triedy.

Toto je konzistentné so Scenárom 2: aj tam dominovali `cylinders` a `displ`. Oba scenáre konvergujú k rovnakému záveru — MPG efektívnosť je primárne určená veľkosťou motora.

#### Klasifikačná výkonnosť: záver

| Metrika | Hodnota |
|---------|---------|
| CV Accuracy | 77.5% ± 0.09% |
| Test Accuracy | 77.9% |
| CV–Test gap | 0.4 pp (žiadny overfitting) |
| Najťažšia trieda | Medium (68.8%) |
| Najľahšia trieda | High (83.0%) |

**Vhodnosť LDA pre tento problém:**
LDA je vhodný model. Separácia tried je takmer lineárna (LD1 dominuje), prediktory sú numericky normalizované a triedy sú relatívne vyvážené. 77.9% accuracy pri 3 triedach (baseline: 33%) potvrdzuje, že fyzikálne vlastnosti motora naozaj predikujú efektívnostnú triedu.

**Obmedzenia:**
1. Tertilové hranice (18 MPG, 22 MPG) sú arbitrárne — Medium trieda je umelo zúžená, čo znižuje jej accuracy
2. LDA predpokladá lineárne hranice — vozidlá blízko hraníc (17–19 MPG, 21–23 MPG) sú štrukturálne ťažko klasifikovateľné bez nelineárneho rozhodovacieho pravidla
3. Porovnanie so Scenárom 2: RMSE 3.36 MPG je väčší ako šírka Medium triedy (4 MPG) — regresný model by tiež chyboval pri hraničných prípadoch

---

### 8.10 Slovník príkazov Scenára 5

#### `quantile()`

```r
quantile(vehicles_model$comb08, probs = c(0, 1/3, 2/3, 1))
```

Vypočíta kvantily vektora. `probs` definuje, ktoré percentily chceme. Výsledok: named vector kde mená sú percentily a hodnoty sú zodpovedajúce hodnoty z distribúcie.

#### `cut()`

```r
cut(comb08, breaks = cuts, labels = c("Low", "Medium", "High"), include.lowest = TRUE)
```

Diskretizuje spojitý vektor do intervalov. `breaks` definuje hranice, `labels` názvy tried. Vracia faktor. `include.lowest = TRUE` zahrnie minimum do prvého intervalu (bez toho by `x == min(breaks)` dalo `NA`).

#### `discrim_linear()` z balíka `discrim`

```r
library(discrim)
lda_spec <- discrim_linear() %>% set_engine("MASS") %>% set_mode("classification")
```

Tidymodels interface pre lineárnu diskriminačnú analýzu. `set_engine("MASS")` špecifikuje backend — balík MASS, funkcia `lda()`. Analóg `linear_reg()` pre regresnú úlohu.

#### `extract_fit_parsnip()` a `pluck()`

```r
mass_lda <- lda_fit %>% extract_fit_parsnip() %>% pluck("fit")
```

Dvojkroková extrakcia interného modelu z workflow:
1. `extract_fit_parsnip(lda_fit)` — vyberie parsnip wrapper objekt
2. `pluck("fit")` — extrahuje pôvodný MASS lda objekt uložený pod kľúčom `"fit"`

`pluck()` je z balíka `purrr` (tidyverse) — bezpečný prístup k prvkom listu. `pluck(x, "fit")` je ekvivalent `x[["fit"]]`, ale neskrachuje pri chýbajúcom kľúči.

#### `prep()` a `bake()` — manuálna aplikácia recipe

```r
prepped     <- prep(lda_recipe, training = lda_train)
train_baked <- bake(prepped, new_data = lda_train)
test_baked  <- bake(prepped, new_data = lda_test)   # rovnaké škály ako z train!
```

- `prep(recipe, training = data)` — "naučí" recipe: vypočíta mediány, škály, dummy úrovne z trénovacích dát
- `bake(prepped, new_data = X)` — aplikuje naučené transformácie na X. Kľúčové: test bake používa *trénovacie škály* — toto zabraňuje data leakage

#### `predict(mass_lda, newdata = ...)` — MASS lda výstup

```r
lda_proj <- predict(mass_lda, newdata = select(train_baked, -efficiency_class))
# $class     — predikované triedy (factor)
# $posterior — matica posteriórnych pravdepodobností (n × K)
# $x         — matica LD súradníc (n × (K-1))
```

MASS `predict()` vracia **list** (nie tibble). `$x` obsahuje projekcie na diskriminačné osi — to sú hodnoty potrebné pre vizualizáciu redukcie dimenzionality.

#### `stat_ellipse()` v ggplot2

```r
stat_ellipse(level = 0.95, linewidth = 1.2)
```

Pridá do grafu elipsu spoľahlivosti. `level = 0.95` nakreslí oblasť, v ktorej by ležalo 95% nových pozorovaní z rovnakej distribúcie (za predpokladu bivariátnej normality). Automaticky sa počíta zvlášť pre každú skupinu definovanú `aes(color = ...)` alebo `aes(group = ...)`.

#### `accuracy()`, `precision()`, `recall()` z `yardstick`

```r
accuracy(data, truth = truth, estimate = pred)
precision(data, truth = truth, estimate = pred, estimator = "macro")
recall(data,    truth = truth, estimate = pred, estimator = "macro")
```

Funkcie z balíka `yardstick` (súčasť tidymodels). Pre multitriednu klasifikáciu:
- `accuracy()` — nevyžaduje `estimator` (je rovnaká bez ohľadu na averaging)
- `precision()` a `recall()` — vyžadujú `estimator`:
  - `"macro"` — aritmetický priemer cez triedy (rovnaká váha)
  - `"macro_weighted"` — vážený priemer podľa veľkosti tried
  - `"micro"` — agregovanie TP/FP/FN cez všetky triedy naraz

Všetky vracajú tibble so stĺpcom `.estimate` obsahujúcim hodnotu metriky.

#### `conf_mat()` a `autoplot()`

```r
cm <- conf_mat(lda_test_pred, truth = truth, estimate = pred)
print(cm)                          # textová matica
autoplot(cm, type = "heatmap")     # grafická heatmapa
```

`conf_mat()` vytvorí konfúznú maticu ako špeciálny objekt triedy `conf_mat`. `autoplot()` s `type = "heatmap"` zobrazí maticu ako farebný grid, kde farba kóduje počet pozorovaní — tmavšia = viac prípadov.

---

*Dokument bude rozšírený o ďalšie scenáre.*
