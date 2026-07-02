# XIII — Spec de périmètre v0.1

> **En une phrase :** XIII est le *linter d'overfitting pour backtests* — le Protocole du
> Treizième Homme transformé en logiciel. On lui donne un backtest, il répond :
> « voici les N façons dont ce backtest se ment probablement à lui-même. »
>
> **Ce n'est PAS** un robot, un signal, ni une promesse de rendement. C'est un
> *pair reviewer* automatisé. Vendeur et acheteur du même côté de la table.

**Statut :** périmètre CONFIRMÉ (2026-07-02). Ce document fige le *quoi*.
L'implémentation des checks est la session suivante.

**Portée v0.1 (volontairement narrow) :** brokers **FTMO + Vantage**, instrument-pilote
**GBPUSD**. On généralisera après avoir prouvé la valeur sur le terrain qu'on connaît
vraiment (3 ans de tests réels dessus).

---

## §1 — Principe : deux couches, une frontière nette

Le protocole a 8 sections (A→H). Certaines sont **calculables par une machine** (le
*linter*), d'autres relèvent du **jugement humain guidé** (le *copilote*). La frontière
est le cœur du design :

| Couche | Nature | Ce qu'elle fait | Modèle éco |
|---|---|---|---|
| **Linter** | Déterministe | Lance les checks automatisables, verdict PASS/WARN/FAIL | Gratuit (OSS) |
| **Copilote** | Guidé / IA | Fait *écrire* à l'utilisateur ses peurs, son pre-mortem, sa fidélité d'exécution, et les teste avec lui | Pro (payant) |

Le protocole (le document, l'histoire, les 8 checks) reste **offert** : c'est le marketing.
On vend son **exécution automatisée** (linter Pro + copilote), pas l'information.

---

## §2 — Périmètre confirmé

### Dans le scope v0.1
- Les **9 checks automatisables** (§3) couvrant les 8 sections A→H.
- **BrokerConfig** pour FTMO (règles prop-firm) et Vantage (coûts d'exécution réels).
- Instrument **GBPUSD** entièrement paramétré (spread mesuré sur tes ticks).
- Entrées : un DataFrame de trades et/ou une courbe d'équité, + optionnellement le
  fichier source de la stratégie (pour les checks statiques).
- Sortie : un `AuditReport` lisible en terminal + exportable JSON.
- CLI : `xiii audit trades.csv --broker ftmo`.

### Hors scope v0.1 (repoussé, assumé)
- Autres brokers que FTMO/Vantage (→ v1.0 : profils broker ajoutables).
- Autres instruments richement paramétrés (GBPUSD d'abord ; les autres = spread générique).
- Le copilote IA automatisé (F/H/G) → Pro, phase 2.
- Dashboard hébergé, multi-utilisateur, runs cloud → Pro, phase 2.
- Toute connexion live/broker (XIII lit des backtests, il ne trade pas).

---

## §3 — Les 8 sections du protocole → 9 checks + 3 guidés

**Couverture : 8/8 sections.** Chaque check cite le fichier du labo d'où sa logique est
extraite (« extraction, pas construction »).

### Linter (automatisé, gratuit)

| id | §Protocole | Ce qu'il détecte | Entrées requises | Extrait de |
|---|---|---|---|---|
| `A1_dummy_scan` | A | `np.random`/`dummy`/`mock`/`placeholder`/`TODO`/`FIXME` dans le script qui produit LE chiffre | `source_file` | grep fourni par le protocole §A (faille CPI `np.random.normal`) |
| `A3_series_alignment` | A | Anti-pattern `concat().dropna()` (inner-join qui effondre la fenêtre) au lieu de `reindex().fillna()` | `source_file` | protocole §A (faille « fenêtre tombée à 23 jours ») |
| `G1_lookahead_scan` | G | `.shift(-n)`, référence au futur, signal calé sur barre N+1 | `source_file` | `baseline_trend_adx.py` & 10+ scripts |
| `B1_window_sensitivity` | B | **Le mirage.** Sharpe recalculé sur fenêtre longue vs courte ; alerte si la fenêtre courte gonfle > seuil | `trades`/`equity` datés | audit 13e homme (Sharpe 2.53→1.22 sur 2.3 ans vs 16 ans) |
| `B3_yearly_breakdown` | B | Décompte année par année ; signale chaque année négative / chaque breach | `trades`/`equity` datés | `portfolio_longhistory.py` |
| `C1_sizing_vs_maxdd` | C | Le sizing déployé défonce-t-il la limite broker ? `k_for_dd` vs `max_total_drawdown` | `equity` + `broker` + `deployed_sizing` | `sizing_fix.py` (`k_for_dd`, `max_dd`) |
| `C2_montecarlo_constraints` | C | Bootstrap de la séquence de trades → P(breach −10%), P(−5% en 1 jour), P(pass challenge) | `trades` + `broker` | nouveau (Monte Carlo standard) sur primitives `sizing_fix.py` |
| `D2_spread_breakeven` | D | Win-rate de rentabilité **net** vs **brut** après spread réel ; marge de la stratégie | `sl_pips`/`tp_pips` + `broker` | `spread_cost_macroedge.py` (be_gross→be_net) |
| `DE_portfolio_correlation` | D+E | Corrélation (Spearman) avec le portefeuille existant, **moyenne ET conditionnelle en crise** | `portfolio` | `portfolio_4bricks_fixed.py` (`df.corr`) + mémoire corr-crise |

### Copilote (guidé — checklist imprimée en v0.1, automatisé en Pro)

| id | §Protocole | Rôle |
|---|---|---|
| `F_falsify_fears` | F | Fait lister à l'utilisateur ses risques affirmés et l'aide à les tester par la donnée (dans les deux sens) |
| `G2_execution_fidelity` | G | Checklist : l'EA reproduit-il EXACTEMENT la règle ? SL en dur chez le broker ? (nécessite l'EA) |
| `H_premortem` | H | Fait écrire « comment ce robot meurt » AVANT le live |

---

## §4 — BrokerConfig (FTMO + Vantage)

Encodé en dur dans `xiii/brokers.py`. Niveaux de confiance annotés dans le code.

### FTMO (prop-firm → contraintes = règles de compte)
- `max_total_drawdown = -0.10` — breach dur *(confiance haute, règle FTMO)*
- `max_daily_loss = -0.05` — breach dur *(confiance haute)*
- `profit_target = +0.10` — objectif Challenge phase 1 *(confiance haute)*
- `min_trading_days = 4` — configurable selon version *(à confirmer)*
- spread GBPUSD : **modélisé** (FTMO ≈ raw + commission) *(placeholder à confirmer)*

### Vantage (broker réel → contraintes = coûts d'exécution, **pas** de règle DD)
- Aucune limite DD (compte réel) → champs risque = `None`
- spread GBPUSD fenêtre d'entrée (07-08h UTC) : **1.6p** *(MESURÉ sur `GBPUSD_ticks_30d`, confiance haute)*
- spread all-day : 1.7p *(mesuré)*
- p95 mauvais-jour, commission RAW, slippage : *(placeholders à confirmer sur les ticks)*

> ⚠️ Les chiffres marqués « placeholder » sont à figer avant le lancement public.
> Les chiffres « mesuré/règle » sont solides.

---

## §5 — API publique (le contrat)

```python
import xiii

report = xiii.audit(
    trades=trades_df,            # DataFrame : date, pnl (ou entry/exit/pips), win…
    equity=equity_series,        # optionnel : courbe d'équité (sinon reconstruite depuis trades)
    source_file="ma_strategie.py",  # optionnel : active les checks statiques (A1, A3, G1)
    broker=xiii.FTMO,            # BrokerConfig
    deployed_sizing=1.08,        # multiplicateur de lot réellement déployé
    portfolio={"MacroEdge": rendements_serie, ...},  # optionnel : active DE
    sl_pips=15, tp_pips=30,      # optionnel : active D2
)

report.print()      # rapport lisible terminal
report.to_json()    # sortie machine
report.passed       # bool global (aucun FAIL)
report.verdicts     # list[CheckResult]
```

Chaque check renvoie un `CheckResult` :

```python
CheckResult(
    check_id: str,          # "B1_window_sensitivity"
    section: str,           # "B"
    status: str,            # "PASS" | "WARN" | "FAIL" | "SKIP"
    headline: str,          # "Sharpe gonflé de +49% par la fenêtre courte"
    detail: str,            # explication chiffrée
    evidence: dict,         # chiffres bruts (pour le JSON / dashboard)
)
```

`SKIP` = entrée requise absente (ex : pas de `source_file` → A1/A3/G1 sautés, signalés).

---

## §6 — Arborescence + carte d'extraction

```
xiii/
├── pyproject.toml
├── README.md              (à écrire — c'est aussi du contenu marketing)
├── SPEC_v0.1.md           (ce fichier)
└── xiii/
    ├── __init__.py        expose audit(), FTMO, VANTAGE
    ├── brokers.py         BrokerConfig, InstrumentCost, FTMO, VANTAGE   ← chiffres réels
    ├── metrics.py         sharpe, max_drawdown, annualized_return, k_for_dd  ← EXTRAIT de sizing_fix.py
    ├── report.py          CheckResult, AuditReport                     (à écrire)
    ├── cli.py             `xiii audit …`                               (à écrire)
    └── checks/
        ├── integrity.py   A1_dummy_scan, A3_series_alignment, G1_lookahead_scan  ← protocole §A + baseline_trend_adx.py
        ├── window.py      B1_window_sensitivity, B3_yearly_breakdown   ← audit 13e homme + portfolio_longhistory.py
        ├── sizing.py      C1_sizing_vs_maxdd, C2_montecarlo_constraints ← sizing_fix.py
        ├── cost.py        D2_spread_breakeven                          ← spread_cost_macroedge.py
        └── correlation.py DE_portfolio_correlation                     ← portfolio_4bricks_fixed.py
```

**Fait (2026-07-02) :** `brokers.py`, `metrics.py`, `__init__.py`, `pyproject.toml`,
`report.py`, `checks/window.py` (B1 ✅ testé), `checks/integrity.py` (A1+A3+G1 ✅ testés,
dogfoodés sur `research/`), `examples/` (demo + spécimen). **4/9 checks actifs.**
**Reste :** `checks/` B3, C1, C2, D2, DE ; `cli.py` ; `README.md` ; figer placeholders §4.

---

## §7 — Découpe open-core

| | Gratuit (OSS) | Pro (payant) |
|---|---|---|
| Les 9 checks du linter | ✅ | ✅ |
| Mono-stratégie, CLI, JSON | ✅ | ✅ |
| BrokerConfig FTMO + Vantage | ✅ | ✅ |
| Corrélation **multi-stratégies** (portefeuille N briques) | — | ✅ |
| Copilote IA (F/H/G automatisés via Claude) | — | ✅ |
| Profils broker personnalisés (au-delà de FTMO/Vantage) | — | ✅ |
| Runs hébergés + dashboard + historique | — | ✅ |
| Walk-forward automatisé | — | ✅ |
| **Services** : audit à la mission (« je red-team ta stratégie ») | — | 💶 mission |

---

## §8 — Reste à faire + définition de « done »

**Prochaine session (implémentation v0.1) :**
1. `report.py` — `CheckResult` + `AuditReport` (print terminal + JSON).
2. `checks/` — implémenter les 9 checks (extraction des logiques citées §3).
3. `cli.py` — `xiii audit`.
4. `README.md` — manifeste + démo (le backtest MacroEdge audité par XIII, screenshot de *rigueur* pas de rendement).
5. Figer les placeholders BrokerConfig (§4) sur les ticks.

**Definition of done v0.1 :** `pip install xiii` fonctionne ; `xiii.audit(...)` sur le
backtest MacroEdge GBPUSD sort les 9 verdicts, dont **B1 qui rattrape le mirage 2.53→1.22**
(le test d'acceptation : XIII doit débusquer l'erreur qui t'a coûté 3 ans).
