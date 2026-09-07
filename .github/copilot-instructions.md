# Project Guidelines

Szakdolgozat repó: LaTeX dolgozat (gyökér) + Python demo-alkalmazás ([code/](../code/)).

## Repo szerkezet

- `Gabrity_Gabor_D9V09Z_Szakdolgozat.tex` — a dolgozat gyökér forrása. Fejezet-változtatás után **két** LaTeX fordítás kell a TOC frissüléséhez.
- `code/` — a dolgozatban bemutatott alkalmazás (adversarial példák generálása/védelme ImageNet modelleken). `code/README.md` írja le a mappastruktúráját és indítását.
- `input/szakdolgozatok/` — korábbi szakdolgozatok txt-i referenciaként (struktúra, stílus).
- `input/kovetelmenyek/` — formai és tartalmi követelmények, általános tanácsok (BSc/MSc).
- `input/cikkek/` — hivatkozott cikkek (Goodfellow, Madry) PDF + txt formában.

## Python kód (`code/`)

- Modulok: `src/model_manager.py`, `src/attack_engine.py`, `src/defense_module.py`, `src/utils.py`, `src/gui/`.
- Virtuális környezet: `code/.venv`. Aktiválás előtte: `.\.venv\Scripts\Activate.ps1`.
- Tesztek futtatása: `cd code; python -m pytest -q` (venv aktiválva legyen, különben `No module named pytest`).
- `attack_engine.py` "functional core": a model/mean/std explicit paraméterként (`AttackContext`) érkezik, nincs rejtett I/O — ezért mockolás nélkül tesztelhető.
- `model_manager.py` a torchvision modelleket (`models.resnet50`, `models.mobilenet_v2`) tölti be — ezek unit tesztben mindig mockolandók (`mocker.patch`), a valódi letöltés/build integrációs szintre való.

## Tesztelési elvek (`code/tests/`)

Lásd [code/tests/conftest.py](../code/tests/conftest.py) és [code/pytest.ini](../code/pytest.ini) markereit (`unit`, `integration`, `gui`).

- **Teszt-piramis**: unit (ms, `tiny_model`/mock-olt backbone) → integration → gui.
- Belső függvény (`_fgsm`, `_pgd`) atomikus, direkt importos tesztelése megengedett, ha önálló algoritmus-identitása van — elkerülve a kombinatorikus robbanást a wrapperen (`attack()`) keresztül. A wrapper tesztje csak a saját felelősségét nézze (validáció, dispatch, eredmény-összeállítás).
- Dispatch-teszt mock spy-jal: `mocker.patch(..., wraps=eredeti_fv)` + `assert_called_once()` / `assert_not_called()`.
- Numerikus algoritmusokra tulajdonság-alapú ÉS oracle/referencia teszt is íródjon, kiegészítve viselkedési bizonyítékkal (`predict()`-en át: nő/csökken-e a bizalom).
- Determinizmus: minden random fixture maga hívja a `torch.manual_seed`-et, fixture-sorrendtől függetlenül.
- Határesetek kötelezők: `epsilon=0`, szélsőséges pixelértékek, irreálisan nagy `step_size`.
- `.detach()` ellenőrzése: `grad_fn is None` + `is_leaf is True` erősebb bizonyíték, mint `requires_grad is False`.
- Egy tulajdonságot mindkét attack-módszerre (`FGSM`, `PGD`) `@pytest.mark.parametrize`-dal tesztelj, ne csak egyikre.
- Nincs kikényszerített 1:1 modul↔tesztfájl megfeleltetés; szervezés felelősség szerint.
- Szakirodalomból eredő "mágikus" konstansot (pl. PGD `step_size` `2.5`-ös szorzója, forrás: Madry et al. 2018) kommentben dokumentálj, ne tűnjön önkényesnek.
- `import torch` ~4s overhead — ez PyTorch natív jellemzője, nem tesztelési hiba.

## Dolgozat konvenciók

Lásd a [repo memóriát](/memories/repo/thesis-requirements.md) is: kötelező önálló `Bevezetes`, `Felhasznaloi dokumentacio`, `Fejlesztoi dokumentacio`, `Teszteles` fejezetek, formai/tartalmi elvárások az `input/kovetelmenyek/` fájlokban.
