# Project Guidelines

Szakdolgozat repó: LaTeX dolgozat ([dolgozat/](../dolgozat/)) + Python demo-alkalmazás ([code/](../code/)).

Ha bármely feladat (kódírás, lektorálás, dolgozatszerkesztés stb.) közben olyan kérdés merül fel, ami nem
dönthető el egyértelműen a meglévő kontextusból (repo memória, jelen fájl, a fájlok tartalma), kérdezz vissza
a felhasználótól végrehajtás előtt, ne találgass.

## Repo szerkezet

- `dolgozat/Gabrity_Gabor_D9V09Z_Szakdolgozat.tex` — a dolgozat egyetlen gyökér forrása. Ezt a meglévő fájlt kell szerkeszteni és fordítani; új dolgozatfájlt nem szabad generálni. A fordítást a `dolgozat/` könyvtárban, vagy a fájl teljes elérési útjával kell indítani. Fejezet-változtatás után **két** LaTeX fordítás kell a TOC frissüléséhez.
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
- LaTeX forrásban minden mondat új sorban kezdődjön; egy mondaton belül sortörés megengedett, ha a sor túllóg egy ésszerű hosszon. Minden mondat végén szerepeljen egy darab szóköz karakter

## ELTE LaTeX referencia

- Az ELTE Informatikai Kar LaTeX-sablon teljes helyi másolata az [input/elteikthesis/](../input/elteikthesis/) könyvtárban található. A könyvtár referenciaanyag; a benne lévő sablonfájlokat a jelenlegi dolgozat szerkesztésekor nem kell módosítani.
- A [sablon összefoglalója](../input/elteikthesis/ELTEIKTHESIS_SUMMARY.md) a sablon formai beállításait, dokumentumszerkezetét, címlap-metaadatait, fordítási folyamatát, opcionális jegyzékeit és a jelenlegi dolgozattal való kapcsolatát írja le.
- Formázási és dokumentumosztály-szintű részletekhez az [elteikthesis.cls](../input/elteikthesis/elteikthesis.cls) fájlt kell használni; a tényleges margó-, sortávolság-, oldalszámozási és címlap-beállítások ott találhatók.
- A magyar használati útmutató és a fordítási lépések az [README_hu.md](../input/elteikthesis/README_hu.md) fájlban vannak.
- A teljes magyar mintadokumentum és annak szerkezete az [elteikthesis_hu.tex](../input/elteikthesis/elteikthesis_hu.tex) fájlban, a magyar mintafejezetek a [samples_hu/](../input/elteikthesis/samples_hu/) könyvtárban találhatók.
- Bibliográfiai példák az [elteikthesis.bib](../input/elteikthesis/elteikthesis.bib), képek és logók az [images/](../input/elteikthesis/images/), forráskód-formázási példa pedig az [elteikthesis_minted.tex](../input/elteikthesis/elteikthesis_minted.tex) fájlban található.
- Az `elteikthesis` sablonra való teljes `report`/`chapter` alapú áttérés nem automatikus feladat; erre csak a felhasználó kifejezett döntése után kerüljön sor. A helyi, hivatalos követelmények az [input/kovetelmenyek/](../input/kovetelmenyek/) fájlokban elsőbbséget élveznek.

## Tartalmi hiányosságok kezelése (nem csak lektoráláskor)

A Tex fordítást én magam végzem el, neked nem kell.
Bármely feladat (nem csak explicit "szöveg lektorálása" kérés) közben, ha a dolgozat szövegében tartalmi hiányosságot
vagy befejezetlenséget észlelsz — pl. egy fogalom/jelölés bevezetésre kerül, de a hozzá logikailag tartozó definíció,
összefüggés vagy lépés kimarad (mint a $z \to p$ kapcsolat hiánya volt), vagy egy `\section`/`\subsection` bevezető
szövege nem fedi le az alatta lévő teljes tartalmat —, a következőképp dönts:
- ha a hiányosság egyértelműen, találgatás nélkül javítható a meglévő kontextusból (fájlok, korábbi szövegrészek,
  hivatkozott cikkek), javítsd ki közvetlenül,
- ha a javításhoz döntés vagy találgatás szükséges (pl. melyik irányba bővítsd, milyen hivatkozással), ne írj bele
  találgatást, hanem kérdezz vissza vagy sorold fel a hiányosságot a chat válaszban.

## Szöveg lektorálása (kulcsszó)

Ha a felhasználó "szöveg lektorálása" kulcsszóval kér lektorálást (adott fejezetre/szakaszra vagy az egész dolgozatra), az alábbiakat végezd el:
- **Tekints a szövegre úgy, mintha a dolgozatot átnéző bíráló lennél**
- **Értelmezően olvasás**: nézd át úgy, hogy a fogalmak bevezetése, felépítése megfelelő sorrendet követ-e. Ha ilyen hibát látsz annak a feloldására írj nekem chat válaszban összefoglalva.
- **Alapvető átnézés**: helyesírás, elgépelés, nyelvtan, mondatszerkezet, magyar szórend javítása.
- **Formulák TeX-esítése**: sima szövegbeWn szereplő matematikai kifejezéseket (pl. "epsilon", "x_adv", "argmax") alakítsd megfelelő LaTeX formulává (`$...$` inline vagy `\[...\]` display), a fájlban már meglévő jelölésekkel konzisztens módon.
- **Jelölések egységesítése**: ellenőrizd, hogy ugyanazt a fogalmat mindenhol ugyanaz a szimbólum/elnevezés jelöli-e (pl. $y_{\mathrm{target}}$, "non-targeted" vs. "untargeted" — lásd a fentebbi terminológiai döntéseket), és javítsd az eltéréseket.
- **Állítások pontosítása**: ha egy kijelentés pontatlan, túlzó vagy alá nem támasztott, pontosítsd — lehetőség szerint a hivatkozott cikkek (`input/cikkek/`) tartalma alapján.
- **Hivatkozási hiányosságok jelzése**: ha egy állítás hivatkozást igényelne, de nincs, vagy egy `\cite` rossz/hiányzó, ne írj bele találgatást a szövegbe — ehelyett a chat válaszban, szövegesen sorold fel ezeket a helyeket a felhasználónak.
- **Egyéb javítások**: bármi más, ami átnézés közben egyértelműen hibának vagy jobbítandó pontnak tűnik (pl. felesleges ismétlés, következetlen igeidő/szóhasználat), az is javítható.
- **Inline utasítások**: a szövegben elhelyezett `%(utasítás)` formájú megjegyzések explicit szerkesztési utasítások — hajtsd végre a bennük foglaltakat, majd töröld magát a `%(...)` jelölést a fájlból.
- Lektoráláskor ne változtass a fejezetstruktúrán vagy a tartalmi mondanivalón, csak a szöveg minőségén, pontosságán és formázásán.
