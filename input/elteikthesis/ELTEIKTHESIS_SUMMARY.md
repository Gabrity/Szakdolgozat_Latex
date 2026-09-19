# ELTE IK LaTeX sablon összefoglaló

A teljes, nyilvános ELTE Informatikai Kari LaTeX-sablon helyi másolata itt található:

- `input/elteikthesis/`

Forrás: `https://github.com/mcserep/elteikthesis`

A helyi másolat a 2.4-es sablont tartalmazza, amelynek verziója az `elteikthesis.cls` szerint `2024/04/26`.

## Fontos fájlok

- `input/elteikthesis/README_hu.md` — magyar használati és fordítási útmutató.
- `input/elteikthesis/elteikthesis_hu.tex` — teljes magyar mintadokumentum.
- `input/elteikthesis/elteikthesis.cls` — a dokumentumosztály tényleges formai beállításai.
- `input/elteikthesis/elteikthesis.bib` — bibliográfiai minta.
- `input/elteikthesis/samples_hu/` — magyar mintafejezetek.
- `input/elteikthesis/images/` — a mintadokumentum képei és logói.
- `input/elteikthesis/elteikthesis_minted.tex` — `minted` alapú forráskód-formázási példa.

## A sablon fontos beállításai

- `report` dokumentumosztály, A4-es papír és 12 pontos betűméret.
- Alapértelmezés szerint egyoldalas nyomtatás; a `twoside` opció kétoldalas tördelést kapcsol be.
- A sablon belső margója 35 mm, a külső, felső és alsó margó 25 mm.
- A Wordben megszokott 1,5-es sortávolságot `\setstretch{1.427465}` közelíti.
- A főfejezetek `\chapter` paranccsal készülnek, és a sablon új oldalra tördel.
- A címlapot metaadatokból állítja elő: cím, év, szerző, szak, témavezető, egyetem, kar, tanszék, város és logó.
- A sablon magyar és angol nyelvet is támogat a `\documentlang` paranccsal.
- A bibliográfiát `biblatex` és `biber` kezeli, numerikus stílussal.

## Mintaszerkezet

A magyar mintafőfájl szerkezete:

1. címlap;
2. témabejelentő;
3. tartalomjegyzék;
4. bevezetés;
5. felhasználói dokumentáció;
6. fejlesztői dokumentáció;
7. összegzés;
8. opcionális köszönetnyilvánítás és mellékletek;
9. irodalomjegyzék;
10. opcionális ábra-, táblázat-, algoritmus-, forráskód- és jelölésjegyzék.

Az ábra-, táblázat-, algoritmus- és kódjegyzék a sablon útmutatója szerint körülbelül 3--5 vagy több releváns elem esetén indokolt.

## Fordítás

A teljes sablon hivatalos fordítási menete:

```text
pdflatex elteikthesis_hu.tex
biber elteikthesis_hu
makeindex -s nomencl.ist -t elteikthesis_hu.nlg -o elteikthesis_hu.nls elteikthesis_hu.nlo
pdflatex elteikthesis_hu.tex
```

A `makeindex` csak akkor szükséges, ha jelölésjegyzék készül. Tartalom- vagy hivatkozásváltozás után további LaTeX-fordítás szükséges a segédfájlok frissítéséhez.

## Hasznos csomagok

A sablon többek között támogatja az `amsmath`, `amsthm`, `amsfonts`, `longtable`, `adjustbox`, `subcaption`, `rotating`, `algorithm`, `algpseudocode`, `listingsutf8`, `minted`, `nomencl` és `todonotes` csomagokat.

## Kapcsolat a jelenlegi dolgozattal

A jelenlegi dolgozat továbbra is `article` alapú, és a forrása a `dolgozat/Gabrity_Gabor_D9V09Z_Szakdolgozat.tex`.

Ez a helyi sablonmásolat referenciaanyag. A teljes `report`/`chapter` alapú átállást nem szabad automatikusan végrehajtani, mert az a címsorszámozás, a címlap, a margók, a bibliográfia és a dokumentum szerkezetének átalakításával járna. A jelenlegi dolgozatban már alkalmazott, kompatibilis megoldásokat kell megőrizni, például a `\setstretch{1.427465}` sortávolságot, a `hyperref` használatát és az irodalomjegyzék tartalomjegyzékbe vételét.

A helyi hivatalos követelmények elsőbbséget élveznek a sablon beállításaival szemben. Különösen a 35 mm-es belső margó eltérését kell figyelembe venni, mert a projekt helyi formai követelményei minden oldalon 25 mm-es margót írnak elő.
