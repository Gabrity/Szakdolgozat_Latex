# Adversarial Image Generator (Desktop GUI)

Asztali alkalmazás adversarial példák generálására és vizsgálatára előre tanított
ImageNet modelleken (ResNet50, MobileNetV2). FGSM és PGD támadásokat,
valamint egyszerű input-sanitization védelmet támogat.

## Telepítés

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

GPU (CUDA 12.1, RTX 2060) esetén:

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## Indítás

```powershell
python main.py
```

## Használat

1. Tölts be egy képet a *Load Image* gombbal.
2. Válaszd ki a modellt (ResNet50 / MobileNetV2) és a támadási algoritmust (FGSM / PGD).
3. Állítsd be az **epsilon** és (PGD esetén) **iterations** csúszkákat.
4. Opcionálisan válassz **target** osztályt a targeted támadáshoz.
5. Kattints **Attack** gombra. A három képmező megjeleníti az eredeti képet,
   a felnagyított zajt és az adversarial képet.
6. A **Defend** gombbal a támadott képet Gaussian Blur / JPEG kompresszióval szűröd
   és újra osztályozod.
7. A **Download original** és **Download adversarial** gombbal elmentheted az eredeti
   (224×224-re vágott) és a támadott képet. A veszteségmentes PNG formátum ajánlott,
   mert a JPEG-mentés megváltoztatná a perturbációt.

## Tesztelés

A tesztek futtatásához a virtuális környezetben a `pytest` és a `pytest-mock`
csomag is szükséges. A parancsokat a `code` könyvtárban kell kiadni, mert
itt található a tesztek importálásához szükséges `pytest.ini`.
Ha a repó gyökérkönyvtárában vagy, előbb válts könyvtárat.

```powershell
Set-Location .\code
.\.venv\Scripts\Activate.ps1
python -m pytest -q
python -m pytest -m gui -q
```

A GUI-tesztek valódi CustomTkinter-widgeteket hoznak létre, ezért grafikus
környezetet igényelnek. Linuxon kijelző nélküli futtatáshoz például Xvfb
szükséges. A tesztablak rejtett, a modell egy kis tesztmodell, a támadási
háttérszál működését pedig ezek a tesztek nem ellenőrzik.

## Mappastruktúra

```
.
├── main.py
├── requirements.txt
├── models/                # Letöltött modell-súlyok cache-elve
└── src/
    ├── model_manager.py
    ├── attack_engine.py
    ├── defense_module.py
    ├── imagenet_classes.py
    ├── utils.py
    └── gui/
        └── app_interface.py
```
