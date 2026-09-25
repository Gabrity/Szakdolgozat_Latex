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
