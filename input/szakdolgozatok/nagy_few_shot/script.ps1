$mappa = $PSScriptRoot
$kimenet = Join-Path $PSScriptRoot "osszefuzott.txt"
$tessdataDir = "$env:USERPROFILE\tessdata"

Add-Type -AssemblyName System.Drawing

# Előfeldolgozás: szürkeárnyalat + kontrasztnövelés + felskálázás, hogy a Tesseract jobban felismerje a szöveget
function Optimize-Image {
    param([string]$InputPath, [string]$OutputPath, [double]$Scale = 2.0, [double]$Contrast = 1.3)

    $src = [System.Drawing.Image]::FromFile($InputPath)
    $newWidth = [int]($src.Width * $Scale)
    $newHeight = [int]($src.Height * $Scale)
    $bitmap = New-Object System.Drawing.Bitmap($newWidth, $newHeight)
    $bitmap.SetResolution($src.HorizontalResolution, $src.VerticalResolution)

    # Kontraszt + szürkeárnyalat mátrix (grayscale = R=G=B átlaga, majd kontraszt a 0.5 középpont körül)
    $c = [float]$Contrast
    $t = [float]((1.0 - $c) / 2.0)
    # Az értékeket előre ki kell számolni külön változóba, mert a [float[]]@(kifejezés*$c, ...)
    # forma miatt a PowerShell parser object[]-ként kezeli az elemeket és a szorzás elhasal
    $r1 = 0.3086 * $c
    $r2 = 0.6094 * $c
    $r3 = 0.0820 * $c
    # ColorMatrix konstruktor float[][] típust vár, PowerShell alap tömbje object[] lenne
    $matrixElements = [float[][]]@(
        [float[]]@($r1, $r1, $r1, 0, 0),
        [float[]]@($r2, $r2, $r2, 0, 0),
        [float[]]@($r3, $r3, $r3, 0, 0),
        [float[]]@(0, 0, 0, 1, 0),
        [float[]]@($t, $t, $t, 0, 1)
    )
    $colorMatrix = New-Object System.Drawing.Imaging.ColorMatrix(,$matrixElements)
    $attributes = New-Object System.Drawing.Imaging.ImageAttributes
    $attributes.SetColorMatrix($colorMatrix)

    $graphics = [System.Drawing.Graphics]::FromImage($bitmap)
    $graphics.InterpolationMode = [System.Drawing.Drawing2D.InterpolationMode]::HighQualityBicubic
    $graphics.DrawImage($src, (New-Object System.Drawing.Rectangle(0, 0, $newWidth, $newHeight)), 0, 0, $src.Width, $src.Height, [System.Drawing.GraphicsUnit]::Pixel, $attributes)

    $graphics.Dispose()
    $src.Dispose()
    $bitmap.Save($OutputPath, [System.Drawing.Imaging.ImageFormat]::Png)
    $bitmap.Dispose()
}

Remove-Item $kimenet -ErrorAction SilentlyContinue

Get-ChildItem "$mappa\*.jpg" | Sort-Object Name | ForEach-Object {
    $tempName = [System.IO.Path]::GetTempFileName() -replace '\.tmp$',''
    $optimizedImage = "$tempName-opt.png"
    Optimize-Image -InputPath $_.FullName -OutputPath $optimizedImage
    & "C:\Program Files\Tesseract-OCR\tesseract.exe" $optimizedImage $tempName -l hun --tessdata-dir $tessdataDir --psm 6 --oem 1 2>$null
    "----- $($_.Name) -----" | Add-Content $kimenet
    Get-Content "$tempName.txt" | Add-Content $kimenet
    Add-Content $kimenet ""
    Remove-Item "$tempName.txt" -ErrorAction SilentlyContinue
    Remove-Item $optimizedImage -ErrorAction SilentlyContinue
}

Write-Output "Kész: $kimenet"