# Gera arquivos de referência com um codificador de terceiros (GDI+ do Windows) e a decodificação do próprio GDI+.
# É a validação cruzada dos decodificadores do motor com um codificador que não é o nosso (docs §1.23).
# Saída: gdiplus_*.jpg/.png, gdiplus_*.rgba e manifest_gdiplus.txt (mesmo formato de manifest.txt).
# Uso (Windows PowerShell 5.1): powershell -ExecutionPolicy Bypass -File make_gdiplus_fixtures.ps1
Add-Type -AssemblyName System.Drawing
$out = Split-Path -Parent $MyInvocation.MyCommand.Path

function New-Scene([int]$w, [int]$h, [bool]$alpha) {
    $fmt = if ($alpha) { [System.Drawing.Imaging.PixelFormat]::Format32bppArgb } else { [System.Drawing.Imaging.PixelFormat]::Format24bppRgb }
    $bmp = New-Object System.Drawing.Bitmap($w, $h, $fmt)
    $bmp.SetResolution(150, 150)
    $g = [System.Drawing.Graphics]::FromImage($bmp)
    $g.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias
    if ($alpha) { $g.Clear([System.Drawing.Color]::FromArgb(0, 0, 0, 0)) }
    $rect = New-Object System.Drawing.Rectangle(0, 0, $w, $h)
    $grad = New-Object System.Drawing.Drawing2D.LinearGradientBrush($rect, [System.Drawing.Color]::FromArgb(($(if ($alpha) { 160 } else { 255 })), 30, 90, 200),
                                                                      [System.Drawing.Color]::FromArgb(255, 250, 220, 40), 30.0)
    $g.FillRectangle($grad, 0, 0, $w, [int]($h * 0.6))
    $g.FillEllipse((New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(255, 220, 30, 40))), [int]($w * 0.45), [int]($h * 0.3), [int]($w * 0.4), [int]($h * 0.55))
    $pen = New-Object System.Drawing.Pen([System.Drawing.Color]::FromArgb(255, 20, 20, 20), 1.5)
    $g.DrawLine($pen, 2, $h - 3, $w - 3, 4)
    $g.Dispose()
    return $bmp
}

function Save-Rgba([string]$file, [string]$rgbaFile) {
    $img = New-Object System.Drawing.Bitmap($file)
    $w = $img.Width; $h = $img.Height
    $data = $img.LockBits((New-Object System.Drawing.Rectangle(0, 0, $w, $h)), [System.Drawing.Imaging.ImageLockMode]::ReadOnly, [System.Drawing.Imaging.PixelFormat]::Format32bppArgb)
    $bytes = New-Object byte[] ($data.Stride * $h)
    [System.Runtime.InteropServices.Marshal]::Copy($data.Scan0, $bytes, 0, $bytes.Length)
    $img.UnlockBits($data); $img.Dispose()
    $rgba = New-Object byte[] ($w * $h * 4)
    for ($y = 0; $y -lt $h; $y++) {
        for ($x = 0; $x -lt $w; $x++) {
            $s = $y * $data.Stride + 4 * $x; $d = 4 * ($y * $w + $x)
            $rgba[$d] = $bytes[$s + 2]; $rgba[$d + 1] = $bytes[$s + 1]; $rgba[$d + 2] = $bytes[$s]; $rgba[$d + 3] = $bytes[$s + 3]
        }
    }
    [System.IO.File]::WriteAllBytes($rgbaFile, $rgba)
}

$codecs = [System.Drawing.Imaging.ImageCodecInfo]::GetImageEncoders()
$jpegCodec = $codecs | Where-Object { $_.MimeType -eq 'image/jpeg' }
$params = New-Object System.Drawing.Imaging.EncoderParameters(1)
$params.Param[0] = New-Object System.Drawing.Imaging.EncoderParameter([System.Drawing.Imaging.Encoder]::Quality, [long]90)

$lines = @()
$photo = New-Scene 61 47 $false
$photo.Save("$out\gdiplus_photo.jpg", $jpegCodec, $params)
$photo.Dispose()
Save-Rgba "$out\gdiplus_photo.jpg" "$out\gdiplus_photo.rgba"
# Tolerância: o GDI+ usa outra IDCT e outra ampliação do croma; a diferença medida é de no máximo 2 níveis.
$lines += "jpeg gdiplus_photo 61 47 150.000000 150.000000 0 3"

$logo = New-Scene 61 47 $true
$logo.Save("$out\gdiplus_logo.png", [System.Drawing.Imaging.ImageFormat]::Png)
$logo.Dispose()
Save-Rgba "$out\gdiplus_logo.png" "$out\gdiplus_logo.rgba"
$lines += "png gdiplus_logo 61 47 150.000000 150.000000 1 0"

[System.IO.File]::WriteAllText("$out\manifest_gdiplus.txt", ($lines -join "`n") + "`n")
"GDI+: $($lines.Count) casos"
