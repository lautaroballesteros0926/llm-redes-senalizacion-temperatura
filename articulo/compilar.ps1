# Compila main.tex en Windows con MiKTeX (pdflatex -> bibtex -> pdflatex x2).
# Uso (PowerShell, desde esta carpeta):   .\compilar.ps1
$ErrorActionPreference = "Continue"
$bin = Get-ChildItem "$env:LOCALAPPDATA\Programs\MiKTeX", "C:\Program Files\MiKTeX" -Recurse -Filter pdflatex.exe -ErrorAction SilentlyContinue |
       Select-Object -First 1 -ExpandProperty DirectoryName
if (-not $bin) { Write-Host "No se encontró MiKTeX. Instálalo con:  winget install MiKTeX.MiKTeX"; exit 1 }
$env:PATH = "$bin;$env:PATH"
Set-Location $PSScriptRoot
pdflatex -interaction=nonstopmode main.tex | Out-Null
bibtex main | Out-Null
pdflatex -interaction=nonstopmode main.tex | Out-Null
pdflatex -interaction=nonstopmode main.tex | Out-Null
if (Test-Path main.pdf) { Write-Host "Listo: main.pdf generado." } else { Write-Host "Falló la compilación; revisa main.log." }
