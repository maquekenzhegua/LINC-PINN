$ErrorActionPreference = 'Stop'
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$src = Join-Path $here 'fig1_architecture.pptx'
$dst = Join-Path $here '..\fig1_architecture.pdf'
$app = New-Object -ComObject PowerPoint.Application
$pres = $app.Presentations.Open($src, $true, $false, $false)
$pres.SaveAs($dst, 32)
$pres.Close()
$app.Quit()
Write-Output "exported $dst"
