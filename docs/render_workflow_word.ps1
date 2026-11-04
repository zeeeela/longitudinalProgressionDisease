$ErrorActionPreference = 'Stop'
$docPath = Join-Path $PSScriptRoot 'Repository_workflow_summary.docx'
$qaDir = Join-Path $PSScriptRoot 'qa_workflow'
New-Item -ItemType Directory -Path $qaDir -Force | Out-Null
$pdfPath = Join-Path $qaDir 'Repository_workflow_summary.pdf'
$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0
try {
    $document = $word.Documents.Open($docPath, $false, $true)
    $document.Repaginate()
    Write-Output ('Pages: ' + $document.ComputeStatistics(2))
    $document.ExportAsFixedFormat($pdfPath, 17)
    $document.Close(0)
} finally {
    $word.Quit()
    [System.Runtime.InteropServices.Marshal]::ReleaseComObject($word) | Out-Null
}
Write-Output $pdfPath
