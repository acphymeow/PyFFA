# Add PyFFA's source and reusable example entry points to this PowerShell session.
$PyFFARoot = $PSScriptRoot
$PyFFASrc = Join-Path $PyFFARoot "src"
$PyFFAExamples = Join-Path $PyFFARoot "examples"
$PyFFAPaths = @($PyFFASrc, $PyFFAExamples)
$PathSeparator = [IO.Path]::PathSeparator

$ExistingEntries = @()
if ($env:PYTHONPATH) {
    $ExistingEntries = @(
        $env:PYTHONPATH -split [regex]::Escape($PathSeparator) |
            Where-Object { $_ -and ($_ -notin $PyFFAPaths) }
    )
}

$env:PYTHONPATH = ($PyFFAPaths + $ExistingEntries) -join $PathSeparator
Write-Host "PyFFA paths enabled for this PowerShell session:"
Write-Host "  $PyFFASrc"
Write-Host "  $PyFFAExamples"
Write-Host "Run examples from the project root so ./Bmaps, ./Emaps, and output paths resolve correctly."
