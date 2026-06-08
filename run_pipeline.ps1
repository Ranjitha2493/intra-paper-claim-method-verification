$ErrorActionPreference = "Stop"

function Move-AllProcessed {
    param (
        [string]$From,
        [string]$To
    )

    if (!(Test-Path $To)) {
        New-Item -ItemType Directory -Path $To | Out-Null
    }

    if (!(Test-Path $From)) {
        Write-Host "Processed folder not found: $From"
        return
    }

    $folders = Get-ChildItem -Path $From -Directory

    foreach ($folder in $folders) {
        $destination = Join-Path $To $folder.Name

        if (Test-Path $destination) {
            $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
            $destination = Join-Path $To "$($folder.Name)_$timestamp"
        }

        Move-Item -Path $folder.FullName -Destination $destination
        Write-Host "Moved $($folder.Name) -> $To"
    }
}

Write-Host "Starting full pipeline..."

# Stage 01
Set-Location ".\01_extract_intro_methods"
python .\01_extract_intro_methods.py
Set-Location ".."
Move-AllProcessed ".\01_extract_intro_methods\01_processed" ".\02_extract_novelty_claims\02_input"

# Stage 02
Set-Location ".\02_extract_novelty_claims"
python .\02_extract_novelty_claims.py
Set-Location ".."
Move-AllProcessed ".\02_extract_novelty_claims\02_processed" ".\03_extract_claim_method_evidence\03_input"

# Stage 03
Set-Location ".\03_extract_claim_method_evidence"
python .\03_extract_claim_method_evidence.py
Set-Location ".."
Move-AllProcessed ".\03_extract_claim_method_evidence\03_processed" ".\04_claim_verification\04_input"

# Stage 04
Set-Location ".\04_claim_verification"
python .\04_claim_verification.py
Set-Location ".."
Move-AllProcessed ".\04_claim_verification\04_processed" ".\05_extract_human_review_categories\05_input"

# Stage 05
Set-Location ".\05_extract_human_review_categories"
python .\05_extract_human_review_categories.py
Set-Location ".."
Move-AllProcessed ".\05_extract_human_review_categories\05_processed" ".\06_generate_review_summaries\06_input"

# Stage 06
Set-Location ".\06_generate_review_summaries"
python .\06_generate_review_summaries.py
Set-Location ".."

Write-Host "Pipeline completed."