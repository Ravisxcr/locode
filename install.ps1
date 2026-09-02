# gocode installer for Windows
# Usage: irm https://raw.githubusercontent.com/AlleyBo55/gocode/main/install.ps1 | iex

$ErrorActionPreference = "Stop"

$repo = "AlleyBo55/gocode"
$binary = "gocode"

# Detect architecture
$arch = if ([Environment]::Is64BitOperatingSystem) {
    if ($env:PROCESSOR_ARCHITECTURE -eq "ARM64") { "arm64" } else { "amd64" }
} else {
    Write-Error "32-bit Windows is not supported"
    exit 1
}

# Get latest release tag
Write-Host "Fetching latest release..." -ForegroundColor Cyan
$release = Invoke-RestMethod -Uri "https://api.github.com/repos/$repo/releases/latest"
$version = $release.tag_name -replace '^v', ''

if (-not $version) {
    Write-Error "Could not determine latest version"
    exit 1
}

$filename = "${binary}_${version}_windows_${arch}.zip"
$url = "https://github.com/$repo/releases/download/v$version/$filename"

# Download
$tmpDir = Join-Path $env:TEMP "gocode-install"
New-Item -ItemType Directory -Force -Path $tmpDir | Out-Null
$zipPath = Join-Path $tmpDir $filename

Write-Host "Downloading gocode v$version for windows/$arch..." -ForegroundColor Cyan
Invoke-WebRequest -Uri $url -OutFile $zipPath

# Extract
Write-Host "Extracting..." -ForegroundColor Cyan
Expand-Archive -Path $zipPath -DestinationPath $tmpDir -Force

# Install to WindowsApps (already in PATH for most users)
$installDir = Join-Path $env:LOCALAPPDATA "Microsoft\WindowsApps"
$exePath = Join-Path $tmpDir "$binary.exe"

if (Test-Path $exePath) {
    Copy-Item $exePath (Join-Path $installDir "$binary.exe") -Force
} else {
    # GoReleaser might nest it in a folder
    $found = Get-ChildItem -Path $tmpDir -Recurse -Filter "$binary.exe" | Select-Object -First 1
    if ($found) {
        Copy-Item $found.FullName (Join-Path $installDir "$binary.exe") -Force
    } else {
        Write-Error "Could not find gocode.exe in the downloaded archive"
        exit 1
    }
}

# Cleanup
Remove-Item -Recurse -Force $tmpDir

Write-Host ""
Write-Host "gocode v$version installed to $installDir\$binary.exe" -ForegroundColor Green
Write-Host ""
Write-Host "Run 'gocode --help' to get started." -ForegroundColor Cyan
