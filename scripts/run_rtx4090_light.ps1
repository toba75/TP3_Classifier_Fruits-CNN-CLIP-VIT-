$ErrorActionPreference = "Stop"

# Get the root directory (one level up from scripts/)
$RootDir = Resolve-Path (Join-Path $PSScriptRoot "..") | Select-Object -ExpandProperty Path

# Change to the project root
Push-Location $RootDir

try {
    Write-Host "Building Docker image (includes light script)..."
    docker build -t tp3-experiments:rtx4090 -f Dockerfile.rtx4090 .
    if ($LASTEXITCODE -ne 0) { throw "Docker build failed" }

    Write-Host "Running LIGHT experiments (Fast Mode)..."
    docker run --rm `
      --gpus all `
      --ipc=host `
      --ulimit memlock=-1 `
      --ulimit stack=67108864 `
      -v "${RootDir}:/workspace" `
      tp3-experiments:rtx4090 bash /usr/local/bin/run_experiments_light.sh

    if ($LASTEXITCODE -ne 0) { throw "Docker run failed" }
}
finally {
    Pop-Location
}
