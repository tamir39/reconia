$ErrorActionPreference = "Stop"

$projectDir = $PSScriptRoot

$appPort = if ($env:APP_PORT) { $env:APP_PORT } else { "8010" }
$postgresPort = if ($env:POSTGRES_PORT) { $env:POSTGRES_PORT } else { "5437" }
$composeProject = if ($env:COMPOSE_PROJECT) { $env:COMPOSE_PROJECT } else { "reconia" }
$seedDemo = if ($env:SEED_DEMO) { $env:SEED_DEMO } else { "1" }

function Test-CommandExists($command) {
  return $null -ne (Get-Command $command -ErrorAction SilentlyContinue)
}

function Get-LanIPv4Address {
  $addresses = Get-NetIPConfiguration -ErrorAction SilentlyContinue |
    Where-Object { $_.IPv4Address -and $_.NetAdapter.Status -eq "Up" } |
    Sort-Object { $_.NetAdapter.InterfaceMetric } |
    ForEach-Object { $_.IPv4Address.IPAddress } |
    Where-Object { $_ -notlike "127.*" -and $_ -notlike "169.254.*" }

  if ($addresses -and $addresses.Count -gt 0) {
    return $addresses[0]
  }

  return $null
}

function Assert-DockerReady {
  if (-not (Test-CommandExists "docker")) {
    Write-Host "Docker CLI is not installed or not available in PATH."
    exit 1
  }

  & docker info *> $null
  if ($LASTEXITCODE -ne 0) {
    Write-Host "Docker Desktop is not running. Start Docker Desktop, wait until it is ready, then run this again."
    exit 1
  }
}

function Assert-PortAvailable($port, $label) {
  # Ports are shared with other projects on this machine. Report the conflict
  # instead of terminating a process that may belong to another project.
  $owners = & netstat -ano -p tcp | Select-String "LISTENING" | ForEach-Object {
    $parts = ($_.ToString().Trim() -split '\s+') | Where-Object { $_ }
    if ($parts.Length -ge 5 -and $parts[1] -match "[:.]$port$") {
      [int]$parts[-1]
    }
  } | Sort-Object -Unique

  if (-not $owners) {
    return
  }

  foreach ($processId in $owners) {
    $owner = Get-Process -Id $processId -ErrorAction SilentlyContinue
    $name = if ($owner) { $owner.ProcessName } else { "unknown" }

    # A container published by this compose project is expected on a re-run.
    if ($name -eq "com.docker.backend" -or $name -eq "docker") {
      continue
    }

    Write-Host ""
    Write-Host "Port $port ($label) is already in use by process $processId ($name)."
    Write-Host "Another project is probably using it. Change $label in run.bat and try again."
    exit 1
  }
}

function Ensure-EnvFile {
  $envPath = Join-Path $projectDir ".env"
  if (-not (Test-Path $envPath)) {
    Copy-Item (Join-Path $projectDir ".env.example") $envPath
    Write-Host "Created .env from .env.example."
  }
}

function Wait-ForHealth($url, $timeoutSeconds) {
  $deadline = (Get-Date).AddSeconds($timeoutSeconds)
  while ((Get-Date) -lt $deadline) {
    try {
      $response = Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec 3
      if ($response.StatusCode -eq 200) {
        return $true
      }
    } catch {
      Start-Sleep -Seconds 1
    }
  }
  return $false
}

Assert-DockerReady
Assert-PortAvailable $appPort "APP_PORT"
Assert-PortAvailable $postgresPort "POSTGRES_PORT"
Ensure-EnvFile

[Environment]::SetEnvironmentVariable("APP_PORT", $appPort, "Process")
[Environment]::SetEnvironmentVariable("POSTGRES_PORT", $postgresPort, "Process")

Write-Host "Building and starting Reconia services..."
& docker compose -p $composeProject up -d --build
if ($LASTEXITCODE -ne 0) {
  Write-Host "Failed to start Reconia services."
  exit 1
}

Write-Host "Waiting for the application to become available..."
if (-not (Wait-ForHealth "http://localhost:$appPort/health" 90)) {
  Write-Host ""
  Write-Host "The application did not become available within 90 seconds."
  Write-Host "Inspect the logs with: docker compose -p $composeProject logs app"
  exit 1
}

if ($seedDemo -eq "1") {
  Write-Host "Loading demonstration data..."
  & docker compose -p $composeProject exec -T app python -m app.seed
  if ($LASTEXITCODE -ne 0) {
    Write-Host "Demonstration data could not be loaded. The application is still running."
  }
}

Write-Host ""
Write-Host "Reconia is running."
Write-Host "  Application:  http://localhost:$appPort"
Write-Host "  Documents:    http://localhost:$appPort/documents"
Write-Host "  Health check: http://localhost:$appPort/health/db"
Write-Host "  PostgreSQL:   localhost:$postgresPort"

$lanIp = Get-LanIPv4Address
if ($lanIp) {
  Write-Host "  On the LAN:   http://$($lanIp):$appPort"
}

Write-Host ""
Write-Host "Follow the logs with: docker compose -p $composeProject logs -f app"
Write-Host "Stop the services with: docker compose -p $composeProject down"
Write-Host "Stop and remove the data volume with: docker compose -p $composeProject down -v"
Write-Host ""
