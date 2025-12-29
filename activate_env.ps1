# p2d venv 环境自动激活脚本
# 如果环境不存在则创建，然后自动激活

# 使用英文避免编码问题
Write-Host "=== p2d Virtual Environment Setup ===" -ForegroundColor Cyan

# 1. Check and create virtual environment
if (-not (Test-Path "venv")) {
    Write-Host "Creating virtual environment..." -ForegroundColor Yellow
    python -m venv venv
    
    if (-not (Test-Path "venv")) {
        Write-Host "Error: Failed to create virtual environment" -ForegroundColor Red
        exit 1
    }
    
    # Activate environment
    Write-Host "Activating environment..." -ForegroundColor Yellow
    & .\venv\Scripts\Activate.ps1
    
    # Check if activation successful
    if ($env:VIRTUAL_ENV -notlike "*venv*") {
        Write-Host "Warning: Environment activation may have failed" -ForegroundColor Red
        Write-Host "Please run: .\venv\Scripts\Activate.ps1" -ForegroundColor White
    }
    
    # Install dependencies
    Write-Host "Installing dependencies..." -ForegroundColor Yellow
    
    # Check if requirements.txt exists
    if (Test-Path "requirements.txt") {
        pip install uv
        uv pip install -r requirements.txt
        Write-Host "Dependencies installed" -ForegroundColor Green
    } else {
        Write-Host "Warning: requirements.txt not found" -ForegroundColor Red
        Write-Host "Please create requirements.txt or install dependencies manually" -ForegroundColor White
    }
} else {
    # Activate existing environment
    Write-Host "Activating existing virtual environment..." -ForegroundColor Green
    & .\venv\Scripts\Activate.ps1
    
    if ($env:VIRTUAL_ENV -notlike "*venv*") {
        Write-Host "Warning: Environment activation failed" -ForegroundColor Red
        Write-Host "Please run manually: .\venv\Scripts\Activate.ps1" -ForegroundColor White
    }
}

# 2. Verify environment
Write-Host "`nVerifying environment..." -ForegroundColor Cyan

try {
    $pythonPath = (Get-Command python).Source
    Write-Host "Python path: $pythonPath" -ForegroundColor White
    
    # Check if in virtual environment
    if ($pythonPath -like "*venv*") {
        Write-Host "Virtual environment activated successfully" -ForegroundColor Green
    } else {
        Write-Host "Python is not in virtual environment" -ForegroundColor Yellow
    }
} catch {
    Write-Host "Error: Cannot find Python command" -ForegroundColor Red
}

Write-Host "`nTo exit environment, run: deactivate" -ForegroundColor Yellow
Write-Host "To reactivate, run: .\activate_env.ps1" -ForegroundColor Yellow