# Скрипт для активации виртуального окружения в DRL

# Переходим в директорию DRL
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptPath

Write-Host "Активация виртуального окружения..." -ForegroundColor Cyan

# Проверяем существование venv
$venvPath = Join-Path $scriptPath "venv\Scripts\Activate.ps1"
if (Test-Path $venvPath) {
    # Устанавливаем политику выполнения для текущего процесса (если нужно)
    $currentPolicy = Get-ExecutionPolicy -Scope Process
    if ($currentPolicy -eq "Restricted") {
        Write-Host "Временно изменяю политику выполнения..." -ForegroundColor Yellow
        Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process -Force
    }
    
    # Активируем окружение
    & $venvPath
    
    # Проверяем активацию
    if ($env:VIRTUAL_ENV) {
        Write-Host "`n[OK] Виртуальное окружение успешно активировано!" -ForegroundColor Green
        Write-Host "   Путь: $env:VIRTUAL_ENV" -ForegroundColor Gray
        Write-Host "`nТеперь вы можете использовать Python из этого окружения." -ForegroundColor Cyan
    } else {
        Write-Host "`n[X] Не удалось активировать окружение. Попробуйте:" -ForegroundColor Red
        Write-Host "   .\venv\Scripts\activate.bat" -ForegroundColor Yellow
    }
} else {
    Write-Host "[X] Виртуальное окружение не найдено по пути: $venvPath" -ForegroundColor Red
    Write-Host "Убедитесь, что виртуальное окружение создано." -ForegroundColor Yellow
}

