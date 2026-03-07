@echo off
REM Создание виртуального окружения и установка зависимостей
cd /d "%~dp0"

if exist venv (
    echo Папка venv уже есть. Удаляю...
    rmdir /s /q venv
)

echo Создаю виртуальное окружение venv...
python -m venv venv
if errorlevel 1 (
    echo Ошибка: убедись, что Python установлен и в PATH.
    pause
    exit /b 1
)

echo Активирую venv и ставлю зависимости...
call venv\Scripts\activate.bat
pip install --upgrade pip
pip install -r requirements.txt

echo.
echo Готово. Чтобы пользоваться этим окружением:
echo   1. В терминале: venv\Scripts\activate
echo   2. В VS Code/Cursor: выбери интерпретатор Python из папки venv (Ctrl+Shift+P - "Python: Select Interpreter")
echo   3. Для Jupyter в ноутбуке выбери ядро с путём ...\investing_hw1_mts\venv\Scripts\python.exe
echo.
pause
