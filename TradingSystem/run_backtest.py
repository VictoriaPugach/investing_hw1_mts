"""
Скрипт запуска: сначала загрузка данных, затем оба бэктеста.
Запуск: из папки TradingSystem выполни: python run_backtest.py
"""
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def run(cmd, cwd=None):
    cwd = cwd or PROJECT_ROOT
    print(f"\n>>> {cmd}")
    r = subprocess.run(cmd, shell=True, cwd=cwd)
    if r.returncode != 0:
        print(f"Ошибка: код выхода {r.returncode}")
        sys.exit(r.returncode)


def main():
    print("1. Загрузка данных...")
    run(f'"{sys.executable}" data/download_data.py')
    print("\n2. Метод 1 — изолированность входа/выхода...")
    run(f'"{sys.executable}" backtest_isolated.py')
    print("\n3. Метод 2 — визуально-графический анализ...")
    run(f'"{sys.executable}" backtest_visual.py')
    print("\nГотово. Результаты в data/: method1_isolated_results.csv, method2_visual_results.csv, method2_visual_plot.png")


if __name__ == "__main__":
    main()
