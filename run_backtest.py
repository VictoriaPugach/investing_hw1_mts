"""
Скрипт запуска: загрузка данных и все основные бэктесты.
Запуск: из корня проекта выполни: python run_backtest.py
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
    print("\n3. Сравнение лучших сетапов по таймфреймам...")
    run(f'"{sys.executable}" backtest_best.py')
    print("\n4. Метод 2 — визуально-графический анализ...")
    run(f'"{sys.executable}" backtest_visual.py')
    print("\n5. Сборка итогового отчёта...")
    run(f'"{sys.executable}" build_final_report.py')
    print("\nГотово.")
    print("Основные результаты в data/:")
    print("  - method1_isolated_results.csv, method1_improved_results.csv")
    print("  - best_setups_comparison.csv, FINAL_REPORT.md, FINAL_RECOMMENDATION.md")
    print("  - method2_visual_results.csv, method2_visual_plot.png")


if __name__ == "__main__":
    main()
