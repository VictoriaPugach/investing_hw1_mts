"""
Метрики эффективности МТС по материалам курса: P, PF, RF, MIDD, APF.
Кривая капитала и дополнительные графики для отчёта.
"""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def equity_curve(trades: pd.DataFrame, initial: float = 1.0) -> pd.Series:
    """Кривая капитала: начальный капитал + кумулятивная сумма PnL."""
    if trades is None or len(trades) == 0:
        return pd.Series(dtype=float)
    pnl = trades["pnl"] if hasattr(trades, "columns") and "pnl" in trades.columns else pd.Series(trades)
    return initial + pnl.cumsum()


def plot_equity_curve(trades: pd.DataFrame, title: str = "Кривая капитала (Eq)", save_path: Path = None, generated_label: str = None):
    """Строит и сохраняет график кривой капитала."""
    eq = equity_curve(trades)
    if eq.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(eq.values, color="steelblue", linewidth=1.5)
    ax.axhline(y=eq.iloc[0], color="gray", linestyle="--", alpha=0.7)
    ax.set_xlabel("Номер сделки")
    ax.set_ylabel("Капитал")
    ax.set_title(title)
    if generated_label:
        ax.text(0.99, 0.02, generated_label, transform=ax.transAxes, fontsize=8, ha="right", va="bottom", color="gray")
    ax.grid(True, alpha=0.3)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_equity_and_drawdown(trades: pd.DataFrame, title: str = "Капитал и просадка", save_path: Path = None):
    """Два графика: кривая капитала и просадка от пика (для отчёта)."""
    eq = equity_curve(trades)
    if eq.empty:
        return
    peak = eq.cummax()
    drawdown = peak - eq
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5), sharex=True, gridspec_kw={"height_ratios": [1.2, 0.8]})
    ax1.plot(eq.values, color="steelblue", linewidth=1.5, label="Капитал")
    ax1.axhline(y=eq.iloc[0], color="gray", linestyle="--", alpha=0.7)
    ax1.set_ylabel("Капитал")
    ax1.set_title(title)
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)
    ax2.fill_between(range(len(drawdown)), drawdown.values, 0, color="coral", alpha=0.6)
    ax2.set_xlabel("Номер сделки")
    ax2.set_ylabel("Просадка")
    ax2.set_title("Просадка от пика (MIDD)")
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_pnl_histogram(trades: pd.DataFrame, title: str = "Распределение PnL по сделкам", save_path: Path = None):
    """Гистограмма прибылей и убытков (визуализация для отчёта)."""
    if trades is None or len(trades) == 0 or "pnl" not in trades.columns:
        return
    pnl = trades["pnl"]
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["green" if x > 0 else "red" for x in pnl]
    ax.bar(range(len(pnl)), pnl, color=colors, alpha=0.7, edgecolor="gray")
    ax.axhline(y=0, color="black", linewidth=0.8)
    ax.set_xlabel("Номер сделки")
    ax.set_ylabel("PnL")
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis="y")
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_exit_reasons(trades: pd.DataFrame, title: str = "Выходы по причинам (TP/SL/BE/REV)", save_path: Path = None):
    """Столбчатая диаграмма: количество выходов по каждой причине (TP, SL, BE, REV, T40, END)."""
    if trades is None or len(trades) == 0 or "exit_reason" not in trades.columns:
        return
    reason = trades["exit_reason"].astype(str)
    counts = reason.value_counts()
    if counts.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    colors_map = {"TP": "green", "SL": "red", "BE": "gray", "REV": "orange", "EN": "navy"}
    colors = [colors_map.get(str(r)[:2], "steelblue") for r in counts.index]
    ax.bar(counts.index.astype(str), counts.values, color=colors, alpha=0.8)
    ax.set_xlabel("Причина выхода")
    ax.set_ylabel("Количество")
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis="y")
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def compute_metrics(trades: pd.DataFrame) -> dict:
    """trades — DataFrame с колонкой pnl или массив PnL."""
    if hasattr(trades, "columns") and "pnl" in trades.columns:
        pnls = trades["pnl"].dropna()
    else:
        pnls = pd.Series(trades).dropna()
    if len(pnls) == 0:
        return {"n_trades": 0, "P": 0.0, "gross_profit": 0.0, "gross_loss": 0.0, "PF": 0.0, "APF": 0.0,
                "MIDD": 0.0, "RF": 0.0, "pct_profitable": 0.0, "avg_win": 0.0, "avg_loss": 0.0,
                "max_win": 0.0, "max_loss": 0.0}
    gross_profit = pnls[pnls > 0].sum()
    gross_loss = abs(pnls[pnls < 0].sum())
    P = float(pnls.sum())
    n = len(pnls)
    pct_profitable = ((pnls > 0).sum() / n * 100) if n else 0
    wins, losses = pnls[pnls > 0], pnls[pnls < 0]
    avg_win = float(wins.mean()) if len(wins) else 0
    avg_loss = float(losses.mean()) if len(losses) else 0
    PF = (gross_profit / gross_loss) if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0)
    Pmax = float(pnls.max())
    APF = ((gross_profit - Pmax) / gross_loss) if gross_loss > 0 else 0
    eq = pnls.cumsum()
    MIDD = float((eq.cummax() - eq).max()) if len(eq) else 0
    RF = (P / MIDD) if MIDD > 0 else (float("inf") if P > 0 else 0)
    return {"n_trades": n, "P": P, "gross_profit": float(gross_profit), "gross_loss": float(gross_loss),
            "PF": PF, "APF": APF, "MIDD": MIDD, "RF": RF, "pct_profitable": pct_profitable,
            "avg_win": avg_win, "avg_loss": avg_loss, "max_win": float(pnls.max()), "max_loss": float(pnls.min())}


def print_metrics(metrics: dict):
    print("--- Метрики ТС ---")
    print(f"Количество сделок: {metrics['n_trades']}, P: {metrics['P']:.2f}, PF: {metrics['PF']:.2f}")
    print(f"MIDD: {metrics['MIDD']:.2f}, RF: {metrics['RF']:.2f}, доля прибыльных: {metrics['pct_profitable']:.1f}%")
