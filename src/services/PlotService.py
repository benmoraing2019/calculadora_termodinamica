"""
PlotService.py
==============
Visualizaciones para resultados de separación flash multicomponente.
Orientado a ingeniería química de planta.

Uso:
    from services.PlotService import PlotService
    ps = PlotService(flash_model)
    ps.dashboard()          # Dashboard completo 2x3
    ps.phase_envelope()     # Solo envolvente de fases
    ps.composition_bar()    # Solo composiciones
    ps.kvalues()            # Solo factores K
    ps.flow_balance()       # Solo balance de flujos
    ps.export_csv("resultados.csv")
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import CoolProp.CoolProp as CP
from CoolProp.CoolProp import AbstractState
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# PALETA Y ESTILO
# ─────────────────────────────────────────────────────────────────────────────
STYLE = {
    "bg":        "#0F1923",
    "panel":     "#162030",
    "border":    "#1E3048",
    "text":      "#E8EEF4",
    "subtext":   "#7A9BB5",
    "accent":    "#00D4FF",
    "vapor":     "#00D4FF",
    "liquid":    "#FF6B35",
    "feed":      "#A8D8A8",
    "crit":      "#FFD700",
    "grid":      "#1A2E42",
    "good":      "#4ECDC4",
    "warn":      "#FFE66D",
    "danger":    "#FF6B6B",
}

COMP_COLORS = [
    "#00D4FF", "#FF6B35", "#A8D8A8", "#FFD700",
    "#C77DFF", "#FF9F1C", "#2EC4B6", "#E71D36",
    "#F72585", "#4CC9F0", "#7B2FBE",
]


def _apply_style(fig, axes_list):
    fig.patch.set_facecolor(STYLE["bg"])
    for ax in axes_list:
        ax.set_facecolor(STYLE["panel"])
        ax.tick_params(colors=STYLE["subtext"], labelsize=8)
        ax.xaxis.label.set_color(STYLE["text"])
        ax.yaxis.label.set_color(STYLE["text"])
        ax.title.set_color(STYLE["accent"])
        for spine in ax.spines.values():
            spine.set_edgecolor(STYLE["border"])
        ax.grid(True, color=STYLE["grid"], linewidth=0.6, linestyle="--", alpha=0.7)


# ─────────────────────────────────────────────────────────────────────────────
# CLASE PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────
class PlotService:
    """
    Recibe una instancia de FlashModel ya ejecutada (con state.update() llamado).

    Atributos esperados en FlashModel:
        - components  : list[str]   nombres CoolProp
        - composition : list[float] fracciones molares globales z
        - mixture_code: str         "Comp1&Comp2&..."
        - T           : float       temperatura K
        - P           : float       presión Pa
        - phase       : int         código de fase CoolProp
        - state       : AbstractState

    Opcionalmente en config_manager:
        - get_mixture_config() retorna lista con 'id_espanol'
        - get_simulation_config() retorna dict con feed.flow_kmol_h
    """

    def __init__(self, flash_model):
        self.fm = flash_model
        self._extract_results()

    # ─────────────────────────────────────────────────────────────────────────
    # EXTRACCIÓN DE RESULTADOS
    # ─────────────────────────────────────────────────────────────────────────
    def _extract_results(self):
        fm = self.fm
        self.n = len(fm.components)

        # Nombres en español si están disponibles
        try:
            mezcla = fm.mezcla_datos
            self.labels = [c.get("id_espanol", c["coolprop_name"]) for c in mezcla]
        except Exception:
            self.labels = fm.components

        self.z = np.array(fm.composition)
        self.is_twophase = (fm.phase == CP.iphase_twophase)

        if self.is_twophase:
            self.beta  = fm.state.Q()                         # fracción molar vapor
            self.x_liq = np.array(fm.state.mole_fractions_liquid())
            self.y_vap = np.array(fm.state.mole_fractions_vapor())
            self.K     = np.where(self.x_liq > 1e-15,
                                  self.y_vap / self.x_liq,
                                  np.nan)
        else:
            self.beta  = 1.0 if fm.phase == CP.iphase_gas else 0.0
            self.x_liq = self.z.copy()
            self.y_vap = self.z.copy()
            self.K     = np.ones(self.n)

        # Flujo de alimentación — usa get_flash_config() que retorna config["simulacion"]
        try:
            sim = fm.config_manager.get_flash_config()
            self.F_total = sim["feed"]["flow_kmol_h"]
        except Exception:
            self.F_total = 100.0   # default si no está configurado

        self.V_flow = self.beta * self.F_total
        self.L_flow = (1 - self.beta) * self.F_total
        self.F_comp = self.z * self.F_total
        self.V_comp = self.y_vap * self.V_flow
        self.L_comp = self.x_liq * self.L_flow

    # ─────────────────────────────────────────────────────────────────────────
    # DASHBOARD PRINCIPAL
    # ─────────────────────────────────────────────────────────────────────────
    def dashboard(self, save_path="flash_dashboard.png", show=True):
        """Dashboard completo 2x3 para ingeniería de planta."""
        fig = plt.figure(figsize=(18, 11))
        fig.patch.set_facecolor(STYLE["bg"])

        gs = gridspec.GridSpec(
            2, 3,
            figure=fig,
            hspace=0.45,
            wspace=0.35,
            left=0.06, right=0.97,
            top=0.88, bottom=0.07
        )

        ax1 = fig.add_subplot(gs[0, 0])   # Envolvente de fases
        ax2 = fig.add_subplot(gs[0, 1])   # Composiciones por fase
        ax3 = fig.add_subplot(gs[0, 2])   # Factores K
        ax4 = fig.add_subplot(gs[1, 0])   # Balance de flujos
        ax5 = fig.add_subplot(gs[1, 1])   # Distribución fracción molar
        ax6 = fig.add_subplot(gs[1, 2])   # Tabla resumen

        _apply_style(fig, [ax1, ax2, ax3, ax4, ax5, ax6])

        self._plot_phase_envelope(ax1)
        self._plot_composition_bar(ax2)
        self._plot_kvalues(ax3)
        self._plot_flow_balance(ax4)
        self._plot_phase_distribution(ax5)
        self._plot_summary_table(ax6)

        # Título principal
        fase_str = "BIFÁSICO (VLE)" if self.is_twophase else "UNA FASE"
        titulo = (
            f"REPORTE DE SEPARACIÓN FLASH  ·  "
            f"T = {self.fm.T:.1f} K  ·  "
            f"P = {self.fm.P/1e5:.2f} bar  ·  "
            f"{fase_str}  ·  "
            f"β = {self.beta:.4f}"
        )
        fig.suptitle(
            titulo,
            fontsize=12, fontweight="bold",
            color=STYLE["accent"],
            y=0.95
        )

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight",
                        facecolor=STYLE["bg"])
            print(f"  ✅ Dashboard guardado: {save_path}")
        if show:
            plt.show()
        return fig

    # ─────────────────────────────────────────────────────────────────────────
    # 1. ENVOLVENTE DE FASES
    # ─────────────────────────────────────────────────────────────────────────
    def phase_envelope(self, save_path=None, show=True):
        fig, ax = plt.subplots(figsize=(8, 6))
        _apply_style(fig, [ax])
        self._plot_phase_envelope(ax)
        fig.patch.set_facecolor(STYLE["bg"])
        ax.set_title("Envolvente de Fases", color=STYLE["accent"], fontsize=13)
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=STYLE["bg"])
        if show:
            plt.show()
        return fig

    def _plot_phase_envelope(self, ax):
        components = self.fm.components
        z          = self.fm.composition
        mezcla_str = self.fm.mixture_code

        # Estimar rango de T razonable
        T_min, T_max = 120, 400
        T_arr = np.linspace(T_min, T_max, 80)

        T_bub, P_bub = [], []
        T_dew, P_dew = [], []

        for T in T_arr:
            for Q, Tl, Pl in [(0.0, T_bub, P_bub), (1.0, T_dew, P_dew)]:
                try:
                    AS = AbstractState("HEOS", mezcla_str)
                    AS.set_mole_fractions(z)
                    AS.update(CP.QT_INPUTS, Q, T)
                    P = AS.p()
                    if 1e3 < P < 1.5e8:
                        Tl.append(T)
                        Pl.append(P / 1e6)
                except Exception:
                    pass

        if T_bub:
            ax.plot(T_bub, P_bub, color=STYLE["liquid"], lw=2,
                    label="Burbuja (Q=0)")
        if T_dew:
            ax.plot(T_dew, P_dew, color=STYLE["vapor"], lw=2,
                    label="Rocío (Q=1)")

        # Rellenar región bifásica
        if T_bub and T_dew:
            T_fill = T_bub + T_dew[::-1]
            P_fill = P_bub + P_dew[::-1]
            ax.fill(T_fill, P_fill, alpha=0.12, color=STYLE["accent"])

        # Punto de operación
        ax.scatter(
            [self.fm.T], [self.fm.P / 1e6],
            s=120, zorder=10,
            color=STYLE["crit"],
            edgecolors="white", linewidths=1.5,
            label=f"Operación ({self.fm.T:.0f} K, {self.fm.P/1e6:.2f} MPa)"
        )

        ax.set_xlabel("Temperatura (K)")
        ax.set_ylabel("Presión (MPa)")
        ax.set_title("Envolvente de Fases", color=STYLE["accent"])
        leg = ax.legend(fontsize=7.5, facecolor=STYLE["panel"],
                        labelcolor=STYLE["text"], edgecolor=STYLE["border"])

    # ─────────────────────────────────────────────────────────────────────────
    # 2. COMPOSICIONES POR FASE
    # ─────────────────────────────────────────────────────────────────────────
    def composition_bar(self, save_path=None, show=True):
        fig, ax = plt.subplots(figsize=(9, 5))
        _apply_style(fig, [ax])
        self._plot_composition_bar(ax)
        fig.patch.set_facecolor(STYLE["bg"])
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=STYLE["bg"])
        if show:
            plt.show()
        return fig

    def _plot_composition_bar(self, ax):
        x     = np.arange(self.n)
        width = 0.25

        bars_z = ax.bar(x - width, self.z,     width, label="Alimentación (z)",
                        color=STYLE["feed"],   alpha=0.85, edgecolor=STYLE["bg"])
        bars_l = ax.bar(x,         self.x_liq, width, label="Líquido (x)",
                        color=STYLE["liquid"], alpha=0.85, edgecolor=STYLE["bg"])
        bars_v = ax.bar(x + width, self.y_vap, width, label="Vapor (y)",
                        color=STYLE["vapor"],  alpha=0.85, edgecolor=STYLE["bg"])

        # Etiquetas de valor encima de cada barra
        for bars in [bars_z, bars_l, bars_v]:
            for bar in bars:
                h = bar.get_height()
                if h > 0.005:
                    ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                            f"{h:.3f}", ha="center", va="bottom",
                            fontsize=6.5, color=STYLE["subtext"])

        ax.set_xticks(x)
        ax.set_xticklabels(self.labels, rotation=25, ha="right", fontsize=8)
        ax.set_ylabel("Fracción molar")
        ax.set_ylim(0, min(1.0, max(np.max(self.z), np.max(self.x_liq),
                                    np.max(self.y_vap)) * 1.18))
        ax.set_title("Composiciones por Fase", color=STYLE["accent"])
        ax.legend(fontsize=8, facecolor=STYLE["panel"],
                  labelcolor=STYLE["text"], edgecolor=STYLE["border"])

    # ─────────────────────────────────────────────────────────────────────────
    # 3. FACTORES K
    # ─────────────────────────────────────────────────────────────────────────
    def kvalues(self, save_path=None, show=True):
        fig, ax = plt.subplots(figsize=(8, 5))
        _apply_style(fig, [ax])
        self._plot_kvalues(ax)
        fig.patch.set_facecolor(STYLE["bg"])
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=STYLE["bg"])
        if show:
            plt.show()
        return fig

    def _plot_kvalues(self, ax):
        x      = np.arange(self.n)
        K_safe = np.where(np.isnan(self.K), 1e-6, self.K)
        colors = [STYLE["vapor"] if k >= 1 else STYLE["liquid"] for k in K_safe]

        bars = ax.bar(x, K_safe, color=colors, alpha=0.85,
                      edgecolor=STYLE["bg"], linewidth=0.8)

        # Línea K=1 (equilibrio)
        ax.axhline(1.0, color=STYLE["crit"], linewidth=1.5,
                   linestyle="--", alpha=0.8, label="K = 1 (equilibrio)")

        # Etiquetas de valor
        for bar, k in zip(bars, K_safe):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 1.05,
                    f"{k:.3f}", ha="center", va="bottom",
                    fontsize=7, color=STYLE["subtext"])

        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels(self.labels, rotation=25, ha="right", fontsize=8)
        ax.set_ylabel("Factor K = y/x  (escala log)")
        ax.set_title("Factores de Equilibrio K", color=STYLE["accent"])

        vapor_patch  = mpatches.Patch(color=STYLE["vapor"],  label="K > 1 → va al vapor")
        liquid_patch = mpatches.Patch(color=STYLE["liquid"], label="K < 1 → va al líquido")
        ax.legend(handles=[vapor_patch, liquid_patch],
                  fontsize=7.5, facecolor=STYLE["panel"],
                  labelcolor=STYLE["text"], edgecolor=STYLE["border"])

    # ─────────────────────────────────────────────────────────────────────────
    # 4. BALANCE DE FLUJOS
    # ─────────────────────────────────────────────────────────────────────────
    def flow_balance(self, save_path=None, show=True):
        fig, ax = plt.subplots(figsize=(9, 5))
        _apply_style(fig, [ax])
        self._plot_flow_balance(ax)
        fig.patch.set_facecolor(STYLE["bg"])
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=STYLE["bg"])
        if show:
            plt.show()
        return fig

    def _plot_flow_balance(self, ax):
        x     = np.arange(self.n)
        width = 0.3

        bars_v = ax.bar(x - width / 2, self.V_comp, width,
                        label=f"Vapor  V={self.V_flow:.1f} kmol/h",
                        color=STYLE["vapor"],  alpha=0.85, edgecolor=STYLE["bg"])
        bars_l = ax.bar(x + width / 2, self.L_comp, width,
                        label=f"Líquido L={self.L_flow:.1f} kmol/h",
                        color=STYLE["liquid"], alpha=0.85, edgecolor=STYLE["bg"])

        # Alimentación como puntos
        ax.scatter(x, self.F_comp, s=60, zorder=10,
                   color=STYLE["feed"], edgecolors="white",
                   linewidths=1, label=f"Feed  F={self.F_total:.1f} kmol/h")

        ax.set_xticks(x)
        ax.set_xticklabels(self.labels, rotation=25, ha="right", fontsize=8)
        ax.set_ylabel("Flujo molar (kmol/h)")
        ax.set_title("Balance de Materia por Componente", color=STYLE["accent"])
        ax.legend(fontsize=8, facecolor=STYLE["panel"],
                  labelcolor=STYLE["text"], edgecolor=STYLE["border"])

    # ─────────────────────────────────────────────────────────────────────────
    # 5. DISTRIBUCIÓN POR FASE (torta apilada)
    # ─────────────────────────────────────────────────────────────────────────
    def _plot_phase_distribution(self, ax):
        """Barras apiladas de fracción de cada componente en cada fase."""
        # Fracción del flujo total que representa cada comp en cada fase
        frac_V = self.V_comp / self.F_total
        frac_L = self.L_comp / self.F_total

        bottom_v, bottom_l = 0.0, 0.0
        for i in range(self.n):
            color = COMP_COLORS[i % len(COMP_COLORS)]
            ax.bar(0, frac_V[i], bottom=bottom_v, width=0.4,
                   color=color, alpha=0.85, edgecolor=STYLE["bg"])
            ax.bar(1, frac_L[i], bottom=bottom_l, width=0.4,
                   color=color, alpha=0.85, edgecolor=STYLE["bg"],
                   label=self.labels[i])
            if frac_V[i] > 0.02:
                ax.text(0, bottom_v + frac_V[i] / 2,
                        f"{self.labels[i][:4]}\n{frac_V[i]:.2f}",
                        ha="center", va="center", fontsize=6,
                        color="white", fontweight="bold")
            if frac_L[i] > 0.02:
                ax.text(1, bottom_l + frac_L[i] / 2,
                        f"{self.labels[i][:4]}\n{frac_L[i]:.2f}",
                        ha="center", va="center", fontsize=6,
                        color="white", fontweight="bold")
            bottom_v += frac_V[i]
            bottom_l += frac_L[i]

        ax.set_xticks([0, 1])
        ax.set_xticklabels(
            [f"VAPOR\nβ={self.beta:.3f}",
             f"LÍQUIDO\n1-β={1-self.beta:.3f}"],
            fontsize=9, color=STYLE["text"]
        )
        ax.set_ylabel("Fracción del feed total")
        ax.set_ylim(0, 1.05)
        ax.set_title("Distribución de Fases", color=STYLE["accent"])

    # ─────────────────────────────────────────────────────────────────────────
    # 6. TABLA RESUMEN
    # ─────────────────────────────────────────────────────────────────────────
    def _plot_summary_table(self, ax):
        ax.axis("off")

        col_labels = ["Componente", "z", "x (liq)", "y (vap)", "K", "F (kmol/h)", "L (kmol/h)", "V (kmol/h)"]
        rows = []
        for i in range(self.n):
            K_str = f"{self.K[i]:.4f}" if not np.isnan(self.K[i]) else "—"
            rows.append([
                self.labels[i],
                f"{self.z[i]:.4f}",
                f"{self.x_liq[i]:.4f}",
                f"{self.y_vap[i]:.4f}",
                K_str,
                f"{self.F_comp[i]:.2f}",
                f"{self.L_comp[i]:.2f}",
                f"{self.V_comp[i]:.2f}",
            ])

        # Fila total
        rows.append([
            "TOTAL",
            f"{sum(self.z):.4f}",
            f"{sum(self.x_liq):.4f}",
            f"{sum(self.y_vap):.4f}",
            "—",
            f"{self.F_total:.2f}",
            f"{self.L_flow:.2f}",
            f"{self.V_flow:.2f}",
        ])

        table = ax.table(
            cellText=rows,
            colLabels=col_labels,
            loc="center",
            cellLoc="center"
        )
        table.auto_set_font_size(False)
        table.set_fontsize(7.2)
        table.scale(1, 1.4)

        # Estilizar celdas
        for (row, col), cell in table.get_celld().items():
            cell.set_facecolor(STYLE["panel"])
            cell.set_edgecolor(STYLE["border"])
            if row == 0:
                cell.set_facecolor(STYLE["border"])
                cell.set_text_props(color=STYLE["accent"], fontweight="bold")
            elif row == len(rows):
                cell.set_facecolor("#1A3050")
                cell.set_text_props(color=STYLE["crit"], fontweight="bold")
            else:
                cell.set_text_props(color=STYLE["text"])

        ax.set_title("Tabla de Resultados", color=STYLE["accent"], pad=10)

    # ─────────────────────────────────────────────────────────────────────────
    # EXPORTAR CSV
    # ─────────────────────────────────────────────────────────────────────────
    def export_csv(self, path="flash_resultados.csv"):
        df = pd.DataFrame({
            "Componente":    self.labels,
            "CoolProp_name": self.fm.components,
            "z_feed":        self.z,
            "x_liquido":     self.x_liq,
            "y_vapor":       self.y_vap,
            "K_factor":      self.K,
            "F_kmolh":       self.F_comp,
            "L_kmolh":       self.L_comp,
            "V_kmolh":       self.V_comp,
        })

        meta = pd.DataFrame({
            "Parametro": ["T (K)", "P (Pa)", "P (bar)", "Beta_vapor",
                          "Beta_liquido", "F_total (kmol/h)",
                          "V_total (kmol/h)", "L_total (kmol/h)", "Fase"],
            "Valor": [
                self.fm.T, self.fm.P, self.fm.P / 1e5,
                self.beta, 1 - self.beta, self.F_total,
                self.V_flow, self.L_flow,
                "Bifásico" if self.is_twophase else "Una fase"
            ]
        })

        with open(path, "w", newline="", encoding="utf-8") as f:
            f.write("# CONDICIONES DE OPERACIÓN\n")
            meta.to_csv(f, index=False)
            f.write("\n# RESULTADOS POR COMPONENTE\n")
            df.to_csv(f, index=False)

        print(f"  ✅ CSV exportado: {path}")
        return df, meta