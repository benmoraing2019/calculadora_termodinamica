import numpy as np
import pandas as pd
import CoolProp.CoolProp as CP
from CoolProp.CoolProp import AbstractState

class FlashModel:
    def __init__(self, temperature: float, pressure: float, config_manager):
        self.T = temperature  # Kelvin
        self.P = pressure     # Pascales

        # ── Guardar referencia al config_manager (lo necesita PlotService) ──
        self.config_manager = config_manager

        # 1. Extraer datos del manager
        self.mezcla_datos = config_manager.get_mixture_config()
        self.components   = [comp['coolprop_name']    for comp in self.mezcla_datos]
        self.composition  = [comp['z_fraccion_molar'] for comp in self.mezcla_datos]

        # 2. Crear el código de CoolProp para la mezcla
        self.mixture_code = '&'.join(self.components)

        # 3. Inicializar el estado termodinámico
        self.state = AbstractState("HEOS", self.mixture_code)
        self.state.set_mole_fractions(self.composition)

        # 4. Actualizar con P y T  →  SIEMPRE antes de leer cualquier propiedad
        self.state.update(CP.PT_INPUTS, self.P, self.T)

        # 5. Leer fase DESPUÉS del update
        self.phase = self.state.phase()

        # Códigos oficiales de CoolProp (CP.iphase_*)
        self.phase_names = {
            CP.iphase_liquid:               "Líquido",
            CP.iphase_gas:                  "Vapor",
            CP.iphase_supercritical:        "Supercrítico",
            CP.iphase_supercritical_gas:    "Gas supercrítico",
            CP.iphase_supercritical_liquid: "Líquido supercrítico",
            CP.iphase_twophase:             "Bifásico (VLE)",
            CP.iphase_unknown:              "Desconocido — revisa T y P",
        }

    def get_phase_name(self) -> str:
        return self.phase_names.get(self.phase, f"Fase código {self.phase}")

    def get_results(self) -> dict:
        """Retorna un dict con los resultados del flash."""
        base = {
            "T_K":       self.T,
            "P_Pa":      self.P,
            "phase_code": self.phase,
            "phase_name": self.get_phase_name(),
        }

        if self.phase == CP.iphase_twophase:
            beta  = self.state.Q()
            x_liq = list(self.state.mole_fractions_liquid())
            y_vap = list(self.state.mole_fractions_vapor())
            K     = [y / x if x > 1e-15 else None
                     for x, y in zip(x_liq, y_vap)]
            base.update({
                "beta_vapor":   beta,
                "beta_liquido": 1 - beta,
                "x_liquido":    x_liq,
                "y_vapor":      y_vap,
                "K_factors":    K,
            })
        else:
            base.update({
                "beta_vapor":   1.0 if self.phase == CP.iphase_gas else 0.0,
                "rho_molar":    self.state.rhomolar(),
                "h_molar":      self.state.hmolar(),
                "s_molar":      self.state.smolar(),
            })

        return base