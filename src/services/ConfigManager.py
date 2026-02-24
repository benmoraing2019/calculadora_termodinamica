import json

class ConfigManager:
    def __init__(self, config_file_path: str = r"src/common/config.json"):
        self.config_file_path = config_file_path
        with open(self.config_file_path, 'r', encoding='utf-8') as file:
            self._config = json.load(file)

    def get_full_config(self) -> dict:
        """Retorna la configuración completa."""
        return self._config
    
    def get_flash_config(self) -> dict:
        """Retorna los parámetros operativos (T, P, Flujo)."""
        return self._config.get("simulacion", {})

    def get_mixture_config(self) -> list:
        """Retorna la lista de componentes y sus fracciones molares."""
        return self._config.get("mezcla", [])