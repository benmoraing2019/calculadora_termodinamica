from .services.FlashModel import FlashModel
from .services.ConfigManager import ConfigManager
from .services.PlotService import PlotService

def test_flash_model():
    cfm = ConfigManager()
    fm  = FlashModel(200, 2e6, cfm)       # T=200K, P=20bar → región bifásica segura
    
    print(fm.get_phase_name())
    print(fm.get_results())

    ps = PlotService(fm)
    ps.dashboard(save_path="flash_dashboard.png")
    ps.export_csv("resultados.csv")

if __name__ == "__main__":
    test_flash_model()