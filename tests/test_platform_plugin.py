from vllm_runtime_sim.platform_plugin import register_simulation_platform


def test_platform_plugin_is_opt_in(monkeypatch):
    monkeypatch.delenv("VLLM_RUNTIME_SIM", raising=False)
    assert register_simulation_platform() is None


def test_platform_plugin_registers_when_enabled(monkeypatch):
    monkeypatch.setenv("VLLM_RUNTIME_SIM", "1")
    assert (
        register_simulation_platform()
        == "vllm_runtime_sim.simulation_platform.SimulationPlatform"
    )
