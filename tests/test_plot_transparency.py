from inspect import signature

from omiclass.plotting.framework import PlottingFramework


class _RecordingFigure:
    def __init__(self):
        self.savefig_kwargs = None

    def savefig(self, *args, **kwargs):
        self.savefig_kwargs = kwargs


def test_plotting_framework_transparency_defaults_to_false():
    assert signature(PlottingFramework).parameters["transparent"].default is False
    assert signature(PlottingFramework).parameters["output_format"].default == "pdf"


def test_framework_save_forwards_transparency(tmp_path, monkeypatch):
    plotter = PlottingFramework.__new__(PlottingFramework)
    plotter.output_dir = tmp_path
    plotter.transparent = True
    plotter.output_format = "png"
    figure = _RecordingFigure()
    monkeypatch.setattr("omiclass.plotting.framework.plt.close", lambda fig: None)

    saved_path = plotter._save(figure, "transparent.pdf")

    assert saved_path == tmp_path / "transparent.png"
    assert figure.savefig_kwargs["format"] == "png"
    assert figure.savefig_kwargs["dpi"] == 300
    assert figure.savefig_kwargs["transparent"] is True
