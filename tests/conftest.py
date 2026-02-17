import matplotlib
import pytest

matplotlib.use("Agg")


@pytest.fixture(autouse=True)
def suppress_plots():
    import matplotlib.pyplot as plt

    plt.ioff()
