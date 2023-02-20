Create axis slice plots of OpenMC geomtry:
  - Interactive plots with hovertext :speech_balloon:
  - Specify the zoom :mag:
  - Color by materials or cells :art:
  - Outline by materials or cells :pencil2:
  - Switch plotting backends between 📉 MatPlotLib and 📈 Plotly

This package is deployed on [xsplot.com](https://www.xsplot.com) as part of the ```openmc_plot``` suite of plotting apps

![openmc geometry plot](https://user-images.githubusercontent.com/8583900/213252783-526fa814-2abd-4aac-bd1d-9cf0024a7039.png)

# Local install

You will need to first install openmc. There are several methods but perhaps the quickest is to use Conda.

```bash
conda install -c conda-forge openmc
```

Then you can install ```openmc_geometry_plot``` with pip

```bash
pip install openmc_geometry_plot
```


# Usage

The package can be used from within your own python script to make plots or via a GUI that is also bundled into the package install.

## Python API script usage

See the [examples folder](https://github.com/fusion-energy/openmc_geometry_plot/tree/master/examples) for example scripts

## Graphical User Interface (GUI) usage

After installing run the ```openmc_geometry_plot``` command from the terminal and the GUI should launch in a new browser window.
