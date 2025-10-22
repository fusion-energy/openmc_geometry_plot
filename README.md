Create axis slice plots of OpenMC geomtry:
  - Interactive plots with hovertext :speech_balloon:
  - Specify the zoom :mag:
  - Color by materials or cells :art:
  - Outline by materials or cells :pencil2:
  - Interactive plotting with 📈 Plotly

This package is deployed on [xsplot.com](https://www.xsplot.com) as part of the ```openmc_plot``` suite of plotting apps

![openmc geometry plot](https://user-images.githubusercontent.com/8583900/213252783-526fa814-2abd-4aac-bd1d-9cf0024a7039.png)

# Local install

You will need to first install OpenMC (0.15.3 or newer).


There are several methods but perhaps the quickest is to use this Pip extra index as it includes a recent version of OpenMC 0.15.3-dev (which is needed to run the app).


```bash
python -m pip install --extra-index-url https://shimwell.github.io/wheels openmc
```


Then you can install ```openmc_geometry_plot``` with pip

```bash
pip install openmc_geometry_plot
```


# Usage

After installing run the ```openmc_geometry_plot``` command from the terminal and the GUI should launch in a new browser window.
