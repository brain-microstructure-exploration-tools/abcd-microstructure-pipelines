# Notebooks

Example notebooks demonstrating `kwneuro` pipeline usage. These are stored as
percent-format Python scripts (compatible with Jupytext) and are excluded from
CI linting.

## Running a notebook

Install Jupytext, then open the `.py` file as a notebook:

```bash
pip install jupytext
jupytext --to notebook example-pipeline.py   # creates example-pipeline.ipynb
jupyter notebook example-pipeline.ipynb
```

Or, if you have Jupytext's Jupyter extension enabled, simply open the `.py` file
directly in Jupyter.
