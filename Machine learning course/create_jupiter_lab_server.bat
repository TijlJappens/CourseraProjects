@echo off
wsl -e bash -ic "conda activate TuriCreateEnv && python -m jupyterlab"

pause