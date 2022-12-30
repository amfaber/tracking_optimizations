"sudo $(maturin build --release)" -match "([^\s]+$)"
pip install $Matches[1] --force-reinstall