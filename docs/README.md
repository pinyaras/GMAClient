# Build Documentation Website

Run the following commands and the static website will be saved in the _build/html folder.
```
rm network_gym_client*.rst
rm modules.rst
sphinx-apidoc -o . ../network_gym_client -d 2
make html

```