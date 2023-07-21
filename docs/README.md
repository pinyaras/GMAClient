# Build Documentation Website

Run the following commands generate rst files for `network_gym_client` library. The rst  files contain modules, classes and functions.
```
rm network_gym_client*.rst
rm modules.rst
sphinx-apidoc -o . ../network_gym_client -d 2
```

Add the classes and functions in the md (markdown) files. E.g.,

````
```{eval-rst}
.. autoclass:: network_gym_client.Env
```
````

Export the static website in the `_build/html` folder.
```
make clean; make html
```