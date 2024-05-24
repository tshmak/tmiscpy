# A Python package

Version: 1.0.0

## To display graphics remotely via ssh

We generally need to install `pyqt5` first, via: 
```
pip install pyqt5
```

Then try running `%matplotlib qt` in `ipython`. 

If there's no error, then we're good. Very often, it will fail with an error like: 
```
Could not load the Qt platform plugin "xcb" in "" even though it was found.
```

Then, we need to set 
```
export QT_DEBUG_PLUGINS=1
```

and run `%matplotlib qt` in `ipython` again. 

It will now tell you which shared library cannot be loaded. You can then run 
```
ldd /path/to/library.so.1 | grep "found"
```
to see which libraries are missing, and install them one by one. 
