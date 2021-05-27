# pyami.core

Core functionality in here

### Core

- Core functionality to operate the simulation

### Instruments

- Default constant definitions here


# Rules

No science related algorithms should be in here

No imports from other moduels should be here (except pyami.io)


i.e. 

in __init__.py:

```
from pyami.core import general.py

my_function = general.my_function

```


in code:
```
pyami.core.my_function()
```
