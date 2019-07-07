# Cartesiang Gnetic Programming
Cartesian Genetic Programming for sklearn

sklearnで，cross validation，grid searchができるようにしたCartesian Genetic Programming．

# Install
clone here
```
git clone https://github.com/shinjikato/cartesian_genetic_programming.git
```

# Run
move directory
```
cd cartesian_genetic_programming
```

run python
```
python
```

import and run
```
import cgp
X,y = data
model = cgp.CGP_regressor()
model.fit(X, y)
print(model.score(X, y))
```

# Best Parameter?