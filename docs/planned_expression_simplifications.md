# Planned expression simplifications

## Always

```text
X + 0       -> X
0 + X       -> X
X - 0       -> X
X * 1       -> X
1 * X       -> X
X / 1       -> X
X ** 1      -> X
-(-X)       -> X
abs(abs(X)) -> abs(X)
```

## With mathematical-support conditions

```text
log(exp(X))    -> X       if X is real
exp(log(X))    -> X       if X is positive
sqrt(X ** 2)   -> abs(X)  if X is real
sin(arcsin(X)) -> X       if X is in [-1, 1]
cos(arccos(X)) -> X       if X is in [-1, 1]
tan(arctan(X)) -> X       if X is real
```

## Stable numerical replacements

```text
log(1 + X)            -> log1p(X)          if X > -1
exp(X) - 1            -> expm1(X)          if X is real
sqrt(X**2 + Y**2)     -> hypot(X, Y)       if X and Y are real
log(exp(X) + exp(Y))  -> logaddexp(X, Y)   if X and Y are real
log(1 + exp(X))       -> logaddexp(0, X)   if X is real
```
