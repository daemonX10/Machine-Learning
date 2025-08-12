Example (Exponential random variable) An Exponential random
variableXwith parameterλ>0 (X∼Exp(λ)) has PDF

```
fX(x)=
```
## ⎧⎪

## ⎪⎨

## ⎪⎪

## ⎩

```
λe−λx, ifx≥ 0 ,
0 , otherwise.
```
We haveE[X]=^1 λand Var(X)=λ^12.

Definition (Cumulative Distribution Function (CDF)) The CDF
of a random variableXisFX(x)=P(X≤x).

In particular, for a continuous random variable, we have

```
FX(x)=
```
```
x
S
−∞
```
```
fX(x)dx,
```
```
fX(x)=
```
```
dFX(x)
dx
```
## .

Properties (Properties of CDF)

- Ify≥x, thenFX(y)≥FX(x).
- lim
    x→−∞

```
FX(x)=0.
```
- lim
    x→∞

```
FX(x)=1.
```
Definition (Normal/Gaussian random variable)A Normal random
variableXwith meanμand varianceσ^2 >0 (X∼N(μ,σ^2 )) has
PDF

```
fX(x)=
```
## 1

## √

```
2 πσ^2
```
```
e−
```
```
1
2 σ^2 (x−μ)
```
```
2
.
```
We haveE[X]=μand Var(X)=σ^2.

Remark (Standard Normal) The standard Normal isN( 0 , 1 ).

Proposition (Linearity of Gaussians) GivenX∼N(μ,σ^2 ), and if
a≠0, thenaX+b∼N(aμ+b,a^2 σ^2 ).

Using thisY=Xσ−μis a standard gaussian.

# Conditioning on an event, and multiple continuous r.v.

Definition (Conditional PDF given an event) Given a continuous
random variableXand eventAwithP(A)>0, we define the
conditional PDF as the function that satisfies

```
P(X∈BSA)=S
B
```
```
fXSA(x)dx.
```
Definition (Conditional PDF givenX∈A) Given a continuous
random variableXand anA⊂R, withP(A)>0:

```
fXSX∈A(x)=
```
## ⎧⎪

## ⎪

## ⎨

## ⎪⎪

## ⎩

```
1
P(A)fX(x), x∈A,
0 , x~∈A.
```
Definition (Conditional expectation) Given a continuous random
variableXand an eventA, withP(A)>0:

## E[XSA]=S

```
∞
−∞
```
```
fXSA(x)dx.
```
Definition (Memorylessness of the exponential random variable)
When we condition an exponential random variableXon the event
X>twe have memorylessness, meaning that the “remaining time”
X−tgiven thatX>tis also geometric with the same parameter
i.e.,
P(X−t>xSX>t)=P(X>x).

```
Theorem (Total probability and expectation theorems) Given a
partition of the space into disjoint eventsA 1 ,A 2 ,...,Ansuch that
∑iP(Ai)=1 we have the following:
```
```
FX(x)=P(A 1 )FXSA 1 (x)+ ⋯ +P(An)FXSAn(x),
```
```
fX(x)=P(A 1 )fXSA 1 (x)+ ⋯ +P(An)fXSAn(x),
```
```
E[X]=P(A 1 )E[XSA 1 ]+ ⋯ +P(An)E[XSAn].
```
```
Definition (Jointly continuous random variables) A pair
(collection) of random variables is jointly continuous if there exists
a joint PDFfX,Ythat describes them, that is, for every setB⊂Rn
```
```
P((X,Y)∈B)=U
B
```
```
fX,Y(x,y)dxdy.
```
```
Properties (Properties of joint PDFs)
```
- fX(x)=

```
∞
∫
−∞
```
```
fX,Y(x,y)dy.
```
- FX,Y(x,y)=P(X≤x,Y≤y)=

```
x
∫
−∞
```
## 

```
y
∫
−∞
```
```
fX,Y(u,v)dv du.
```
- fX,Y(x)=

```
∂^2 FX,Y(x,y)
∂x ∂y.
Example (Uniform joint PDF on a setS) LetS⊂R^2 with area
s>0, then the random variable(X,Y)is uniform overSif it has
PDF
```
```
fX,Y(x,y)=
```
## ⎧⎪

## ⎪

## ⎨

## ⎪⎪

## ⎩

```
1
s, (x,y)∈S,
0 , (x,y) ~∈S.
```
# Conditioning on a random variable, independence, Bayes’ rule

```
Definition (Conditional PDF given another random variable)
Given jointly continuous random variablesX,Yand a valueysuch
thatfY(y)>0, we define the conditional PDF as
```
```
fXSY(xSy)
```
```
△
=
```
```
fX,Y(x,y)
fY(y)
```
## .

```
Additionally we defineP(X∈ASY=y)∫AfXSY(xSy)dx.
Proposition (Multiplication rule) Given jointly continuous
random variablesX,Y, whenever possible we have
```
```
fX,Y(x,y)=fX(x)fYSX(ySx)=fY(y)fXSY(xSy).
```
```
Definition (Conditional expectation) Given jointly continuous
random variablesX,Y, andysuch thatfY(y)>0, we define the
conditional expected value as
```
```
E[XSY=y]=S
```
```
∞
−∞
```
```
xfXSY(xSy)dx.
```
```
Additionally we have
```
```
E[g(X)SY=y]=S
```
```
∞
−∞
```
```
g(x)fXSY(xSy)dx.
```
```
Theorem (Total probability and total expectation theorems)
```
```
fX(x)=S
```
```
∞
−∞
```
```
fY(y)fXSY(xSy)dy,
```
## E[X]=S

```
∞
−∞
```
```
fY(y)E[XSY=y]dy.
```
```
Definition (Independence) Jointly continuous random variables
X,Yare independent iffX,Y(x,y)=fX(x)fY(y)for allx,y.
```
```
Proposition (Expectation of product of independent r.v.) IfX
andYare independent continuous random variables,
```
## E[XY]=E[X]E[Y].

```
Remark IfXandYare independent,
E[g(X)h(Y)]=E[g(X)]E[h(Y)].
```
```
Proposition (Variance of sum of independent random variables)
IfXandYare independent continuous random variables,
```
```
Var(X+Y)=Var(X)+Var(Y).
```
```
Proposition (Bayes’ rule summary)
```
- ForX,Ydiscrete:pXSY(xSy)=

```
pX(x)pYSX(ySx)
pY(y).
```
- ForX,Ycontinuous:fXSY(xSy)=

```
fX(x)fYSX(ySx)
fY(y).
```
- ForXdiscrete,Ycontinuous:pXSY(xSy)=

```
pX(x)fYSX(ySx)
fY(y).
```
- ForXcontinuous,Ydiscrete:fXSY(xSy)=

```
fX(x)pYSX(ySx)
pY(y).
```
# Derived distributions

```
Proposition (Discrete case) Given a discrete random variableX
and a functiong, the r.v.Y=g(X)has PMF
```
```
pY(y)= Q
x∶g(x)=y
```
```
pX(x).
```
```
Remark (Linear function of discrete random variable) If
g(x)=ax+b, thenpY(y)=pXy−ab.
```
```
Proposition (Linear function of continuous r.v.) Given a
continuous random variableXandY=aX+b, witha≠0, we have
```
```
fY(y)=
```
## 1

```
SaS
```
```
fX
```
```
y−b
a
```
## .

```
Corollary (Linear function of normal r.v.) IfX∼N(μ,σ^2 )and
Y=aX+b, witha≠0, thenY∼N(aμ+b,a^2 σ^2 ).
```
```
Example (General function of a continuous r.v.) IfXis a
continuous random variable andgis any function, to obtain the
pdf ofY=g(X)we follow the two-step procedure:
```
1. Find the CDF ofY:FY(y)=P(Y≤y)=P(g(X)≤y).
2. Differentiate the CDF ofYto obtain the PDF:
    fY(y)=dFdYy(y).

```
Proposition (General formula for monotonicg) LetXbe a
continuous random variable andga function that is monotonic
whereverfX(x)>0. The PDF ofY=g(X)is given by
```
```
fY(y)=fX(h(y)) W
```
```
dh
dy
```
```
(y)W.
```
```
whereh=g−^1 in the interval where g is monotonic.
```

