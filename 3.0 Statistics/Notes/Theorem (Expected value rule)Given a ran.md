Theorem (Expected value rule)Given a random variableXand a
functiong∶R→R, we construct the random variableY=g(X).
Then
Q
y

```
ypY(y)=E[Y]=E[g(X)]=Q
x
```
```
g(x)pX(x).
```
Remark (PMF ofY=g(X)) The PMF ofY=g(X)is
pY(y)= ∑
x∶g(x)=y

```
pX(x).
```
Remark In generalg(E[X])≠E[g(X)]. They are equal if
g(x)=ax+b.

# Variance, conditioning on an event, multiple r.v.

Definition (Variance of a random variable) Given a random
variableXwithμ=E[X], its variance is a measure of the spread
of the random variable and is defined as

```
Var(X)△=E(X−μ)^2 =Q
x
```
```
(x−μ)^2 pX(x).
```
Definition (Standard deviation)

```
σX=
```
## »

```
Var(X).
```
Properties (Properties of the variance)

- Var(aX)=a^2 Var(X),for alla∈R.
- Var(X+b)=Var(X),for allb∈R.
- Var(aX+b)=a^2 Var(X).
- Var(X)=E[X^2 ]−(E[X])^2.

Example (Variance of known r.v.)

- IfX∼Ber(p),then Var(X)=p( 1 −p).
- IfX∼Uni[a,b],then Var(X)=

```
(b−a)(b−a+ 2 )
12.
```
- IfX∼Bin(n,p),then Var(X)=np( 1 −p).
- IfX∼Geo(p),then Var(X)=^1 p− 2 p

Proposition (Conditional PMF and expectation, given an event)
Given the eventA, withP(A)>0, we have the following

- pXSA(x)=P(X=xSA).
- IfAis a subset of the range ofX, then:

```
pXSA(x)
```
```
△
=pXS{X∈A}(x)=
```
## ⎧⎪

## ⎪

## ⎨

## ⎪⎪

## ⎩

```
1
P(A)pX(x), ifx∈A,
0 , otherwise.
```
- ∑xpXSA(x)=1.
- E[XSA]=∑xxpXSA(x).
- E[g(X)SA]=∑xg(x)pXSA(x).

Proposition (Total expectation rule) Given a partition of disjoint
eventsA 1 ,...,Ansuch that∑iP(Ai)=1, andP(Ai)>0,

```
E[X]=P(A 1 )E[XSA 1 ]+ ⋯ +P(An)E[XSAn].
```
Definition (Memorylessness of the geometric random variable)
When we condition a geometric random variableXon the event
X>nwe have memorylessness, meaning that the “remaining time”
X−n, given thatX>n, is also geometric with the same parameter.
Formally,
pX−nSX>n(i)=pX(i).

Definition (Joint PMF) The joint PMF of random variables
X 1 ,X 2 ,...,Xnis
pX 1 ,X 2 ,...,Xn(x 1 ,...,xn)=P(X 1 =x 1 ,...,Xn=xn).

```
Properties (Properties of joint PMF)
```
- ∑
    x 1

## ⋯∑

```
xn
```
```
pX 1 ,...,Xn(x 1 ,...,xn)=1.
```
- pX 1 (x 1 )=∑
    x 2

## ⋯∑

```
xn
```
```
pX 1 ,...,Xn(x 1 ,x 2 ,...,xn).
```
- pX 2 ,...,Xn(x 2 ,...,xn)=∑
    x 1

```
pX 1 ,X 2 ,...,Xn(x 1 ,x 2 ,...,xn).
```
```
Definition (Functions of multiple r.v.) IfZ=g(X 1 ,...,Xn),
whereg∶Rn→R, thenpZ(z)=P(g(X 1 ,...,Xn)=z).
Proposition (Expected value rule for multiple r.v.) Given
g∶Rn→R,
```
```
E[g(X 1 ,...,Xn)]= Q
x 1 ,...,xn
```
```
g(x 1 ,...,xn)pX 1 ,...,Xn(x 1 ,...,xn).
```
```
Properties (Linearity of expectations)
```
- E[aX+b]=aE[X]+b.
- E[X 1 + ⋯ +Xn]=E[X 1 ]+ ⋯ +E[Xn].

# Conditioning on a random variable, independence

```
Definition (Conditional PMF given another random variable)
Given discrete random variablesX,Yandysuch thatpY(y)> 0
we define
pXSY(xSy)
```
```
△
=
```
```
pX,Y(x,y)
pY(y)
```
## .

```
Proposition (Multiplication rule) Given jointly discrete random
variablesX,Y, and whenever the conditional probabilities are
defined,
```
```
pX,Y(x,y)=pX(x)pYSX(ySx)=pY(y)pXSY(xSy).
```
```
Definition (Conditional expectation) Given discrete random
variablesX,Yandysuch thatpY(y)>0 we define
```
```
E[XSY=y]=Q
x
```
```
xpXSY(xSy).
```
```
Additionally we have
```
```
E[g(X)SY=y]=Q
x
```
```
g(x)pXSY(xSy).
```
```
Theorem (Total probability and expectation theorems)
IfpY(y)> 0 ,then
```
```
pX(x)=Q
y
```
```
pY(y)pXSY(xSy),
```
## E[X]=Q

```
y
```
```
pY(y)E[XSY=y].
```
```
Definition (Independence of a random variable and an event) A
discrete random variableXand an eventAare independent if
P(X=xandA)=pX(x)P(A),for allx.
Definition (Independence of two random variables) Two discrete
random variablesXandYare independent if
pX,Y(x,y)=pX(x)pY(y)for allx,y.
Remark (Independence of a collection of random variables) A
collectionX 1 ,X 2 ,...,Xnof random variables are independent if
```
```
pX 1 ,...,Xn(x 1 ,...,xn)=pX 1 (x 1 )⋯pXn(xn),∀x 1 ,...,xn.
```
```
Remark (Independence and expectation) In general,
E[g(X,Y)]≠g(E[X],E[Y]). An exception is for linear functions:
E[aX+bY]=aE[X]+bE[Y].
```
```
Proposition (Expectation of product of independent r.v.) IfX
andYare discrete independent random variables,
```
```
E[XY]=E[X]E[Y].
```
```
Remark IfXandYare independent,
E[g(X)h(Y)]=E[g(X)]E[h(Y)].
Proposition (Variance of sum of independent random variables)
IFXandYare discrete independent random variables,
```
```
Var(X+Y)=Var(X)+Var(Y).
```
# Continuous random variables

# PDF, Expectation, Variance, CDF

```
Definition (Probability density function (PDF)) A probability
density function of a r.v.Xis a non-negative real valued function
fXthat satisfies the following
```
-

```
∞
∫
−∞
```
```
fX(x)dx=1.
```
- P(a≤X≤b)=

```
b
∫
a
```
```
fX(x)dxfor some random variableX.
```
```
Definition (Continuous random variable) A random variableXis
continuous if its probability law can be described by a PDFfX.
Remark Continuous random variables satisfy:
```
- For smallδ>0,P(a≤X≤a+δ)≈fX(a)δ.
- P(X=a)= 0 ,∀a∈R.
Definition (Expectation of a continuous random variable) The
expectation of a continuous random variable is

```
E[X]
```
```
△
=S
```
```
∞
−∞
```
```
xfX(x)dx.
```
```
assuming
```
```
∞
∫
−∞
```
```
SxSfX(x)dx<∞.
```
```
Properties (Properties of expectation)
```
- IfX≥0 thenE[X]≥0.
- Ifa≤X≤bthena≤E[X]≤b.
- E[g(X)]=

```
∞
∫
−∞
```
```
g(x)fX(x)dx.
```
- E[aX+b]=aE[X]+b.
Definition (Variance of a continuous random variable) Given a
continuous random variableXwithμ=E[X], its variance is

```
Var(X)=E(X−μ)^2 =S
```
```
∞
−∞
```
```
(x−μ)^2 fX(x)dx.
```
```
It has the same properties as the variance of a discrete random
variable.
Example (Uniform continuous random variable) A Uniform
continuous random variableXbetweenaandb, witha<b,
(X∼Uni(a,b)) has PDF
```
```
fX(x)=
```
## ⎧⎪

## ⎪

## ⎨

## ⎪⎪

## ⎩

```
1
b−a, ifa<x<b,
0 , otherwise.
```
```
We haveE[X]=a+ 2 band Var(X)=(b−a)
```
```
2
12.
```

