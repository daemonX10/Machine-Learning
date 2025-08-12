## Sums of independent r.v., covariance and correlation

Proposition (Discrete case) LetX,Ybe discrete independent
random variables andZ=X+Y, then the PMF ofZis

```
pZ(z)=Q
x
```
```
pX(x)pY(z−x).
```
Proposition (Continuous case) LetX,Ybe continuous
independent random variables andZ=X+Y, then the PDF ofZis

```
fZ(z)=S
```
```
∞
−∞
```
```
fX(x)fY(z−x)dx.
```
Proposition (Sum of independent normal r.v.) LetX∼N(μx,σ^2 x)
andY∼N(μy,σ^2 y)independent. Then

Z=X+Y∼N(μx+μy,σ^2 x+σ^2 y).

Definition (Covariance) We define the covariance of random
variablesX,Yas

```
Cov(X,Y)
```
```
△
=E[(X−E[X]) (Y−E[Y])].
```
Properties (Properties of covariance)

- IfX,Yare independent, then Cov(X,Y)=0.
- Cov(X,X)=Var(X).
- Cov(aX+b,Y)=aCov(X,Y).
- Cov(X,Y+Z)=Cov(X,Y)+Cov(X,Z).
- Cov(X,Y)=E[XY]−E[X]E[Y].

Proposition (Variance of a sum of r.v.)

```
Var(X 1 + ⋯ +Xn)=Q
i
```
```
Var(Xi)+Q
i≠j
```
```
Cov(Xi,Xj).
```
Definition (Correlation coefficient) We define the correlation
coefficient of random variablesX,Y, withσX,σY>0, as

```
ρ(X,Y)
```
```
△
=
```
```
Cov(X,Y)
σXσY
```
### .

Properties (Properties of the correlation coefficient)

- − 1 ≤ρ≤1.
- IfX,Yare independent, thenρ=0.
- SρS=1 if and only ifX−E[X]=c(Y−E[Y]).
- ρ(aX+b,Y)=sign(a)ρ(X,Y).

## Conditional expectation and variance, sum of

## random number of r.v.

Definition (Conditional expectation as a random variable) Given
random variablesX,Ythe conditional expectationE[XSY]is the
random variable that takes the valueE[XSY=y]wheneverY=y.

Theorem (Law of iterated expectations)

```
E[E[XSY]]=E[X].
```
```
Definition (Conditional variance as a random variable) Given
random variablesX,Ythe conditional variance Var(XSY)is the
random variable that takes the value Var(XSY=y)whenever
Y=y.
Theorem (Law of total variance)
```
```
Var(X)=E[Var(XSY)]+Var(E[XSY]).
```
```
Proposition (Sum of a random number of independent r.v.)
LetNbe a nonnegative integer random variable.
LetX,X 1 ,X 2 ,...,XNbe i.i.d. random variables.
LetY=∑iXi. Then
```
```
E[Y]=E[N]E[X],
```
```
Var(Y)=E[N]Var(X)+(E[X])^2 Var(N).
```
# Convergence of random variables

## Inequalities, convergence, and the Weak Law of

## Large Numbers

```
Theorem (Markov inequality)Given a random variableX≥0 and,
for everya>0 we have
```
```
P(X≥a)≤
```
### E[X]

```
a
```
### .

```
Theorem (Chebyshev inequality) Given a random variableXwith
E[X]=μand Var(X)=σ^2 , for every>0 we have
```
```
P(SX−μS≥)≤
```
```
σ^2
^2
```
### .

```
Theorem (Weak Law of Large Number (WLLN)) Given a
sequence of i.i.d. random variables{X 1 ,X 2 ,...}withE[Xi]=μ
and Var(Xi)=σ^2 , we define
```
```
Mn=
```
### 1

```
n
```
```
n
Q
i= 1
```
```
Xi,
```
```
for every>0 we have
```
```
lim
n→∞
```
```
P(SMn−μS≥)= 0.
```
```
Definition (Convergence in probability) A sequence of random
variables{Yi}converges in probability to the random variableYif
```
```
nlim→∞P(SYi−YS≥)=^0 ,
for every>0.
Properties (Properties of convergence in probability) IfXn→a
andYn→bin probability, then
```
- Xn+Yn→a+b.
- Ifgis a continuous function, theng(Xn)→g(a).
- E[Xn]does not always converge toa.

## The Central Limit Theorem

```
Theorem (Central Limit Theorem (CLT)) Given a sequence of
independent random variables{X 1 ,X 2 ,...}withE[Xi]=μand
Var(Xi)=σ^2 , we define
```
```
Zn=
```
### 1

```
σ
```
### √

```
n
```
```
n
Q
i= 1
```
```
(Xi−μ).
```
```
Then, for everyz, we have
```
```
lim
n→∞
```
```
P(Zn≤z)=P(Z≤z),
```
```
whereZ∼N( 0 , 1 ).
Corollary (Normal approximation of a binomial) Let
X∼Bin(n,p)withnlarge. ThenSncan be approximated by
Z∼N(np,np( 1 −p)).
Remark (De Moivre-Laplace 1/2 approximation) LetX∼Bin,
thenP(X=i)=Pi−^12 ≤X≤i+^12 and we can use the CLT to
approximate the PMF ofX.
```

