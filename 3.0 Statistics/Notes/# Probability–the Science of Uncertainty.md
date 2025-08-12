# Probability–the Science of Uncertainty

# and Data

```
by Fabi ́an Kozynski
```
## Probability

## Probability models and axioms

Definition (Sample space) A sample space Ω is the set of all
possible outcomes. The set’s elements must be mutually exclusive,
collectively exhaustive and at the right granularity.


Definition (Event) An event is a subset of the sample space.
Probability is assigned to events.

Definition (Probability axioms) A probability lawPassigns
probabilities to events and satisfies the following axioms:

NonnegativityP(A)≥0 for all eventsA.

NormalizationP(Ω)=1.

(Countable) additivityFor every sequence of eventsA 1 ,A 2 ,...

such thatAi∩Aj=∅:P⋃
i

```
Ai=∑
i
```
```
P(Ai).
```
Corollaries (Consequences of the axioms)

- P(∅)=0.
- For any finite collection of disjoint eventsA 1 ,...,An,

```
P
```
```
n
⋃
i= 1
```
```
Ai=
```
```
n
∑
i= 1
```
```
P(Ai).
```
- P(A)+P(Ac)=1.
- P(A)≤1.
- IfA⊂B, thenP(A)≤P(B).
- P(A∪B)=P(A)+P(B)−P(A∩B).
- P(A∪B)≤P(A)+P(B).

Example (Discrete uniform law) Assume Ω is finite and consists
ofnequally likely elements. Also, assume thatA⊂Ω withk
elements. ThenP(A)=kn.

## Conditioning and Bayes’ rule

Definition (Conditional probability) Given that eventBhas
occurred and thatP(B)>0, the probability thatAoccurs is

```
P(ASB)△=
```
### P(A∩B)

### P(B)

### .

Remark (Conditional probabilities properties) They are the same
as ordinary probabilities. AssumingP(B)>0:

- P(ASB)≥0.
- P(ΩSB)= 1
- P(BSB)=1.
- IfA∩C=∅,P(A∪CSB)=P(ASB)+P(CSB).

Proposition (Multiplication rule)

P(A 1 ∩A 2 ∩⋯∩An)=P(A 1 )⋅P(A 2 SA 1 )⋯P(AnSA 1 ∩A 2 ∩⋯∩An− 1 ).

Theorem (Total probability theorem) Given a partition
{A 1 ,A 2 ,...}of the sample space, meaning that⋃
i

```
Ai=Ω and the
```
events are disjoint, and for every eventB, we have

```
P(B)=Q
i
```
```
P(Ai)P(BSAi).
```
```
Theorem (Bayes’ rule) Given a partition{A 1 ,A 2 ,...}of the
sample space, meaning that⋃
i
```
```
Ai=Ω and the events are disjoint,
```
```
and ifP(Ai)>0 for alli, then for every eventB, the conditional
probabilitiesP(AiSB)can be obtained from the conditional
probabilitiesP(BSAi)and the initial probabilitiesP(Ai)as follows:
```
```
P(AiSB)=
```
```
P(Ai)P(BSAi)
∑jP(Aj)P(BSAj)
```
### .

## Independence

```
Definition (Independence of events) Two events are independent
if occurrence of one provides no information about the other. We
say thatAandBare independent if
```
```
P(A∩B)=P(A)P(B).
```
```
Equivalently, as long asP(A)>0 andP(B)>0,
```
```
P(BSA)=P(B) P(ASB)=P(A).
```
```
Remarks
```
- The definition of independence is symmetric with respect to
    AandB.
- The product definition applies even ifP(A)=0 orP(B)=0.
Corollary IfAandBare independent, thenAandBcare
independent. Similarly forAcandB, or forAcandBc.
Definition (Conditional independence) We say thatAandBare
independent conditioned onC, whereP(C)>0, if

```
P(A∩BSC)=P(ASC)P(BSC).
```
```
Definition (Independence of a collection of events) We say that
eventsA 1 ,A 2 ,...,Anare independent if for every collection of
distinct indicesi 1 ,i 2 ,...,ik, we have
```
```
P(Ai 1 ∩...∩Aik)=P(Ai 1 )⋅P(Ai 2 )⋯P(Aik).
```
## Counting

```
This section deals with finite sets with uniform probability law. In
this case, to calculateP(A), we need to count the number of
elements inAand in Ω.
Remark (Basic counting principle) For a selection that can be
done inrstages, withnichoices at each stagei, the number of
possible selections isn 1 ⋅n 2 ⋯nr.
Definition (Permutations) The number of permutations
(orderings) ofndifferent elements is
```
```
n!= 1 ⋅ 2 ⋅ 3 ⋯n.
```
```
Definition (Combinations) Given a set ofnelements, the number
of subsets with exactlykelements is
```
### 

```
n
k
```
### =

```
n!
k!(n−k)!
```
### .

```
Definition (Partitions) We are given ann−element set and
nonnegative integersn 1 ,n 2 ,...,nr, whose sum is equal ton. The
number of partitions of the set intordisjoint subsets, with theith
subset containing exactlynielements, is equal to
```
```

```
```
n
n 1 ,...,nr
```
### =

```
n!
n 1 !n 2 !⋯nr!
```
### .

```
Remark This is the same as counting how to assignndistinct
elements torpeople, giving each personiexactlynielements.
```
## Discrete random variables

## Probability mass function and expectation

```
Definition (Random variable) A random variableXis a function
of the sample space Ω into the real numbers (orRn). Its range can
be discrete or continuous.
Definition (Probability mass funtion (PMF)) The probability law
of a discrete random variableXis called its PMF. It is defined as
```
```
pX(x)=P(X=x)=P({ω∈Ω∶X(ω)=x}).
```
```
Properties
```
```
pX(x)≥ 0 ,∀x.
```
```
∑xpX(x)=1.
Example (Bernoulli random variable) A Bernoulli random
variableXwith parameter 0≤p≤1 (X∼Ber(p)) takes the
following values:
```
```
X=
```
### ⎧⎪

### ⎪

### ⎨

### ⎪⎪

### ⎩

```
1 w.p.p,
0 w.p. 1−p.
```
```
An indicator random variable of an event (IA=1 ifAoccurs) is an
example of a Bernoulli random variable.
Example (Discrete uniform random variable) A Discrete uniform
random variableXbetweenaandbwitha≤b(X∼Uni[a,b])
takes any of the values in{a,a+ 1 ,...,b}with probabilityb−^1 a+ 1.
Example (Binomial random variable) A Binomial random
variableXwith parametersn(natural number) and 0≤p≤ 1
(X∼Bin(n,p)) takes values in the set{ 0 , 1 ,...,n}with
probabilitiespX(i)=nipi( 1 −p)n−i.
It represents the number of successes innindependent trials where
each trial has a probability of successp. Therefore, it can also be
seen as the sum ofnindependent Bernoulli random variables, each
with parameterp.
Example (Geometric random variable) A Geometric random
variableXwith parameter 0≤p≤1 (X∼Geo(p)) takes values in
the set{ 1 , 2 ,...}with probabilitiespX(i)=( 1 −p)i−^1 p.
It represents the number of independent trials until (and including)
the first success, when the probability of success in each trial isp.
Definition (Expectation/mean of a random variable) The
expectation of a discrete random variable is defined as
```
```
E[X]△=Q
x
```
```
xpX(x).
```
```
assuming∑xSxSpX(x)<∞.
Properties (Properties of expectation)
```
- IfX≥0 thenE[X]≥0.
- Ifa≤X≤bthena≤E[X]≤b.
- IfX=cthenE[X]=c.
Example Expected value of know r.v.
- IfX∼Ber(p)thenE[X]=p.
- IfX=IAthenE[X]=P(A).
- IfX∼Uni[a,b]thenE[X]=a+ 2 b.
- IfX∼Bin(n,p)thenE[X]=np.
- IfX∼Geo(p)thenE[X]=^1 p.


