Attack definitions
==================

We define here formally re-identification and attribute inference attacks.

## Definitions
- Let $\mathcal{A} = \{a_1, \ldots, a_k\}$ be the set of attributes, and let ${\sf dom}(a_i)$ be the domain of attribute $a_i$.
- A dataset $D = \{r_1, \ldots, r_n\}$ is a set of rows where $r_i \in {\sf dom}(a_1) \times \ldots \times {\sf dom}(a_k)$ is a tuple of $k$ values and represents one individual in the population.
- Let $\mathcal{S} \subseteq \mathcal{A}$ be a subset of attributes. We say that $r_i[\mathcal{S}]$ are the attribute values of the individual $r_i$ for attributes in $\mathcal{S}$, e.g., $r_2[\{\text{age}, \text{education}\}] = (23, Bachelor)$.

Assumptions about the adversary:
  - She knows $D$, $\mathcal{A}$ and all ${\sf dom}(a_i)$.
  - She has a single target $r_t \in D$ (random person from the population) and knows some quasi-identifiers (QIDs) about the target, i.e., there is a subset of attributes $\mathcal{Q} \subseteq \mathcal{A}$ and she knows $r_t[\mathcal{Q}]$.
  - For re-identification: The adversary is trying to find out which row correspond to her target.
  - For attribute-inference: The adversary is trying to infer the value of a sensitive attribute $a_s \in \mathcal{A}$.

---
## Example dataset
| age | education   | income   |
|-----|-------------|----------|
| 20  | Master      | low     |
| 30  | High School | medium|
| 30  | High School | low     |
| 30  | PhD         | medium|
| 30  | PhD         | medium|
| 55  | Bachelor    | high     |
| 55  | Bachelor    | high     |
| 55  | Bachelor    | medium|

---
## Re-identification

**Prior vulnerability:** The adversary's chance of re-identifying a random target before observing the data.

Before learning the QIDs of her target the best the adversary can do is to guess one of the $n$ rows, thus her expected probability of guessing correctly the row is

$\begin{equation}
1/n \text{ .}\nonumber
\end{equation}$

In the example the prior vulnerability is $1/ 8$.

**Posterior vulnerability:** The adversary's chance of re-identifying a random target after observing the data.

After the adversary learned $r_t[\mathcal{Q}]$, she will filter all records $\{r_i~:~r_i[\mathcal{Q}] = r_t[\mathcal{Q}]\}$ and from this subset the best she can do is guessing one of the rows. Her expected chance of success is

$\begin{equation}
\frac{1}{n} \sum\limits_{r_t \in D} \frac{1}{|~\{r_i~:~r_i[\mathcal{Q}] = r_t[\mathcal{Q}]\}~|}\nonumber
\end{equation}$

In the example, assume the adversary knows the age and education of her target. So the possible targets are:
- (20, Master)
- (30, High School)
- (30, PhD)
- (55, Bachelor)

The posterior is then $\Large \frac{1 + 2\cdot \frac{1}{2} + 2\cdot \frac{1}{2} + 3\cdot \frac{1}{3}}{8} = \frac{4}{8} = \frac{1}{2}$

---
## Attribute Inference

Assume the sensitive attribute is *income*.

**Prior vulnerability:** The adversary's chance of guessing correctly the sensitive attribute value of a random target before observing the data.

Before learning the QIDs of her target the best she can do is guessing the most frequent attribute value in the dataset, thus her expected probabilty of success is

$\begin{equation}
\max\limits_{v \in \mathcal{D}(a_s)} \frac{|r_i~:~r_i[a_s] = v|}{n}\text{ .}\nonumber
\end{equation}$

In the example, the most frequent income is "medium" that appears in 4 records, so the prior vulnerability will be $4/8 = 1/2$.

**Posterior vulnerability:** The adversary's chance of guessing correctly the sensitive attribute value of a random target after observing the data.

After the adversary learned $r_t[\mathcal{Q}]$, she will filter all records $\{r_i:~r_i[\mathcal{Q}] = r_t[\mathcal{Q}]\}$ and from this subset the best she can do is to guess the attribute value with the highest frequency in the subset. Her expected chance of sucess is

$\begin{equation}
\frac{1}{n} \sum\limits_{r_t \in D} \max\limits_{v \in \mathcal{D}(a_s)} \frac{|~\{r_i~:~r_i[\mathcal{Q}] = r_t[\mathcal{Q}]\} \wedge r_i[a_s] = v~|}{|~\{r_i~:~r_i[\mathcal{Q}] = r_t[\mathcal{Q}]\}~|}
\end{equation}$

In the example the posterior vulnerability will be $\Large\frac{1 + 2\cdot \frac{1}{2} + 2\cdot 1 + 3\cdot \frac{2}{3}}{8} = \frac{6}{8} = \frac{3}{4}$.

