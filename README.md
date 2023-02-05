# Hidden-Markov-Models

 # General information
 A Hidden Markov Model (HMM) is a statistical model used to represent a sequence of observations generated from an underlying, unobservable (hidden) state. It is a type of probabilistic graphical model where the hidden states sequence is related to the observed data through emission probabilities.


- $\mathrm{A}=$ transition matrix ( Hidden in HMM)

- $B=$ emission matrix ( observed variables) (Visible in HMM)

- $\mathrm{HMM}=$ hidden $\mathrm{MC}+$ observed variables

- $\mathrm{O}=$ sequence of emissions

- $\pi=$ probability of the states (probability distribution of the states)

- $\pi_{0} A=$ future probability states from current state $=>\pi A=\mathrm{A}$

**Markov chain**

- **Reducible:** Markov chain includes states not reachable by some states

- **Irreducible:** not reducible

- **Transient state:** can't be reached by other states, less than 1

- **Recurrence state:** Bound to revisit state, equal to 1

- Probability to reach state $\mathrm{i}->\mathrm{j}$ in $\mathrm{n}$ steps is calculated by: $A_{i j}^{n}$

$$
P_{i j}(\mathrm{n})=A_{i j}^{n}
$$

**Forward:**

- Gives the probability of an observed sequence $(\mathrm{O})$ when given the parameters: $\{A, B, \pi\}$.

- Naive solution would have complexity $2 T N^{T}$

- Using $\alpha$ we can reduce the complexity to $T N^{2}$

- We have to start $\alpha$-pass by calculating base case:

$$
\alpha_{1}(i)=P\left(O_{1}=o_{1}, X_{1}=x_{i}\right)=b_{i}\left(o_{1}\right) \pi_{1}
$$

- After we calculate $\alpha_{1}$ we can now calculate for all the hidden states as followed:

$$
\alpha_{t}(i)=P\left(O_{1: t}=o_{1: t^{\prime}} X_{t}=x_{i}\right)=b_{i}\left(o_{t}\right)\left[\sum_{j=1}^{N} a_{j, i} \alpha_{t-1}(j)\right]
$$

- Now we can calculate the probability of having observed the given observation sequence $O_{1: T}$ where $\mathrm{N}$ is the number of hidden states.

$$
P\left(O_{1: T}=o_{1: T}\right)=\sum_{j=1}^{N} \alpha_{T}(j)
$$



**Backward:**

- $\beta$ represent how the present relates to the future

- $\beta_{t}(i)$ is the probability that the model is in the hidden state $x_{t}(i)$ (i in $[1,2, \ldots \mathrm{N}])$

$$
\beta_{t}(i)=p\left(O_{t+1: T} \mid X_{t}=i \lambda\right)
$$

- First it has to be initialized

$$
\beta_{T}(i)=1, \forall i=1, \ldots, N
$$

- Now to compute on how the present relates to the future:

$$
\beta_{t}(i)=\sum_{j=1}^{N} a_{j, i} b_{j}\left(O_{t+1}\right) \beta_{t+1}(j)
$$

**Viterbir:**

- Computes the most likely sequence of hidden states that give the observation.

- We start to initialize $\delta_{1}(i)$

$$
\delta_{1}(i)=b_{i}\left(o_{1}\right) \pi_{i} \text { where } \mathrm{i}=1 \ldots \mathrm{N}
$$

- $\delta_{t}(i)$ also known as delta

o For each $\mathrm{t}>1$ we always keep the max

$$
\delta_{t}(i)=\max \left[\delta_{t-1}(j) a_{i j} b_{i}\left(o_{t}\right)\right] \text { where } j \in\{1, \ldots ., N\}
$$

- $\delta_{t}^{i d x}(i)$ also known as deltalndex

$$
\delta_{t}^{i d x}(i)=\operatorname{argmax}_{j \in[1, . N]} a_{j, i} \delta_{t-1}(j) b_{i}\left(o_{t}\right) \text { for } i \in[1, . . N]
$$

**Baum-Welch:**

- By giving an observation we want to improve the parameters. By tuning the parameters we then get a better model that better represents the observed data.

- **4 steps in Alg**:

o Initialize $\lambda=(A, B, \pi)$

o Compute: $\alpha_{t}(i), \beta_{t}(k), \gamma_{t}(i, j), \gamma_{t}(i)$

o Re-estimate the model $\lambda=(A, B, \pi)$ using the values from step 2

o Repeat from step 2 until $P(O \mid \lambda)$ converges to find a local optimum - Scaled Baum-Welch

- Scaling is usually necessary in the implementation of Baum-Welch re-estimation process. It's to tackle the problem when terms are significantly less than one. The solution is that at each induction step of the forward algorithm, we need to scale all $\alpha(i)$ appropriately. This scaling factor should only depend on the current time index $t$, but be independent of the state i.

- **Gamma & Di-Gamma** 

- With Di-Gamma and Gamma we can together find out if given an observation sequence and the current estimate of $\mathrm{HMM}$ what is the probability at time $t$ that the hidden state is $i$ or what is the probability at time t we will transition from state $i$ to $j$. All while knowing what came before and after.

o Gamma

$$
\gamma_{t}(i)=\sum_{j=1}^{N} \gamma_{t}(i, j)=p\left(X_{t}=i \mid O_{t+T}, \lambda\right)
$$

o Di-Gamma

- We are interested in the probability of transition from i $->\mathrm{j}$ given that we know all past states of $i$ and future states of $j$.

$$
\gamma_{t}(i, j)=\frac{\alpha_{t}(i) a_{i j} b_{j}\left(O_{t+1}\right) \beta_{t+1}(j)}{\sum_{i=1}^{N} \alpha_{T}(i)}=p\left(X_{t}=i, X_{t+1}=j \mid O_{t+T}, \lambda\right)
$$

- Re-estimate

o Initial estimate

$$
\pi_{t}=\gamma_{t}(i) \quad \forall i=1, \ldots, N
$$

o Transition estimate

$$
a_{i j}=\frac{\sum_{t=1}^{T-1} \gamma_{t}(i, j)}{\sum_{t=1}^{T-1} \gamma_{t}(i)} \forall i, j=1, \ldots, N
$$

o Emission estimate

$$
b_{j}(k)=\frac{\sum_{t=1}^{T-1} 1\left(O_{t}=k\right) Y_{t}(j)}{\sum_{t=1}^{T-1} \gamma_{t}(j)} \forall j=1, \ldots, N, k=1, \ldots, K
$$



**Lab 2 - HMM**

**Question 1 This problem can be formulated in matrix form. Please specify the initial probability vector π, the transition probability matrix A and the observation probability matrix B.**

    pi = [ 0.5 0.5 ]

    A = [ 0.5 0.5 ]
        [ 0.5 0.5 ]    
     
    B = [ 0.9 0.1 ]
        [ 0.5 0.5 ]       



**Question 2 What is the result of this operation?**

    result = [ 0.5 0.5 ]

**Question 3 What is the result of this operation?**

    [ 0.5 0.5 ]  [ 0.9 0.1 ]   = [ 0.7 0.3 ] 
                 [ 0.5 0.5 ]  
        

**Question 4 Why is it valid to substitute $01: t=01: t$ with Ot $=$ ot when we condition on the state $\mathbf{X t}=\mathbf{x i}$ ?**

In order to get the prob of Xt we substitute O1:t with Ot and as a result transfer the O1:t-1 to Xt. The reason for its validity is due to the fact that we base their probability off the most probable hidden state.

It's because the observations are conditionally independent from each other. So if we need any information of a previous observation they exist in xi.

**Question 5 How many values are stored in the matrices $\delta$ and $\delta^{\wedge}$ idx respectively?**

Here, the prob. of each state at each time step is stored in the $\delta$ matrix. Hence, every possible time step t multiplied by its corresponding values; one value per state. This means $\mathrm{T} \times \mathrm{N}$ values for the $\delta$ matrix (where $T$ accounts for all possible time steps).

As for the second matrix $\delta^{\wedge} i d x$ this one saves the most probable hidden state at a specific time step t-1.

Overall we get:

$\delta: ~ T \times N$

$\delta^{\wedge} i d x:(T-1) \times N$ 

**Question 6 Why we do we need to divide by the sum over the final $\alpha$ values for the di-gamma function?**

When we are looking for a probability value we have to consider a normalizing factor. Which in this case is the sum of all possible states. We get to this formula through bayes theorem. So the only clarification needed is to understand that: $P\left(O_{1: T}=o_{1: T}\right)=\sum_{k=1}^{N} \alpha_{T}(k)$
