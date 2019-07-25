# social-relationship based reviewer assignment method

## introduction

With the development of the number of articles journals and conferences having been received. Assigning proper reviewers to review them became a heavy work for editors. Many method have been proposed to help solve this problem, but almost all of them focus on the professional fitness between reviewers and paper and ignore the social relationship between authors of the paper and experts who were allocated to review it. However, this could have some serious influence on the fairness of the review result of the atricle. Hence, we try to design an algorithm to automatically assign the most appropriate reviewers based on both the professional fitness, their social relationship with the authors and capability limit, i.e. the maximum number of papers the reviewer can check.

## problem definition

<!-- so what kind of dataset shall we have?
	
	First we need the social relationship network that concludes all reviewers and, if possible, authors (becasue some authors who are publishing their first paper may have no relation with others before) 
	
	Second we need to know the profession skills of each reviewer, this could ensure us to give them the paper that solving questions that are most relate to their research interest
	
	At last we need to know the upperboud of the number of paper every reviewer could read-->

Given the set of paper $P$ and the set of reviewers $R$, our goal is to assign every article $p \in P$ to a certain number of reviewers in $R$ that satisfied:

1. every paper should be assigned to a certain number ($k$) of reviewers;
2. every reviewer should not be assigned papers more than upper limit of their willing;
3. the research area of reviewers and research topic of aritcles should be as similar as possible;
4. the social relationship among authors and reivewers should be as weak as possible.

If we use a matrix $M \in R^{n\times m}$ to denote the allocation scheme, where $n=|P|$ is the total number of atricles and $m=|R|$ is the number of reviewers. For every element $m_{ij}$ in $M$, if $m_{ij}=1$ means reviewer $j$ is assigned to atricle $i$, otherwise $m_{ij}=0$. $M^i$ is the $i$-th row of $M$ and $M_j$ is the $j$-th column of $M$. Then we should ensure that

1. $\forall i>0 \And i \le n,|M^i|=k$, and
2. $\forall j>0 \And j \le m,|M_j| \le l_j$,

where $l_j$ is the upper limit of reviewer $j$'s willing.

Besides, all authors and reviewers could construct a network ($G$) consisting their social relationship. This network could be built according to several principles. For example, we could use a vertex to denote an author or a reviewer, and an edge to illustrate that there exists co-author relationship between them, or we could build the network based on their following-follower relationship in social media like twitter.

Based on the social network, we could measure the distance of their social relationships. Every article in $P$ is related to several authors $A_{p_i}\rightarrow \{ a_{i1},a_{i2},\dotsc,a_{il}\}$, where $a_{ik}$ is the $k$-th author of paper $i$. And we can get the reviewers of this paper from $M^i$, which is $b_i \rightarrow \{m_{ij} \in M^i|m_{ij}=1\}$. Then authors and reviewers of this article could form a subgraph of the social network $G$. Then we could use k-density to measure the structural tenuous of this subgraph.

$$
k-density(F)=\frac{\sum_{u\in V_F}|N_G^k(u)\cap V_F|}{|V_F|(|V_F|-1)}
$$
, where $V_F=\{v|v \in A_{p_i} \cup b_i\}$.

and we use a simple similarity measure to measure the similarity between a reviewer's professionism in the face of an article.

$$
w(p_i,r_j) = \frac{|K^p_i\cap K^r_j|}{|K^p_i|}
$$
, where $K^p_i=\{k_{i1},k_{i2},\dotsb,k_{iu}\}$ and $k^r_j=\{k_{j1},k_{j2},\dotsb,k_{jt}\}$ are the keywords set of paper $i$ and reviewer $j$, seperately.

<!-- what is the input and output of this method
	Input: P,R,(G?),keyword set K, parameter k
	Output: Allocation scheme M -->
