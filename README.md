<h1>Ranking metrics implementation</h1>

<h4>Some examples of how listed metrics work and code to implement them.</h4>

<ul>
<li> **Precision@k** -  Precision at k is the proportion of recommended items in the top-k set that are relevant.</li>

<li> **Recall@k** - Recall at k is the proportion of relevant items found in the top-k recommendations.</li>

<li> **Specificity@k** - Specificity at k is the proportion of non relevant items than our system didn't recommend.</li>

<li> **f1@k** - f1 at k is a harmonic mean of precision and recall at k items.</li>

<li> **cg@k** - Cumulative Gain is the sum of the graded relevance values of all results in a search result list.</li>

<li> **dcg@k** - The premise of DCG is that highly relevant documents appearing lower in a search result list should be penalized as the graded relevance value is reduced logarithmically proportional to the position of the result.</li>

<li> **ndcg@k** - The metric that can show us how good our dcg comparing with ideal dcg.
</ul>


*Have a good day :)*