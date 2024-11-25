# Adapted from https://github.com/google-deepmind/mctx
# Batched tree data structure to hold search data

from __future__ import annotations
from typing import Any, ClassVar, Generic, TypeVar

import chex
import jax
import jax.numpy as jnp


T = TypeVar("T")


@chex.dataclass(frozen=True,kw_only=True)
class Tree(Generic[T]):
  """State of a search tree.

  The `Tree` dataclass is used to hold and inspect search data for a batch of inputs. In the fields below: 
    `B` denotes the batch dimension, 
    `N` represents the current number of decision nodes in the tree (maximum across batch), 
    `M` represents the current number of chance nodes in the tree (maximum across batch),
    `A` is the number of discrete actions, 
    `O` is the maximum number of discrete outcomes, 
    `E` is size of all embeddings.

  TODO: double check below paragraph is correct

  I've assumed all embeddings are of 1D size `E`, but they should be able to be arbitrary number of dimensions, and it 
  is assumed that each action is an integer in the range [0,1,...,A-1].

  For the THTX adaptation, going to split data values into decision and chance node variables. The a'th child for 
  dnode at index [b,i] is [b,i,a]. From reading `search.py` it seems that there is some way to use mctx in stochastic 
  domains, but it doesn't seem intuitive/obvious to me, and I want to turn this into THTS style anyway.
  
  Additionally, because we're removing the `mcts_mode` from search, each tree may be of a different size, so we need to
  keep 

  TODO: update c_node_children_outcomes -> something including embeddings

  num_d_nodes: `[B]` the number of decision nodes in each tree of the batch
  num_c_nodes: `[B]` the number of chance nodes in each tree of the batch 

  d_node_state_embedding: `[B,N,E]` the state embeddings for each decision node.
  d_node_visits: `[B,N]` the visit counts for each decision node.
  d_node_raw_value: `[B,N]` the raw (V) value for each decision node. (I.e. output from neural net).
  d_node_value: `[B,N]` the search (V) value for each decision node. (I.e. value updating in search).
    TODO: d_node_value could be `[B,N,K]` if want K search values. (E.g. if want a UCT value for search and an advantage estimate).
  d_node_prior_policy_logits: `[B,M]` logit for the prior (search) policy taking the action corresponding to each 
    chance node.
  d_node_parent_index: `[B,N]` the node index for the parent (chance nodes) for each decision node. That is, the parent 
    of a decision node at index [b,i] has index [b,j] where j = d_node_parent[b,i].
  d_node_children_index: `[B,N,A]` the node index for the children (chance nodes) for each decision node for each 
    action. That is, the child of decision node at index [b,i] from taking action a has index [b,j] where 
    j = d_node_children_index[b,i,a].

  c_node_action: `[B,M]` the action corresponding to each chance node.
  c_node_visits: `[B,M]` the visit counts for each chance node.
  c_node_reward: `[B,M]` the immediate reward corresponding to this chance node.
  c_node_search_policy_logit: `[B,M]` logit for the prior (search) policy taking the action corresponding to each 
    chance node.
  c_node_raw_value: `[B,M]` the raw (Q) value for each chance node. (I.e. output from neural net).
  c_node_value: `[B,M]` the search (Q) value for each chance node. (I.e. value updating in search).
    TODO: c_node_value could be `[B,M,K]` if want K search values. (E.g. if want a UCT value for search and an advantage estimate).
  c_node_discount: `[B,M]` the discount between the current `c_node_reward` and future values from `c_node_value`. 
  c_node_parent_index: `[B,M]` the node index for the parent (decision nodes) for each chance node. That is, the parent 
    of a chance node at index [b,j] has index [b,i] where i = c_node_parent[b,j].
  c_node_children_index: `[B,M,O]` the node index for the children (decision nodes) for each chance node for each 
    outcome. If chance node index is [b,j] and the observation embedding is e ((next) state embedding in fully 
    observable). Then the outcome o is such that c_node_children_outcomes[b,j,o] == e. The index of the child decision 
    node is then [b,i] where i = c_node_children_index[b,j,o].
  c_node_children_outcomes: `[B,M,O,E]` the observation embedding ((next) state embedding if fully observable) 
    corresponding to each outcome o. If the outcome (embedding) observerd is e, at chance node [b,j], then the outcome 
    is o such that c_node_children_outcomes[b,j,o] == e. 

  TODO: work out if below are necessary?
  root_invalid_actions: `[B,A]` a mask with invalid actions at the root. In the mask, invalid actions have ones, and 
    valid actions have zeros.
  extra_data: `[B, ...]` extra data passed to the search.
  """
  num_trials: chex.Array                        # [B,1] (really just 1d, but duplicated for batching)

  num_d_nodes: chex.Array                       # [B,1]
  num_c_nodes: chex.Array                       # [B,1]

  d_node_state_embedding: chex.Array            # [B,N,E]
  d_node_visits: chex.Array                     # [B,N]
  d_node_raw_value: chex.Array                  # [B,N]
  d_node_value: chex.Array                      # [B,N]
  d_node_prior_policy_logits: chex.Array        # [B,N,A]
  d_node_parent_index: chex.Array               # [B,N]
  d_node_children_index: chex.Array             # [B,N,A]

  c_node_action: chex.Array                     # [B,M]
  c_node_visits: chex.Array                     # [B,M]
  c_node_reward: chex.Array                     # [B,M]
  c_node_raw_value: chex.Array                  # [B,M]
  c_node_value: chex.Array                      # [B,M]
  c_node_discount: chex.Array                   # [B,M]
  c_node_children_index: chex.Array             # [B,M,O]
  c_node_children_outcomes: chex.Array          # [B,M,O,E]
  
  root_invalid_actions: chex.Array              # [B,A]
  extra_data: chex.Array                        # [B,...]

  # The following attributes are class variables (and should not be set on
  # Tree instances).
  ROOT_INDEX: ClassVar[int] = 0
  NO_PARENT: ClassVar[int] = -1
  UNVISITED: ClassVar[int] = -1

  @property
  def num_actions(self):
    return self.d_node_children_index.shape[-1]
  
  @property
  def num_d_nodes_allocated(self):
    return self.d_node_visits.shape[1]
  
  @property
  def num_c_nodes_allocated(self):
    return self.c_node_visits.shape[1]

  def qvalues(self, dnode_indices):
    """Compute q-values for any node dnode indices in the tree."""
    # pytype: disable=wrong-arg-types  # jnp-type
    if jnp.asarray(dnode_indices).shape:
      return jax.vmap(_unbatched_qvalues)(self, dnode_indices)
    else:
      return _unbatched_qvalues(self, dnode_indices)
    # pytype: enable=wrong-arg-types

  # TODO: summary stuff are like the data extractors want to do
  def summary(self) -> SearchSummary:
    """Extract summary statistics for the root node."""
    # Get state and action values for the root nodes.
    chex.assert_rank(self.d_node_value, 2)
    value = self.d_node_value[:, Tree.ROOT_INDEX]
    batch_size, = value.shape[0]
    root_indices = jnp.full((batch_size,), Tree.ROOT_INDEX)
    qvalues = self.qvalues(root_indices)
    # Extract visit counts and induced probabilities for the root nodes.
    children_indices = self.d_node_children_index[:, Tree.ROOT_INDEX]
    visit_counts = jnp.take_along_axis(self.c_node_visits, children_indices, axis=1)
    total_counts = jnp.sum(visit_counts, axis=-1, keepdims=True) 
    visit_probs = visit_counts / jnp.maximum(total_counts, 1) 
    visit_probs = jnp.where(total_counts > 0, visit_probs, 1 / self.num_actions) 
    # Return relevant stats.
    return SearchSummary(  # pytype: disable=wrong-arg-types  # numpy-scalars
        visit_counts=visit_counts,
        visit_probs=visit_probs,
        value=value,
        qvalues=qvalues)


def infer_batch_size(tree: Tree) -> int:
  """Recovers batch size from `Tree` data structure."""
  if tree.num_d_nodes.ndim != 2:
    raise ValueError("Input tree is not batched.")
  chex.assert_equal_shape_prefix(jax.tree_util.tree_leaves(tree), 1)
  return tree.num_d_nodes.shape[0]


# A number of aggregate statistics and predictions are extracted from the
# search data and returned to the user for further processing.
@chex.dataclass(frozen=True)
class SearchSummary:
  """Stats from MCTS search."""
  visit_counts: chex.Array
  visit_probs: chex.Array
  value: chex.Array
  qvalues: chex.Array


# N.B. unbatched means no `B` dimension
def _unbatched_qvalues(tree: Tree, dnode_index: int) -> int:
  chex.assert_rank(tree.children_discounts, 2)
  children_indices = tree.d_node_children_index[dnode_index]
  rewards = tree.c_node_reward[children_indices]
  discounts = tree.c_node_discount[children_indices]
  children_values = tree.c_node_value[children_indices]
  return (  # pytype: disable=bad-return-type  # numpy-scalars
      rewards + discounts * children_values
  )