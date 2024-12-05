# Adapted from https://github.com/google-deepmind/mctx 
"""A JAX implementation of batched MCTS."""
import functools
from typing import Any, NamedTuple, Optional, Tuple, TypeVar

import chex
import jax
import jax.numpy as jnp

# from thtx._src import action_selection
from thtx._src import base
from thtx._src import tree as tree_lib
from thtx._src import util

Tree = tree_lib.Tree
T = TypeVar("T")


def create_tree_from_root_init(
    root_init: base.RootInitFnOutput,
    num_trials: int,
    init_tree_size: int,
    root_invalid_actions: Optional[chex.Array] = None,
    extra_data: Optional[Any] = None) -> Tree:
  """
  Wrapper to contain non jitable code

  TODO: docstring from splitting this function up
  """
  if root_invalid_actions is None:
    root_invalid_actions = jnp.zeros_like(root_init.prior_policy_logits)
  return _create_tree_from_root_init(root_init, num_trials, init_tree_size, root_invalid_actions, extra_data)

@functools.partial(jax.jit,static_argnums=(2,))
def _create_tree_from_root_init(
    root_init: base.RootInitFnOutput,
    num_trials: int,
    init_tree_size: int,
    root_invalid_actions: chex.Array,
    extra_data: Any) -> Tree:
  """Create a search tree and initialise state at root node.

  TODO: for now, just setting num_outcomes to 1

  Batched calls only.
  
  Args:
    root_values: values to use to initialise the root node
    num_trials: the number of trials that will be run
    init_tree_size: the number of (decision and chance) nodes to initialise in the tree
    root_invalid_actions: a mask of actions that are invalid at the root node
    extra_data: any extra data to store in the tree
  
  Returns:
    A new search tree
  """
  chex.assert_rank(root_init.prior_policy_logits, 2)
  batch_size, num_actions = root_init.prior_policy_logits.shape
  chex.assert_shape(root_init.value, [batch_size])
  num_nodes = init_tree_size
  data_dtype = root_init.value.dtype
  batch_one_shape = (batch_size, 1)
  batch_node_shape = (batch_size, num_nodes)
  batch_node_action_shape = (batch_size, num_nodes, num_actions)
  num_outcomes = 1
  batch_node_outcome_shape = (batch_size, num_nodes, num_outcomes)

  # if root_invalid_actions is None:
  #   root_invalid_actions = jnp.zeros_like(root_init.prior_policy_logits)

  def _embedding_zeros(root_node_embedding, shape_prefix):
    return jnp.zeros(shape_prefix + root_node_embedding.shape[1:], dtype=root_node_embedding.dtype)

  tree = Tree(
    num_trials=jnp.full(batch_one_shape, num_trials, dtype=jnp.int32),
    num_d_nodes=jnp.ones(batch_one_shape, dtype=jnp.int32),
    num_c_nodes=jnp.zeros(batch_one_shape, dtype=jnp.int32),
    d_node_state_embeddings=jax.tree.map(
      _embedding_zeros, 
      root_init.state_embeddings, 
      util.pytree_broadcast(batch_node_shape, like_pytree=root_init.state_embeddings)),
    d_node_visits=jnp.zeros(batch_node_shape, dtype=jnp.int32),
    d_node_raw_value=jnp.zeros(batch_node_shape, dtype=data_dtype),
    d_node_value=jnp.zeros(batch_node_shape, dtype=data_dtype),
    d_node_prior_policy_logits=jnp.zeros(batch_node_action_shape, dtype=root_init.prior_policy_logits.dtype),
    d_node_parent_index=jnp.full(batch_node_shape, Tree.NO_PARENT, dtype=jnp.int32),
    d_node_children_index=jnp.full(batch_node_action_shape, Tree.UNVISITED, dtype=jnp.int32),
    c_node_action=jnp.zeros(batch_node_shape, dtype=jnp.int32),
    c_node_visits=jnp.zeros(batch_node_shape, dtype=jnp.int32),
    c_node_reward=jnp.zeros(batch_node_shape, dtype=data_dtype),
    c_node_raw_value=jnp.zeros(batch_node_shape, dtype=data_dtype),
    c_node_value=jnp.zeros(batch_node_shape, dtype=data_dtype),
    c_node_discount=jnp.zeros(batch_node_shape, dtype=data_dtype),
    c_node_children_index=jnp.full(batch_node_outcome_shape, Tree.UNVISITED, dtype=jnp.int32),
    c_node_children_outcome_embeddings=jax.tree.map(
      _embedding_zeros, 
      root_init.state_embeddings, 
      util.pytree_broadcast(batch_node_outcome_shape, like_pytree=root_init.state_embeddings)),
    root_invalid_actions=root_invalid_actions,
    extra_data=extra_data
  )

  root_index = jnp.full([batch_size], Tree.ROOT_INDEX)
  tree = initialise_d_node_values(tree, root_index, root_init.prior_policy_logits, root_init.value, root_init.state_embeddings)
  return tree


@jax.jit
def initialise_d_node_values(
    tree: Tree[T],
    node_index: chex.Array,
    prior_policy_logits: chex.Array,
    value: chex.Array,
    state_embeddings: chex.Array) -> Tree[T]:
  """Initialises decision nodes at 'node_index'.

  Batched calls only

  Args:
    tree: `Tree` to whose node is to be updated.
    node_index: the index of the decision node to update. Shape `[B]`.
    prior_policy_logits: the prior policy logits to fill in for the new node, of shape `[B, A]`.
    raw_value: the raw value to fill in for the decision node. Shape `[B]`.
    embedding: the state embeddings for the node. Shape `[B, ...]`.

  Returns:
    The new tree with updated nodes.
  """
  batch_size = tree_lib.infer_batch_size(tree)
  batch_range = jnp.arange(batch_size)
  chex.assert_shape(prior_policy_logits, (batch_size, tree.num_actions))

  # if at max depth, then may try to 'initialise' a node multiple times
  new_visits = tree.d_node_visits[batch_range, node_index] + 1
  updates = dict( 
      d_node_state_embeddings=jax.tree.map(
          lambda t, s: util.batch_update(t, s, node_index),
          tree.d_node_state_embeddings, state_embeddings),
      d_node_visits=util.batch_update(tree.d_node_visits, new_visits, node_index),
      d_node_raw_value=util.batch_update(tree.d_node_raw_value, value, node_index),
      d_node_value=util.batch_update(tree.d_node_value, value, node_index),
      d_node_prior_policy_logits=util.batch_update(tree.d_node_prior_policy_logits, prior_policy_logits, node_index),
  )

  return tree.replace(**updates)



# @functools.partial(jax.jit,static_argnums=(1,2))
def extend_tree(
    tree: Tree[T],
    num_extra_d_nodes: int,
    num_extra_c_nodes: int) -> Tree[T]:
  """Extends 'tree' with extra decision and chance nodes

  Batched calls only

  TODO: for now, just setting num_outcomes to 1
  
  Args:
    tree: The tree to add more nodes to
    num_extra_d_nodes: The number of decision nodes to add
    extra_c_nodes: The number of chance nodes to add
  
  Returns:
    The tree with additional decision and chance nodes
  """
  batch_size = tree_lib.infer_batch_size(tree)
  num_actions = tree.num_actions
  data_dtype = tree.d_node_value.dtype
  logits_dtype = tree.d_node_prior_policy_logits.dtype
  batch_dnode_shape = (batch_size, num_extra_d_nodes)
  batch_cnode_shape = (batch_size, num_extra_c_nodes)
  batch_dnode_action_shape = (batch_size, num_extra_d_nodes, num_actions)
  num_outcomes = 1
  batch_cnode_outcome_shape = (batch_size, num_extra_c_nodes, num_outcomes)

  def _get_root_embedding(_batched_embedding):
    return _batched_embedding[:,Tree.ROOT_INDEX]
  
  root_node_state_embeddings = jax.tree.map(_get_root_embedding, tree.d_node_state_embeddings)

  def _embedding_zeros(_batched_root_embedding, shape_prefix):
    return jnp.zeros(shape_prefix + _batched_root_embedding.shape[1:], dtype=_batched_root_embedding.dtype)
  
  xtra_d_node_state_embeddings = jax.tree.map(
    _embedding_zeros, 
    root_node_state_embeddings, 
    util.pytree_broadcast(batch_dnode_shape, like_pytree=root_node_state_embeddings))
  xtra_d_node_visits = jnp.zeros(batch_dnode_shape, dtype=jnp.int32)
  xtra_d_node_raw_value = jnp.zeros(batch_dnode_shape, dtype=data_dtype)
  xtra_d_node_value = jnp.zeros(batch_dnode_shape, dtype=data_dtype)
  xtra_d_node_prior_policy_logits = jnp.zeros(batch_dnode_action_shape, dtype=logits_dtype)
  xtra_d_node_parent_index = jnp.full(batch_dnode_shape, Tree.NO_PARENT, dtype=jnp.int32)
  xtra_d_node_children_index = jnp.full(batch_dnode_action_shape, Tree.UNVISITED, dtype=jnp.int32)
  xtra_c_node_action = jnp.zeros(batch_cnode_shape, dtype=jnp.int32)
  xtra_c_node_visits = jnp.zeros(batch_cnode_shape, dtype=jnp.int32)
  xtra_c_node_reward = jnp.zeros(batch_cnode_shape, dtype=data_dtype)
  xtra_c_node_raw_value = jnp.zeros(batch_cnode_shape, dtype=data_dtype)
  xtra_c_node_value = jnp.zeros(batch_cnode_shape, dtype=data_dtype)
  xtra_c_node_discount = jnp.zeros(batch_cnode_shape, dtype=data_dtype)
  xtra_c_node_children_index = jnp.full(batch_cnode_outcome_shape, Tree.UNVISITED, dtype=jnp.int32)
  xtra_c_node_children_outcome_embeddings = jax.tree.map(
    _embedding_zeros, 
    root_node_state_embeddings, 
    util.pytree_broadcast(batch_cnode_outcome_shape,like_pytree=root_node_state_embeddings))

  updates = dict(
    d_node_state_embeddings=jax.tree.map(
          lambda t, s: jnp.concatenate([t,s], axis=1),
          tree.d_node_state_embeddings, xtra_d_node_state_embeddings),
    d_node_visits=jnp.concatenate([tree.d_node_visits,xtra_d_node_visits], axis=1),
    d_node_raw_value=jnp.concatenate([tree.d_node_raw_value,xtra_d_node_raw_value], axis=1),
    d_node_value=jnp.concatenate([tree.d_node_value,xtra_d_node_value], axis=1),
    d_node_prior_policy_logits=jnp.concatenate(
      [tree.d_node_prior_policy_logits,xtra_d_node_prior_policy_logits], axis=1),
    d_node_parent_index=jnp.concatenate([tree.d_node_parent_index,xtra_d_node_parent_index], axis=1),
    d_node_children_index=jnp.concatenate([tree.d_node_children_index,xtra_d_node_children_index], axis=1),
    c_node_action=jnp.concatenate([tree.c_node_action,xtra_c_node_action], axis=1),
    c_node_visits=jnp.concatenate([tree.c_node_visits,xtra_c_node_visits], axis=1),
    c_node_reward=jnp.concatenate([tree.c_node_reward,xtra_c_node_reward], axis=1),
    c_node_raw_value=jnp.concatenate([tree.c_node_raw_value,xtra_c_node_raw_value], axis=1),
    c_node_value=jnp.concatenate([tree.c_node_value,xtra_c_node_value], axis=1),
    c_node_discount=jnp.concatenate([tree.c_node_discount,xtra_c_node_discount], axis=1),
    c_node_children_index=jnp.concatenate([tree.c_node_children_index,xtra_c_node_children_index], axis=1),
    c_node_children_outcome_embeddings=jax.tree.map(
          lambda t, s: jnp.concatenate([t,s], axis=1),
          tree.c_node_children_outcome_embeddings, xtra_c_node_children_outcome_embeddings),
  )

  return tree.replace(**updates)


# @functools.partial(jax.jit,static_argnums=(0,1,2))
# Cant jit this because need to pass traced arguments to arguments (1,2) of extend_tree, which need to be static
def ensure_space_for_trial(
    tree: Tree[T],
    max_depth: int,
    mcts_mode: bool) -> Tree[T]:
  """Ensures tree has enough space for a trial with max depth 'max_depth'

  If running in mcts_mode, tree doesn't need to be extended ever, as know that there will be at most num_trials+1 many 
  nodes at the end of the search 

  Batched calls only

  NB: arrays can have length 0 in a dimension. So this *should* extend by a length of 0 if no extending needed 
  NBB: need to make 'mcts_mode' a static arg for jax.jit so can use it in conditional statements

  Args:
    tree: current search tree
    max_depth: max depth for each trial
    mcts_mode: are we running in mcts mode

  Returns:
    Updated search tree with enough space for the next trial
  """
  if mcts_mode:
    return tree
  d_nodes_space = tree.num_d_nodes_allocated - jnp.max(tree.num_d_nodes)
  extra_d_nodes_needed = int(jax.lax.max(0, max_depth - d_nodes_space))
  c_nodes_space = tree.num_c_nodes_allocated - jnp.max(tree.num_c_nodes)
  extra_c_nodes_needed = int(jax.lax.max(0, max_depth - c_nodes_space))
  return extend_tree(tree, extra_d_nodes_needed, extra_c_nodes_needed)

T = TypeVar("T")

@chex.dataclass(frozen=True,kw_only=True)
class _TrialBuf:
  """
  Buffer to contain data in selection phase

  'D' is the maximum length of trial

  TODO: write this out properly
  TODO: this is unbatched?
  """
  d_node_indices: chex.Array                # [D+1]
  d_node_state_embeddings: chex.Array       # [D+1,...]
  d_node_visits: chex.Array                 # [D+1]
  d_node_raw_values: chex.Array             # [D+1]
  d_node_values: chex.Array                 # [D+1]
  d_node_prior_policy_logits: chex.Array    # [D+1,A]
  d_node_parent_indices: chex.Array         # [D+1]
  d_node_action_selected: chex.Array       # [D+1] 

  c_node_indices: chex.Array                    # [D]
  c_node_visits: chex.Array                     # [D]
  c_node_rewards: chex.Array                    # [D]
  c_node_raw_values: chex.Array                 # [D]
  c_node_values: chex.Array                     # [D]
  c_node_discounts: chex.Array                  # [D]
  c_node_outcome_selected: chex.Array           # [D]
  c_node_outcome_embeddings: chex.Array         # [D,...]


class _SelectionState(NamedTuple):
  """
  State for the while loop in selection phase.
  
  Unbatched.

  TODO: write proper docstring
  """
  rng_key: chex.PRNGKey
  d_node_index: int
  depth: int
  is_continuing: bool
  trial_buffer: _TrialBuf


# TODO: organise base.ActionSelectionFn into THTX interface, 
# TODO: also want it to include the state embeddings?

# TODO: next = simulate -> selection etc. Wrote some things in the dev notebook

@functools.partial(jax.jit,static_argnums=(1,))
def make_new_trialbuf(
    tree: Tree[T], 
    max_depth: int):
  """
  Creates a new _TrialBuf to store data for a single trial
  Unbatched only calls.

  TODO: write proper docstring with args and return
  """
  # Shapes and dtypes
  num_actions = tree.num_actions
  depth_p1_shape = (max_depth+1,)
  depth_p1_action_shape = (max_depth+1, num_actions)
  depth_shape = (max_depth,)

  # Create arrays for trial buff
  # Note that tree.d_node_state_embeddings.shape == [N,...] here (or pytree with leaves [N,...]), and want the ... bit
  def _embedding_zeros(_d_node_state_embedding, shape_prefix):
    return jnp.zeros(shape_prefix + _d_node_state_embedding.shape[1:], dtype=_d_node_state_embedding.dtype)
  
  d_node_indices = jnp.full(depth_p1_shape, Tree.NULL_INDEX, dtype=jnp.int32)
  d_node_state_embeddings = jax.tree.map(
    _embedding_zeros, 
    tree.d_node_state_embeddings, 
    util.pytree_broadcast(depth_p1_shape,like_pytree=tree.d_node_state_embeddings))
  d_node_visits = jnp.zeros(depth_p1_shape, dtype=jnp.int32)
  d_node_raw_values = jnp.zeros(depth_p1_shape, dtype=tree.d_node_raw_value.dtype)
  d_node_values = jnp.zeros(depth_p1_shape, dtype=tree.d_node_value.dtype)
  d_node_prior_policy_logits = jnp.zeros(depth_p1_action_shape, dtype=tree.d_node_prior_policy_logits.dtype)
  d_node_parent_indices = jnp.zeros(depth_p1_shape, dtype=jnp.int32)
  d_node_action_selected = jnp.zeros(depth_p1_shape, dtype=jnp.int32)
  c_node_indices = jnp.full(depth_shape, Tree.NULL_INDEX, dtype=jnp.int32)
  c_node_visits = jnp.zeros(depth_shape, dtype=jnp.int32)
  c_node_rewards = jnp.zeros(depth_shape, dtype=tree.c_node_reward.dtype)
  c_node_raw_values = jnp.zeros(depth_shape, dtype=tree.c_node_raw_value.dtype)
  c_node_values = jnp.zeros(depth_shape, dtype=tree.c_node_value.dtype)
  c_node_discounts = jnp.zeros(depth_shape, dtype=tree.c_node_discount.dtype)
  c_node_outcome_selected = jnp.zeros(depth_shape, dtype=jnp.int32)
  c_node_outcome_embeddings = jax.tree.map(
    _embedding_zeros, 
    tree.c_node_children_outcome_embeddings, 
    util.pytree_broadcast(depth_shape, like_pytree=tree.c_node_children_outcome_embeddings))

  # Copy values for root node from tree to trialbuf
  def _root_node_index(emb):
    return emb[Tree.ROOT_INDEX]
  def _update_first_d_node_embedding(emb,root_emb):
    return emb.at[0].set(root_emb)
  
  root_node_index = Tree.ROOT_INDEX
  root_node_state_embeddings = jax.tree.map(_root_node_index, tree.d_node_state_embeddings)
  root_node_visits = tree.d_node_visits[Tree.ROOT_INDEX]
  root_node_raw_value = tree.d_node_raw_value[Tree.ROOT_INDEX]
  root_node_value = tree.d_node_value[Tree.ROOT_INDEX]
  root_node_prior_policy_logits = tree.d_node_prior_policy_logits[Tree.ROOT_INDEX]
  root_node_parent_index = Tree.NO_PARENT

  d_node_indices = d_node_indices.at[0].set(root_node_index)
  d_node_state_embeddings = jax.tree.map(
    _update_first_d_node_embedding, d_node_state_embeddings, root_node_state_embeddings)
  d_node_visits = d_node_visits.at[0].set(root_node_visits)
  d_node_raw_values = d_node_raw_values.at[0].set(root_node_raw_value)
  d_node_values = d_node_values.at[0].set(root_node_value)
  d_node_prior_policy_logits = d_node_prior_policy_logits.at[0].set(root_node_prior_policy_logits)
  d_node_parent_indices.at[0].set(root_node_parent_index)

  # Return new trialbuf
  return _TrialBuf(
    d_node_indices=d_node_indices,
    d_node_state_embeddings=d_node_state_embeddings,
    d_node_visits=d_node_visits,
    d_node_raw_values=d_node_raw_values,
    d_node_values=d_node_values,
    d_node_prior_policy_logits=d_node_prior_policy_logits,
    d_node_parent_indices=d_node_parent_indices,
    d_node_action_selected=d_node_action_selected,
    c_node_indices=c_node_indices,
    c_node_visits=c_node_visits,
    c_node_rewards=c_node_rewards,
    c_node_raw_values=c_node_raw_values,
    c_node_values=c_node_values,
    c_node_discounts=c_node_discounts,
    c_node_outcome_selected=c_node_outcome_selected,
    c_node_outcome_embeddings=c_node_outcome_embeddings,
  )


@functools.partial(jax.jit,static_argnums=(2,3,4,5))
@functools.partial(jax.vmap, in_axes=[0,0,None,None,None,None], out_axes=0)
def selection(
  rng_key: chex.PRNGKey,
  tree: Tree,
  action_selection_fn: base.ActionSelectionFn,
  outcome_sample_fn: base.OutcomeSampleFn,
  max_depth: int,
  mcts_mode: bool) -> _TrialBuf:
  """
  Performs selection phase 

  Call batched only, written unbatched using vmap

  TODO: update more of the things in trial buf?
  TODO: play around with this a bit, does everything in loop state have to be a jnp datatype?
  TODO: fix when not mcts, that d_node_index will often be -1
  TODO: generally, would quite like to be able to specify arg names to functions,
  """
  jnp_not_mcts_mode = jnp.array(not mcts_mode)

  def cond_fun(loop_state):
    return loop_state.is_continuing
  
  def body_fun(loop_state):
    d_node_index = loop_state.d_node_index
    depth = loop_state.depth

    rng_key, action_selection_rng_key = jax.random.split(loop_state.rng_key)
    action = action_selection_fn(action_selection_rng_key, tree, d_node_index, loop_state.depth)
    c_node_index = tree.d_node_children_index[d_node_index,action]

    rng_key, outcome_sample_rng_key = jax.random.split(rng_key)
    outcome = outcome_sample_fn(outcome_sample_rng_key, tree, c_node_index, loop_state.depth)
    next_d_node_index = tree.c_node_children_index[c_node_index,outcome]

    is_before_depth_cutoff = (depth < max_depth)
    is_visited = (next_d_node_index != Tree.UNVISITED)
    # is_continuing = is_before_depth_cutoff and (mcts_mode or is_visited)
    is_continuing = jnp.logical_and(is_before_depth_cutoff, jnp.logical_or(jnp_not_mcts_mode, is_visited))

    tb = loop_state.trial_buffer

    return _SelectionState(
      rng_key=rng_key,
      d_node_index=next_d_node_index,
      depth=depth+1,
      is_continuing=is_continuing,
      trial_buffer=tb.replace(
        d_node_indices=tb.d_node_indices.at[depth+1].set(next_d_node_index),
        d_node_action_selected=tb.d_node_action_selected.at[depth].set(action),
        c_node_indices=tb.c_node_indices.at[depth].set(c_node_index),
        c_node_outcome_selected=tb.c_node_outcome_selected.at[depth].set(outcome),
      ),
    )
  
  root_d_node_index = jnp.array(Tree.ROOT_INDEX, dtype=jnp.int32)
  depth = jnp.zeros((), dtype=jnp.int32)
  init_loop_state = _SelectionState(
    rng_key = rng_key,
    d_node_index=root_d_node_index,
    depth=depth,
    is_continuing=jnp.array(True),
    trial_buffer=make_new_trialbuf(tree,max_depth),
  )

  end_loop_state = jax.lax.while_loop(cond_fun, body_fun, init_loop_state)

  return end_loop_state.trial_buffer


































def update_tree_node(
    tree: Tree[T],
    node_index: chex.Array,
    prior_logits: chex.Array,
    value: chex.Array,
    embedding: chex.Array) -> Tree[T]:
  """Updates the tree at node index.

  Args:
    tree: `Tree` to whose node is to be updated.
    node_index: the index of the expanded node. Shape `[B]`.
    prior_logits: the prior logits to fill in for the new node, of shape
      `[B, num_actions]`.
    value: the value to fill in for the new node. Shape `[B]`.
    embedding: the state embeddings for the node. Shape `[B, ...]`.

  Returns:
    The new tree with updated nodes.
  """
  batch_size = tree_lib.infer_batch_size(tree)
  batch_range = jnp.arange(batch_size)
  chex.assert_shape(prior_logits, (batch_size, tree.num_actions))

  # When using max_depth, a leaf can be expanded multiple times.
  new_visit = tree.node_visits[batch_range, node_index] + 1
  updates = dict(  # pylint: disable=use-dict-literal
      children_prior_logits=util.batch_update(
          tree.children_prior_logits, prior_logits, node_index),
      raw_values=util.batch_update(
          tree.raw_values, value, node_index),
      node_values=util.batch_update(
          tree.node_values, value, node_index),
      node_visits=util.batch_update(
          tree.node_visits, new_visit, node_index),
      embeddings=jax.tree.map(
          lambda t, s: util.batch_update(t, s, node_index),
          tree.embeddings, embedding))

  return tree.replace(**updates)





def search(
    params: base.Params,
    rng_key: chex.PRNGKey,
    *,
    root: base.RootFnOutput,
    recurrent_fn: base.RecurrentFn,
    root_action_selection_fn: base.RootActionSelectionFn,
    interior_action_selection_fn: base.InteriorActionSelectionFn,
    num_simulations: int,
    max_depth: Optional[int] = None,
    invalid_actions: Optional[chex.Array] = None,
    extra_data: Any = None) -> Tree:
  """Performs a full search and returns sampled actions.

  In the shape descriptions, `B` denotes the batch dimension.

  Args:
    params: params to be forwarded to root and recurrent functions.
    rng_key: random number generator state, the key is consumed.
    root: a `(prior_logits, value, embedding)` `RootFnOutput`. The
      `prior_logits` are from a policy network. The shapes are
      `([B, num_actions], [B], [B, ...])`, respectively.
    recurrent_fn: a callable to be called on the leaf nodes and unvisited
      actions retrieved by the simulation step, which takes as args
      `(params, rng_key, action, embedding)` and returns a `RecurrentFnOutput`
      and the new state embedding. The `rng_key` argument is consumed.
    root_action_selection_fn: function used to select an action at the root.
    interior_action_selection_fn: function used to select an action during
      simulation.
    num_simulations: the number of simulations.
    max_depth: maximum search tree depth allowed during simulation, defined as
      the number of edges from the root to a leaf node.
    invalid_actions: a mask with invalid actions at the root. In the
      mask, invalid actions have ones, and valid actions have zeros.
      Shape `[B, num_actions]`.
    extra_data: extra data passed to `tree.extra_data`. Shape `[B, ...]`.

  Returns:
    `SearchResults` containing outcomes of the search, e.g. `visit_counts`
    `[B, num_actions]`.
  """
  action_selection_fn = action_selection.switching_action_selection_wrapper(
      root_action_selection_fn=root_action_selection_fn,
      interior_action_selection_fn=interior_action_selection_fn
  )

  # Do simulation, expansion, and backward steps.
  batch_size = root.value.shape[0]
  batch_range = jnp.arange(batch_size)
  if max_depth is None:
    max_depth = num_simulations
  if invalid_actions is None:
    invalid_actions = jnp.zeros_like(root.prior_logits)

  def body_fun(sim, loop_state):
    rng_key, tree = loop_state
    rng_key, simulate_key, expand_key = jax.random.split(rng_key, 3)
    # simulate is vmapped and expects batched rng keys.
    simulate_keys = jax.random.split(simulate_key, batch_size)
    parent_index, action = simulate(
        simulate_keys, tree, action_selection_fn, max_depth)
    # A node first expanded on simulation `i`, will have node index `i`.
    # Node 0 corresponds to the root node.
    next_node_index = tree.children_index[batch_range, parent_index, action]
    next_node_index = jnp.where(next_node_index == Tree.UNVISITED,
                                sim + 1, next_node_index)
    tree = expand(
        params, expand_key, tree, recurrent_fn, parent_index,
        action, next_node_index)
    tree = backward(tree, next_node_index)
    loop_state = rng_key, tree
    return loop_state

  # Allocate all necessary storage.
  tree = instantiate_tree_from_root(root, num_simulations,
                                    root_invalid_actions=invalid_actions,
                                    extra_data=extra_data)
  _, tree = jax.lax.fori_loop(
      0, num_simulations, body_fun, (rng_key, tree))

  return tree


class _SimulationState(NamedTuple):
  """The state for the simulation while loop."""
  rng_key: chex.PRNGKey
  node_index: int
  action: int
  next_node_index: int
  depth: int
  is_continuing: bool


@functools.partial(jax.vmap, in_axes=[0, 0, None, None], out_axes=0)
def simulate(
    rng_key: chex.PRNGKey,
    tree: Tree,
    action_selection_fn: base.InteriorActionSelectionFn,
    max_depth: int) -> Tuple[chex.Array, chex.Array]:
  """Traverses the tree until reaching an unvisited action or `max_depth`.

  Each simulation starts from the root and keeps selecting actions traversing
  the tree until a leaf or `max_depth` is reached.

  Args:
    rng_key: random number generator state, the key is consumed.
    tree: _unbatched_ MCTS tree state.
    action_selection_fn: function used to select an action during simulation.
    max_depth: maximum search tree depth allowed during simulation.

  Returns:
    `(parent_index, action)` tuple, where `parent_index` is the index of the
    node reached at the end of the simulation, and the `action` is the action to
    evaluate from the `parent_index`.
  """
  def cond_fun(state):
    return state.is_continuing

  def body_fun(state):
    # Preparing the next simulation state.
    node_index = state.next_node_index
    rng_key, action_selection_key = jax.random.split(state.rng_key)
    action = action_selection_fn(action_selection_key, tree, node_index,
                                 state.depth)
    next_node_index = tree.children_index[node_index, action]
    # The returned action will be visited.
    depth = state.depth + 1
    is_before_depth_cutoff = depth < max_depth
    is_visited = next_node_index != Tree.UNVISITED
    is_continuing = jnp.logical_and(is_visited, is_before_depth_cutoff)
    return _SimulationState(  # pytype: disable=wrong-arg-types  # jax-types
        rng_key=rng_key,
        node_index=node_index,
        action=action,
        next_node_index=next_node_index,
        depth=depth,
        is_continuing=is_continuing)

  node_index = jnp.array(Tree.ROOT_INDEX, dtype=jnp.int32)
  depth = jnp.zeros((), dtype=tree.children_prior_logits.dtype)
  # pytype: disable=wrong-arg-types  # jnp-type
  initial_state = _SimulationState(
      rng_key=rng_key,
      node_index=tree.NO_PARENT,
      action=tree.NO_PARENT,
      next_node_index=node_index,
      depth=depth,
      is_continuing=jnp.array(True))
  # pytype: enable=wrong-arg-types
  end_state = jax.lax.while_loop(cond_fun, body_fun, initial_state)

  # Returning a node with a selected action.
  # The action can be already visited, if the max_depth is reached.
  return end_state.node_index, end_state.action


def expand(
    params: chex.Array,
    rng_key: chex.PRNGKey,
    tree: Tree[T],
    recurrent_fn: base.RecurrentFn,
    parent_index: chex.Array,
    action: chex.Array,
    next_node_index: chex.Array) -> Tree[T]:
  """Create and evaluate child nodes from given nodes and unvisited actions.

  Args:
    params: params to be forwarded to recurrent function.
    rng_key: random number generator state.
    tree: the MCTS tree state to update.
    recurrent_fn: a callable to be called on the leaf nodes and unvisited
      actions retrieved by the simulation step, which takes as args
      `(params, rng_key, action, embedding)` and returns a `RecurrentFnOutput`
      and the new state embedding. The `rng_key` argument is consumed.
    parent_index: the index of the parent node, from which the action will be
      expanded. Shape `[B]`.
    action: the action to expand. Shape `[B]`.
    next_node_index: the index of the newly expanded node. This can be the index
      of an existing node, if `max_depth` is reached. Shape `[B]`.

  Returns:
    tree: updated MCTS tree state.
  """
  batch_size = tree_lib.infer_batch_size(tree)
  batch_range = jnp.arange(batch_size)
  chex.assert_shape([parent_index, action, next_node_index], (batch_size,))

  # Retrieve states for nodes to be evaluated.
  embedding = jax.tree.map(
      lambda x: x[batch_range, parent_index], tree.embeddings)

  # Evaluate and create a new node.
  step, embedding = recurrent_fn(params, rng_key, action, embedding)
  chex.assert_shape(step.prior_logits, [batch_size, tree.num_actions])
  chex.assert_shape(step.reward, [batch_size])
  chex.assert_shape(step.discount, [batch_size])
  chex.assert_shape(step.value, [batch_size])
  tree = update_tree_node(
      tree, next_node_index, step.prior_logits, step.value, embedding)

  # Return updated tree topology.
  return tree.replace(
      children_index=util.batch_update(
          tree.children_index, next_node_index, parent_index, action),
      children_rewards=util.batch_update(
          tree.children_rewards, step.reward, parent_index, action),
      children_discounts=util.batch_update(
          tree.children_discounts, step.discount, parent_index, action),
      parents=util.batch_update(tree.parents, parent_index, next_node_index),
      action_from_parent=util.batch_update(
          tree.action_from_parent, action, next_node_index))


@jax.vmap
def backward(
    tree: Tree[T],
    leaf_index: chex.Numeric) -> Tree[T]:
  """Goes up and updates the tree until all nodes reached the root.

  Args:
    tree: the MCTS tree state to update, without the batch size.
    leaf_index: the node index from which to do the backward.

  Returns:
    Updated MCTS tree state.
  """

  def cond_fun(loop_state):
    _, _, index = loop_state
    return index != Tree.ROOT_INDEX

  def body_fun(loop_state):
    # Here we update the value of our parent, so we start by reversing.
    tree, leaf_value, index = loop_state
    parent = tree.parents[index]
    count = tree.node_visits[parent]
    action = tree.action_from_parent[index]
    reward = tree.children_rewards[parent, action]
    leaf_value = reward + tree.children_discounts[parent, action] * leaf_value
    parent_value = (
        tree.node_values[parent] * count + leaf_value) / (count + 1.0)
    children_values = tree.node_values[index]
    children_counts = tree.children_visits[parent, action] + 1

    tree = tree.replace(
        node_values=update(tree.node_values, parent_value, parent),
        node_visits=update(tree.node_visits, count + 1, parent),
        children_values=update(
            tree.children_values, children_values, parent, action),
        children_visits=update(
            tree.children_visits, children_counts, parent, action))

    return tree, leaf_value, parent

  leaf_index = jnp.asarray(leaf_index, dtype=jnp.int32)
  loop_state = (tree, tree.node_values[leaf_index], leaf_index)
  tree, _, _ = jax.lax.while_loop(cond_fun, body_fun, loop_state)

  return tree





def update_tree_node(
    tree: Tree[T],
    node_index: chex.Array,
    prior_logits: chex.Array,
    value: chex.Array,
    embedding: chex.Array) -> Tree[T]:
  """Updates the tree at node index.

  Args:
    tree: `Tree` to whose node is to be updated.
    node_index: the index of the expanded node. Shape `[B]`.
    prior_logits: the prior logits to fill in for the new node, of shape
      `[B, num_actions]`.
    value: the value to fill in for the new node. Shape `[B]`.
    embedding: the state embeddings for the node. Shape `[B, ...]`.

  Returns:
    The new tree with updated nodes.
  """
  batch_size = tree_lib.infer_batch_size(tree)
  batch_range = jnp.arange(batch_size)
  chex.assert_shape(prior_logits, (batch_size, tree.num_actions))

  # When using max_depth, a leaf can be expanded multiple times.
  new_visit = tree.node_visits[batch_range, node_index] + 1
  updates = dict(  # pylint: disable=use-dict-literal
      children_prior_logits=util.batch_update(
          tree.children_prior_logits, prior_logits, node_index),
      raw_values=util.batch_update(
          tree.raw_values, value, node_index),
      node_values=util.batch_update(
          tree.node_values, value, node_index),
      node_visits=util.batch_update(
          tree.node_visits, new_visit, node_index),
      embeddings=jax.tree.map(
          lambda t, s: util.batch_update(t, s, node_index),
          tree.embeddings, embedding))

  return tree.replace(**updates)


def instantiate_tree_from_root(
    root: base.RootFnOutput,
    num_simulations: int,
    root_invalid_actions: chex.Array,
    extra_data: Any) -> Tree:
  """Initializes tree state at search root."""
  chex.assert_rank(root.prior_logits, 2)
  batch_size, num_actions = root.prior_logits.shape
  chex.assert_shape(root.value, [batch_size])
  num_nodes = num_simulations + 1
  data_dtype = root.value.dtype
  batch_node = (batch_size, num_nodes)
  batch_node_action = (batch_size, num_nodes, num_actions)

  def _zeros(x):
    return jnp.zeros(batch_node + x.shape[1:], dtype=x.dtype)

  # Create a new empty tree state and fill its root.
  tree = Tree(
      node_visits=jnp.zeros(batch_node, dtype=jnp.int32),
      raw_values=jnp.zeros(batch_node, dtype=data_dtype),
      node_values=jnp.zeros(batch_node, dtype=data_dtype),
      parents=jnp.full(batch_node, Tree.NO_PARENT, dtype=jnp.int32),
      action_from_parent=jnp.full(
          batch_node, Tree.NO_PARENT, dtype=jnp.int32),
      children_index=jnp.full(
          batch_node_action, Tree.UNVISITED, dtype=jnp.int32),
      children_prior_logits=jnp.zeros(
          batch_node_action, dtype=root.prior_logits.dtype),
      children_values=jnp.zeros(batch_node_action, dtype=data_dtype),
      children_visits=jnp.zeros(batch_node_action, dtype=jnp.int32),
      children_rewards=jnp.zeros(batch_node_action, dtype=data_dtype),
      children_discounts=jnp.zeros(batch_node_action, dtype=data_dtype),
      embeddings=jax.tree.map(_zeros, root.embedding),
      root_invalid_actions=root_invalid_actions,
      extra_data=extra_data)

  root_index = jnp.full([batch_size], Tree.ROOT_INDEX)
  tree = update_tree_node(
      tree, root_index, root.prior_logits, root.value, root.embedding)
  return tree
