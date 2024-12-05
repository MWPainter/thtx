"""Utility functions"""

import jax


#####
# Updating (batched) arrays 
#####

def update(x, vals, *indices):
  """Update 'x[indices]' to the values 'vals'"""
  return x.at[indices].set(vals)

# vmapped version of 'update' to operate on batches
batch_update = jax.vmap(update)


#####
# Broadcasting helper for py trees
#####

def pytree_broadcast(leaf, like_pytree):
  """
  Example, if like_pytree == {'a': 0, 'b': 1}, and leaf == 7, then returns the pytree {'a': 7, 'b': 7}
  Useful if want to pass a 'scalar' or pytree leaf object to jax.tree.map
  Appreciate this is a one liner, but makes things more readable :)
  """
  return jax.tree.map(lambda _: leaf, like_pytree)