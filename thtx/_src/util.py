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