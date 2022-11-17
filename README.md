A Lattice Toolbox
with Lattice Generator, Plot Functions, etc. for triangular, honeycomb and Kagome lattices

The toolbox consists of the following files:


lattice_aux.py:
  - lattice vectors for triangular and hexagonal lattice
  - calculate the euclidean distance for triangular lattices on a torus
  
  lattice_generator.py:
  - Generate a dictionary for lattices for honeycomb and triangular lattice based on the utlf matrix of the lattice. Some simulation tori are hardcoded for the utlf matrices, others are read from the matrix components. 
  - - Stores the neighbor indices for each lattice points, offers functionality to find specific neighbors of lattice sites
  - calculate adjacency matrices and simple cycles
  - plot the lattice with highlighted nodes and faces or plot multiple instances of the lattices next to each other to see the effect of periodic boundary conditions
  - calculate 3D coordinates for projection of the 2D lattice on a torus
  - plot the lattice on a torus (experimental)

lattice_sets.py
- Different sets of lattices with number of sites, utlf matrix encoded as ID and symmetry point group


plot_colored_lattices.py
- Visualize the local action of a Pauli operator acting on the associated ${\mathbb{C}^2}^{\otimes N}$ Hilber space
- Plot the triangular lattice with colored faces, colored nodes, additional information, etc.

sublattice_finder.py
- Given a Triangular Lattice with the K-Point in Brillouin Zone, find its three sublattices
