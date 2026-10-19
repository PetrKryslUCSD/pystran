.. pystran documentation master file, created by
   sphinx-quickstart on Thu Mar 13 17:00:01 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

pystran documentation
=====================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   modules

Description
============

A simple structural analysis tool in Python, for structures consisting of truss
and beam members, springs, and rigid bodies, both in two dimensions and in
three dimensions. This package is intended for educational purposes, and hence
attention was paid to simplicity and clarity, and not so much to computational
efficiency.

Approach
--------

The approach is based on classical weighted residual formulation (Galerkin).
The formulations are derived in the `"Finite element modeling with shells and
beams" book <http://hogwarts.ucsd.edu/~pkrysl/femstructures-book/>`_.

The approach here is modern as opposed to classic.

Classically, the geometrical transformations are developed explicitly to push
the stiffness and mass matrices from special orientations to the real
orientation in space. This requires multiplication of the stiffness and mass
matrices by large transformation matrices on the left and right. The matrices
in special orientations are usually derived analytically, and these explicit
expressions become the starting point for developing computations. So for
instance for spatial beams, the starting point are 12x12 matrices.

The modern approach develops an expression for the strains in a basic element,
for instance curvature in beams. This leads to a small basic stiffness matrix,
4x4 matrix in the case of a 2D beam. The geometrical transformation is then
introduced implicitly by projecting displacements in the real space onto the
local basis vectors of the element. The Galerkin weighted residual method then
naturally completes the development of the matrices.

The three dimensional beam is in such a modern framework treated as a
superposition of four stiffness mechanisms, each with its own
strain-displacement matrix. The stiffness and mass matrices are obtained
readily using numerical integration.


Features and limitations
--------

- The package analyzes two-dimensional and three-dimensional structures made up
  of truss (axial) members and beams (possibly in combination), rigid links, and
  general springs. Concentrated masses can be added at joints.
- Linear statics and dynamics (free vibration) solvers are included.
- Only elastic models can be solved.
- For beams, only the Bernoulli-Euler model is implemented, so no shear
  deformation is taken into account.
- Only straight members are treated.
- It is assumed that the cross sections are doubly symmetric, and there is no coupling between the bending actions in the
  two orthogonal principal planes.
- Coupling of axial and bending action is not implemented. This means that the
  neutral axis must pass through the centroid.
- Warping of the cross sections is not modelled, hence only free torsion
  effects are included.
- Member loading is not considered. All member loading needs to be converted to
  nodal forces by the user.
- Member end releases (hinges)
  are not implemented. Internal hinges can be modelled with linked joints. 
- Degrees of freedom are only along the global Cartesian axes. Skew supports
  are not included (except with a penalty method based on springs)
- Offsets of the beams from the joints are currently not implemented.
- Rigid links between pairs of joints can be modeled with a penalty approach.
