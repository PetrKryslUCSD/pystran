"""
pystran - Python package for structural analysis with trusses and beams 

(C) 2025, Petr Krysl

This package is intended for educational purposes only. Professional users may
find it too bare bones.

The approach is based on classical weighted residual formulation (Galerkin). The
formulations are derived in the "Finite element modeling with shells and beams"
[book](http://hogwarts.ucsd.edu/~pkrysl/femstructures-book/).

The approach here is modern as opposed to classic.

Classically, the geometrical transformations are developed explicitly to push
the stiffness and mass matrices from special orientations to the real
orientation in space. This requires multiplication of the stiffness and mass
matrices by large transformation matrices on the left and right. The matrices in
special orientations are usually developed analytically, and these explicit
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
strain-displacement matrix. The stiffness and mass matrices are obtained readily
using numerical integration.


## Features and limitations

- The package analyzes two-dimensional and three-dimensional structures made up
  of truss (axial) members and beams (even in combination), with added masses
  and springs at joints.
- Linear statics and dynamics (free vibration) solvers are included.
- The Bernoulli-Euler model is implemented, so no shear deformation is taken into account.
- Only elastic models can be solved.
- Only straight members are treated.
- Only doubly symmetric cross sections can be handled in three dimensions. Hence
  there is no coupling between the bending actions in the two orthogonal planes.
- Warping of the cross sections is not modelled, hence only free torsion effects are included.
- Member loading is not considered. All member loading needs to be converted to nodal forces.
- Internal hinges can be modelled with linked joints. No member end releases are implemented.
- Degrees of freedom are only along the cartesian axes. Skew supports are not
  included (except with a penalty method based on springs)
- Offsets are currently not implemented.

"""
