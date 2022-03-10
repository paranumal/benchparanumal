/*

The MIT License (MIT)

Copyright (c) 2020 Tim Warburton, Noel Chalmers, Jesse Chan, Ali Karakus

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

*/

#ifndef MESH_HPP
#define MESH_HPP 1

#include "core.hpp"
#include "platform.hpp"
#include "settings.hpp"
#include "ogs.hpp"

namespace libp {

class mesh_t {
public:
  platform_t platform;
  settings_t settings;
  properties_t props;

  comm_t comm;
  int rank, size;

  /*************************/
  /* Element Data          */
  /*************************/
  int dim;
  int Nverts, Nfaces, NfaceVertices;
  int elementType;

  // indices of vertex nodes
  memory<int> vertexNodes;

  hlong Nnodes=0; //global number of element vertices
  memory<dfloat> EX; // coordinates of vertices for each element
  memory<dfloat> EY;
  memory<dfloat> EZ;

  dlong Nelements=0;       //local element count
  hlong NelementsGlobal=0; //global element count
  memory<hlong> EToV;      // element-to-vertex connectivity
  memory<dlong> EToE;      // element-to-element connectivity
  memory<int>   EToF;      // element-to-(local)face connectivity
  memory<int>   EToP;      // element-to-partition/process connectivity

  memory<dlong> VmapM;  // list of vertices on each face
  memory<dlong> VmapP;  // list of vertices that are paired with face vertices

  /*************************/
  /* FEM Space             */
  /*************************/
  int N=0, Np=0;             // N = Polynomial order and Np = Nodes per element
  memory<dfloat> r, s, t;    // coordinates of local nodes

  int Nq=0;                 // N = Polynomial order, Nq=N+1
  memory<dfloat> gllz;      // 1D GLL quadrature nodes
  memory<dfloat> gllw;      // 1D GLL quadrature weights

  // face node info
  int Nfp=0;                // number of nodes per face
  memory<int> faceNodes;    // list of element reference interpolation nodes on element faces
  memory<int> faceVertices; // list of mesh vertices on each face

  /*************************/
  /* FEM Operators         */
  /*************************/
  memory<dfloat> Dr, Ds, Dt;    // collocation differentiation matrices
  memory<dfloat> D;
  deviceMemory<dfloat> o_D;
  memory<dfloat> MM, invMM;     // reference mass matrix
  deviceMemory<dfloat> o_MM;
  memory<dfloat> LIFT;          // lift matrix
  deviceMemory<dfloat> o_LIFT;
  memory<dfloat> sM;            // surface mass (MM*LIFT)^T
  deviceMemory<dfloat> o_sM;
  memory<dfloat> Srr, Srs, Srt; //element stiffness matrices
  memory<dfloat> Ssr, Sss, Sst;
  memory<dfloat> Str, Sts, Stt;
  memory<dfloat> S;
  deviceMemory<dfloat> o_S;

  /*************************/
  /* Cubature              */
  /*************************/
  // cubature
  int cubN, cubNp, cubNfp, cubNq;
  memory<dfloat> cubr, cubs, cubt, cubw; // coordinates and weights of local cubature nodes

  memory<dfloat> cubInterp;    // interpolate from W&B to cubature nodes
  deviceMemory<dfloat> o_cubInterp;
  memory<dfloat> cubProject;   // projection matrix from cubature nodes to W&B nodes
  deviceMemory<dfloat> o_cubProject;
  memory<dfloat> cubD;         // 1D differentiation matrix
  deviceMemory<dfloat> o_cubD;
  memory<dfloat> cubDW;        // 1D weak differentiation matrix
  memory<dfloat> cubDrW;       // 'r' weak differentiation matrix
  memory<dfloat> cubDsW;       // 's' weak differentiation matrix
  memory<dfloat> cubDtW;       // 't' weak differentiation matrix
  memory<dfloat> cubDWmatrices;

  /*************************/
  /* Plotting              */
  /*************************/
  // ploting info for generating field vtu
  int plotN;
  int plotNverts;    // number of vertices for each plot element
  int plotNelements; // number of "plot elements" per element
  memory<int>   plotEToV;      // triangulation of plot nodes

  /*************************/
  /* Physical Space        */
  /*************************/
  memory<dlong> vmapM;      // list of volume nodes that are face nodes
  memory<dlong> vmapP;      // list of volume nodes that are paired with face nodes

  // volume node info
  memory<dfloat> x, y, z;    // coordinates of physical nodes
  deviceMemory<dfloat> o_x, o_y, o_z;    // coordinates of physical nodes

  // Jacobian
  memory<dfloat> wJ;
  deviceMemory<dfloat> o_wJ;
  // volumeGeometricFactors;
  dlong Nvgeo;
  memory<dfloat> vgeo;
  deviceMemory<dfloat> o_vgeo;
  // surfaceGeometricFactors;
  dlong   Nsgeo;
  memory<dfloat> sgeo;
  deviceMemory<dfloat> o_sgeo;
  // second order volume geometric factors
  dlong Nggeo;
  memory<dfloat> ggeo;
  deviceMemory<dfloat> o_ggeo;

  memory<dfloat> cubx, cuby, cubz; // coordinates of physical nodes
  memory<dfloat> cubwJ;            //Jacobian at cubature points
  deviceMemory<dfloat> o_cubwJ;
  memory<dfloat> cubvgeo;          //volume geometric data at cubature points
  deviceMemory<dfloat> o_cubvgeo;
  memory<dfloat> cubsgeo;          //surface geometric data at cubature points
  deviceMemory<dfloat> o_cubsgeo;
  memory<dfloat> cubggeo;          //second type volume geometric data at cubature points
  deviceMemory<dfloat> o_cubggeo;

  /*************************/
  /* MPI Data              */
  /*************************/
  // MPI halo exchange info
  ogs::halo_t halo;                          // halo exchange pointer
  dlong NinternalElements;                   // number of elements that can update without halo exchange
  dlong NhaloElements;                       // number of elements that cannot update without halo exchange
  dlong  totalHaloPairs;                     // number of elements to be received in halo exchange
  memory<dlong> internalElementIds;          // list of elements that can update without halo exchange
  memory<dlong> haloElementIds;              // list of elements to be sent in halo exchange
  deviceMemory<dlong> o_internalElementIds;  // list of elements that can update without halo exchange
  deviceMemory<dlong> o_haloElementIds;      // list of elements to be sent in halo exchange

  // CG gather-scatter info
  memory<hlong> globalIds;

  // list of elements that are needed for global gather-scatter
  dlong NglobalGatherElements;
  memory<dlong> globalGatherElementList;
  deviceMemory<dlong> o_globalGatherElementList;

  // list of elements that are not needed for global gather-scatter
  dlong NlocalGatherElements;
  memory<dlong> localGatherElementList;
  deviceMemory<dlong> o_localGatherElementList;

  mesh_t() = default;
  mesh_t(platform_t& _platform, settings_t& _settings,
         comm_t _comm) {
    Setup(_platform, _settings, _comm);
  }

  // generic mesh setup
  void Setup(platform_t& _platform, settings_t& _settings,
             comm_t _comm);

   // Compute physical nodes
  void PhysicalNodes() {
    switch (elementType) {
      case TRIANGLES:
        PhysicalNodesTri2D();
        break;
      case QUADRILATERALS:
        PhysicalNodesQuad2D();
        break;
      case TETRAHEDRA:
        PhysicalNodesTet3D();
        break;
      case HEXAHEDRA:
        PhysicalNodesHex3D();
        break;
    }
  }

  // Setup geofactors
  void GeometricFactors() {
    switch (elementType) {
      case TRIANGLES:
        GeometricFactorsTri2D();
        break;
      case QUADRILATERALS:
        GeometricFactorsQuad2D();
        break;
      case TETRAHEDRA:
        GeometricFactorsTet3D();
        break;
      case HEXAHEDRA:
        GeometricFactorsHex3D();
        break;
    }
  }

  // Setup cubature
  void CubatureSetup() {
    switch (elementType) {
      case TRIANGLES:
        CubatureSetupTri2D();
        break;
      case QUADRILATERALS:
        CubatureSetupQuad2D();
        break;
      case TETRAHEDRA:
        CubatureSetupTet3D();
        break;
      case HEXAHEDRA:
        CubatureSetupHex3D();
        break;
    }
  }

  // Plot
  void PlotFields(memory<dfloat> Q, int Nfields, std::string fileName) {
    switch (dim) {
      case 2:
        PlotFields2D(Q, Nfields, fileName);
        break;
      case 3:
        PlotFields3D(Q, Nfields, fileName);
        break;
    }
  }

  /* Build a masked ogs handle*/
  ogs::ogs_t MaskedGatherScatterSetup(int Nfields);

  /*Element types*/
  static constexpr int TRIANGLES     =3;
  static constexpr int QUADRILATERALS=4;
  static constexpr int TETRAHEDRA    =6;
  static constexpr int HEXAHEDRA     =12;

 private:
  int G00ID, G01ID, G02ID, G11ID, G12ID, G22ID;

  /*Set the type of mesh*/
  void SetElementType(const int eType);

  // box mesh
  void SetupBox() {
    switch (elementType) {
      case TRIANGLES:
        SetupBoxTri2D();
        break;
      case QUADRILATERALS:
        SetupBoxQuad2D();
        break;
      case TETRAHEDRA:
        SetupBoxTet3D();
        break;
      case HEXAHEDRA:
        SetupBoxHex3D();
        break;
    }
  }
  void SetupBoxTri2D();
  void SetupBoxQuad2D();
  void SetupBoxTet3D();
  void SetupBoxHex3D();

  // reference nodes and operators
  void ReferenceNodes() {
    switch (elementType) {
      case TRIANGLES:
        ReferenceNodesTri2D();
        break;
      case QUADRILATERALS:
        ReferenceNodesQuad2D();
        break;
      case TETRAHEDRA:
        ReferenceNodesTet3D();
        break;
      case HEXAHEDRA:
        ReferenceNodesHex3D();
        break;
    }
  }
  void ReferenceNodesTri2D();
  void ReferenceNodesQuad2D();
  void ReferenceNodesTet3D();
  void ReferenceNodesHex3D();


  /* build parallel face connectivity */
  void Connect();

  // face-vertex to face-vertex connection
  void ConnectFaceVertices();

  // face-node to face-node connection
  void ConnectFaceNodes();

  // setup halo region
  void HaloSetup();

  /* build global connectivity in parallel */
  void ConnectNodes();

  /* build global gather scatter ops */
  void GatherScatterSetup();

  // Compute physical nodes
  void PhysicalNodesTri2D();
  void PhysicalNodesQuad2D();
  void PhysicalNodesTet3D();
  void PhysicalNodesHex3D();

  // Setup geofactors
  void GeometricFactorsTri2D();
  void GeometricFactorsQuad2D();
  void GeometricFactorsTet3D();
  void GeometricFactorsHex3D();

  // Setup cubature
  void CubatureSetupTri2D();
  void CubatureSetupQuad2D();
  void CubatureSetupTet3D();
  void CubatureSetupHex3D();

  // Plot
  void PlotFields2D(memory<dfloat> Q, int Nfields, std::string fileName);
  void PlotFields3D(memory<dfloat> Q, int Nfields, std::string fileName);


  /*************************/
  /* FEM Space             */
  /*************************/
  //1D
  void Nodes1D(int N, dfloat r[]);
  void OrthonormalBasis1D(dfloat a, int i, dfloat &P);
  void GradOrthonormalBasis1D(dfloat a, int i, dfloat &Pr);
  void Vandermonde1D(int N, int Npoints, dfloat r[], dfloat V[]);
  void GradVandermonde1D(int N, int Npoints, dfloat r[], dfloat Vr[]);

  void MassMatrix1D(int _Np, dfloat V[], dfloat MM[]);
  void Dmatrix1D(int _N, int NpointsIn, dfloat _rIn[],
                 int NpointsOut, dfloat _rOut[], dfloat _Dr[]);
  void InterpolationMatrix1D(int _N,
                             int NpointsIn, dfloat rIn[],
                             int NpointsOut, dfloat rOut[],
                             dfloat I[]);

  //Jacobi polynomial evaluation
  dfloat JacobiP(dfloat a, dfloat alpha, dfloat beta, int N);
  dfloat GradJacobiP(dfloat a, dfloat alpha, dfloat beta, int N);

  //Gauss-Legendre-Lobatto quadrature nodes
  void JacobiGLL(int N, dfloat x[], dfloat w[]=nullptr);

  //Nth order Gauss-Jacobi quadrature nodes and weights
  void JacobiGQ(dfloat alpha, dfloat beta, int N, dfloat x[], dfloat w[]);

  //Quads
  void NodesQuad2D(int _N, dfloat _r[], dfloat _s[]);
  void FaceNodesQuad2D(int _N, dfloat _r[], dfloat _s[], int _faceNodes[]);
  void VertexNodesQuad2D(int _N, dfloat _r[], dfloat _s[], int _vertexNodes[]);
  void FaceNodeMatchingQuad2D(int _N, dfloat _r[], dfloat _s[],
                              int _faceNodes[], int R[]);
  void EquispacedNodesQuad2D(int _N, dfloat _r[], dfloat _s[]);
  void EquispacedEToVQuad2D(int _N, int _EToV[]);

  //Hexs
  void NodesHex3D(int _N, dfloat _r[], dfloat _s[], dfloat _t[]);
  void FaceNodesHex3D(int _N, dfloat _r[], dfloat _s[], dfloat _t[],  int _faceNodes[]);
  void VertexNodesHex3D(int _N, dfloat _r[], dfloat _s[], dfloat _t[], int _vertexNodes[]);
  void FaceNodeMatchingHex3D(int _N, dfloat _r[], dfloat _s[], dfloat _t[],
                             int _faceNodes[], int R[]);
  void EquispacedNodesHex3D(int _N, dfloat _r[], dfloat _s[], dfloat _t[]);
  void EquispacedEToVHex3D(int _N, int _EToV[]);
};

void meshAddSettings(settings_t& settings);
void meshReportSettings(settings_t& settings);

} //namespace libp

#endif

