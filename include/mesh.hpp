/*

The MIT License (MIT)

Copyright (c) 2017-2022 Tim Warburton, Noel Chalmers, Jesse Chan, Ali Karakus

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
  int RXID, RYID, RZID;
  int SXID, SYID, SZID;
  int TXID, TYID, TZID;
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
  static void Nodes1D(const int _N, memory<dfloat>& _r);
  static void EquispacedNodes1D(const int _N, memory<dfloat>& _r);
  static void OrthonormalBasis1D(const dfloat a, const int i, dfloat& P);
  static void GradOrthonormalBasis1D(const dfloat a, const int i, dfloat& Pr);
  static void Vandermonde1D(const int _N,
                            const memory<dfloat> _r,
                            memory<dfloat>& V);
  static void GradVandermonde1D(const int _N,
                                const memory<dfloat> _r,
                                memory<dfloat>& Vr);
  static void MassMatrix1D(const int _Np,
                           const memory<dfloat> V,
                           memory<dfloat>& _MM);
  static void Dmatrix1D(const int _N,
                        const memory<dfloat> _rIn,
                        const memory<dfloat> _rOut,
                        memory<dfloat>& _Dr);
  static void InterpolationMatrix1D(const int _N,
                                    const memory<dfloat> _rIn,
                                    const memory<dfloat> _rOut,
                                    memory<dfloat>& I);

  //Jacobi polynomial evaluation
  static dfloat JacobiP(const dfloat a, const dfloat alpha,
                        const dfloat beta, const int _N);
  static dfloat GradJacobiP(const dfloat a, const dfloat alpha,
                            const dfloat beta, const int _N);

  //Gauss-Legendre-Lobatto quadrature nodes
  static void JacobiGLL(const int _N,
                        memory<dfloat>& _x);
  static void JacobiGLL(const int _N,
                        memory<dfloat>& _x,
                        memory<dfloat>& _w);

  //Nth order Gauss-Jacobi quadrature nodes and weights
  static void JacobiGQ(const dfloat alpha, const dfloat beta,
                       const int _N,
                       memory<dfloat>& _x,
                       memory<dfloat>& _w);

  //Tris
  static void NodesTri2D(const int _N,
                         memory<dfloat>& _r,
                         memory<dfloat>& _s);
  static void FaceNodesTri2D(const int _N,
                             const memory<dfloat> _r,
                             const memory<dfloat> _s,
                             memory<int>& _faceNodes);
  static void VertexNodesTri2D(const int _N,
                               const memory<dfloat> _r,
                               const memory<dfloat> _s,
                               memory<int>& _vertexNodes);
  static void FaceNodeMatchingTri2D(const memory<dfloat> _r,
                                    const memory<dfloat> _s,
                                    const memory<int> _faceNodes,
                                    const memory<int> _faceVertices,
                                    memory<int>& R);
  static void EquispacedNodesTri2D(const int _N,
                                   memory<dfloat>& _r,
                                   memory<dfloat>& _s);
  static void EquispacedEToVTri2D(const int _N, memory<int>& _EToV);
  static void OrthonormalBasisTri2D(const dfloat _r, const dfloat _s,
                                    const int i, const int j,
                                    dfloat& P);
  static void GradOrthonormalBasisTri2D(const dfloat _r, const dfloat _s,
                                        const int i, const int j,
                                        dfloat& Pr, dfloat& Ps);
  static void VandermondeTri2D(const int _N,
                               const memory<dfloat> _r,
                               const memory<dfloat> _s,
                               memory<dfloat>& V);
  static void GradVandermondeTri2D(const int _N,
                                   const memory<dfloat> _r,
                                   const memory<dfloat> _s,
                                   memory<dfloat>& Vr,
                                   memory<dfloat>& Vs);
  static void MassMatrixTri2D(const int _Np,
                              const memory<dfloat> V,
                              memory<dfloat>& _MM);
  static void invMassMatrixTri2D(const int _Np,
                                 const memory<dfloat> V,
                                 memory<dfloat>& _invMM);
  static void DmatrixTri2D(const int _N,
                           const memory<dfloat> _r,
                           const memory<dfloat> _s,
                           memory<dfloat>& _D);
  static void SmatrixTri2D(const int _N,
                           const memory<dfloat> _Dr,
                           const memory<dfloat> _Ds,
                           const memory<dfloat> _MM,
                           memory<dfloat>& _S);
  static void InterpolationMatrixTri2D(const int _N,
                                       const memory<dfloat> rIn,
                                       const memory<dfloat> sIn,
                                       const memory<dfloat> rOut,
                                       const memory<dfloat> sOut,
                                       memory<dfloat>& I);
  static void CubatureNodesTri2D(const int cubTriN,
                                 int& _cubNp,
                                 memory<dfloat>& cubTrir,
                                 memory<dfloat>& cubTris,
                                 memory<dfloat>& cubTriw);
  static void CubaturePmatrixTri2D(const int _N,
                                   const memory<dfloat> _r,
                                   const memory<dfloat> _s,
                                   const memory<dfloat> _cubr,
                                   const memory<dfloat> _cubs,
                                   memory<dfloat>& _cubProject);
  static void CubatureWeakDmatricesTri2D(const int _N,
                                         const memory<dfloat> _r,
                                         const memory<dfloat> _s,
                                         const memory<dfloat> _cubr,
                                         const memory<dfloat> _cubs,
                                         memory<dfloat>& _cubPDT);

  static void Warpfactor(const int _N,
                         const memory<dfloat> _r,
                         memory<dfloat> warp);
  static void WarpBlendTransformTri2D(const int _N,
                                      memory<dfloat> _r,
                                      memory<dfloat> _s,
                                      const dfloat alphaIn=-1);


  //Quads
  static void NodesQuad2D(const int _N,
                          memory<dfloat>& _r,
                          memory<dfloat>& _s);
  static void FaceNodesQuad2D(const int _N,
                              const memory<dfloat> _r,
                              const memory<dfloat> _s,
                              memory<int>& _faceNodes);
  static void VertexNodesQuad2D(const int _N,
                                const memory<dfloat> _r,
                                const memory<dfloat> _s,
                                memory<int>& _vertexNodes);
  static void FaceNodeMatchingQuad2D(const memory<dfloat> _r,
                                     const memory<dfloat> _s,
                                     const memory<int> _faceNodes,
                                     const memory<int> _faceVertices,
                                     memory<int>& R);
  static void EquispacedNodesQuad2D(const int _N,
                                    memory<dfloat>& _r,
                                    memory<dfloat>& _s);
  static void EquispacedEToVQuad2D(const int _N, memory<int>& _EToV);

  //Tets
  static void NodesTet3D(const int _N,
                         memory<dfloat>& _r,
                         memory<dfloat>& _s,
                         memory<dfloat>& _t);
  static void FaceNodesTet3D(const int _N,
                             const memory<dfloat> _r,
                             const memory<dfloat> _s,
                             const memory<dfloat> _t,
                             memory<int>& _faceNodes);
  static void VertexNodesTet3D(const int _N,
                               const memory<dfloat> _r,
                               const memory<dfloat> _s,
                               const memory<dfloat> _t,
                               memory<int>& _vertexNodes);
  static void FaceNodeMatchingTet3D(const memory<dfloat> _r,
                                    const memory<dfloat> _s,
                                    const memory<dfloat> _t,
                                    const memory<int> _faceNodes,
                                    const memory<int> _faceVertices,
                                    memory<int>& R);
  static void EquispacedNodesTet3D(const int _N,
                                   memory<dfloat>& _r,
                                   memory<dfloat>& _s,
                                   memory<dfloat>& _t);
  static void EquispacedEToVTet3D(const int _N, memory<int>& _EToV);

  static void OrthonormalBasisTet3D(const dfloat _r, const dfloat _s, const dfloat _t,
                                    const int i, const int j, const int k,
                                    dfloat& P);
  static void GradOrthonormalBasisTet3D(const dfloat _r, const dfloat _s, const dfloat _t,
                                        const int i, const int j, const int k,
                                        dfloat& Pr, dfloat& Ps, dfloat& Pt);
  static void VandermondeTet3D(const int _N,
                               const memory<dfloat> _r,
                               const memory<dfloat> _s,
                               const memory<dfloat> _t,
                               memory<dfloat>& V);
  static void GradVandermondeTet3D(const int _N,
                                   const memory<dfloat> _r,
                                   const memory<dfloat> _s,
                                   const memory<dfloat> _t,
                                   memory<dfloat>& Vr,
                                   memory<dfloat>& Vs,
                                   memory<dfloat>& Vt);
  static void MassMatrixTet3D(const int _Np,
                              const memory<dfloat> V,
                              memory<dfloat>& _MM);
  static void invMassMatrixTet3D(const int _Np,
                                 const memory<dfloat> V,
                                 memory<dfloat>& _invMM);
  static void DmatrixTet3D(const int _N,
                           const memory<dfloat> _r,
                           const memory<dfloat> _s,
                           const memory<dfloat> _t,
                           memory<dfloat>& _D);
  static void SmatrixTet3D(const int _N,
                           const memory<dfloat> _Dr,
                           const memory<dfloat> _Ds,
                           const memory<dfloat> _Dt,
                           const memory<dfloat> _MM,
                           memory<dfloat>& _S);
  static void InterpolationMatrixTet3D(const int _N,
                                       const memory<dfloat> rIn,
                                       const memory<dfloat> sIn,
                                       const memory<dfloat> tIn,
                                       const memory<dfloat> rOut,
                                       const memory<dfloat> sOut,
                                       const memory<dfloat> tOut,
                                       memory<dfloat>& I);

  static void CubatureNodesTet3D(const int cubTetN,
                                 int& _cubNp,
                                 memory<dfloat>& _cubr,
                                 memory<dfloat>& _cubs,
                                 memory<dfloat>& _cubt,
                                 memory<dfloat>& _cubw);
  static void CubaturePmatrixTet3D(const int _N,
                                   const memory<dfloat> _r,
                                   const memory<dfloat> _s,
                                   const memory<dfloat> _t,
                                   const memory<dfloat> _cubr,
                                   const memory<dfloat> _cubs,
                                   const memory<dfloat> _cubt,
                                   memory<dfloat>& _cubProject);
  static void CubatureWeakDmatricesTet3D(const int _N,
                                         const memory<dfloat> _r,
                                         const memory<dfloat> _s,
                                         const memory<dfloat> _t,
                                         const memory<dfloat> _cubr,
                                         const memory<dfloat> _cubs,
                                         const memory<dfloat> _cubt,
                                         memory<dfloat>& _cubPDT);

  static void WarpShiftFace3D(const int _N, const dfloat alpha,
                              const memory<dfloat> L1,
                              const memory<dfloat> L2,
                              const memory<dfloat> L3,
                              memory<dfloat> w1,
                              memory<dfloat> w2);
  static void WarpBlendTransformTet3D(const int _N,
                                      memory<dfloat> _r,
                                      memory<dfloat> _s,
                                      memory<dfloat> _t,
                                      const dfloat alphaIn=-1);


  //Hexs
  static void NodesHex3D(const int _N,
                         memory<dfloat>& _r,
                         memory<dfloat>& _s,
                         memory<dfloat>& _t);
  static void FaceNodesHex3D(const int _N,
                             const memory<dfloat> _r,
                             const memory<dfloat> _s,
                             const memory<dfloat> _t,
                             memory<int>& _faceNodes);
  static void VertexNodesHex3D(const int _N,
                               const memory<dfloat> _r,
                               const memory<dfloat> _s,
                               const memory<dfloat> _t,
                               memory<int>& _vertexNodes);
  static void FaceNodeMatchingHex3D(const memory<dfloat> _r,
                                    const memory<dfloat> _s,
                                    const memory<dfloat> _t,
                                    const memory<int> _faceNodes,
                                    const memory<int> _faceVertices,
                                    memory<int>& R);
  static void EquispacedNodesHex3D(const int _N,
                                   memory<dfloat>& _r,
                                   memory<dfloat>& _s,
                                   memory<dfloat>& _t);
  static void EquispacedEToVHex3D(const int _N, memory<int>& _EToV);

};

void meshAddSettings(settings_t& settings);
void meshReportSettings(settings_t& settings);

} //namespace libp

#endif

