#include <Qt3DIncludes.h>
#include <GaussIncludes.h>
#include <FEMIncludes.h>

//Hexahedral FEM Test
#include <ConstraintFixedPoint.h>
#include <TimeStepperEulerImplicitLinear.h>

using namespace Gauss;
using namespace FEM;
using namespace ParticleSystem; //For Force Spring

//typedef physical entities I need

//typedef scene
typedef PhysicalSystemFEM<double, LinearHex> FEMLinearHexes;

typedef World<double, std::tuple<FEMLinearHexes *>, std::tuple<ForceSpring<double> *>, std::tuple<ConstraintFixedPoint<double> *> > MyWorld;
typedef TimeStepperEulerImplictLinear<double, AssemblerEigenSparseMatrix<double>,
AssemblerEigenVector<double> > MyTimeStepper;

typedef Scene<MyWorld, MyTimeStepper> MyScene;

int main(int argc, char **argv) {
    
    std::cout<<"Test Linear FEM \n";
    
    //Setup Physics
    MyWorld world;
    
    
    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
    
    /*V.resize(8,3);
    F.resize(1,8);
    
    V <<    0,0,0,
            1,0,0,
            1,0,1,
            0,0,1,
            0,1,0,
            1,1,0,
            1,1,1,
            0,1,1;
    
    F << 0,1,2,3,4,5,6,7;*/
    
    //Voxel grid from libigl
    igl::grid(Eigen::RowVector3i(15, 5,5),  V);
    
    elementsFromGrid(Eigen::RowVector3i(15, 5,5), V, F);
    
    FEMLinearHexes *test = new FEMLinearHexes(V,F);
    
    world.addSystem(test);
    fixDisplacementMin(world, test);
    world.finalize(); //After this all we're ready to go (clean up the interface a bit later)
    
    auto q = mapStateEigen(world);
    q.setZero();
    
    MyTimeStepper stepper(0.01);
    MyScene *scene = new MyScene(&world, &stepper);
    
    //Display
    QGuiApplication app(argc, argv);
    
    GAUSSVIEW(scene);
    gaussView.startScene();
    
    return app.exec();
}
