//
//  TimeStepperEulerImplicitLinearFEPR.h
//  Gauss
//
//  Created by Qiqin Fang on 2/10/17.
//
//

#ifndef TimeStepperEulerImplicitLinearFEPR_h
#define TimeStepperEulerImplicitLinearFEPR_h

#include <World.h>
#include <Assembler.h>
#include <TimeStepper.h>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <UtilitiesMATLAB.h>
#include <Eigen/SparseCholesky>
#include <Eigen/SparseLU>
#include <SolverPardiso.h>
#include <AssemblerParallel.h>

// #include <Eigen/IterativeLinearSolvers>


//TODO Solver Interface
namespace Gauss {

    //Given Initial state, step forward in time using linearly implicit Euler Integrator
    template<typename DataType, typename MatrixAssembler, typename VectorAssembler>
    class TimeStepperImplEulerImplicitLinearFEPR
    {
    public:

        TimeStepperImplEulerImplicitLinearFEPR(bool refactor = true) {
            m_factored = false;
            m_refactor = refactor;
        }

        TimeStepperImplEulerImplicitLinearFEPR(const TimeStepperImplEulerImplicitLinearFEPR &toCopy) {
            m_factored = false;
        }

        ~TimeStepperImplEulerImplicitLinearFEPR() {
            #ifdef GAUSS_PARDISO
                m_pardiso.cleanup();
            #endif
        }

        //Methods
        template<typename World>
        void step(World &world, double dt, double t);

        inline typename VectorAssembler::MatrixType & getLagrangeMultipliers() { return m_lagrangeMultipliers; }

    protected:

        MatrixAssembler m_massMatrix;
        MatrixAssembler m_stiffnessMatrix;
        VectorAssembler m_forceVector;

        //storage for lagrange multipliers
        typename VectorAssembler::MatrixType m_lagrangeMultipliers;

        bool m_factored, m_refactor;

#ifdef GAUSS_PARDISO

        SolverPardiso<Eigen::SparseMatrix<DataType, Eigen::RowMajor> > m_pardiso;

#endif

    private:
    };
}

template<typename DataType, typename MatrixAssembler, typename VectorAssembler>
template<typename World>
void TimeStepperImplEulerImplicitLinearFEPR<DataType, MatrixAssembler, VectorAssembler>::step(World &world, double dt, double t) {
    Eigen::VectorXd x_ref = mapStateEigen<0>(world);
    Eigen::VectorXd geo_x(x_ref.rows());
    Eigen::MatrixXd V(x_ref.rows() / 3, 3);
    int idx = 0;
    forEachIndex(world.getSystemList(), [&world, &geo_x, &idx](auto type, auto index, auto &a) {
        Eigen::MatrixXd geo = a->getGeometry().first;
        geo_x.block(index, 0, geo.size(), 1) = Eigen::VectorXd(geo);
        idx += geo.size();
    });
    V << geo_x;
    std::cout << t << std::endl;

    Eigen::VectorXd x_cur = geo_x + mapStateEigen<0>(world);
    Eigen::VectorXd v_cur = mapStateEigen<1>(world);
    double energy_cur = getEnergy(world);

    Eigen::SparseMatrix<DataType, Eigen::RowMajor> systemMatrix;
    Eigen::VectorXd x0;

    if(m_refactor || !m_factored) {

        //First two lines work around the fact that C++11 lambda can't directly capture a member variable.
        MatrixAssembler &massMatrix = m_massMatrix;
        MatrixAssembler &stiffnessMatrix = m_stiffnessMatrix;

        //get mass matrix
        ASSEMBLEMATINIT(massMatrix, world.getNumQDotDOFs()+world.getNumConstraints(), world.getNumQDotDOFs()+world.getNumConstraints());
        ASSEMBLELIST(massMatrix, world.getSystemList(), getMassMatrix);

        //add in the constraints
        ASSEMBLELISTOFFSET(massMatrix, world.getConstraintList(), getGradient, world.getNumQDotDOFs(), 0);
        ASSEMBLELISTOFFSETTRANSPOSE(massMatrix, world.getConstraintList(), getGradient, 0, world.getNumQDotDOFs());
        ASSEMBLEEND(massMatrix);


        //get stiffness matrix
        ASSEMBLEMATINIT(stiffnessMatrix, world.getNumQDotDOFs()+world.getNumConstraints(), world.getNumQDotDOFs()+world.getNumConstraints());
        ASSEMBLELIST(stiffnessMatrix, world.getSystemList(), getStiffnessMatrix);
        ASSEMBLELIST(stiffnessMatrix, world.getForceList(), getStiffnessMatrix);
        ASSEMBLEEND(stiffnessMatrix);

        systemMatrix = (*m_massMatrix)- dt*dt*(*m_stiffnessMatrix);

    }

    VectorAssembler &forceVector = m_forceVector;

    ASSEMBLEVECINIT(forceVector, world.getNumQDotDOFs()+world.getNumConstraints());
    ASSEMBLELIST(forceVector, world.getForceList(), getForce);
    ASSEMBLELIST(forceVector, world.getSystemList(), getForce);
    ASSEMBLELISTOFFSET(forceVector, world.getConstraintList(), getDbDt, world.getNumQDotDOFs(), 0);
    ASSEMBLEEND(forceVector);

    //Grab the state
    Eigen::Map<Eigen::VectorXd> q = mapStateEigen<0>(world);
    Eigen::Map<Eigen::VectorXd> qDot = mapStateEigen<1>(world);

    //setup RHS
    (*forceVector).head(world.getNumQDotDOFs()) = (*m_massMatrix).block(0,0, world.getNumQDotDOFs(), world.getNumQDotDOFs())*qDot + dt*(*forceVector).head(world.getNumQDotDOFs());

#ifdef GAUSS_PARDISO
    if(m_refactor || !m_factored) {
        m_pardiso.symbolicFactorization(systemMatrix);
        m_pardiso.numericalFactorization();
        m_factored = true;
    }

    m_pardiso.solve(*forceVector);
    x0 = m_pardiso.getX();
    //m_pardiso.cleanup();
#else
    //solve system (Need interface for solvers but for now just use Eigen LLt)
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double> > solver;

    if(m_refactor || !m_factored) {
        solver.compute(systemMatrix);
    }

    if(solver.info()!=Eigen::Success) {
        // decomposition failed
        assert(1 == 0);
        std::cout<<"Decomposition Failed \n";
        exit(1);
    }

    if(solver.info()!=Eigen::Success) {
        // solving failed
        assert(1 == 0);
        std::cout<<"Solve Failed \n";
        exit(1);
    }


    x0 = solver.solve((*forceVector));
#endif

    qDot = x0.head(world.getNumQDotDOFs());

    m_lagrangeMultipliers = x0.tail(world.getNumConstraints());

    // get mass matrix
    AssemblerEigenSparseMatrix<double> M;
    getMassMatrix(M, world);
    Eigen::SparseMatrix<double, Eigen::RowMajor> massMatrix = (*M);
    
    // Update world state
    updateState(world, world.getState(), dt);
    
    Eigen::VectorXd x_next = geo_x + mapStateEigen<0>(world);
    Eigen::VectorXd v_next = mapStateEigen<1>(world);
    double energy_next = getEnergy(world);

    Eigen::VectorXd x = x_next;
    Eigen::VectorXd v = v_next;
    double S = 0;
    double T = 0;
    Eigen::VectorXd q_iter(x_next.rows() + v_next.rows() + 2);
    q_iter << x, v, S, T;

    Eigen::SparseMatrix<double, Eigen::ColMajor> D(q_iter.rows(), q_iter.rows());
    double h = dt;
    double C = 0.01;
    double m = massMatrix.sum() / 3;
    double r_2 = pow(0.5 * (V.col(0).maxCoeff() - V.col(0).minCoeff()), 2) 
               + pow(0.5 * (V.col(1).maxCoeff() - V.col(1).minCoeff()), 2)
               + pow(0.5 * (V.col(2).maxCoeff() - V.col(2).minCoeff()), 2);
    double eps = C * m * r_2;

    Eigen::SparseMatrix<double, Eigen::RowMajor> Dtemp1(q_iter.rows(), massMatrix.cols());
    Dtemp1.topRows(massMatrix.rows()) = massMatrix;
    Eigen::SparseMatrix<double, Eigen::ColMajor> Dtemp1_col(Dtemp1);
    Eigen::SparseMatrix<double, Eigen::RowMajor> Dtemp2(D.rows(), massMatrix.cols());
    Dtemp2.middleRows(massMatrix.rows(), massMatrix.rows()) = pow(h, 2) * massMatrix;
    Eigen::SparseMatrix<double, Eigen::ColMajor> Dtemp2_col(Dtemp2);
    Eigen::SparseMatrix<double, Eigen::ColMajor> Dtemp3_col(D.rows(), 2);
    Dtemp3_col.insert(x.rows() + v.rows(), 0) = eps;
    Dtemp3_col.insert(x.rows() + v.rows() + 1, 1) = eps;
    D.leftCols(Dtemp1_col.cols()) = Dtemp1_col;
    D.middleCols(Dtemp1_col.cols(), Dtemp2_col.cols()) = Dtemp2_col;
    D.rightCols(Dtemp3_col.cols()) = Dtemp3_col;
    
    Eigen::Vector3d sgrad;
    sgrad << 0, 0, 0;
    for(int i = 0; i < v.rows() / 3; i++)
    {
        double m_i = massMatrix.coeff(3 * i, 3 * i);
        Eigen::Vector3d v_next_i;
        v_next_i << v_next(3 * i + 0), v_next(3 * i + 1), v_next(3 * i + 2);
        Eigen::Vector3d v_cur_i;
        v_cur_i << v_cur(3 * i + 0), v_cur(3 * i + 1), v_cur(3 * i + 2);
        sgrad -= m_i * (v_cur_i - v_next_i);
    }

    Eigen::Vector3d tgrad;
    tgrad << 0, 0, 0;
    for(int i = 0; i < x.rows() / 3; i++)
    {
        double m_i = massMatrix.coeff(3 * i, 3 * i);
        Eigen::Vector3d x_next_i;
        x_next_i << x_next(3 * i + 0), x_next(3 * i + 1), x_next(3 * i + 2);
        Eigen::Vector3d x_cur_i;
        x_cur_i << x_cur(3 * i + 0), x_cur(3 * i + 1), x_cur(3 * i + 2);
        Eigen::Vector3d v_next_i;
        v_next_i << v_next(3 * i + 0), v_next(3 * i + 1), v_next(3 * i + 2);
        Eigen::Vector3d v_cur_i;
        v_cur_i << v_cur(3 * i + 0), v_cur(3 * i + 1), v_cur(3 * i + 2);
        tgrad -= m_i * (x_cur_i.cross(v_cur_i) - x_next_i.cross(v_next_i));
    }
    
    for (int i = 0; i < 1000000; i++) {
        x = q_iter.block(0, 0, x.rows(), 1);
        v = q_iter.block(x.rows(), 0, v.rows(), 1);
        S = q_iter(x.rows() + v.rows());
        T = q_iter(x.rows() + v.rows() + 1);
        double energy_iter = getEnergy(world);
        
        Eigen::SparseMatrix<double, Eigen::ColMajor> gradP(x.rows() + v.rows() + 2, 3);
        Eigen::SparseMatrix<double, Eigen::ColMajor> gradL(x.rows() + v.rows() + 2, 3);
        typedef Eigen::Triplet<double> Triplet;
        std::vector<Triplet> gradPTripletList;
        gradPTripletList.reserve(v.rows() + 3);
        for(int i = 0; i < v.rows() / 3; i++)
        {
            double m_i = massMatrix.coeff(3 * i, 3 * i);
            gradPTripletList.push_back(Triplet(x.rows() + 3 * i + 0, 0, m_i));
            gradPTripletList.push_back(Triplet(x.rows() + 3 * i + 1, 1, m_i));
            gradPTripletList.push_back(Triplet(x.rows() + 3 * i + 2, 2, m_i));
        }
        gradPTripletList.push_back(Triplet(x.rows() + v.rows(), 0, sgrad(0)));
        gradPTripletList.push_back(Triplet(x.rows() + v.rows(), 1, sgrad(1)));
        gradPTripletList.push_back(Triplet(x.rows() + v.rows(), 2, sgrad(2)));
        gradP.setFromTriplets(gradPTripletList.begin(), gradPTripletList.end());
        
        std::vector<Triplet> gradLTripletList;
        gradLTripletList.reserve(2 * (x.rows() + v.rows()) + 3);
        for ( int i = 0; i < x.rows() / 3; i ++ ) 
        {
            double m_i = massMatrix.coeff(3 * i, 3 * i);
            double v_i1 = v(3 * i + 0);
            double v_i2 = v(3 * i + 1);
            double v_i3 = v(3 * i + 2);
            gradLTripletList.push_back(Triplet(3 * i + 0, 1,  m_i * v_i3));
            gradLTripletList.push_back(Triplet(3 * i + 0, 2, -m_i * v_i2));
            gradLTripletList.push_back(Triplet(3 * i + 1, 2,  m_i * v_i1));
            gradLTripletList.push_back(Triplet(3 * i + 1, 0, -m_i * v_i3));
            gradLTripletList.push_back(Triplet(3 * i + 2, 0,  m_i * v_i2));
            gradLTripletList.push_back(Triplet(3 * i + 2, 1, -m_i * v_i1));
        }
        for(int i = 0; i < v.rows() / 3; i++)
        {
            double m_i = massMatrix.coeff(3 * i, 3 * i);
            double x_i1 = x(3 * i + 0);
            double x_i2 = x(3 * i + 1);
            double x_i3 = x(3 * i + 2);
            gradLTripletList.push_back(Triplet(x.rows() + 3 * i + 0, 1, -m_i * x_i3));
            gradLTripletList.push_back(Triplet(x.rows() + 3 * i + 0, 2,  m_i * x_i2));
            gradLTripletList.push_back(Triplet(x.rows() + 3 * i + 1, 2, -m_i * x_i1));
            gradLTripletList.push_back(Triplet(x.rows() + 3 * i + 1, 0,  m_i * x_i3));
            gradLTripletList.push_back(Triplet(x.rows() + 3 * i + 2, 0, -m_i * x_i2));
            gradLTripletList.push_back(Triplet(x.rows() + 3 * i + 2, 1,  m_i * x_i1));
        }
        gradLTripletList.push_back(Triplet(x.rows() + v.rows() + 1, 0, tgrad(0)));
        gradLTripletList.push_back(Triplet(x.rows() + v.rows() + 1, 1, tgrad(1)));
        gradLTripletList.push_back(Triplet(x.rows() + v.rows() + 1, 2, tgrad(2)));
        gradL.setFromTriplets(gradLTripletList.begin(), gradLTripletList.end());

        Eigen::VectorXd gradH(x.rows() + v.rows() + 2);
        AssemblerEigenVector<double> force;
        getForceVector(force, world);
        Eigen::VectorXd gradPotential = -(*force);
        gradH << gradPotential, massMatrix * v, 0, 0;
        
        Eigen::MatrixXd gradC(gradP.rows(), gradP.cols() + gradL.cols() + gradH.cols());
        gradC.leftCols(gradH.cols()) = gradH;
        gradC.middleCols(1, gradP.cols()) = gradP.toDense();
        gradC.rightCols(gradL.cols()) = gradL.toDense();
        
        ///////////////////////////////////////////////

        Eigen::Vector3d p;
        p << 0, 0, 0;
        for (int i = 0; i < v.rows() / 3; i++)
        {
            double m_i = massMatrix.coeff(3 * i, 3 * i);
            Eigen::Vector3d v_i;
            v_i << v(3 * i), v(3 * i + 1), v(3 * i + 2);
            Eigen::Vector3d v_i_next;
            v_i_next << v_next(3 * i), v_next(3 * i + 1), v_next(3 * i + 2);
            Eigen::Vector3d v_i_cur;
            v_i_cur << v_cur(3 * i), v_cur(3 * i + 1), v_cur(3 * i + 2);
            p += m_i * ((v_i - v_i_next) - S * (v_i_cur - v_i_next));
        }
        
        Eigen::Vector3d l;
        l << 0, 0, 0;
        for (int i = 0; i < x.rows() / 3; i++)
        {
            double m_i = massMatrix.coeff(3 * i, 3 * i);
            Eigen::Vector3d v_i;
            v_i << v(3 * i), v(3 * i + 1), v(3 * i + 2);
            Eigen::Vector3d v_i_next;
            v_i_next << v_next(3 * i), v_next(3 * i + 1), v_next(3 * i + 2);
            Eigen::Vector3d v_i_cur;
            v_i_cur << v_cur(3 * i), v_cur(3 * i + 1), v_cur(3 * i + 2);
            Eigen::Vector3d x_i;
            x_i << x(3 * i), x(3 * i + 1), x(3 * i + 2);
            Eigen::Vector3d x_i_next;
            x_i_next << x_next(3 * i), x_next(3 * i + 1), x_next(3 * i + 2);
            Eigen::Vector3d x_i_cur;
            x_i_cur << x_cur(3 * i), x_cur(3 * i + 1), x_cur(3 * i + 2);
            l += m_i * ((x_i.cross(v_i) - x_i_next.cross(v_i_next)) - T * (x_i_cur.cross(v_i_cur) - x_i_next.cross(v_i_next)));
        }
        Eigen::VectorXd c(7);
        c << energy_iter - energy_cur, p(0), p(1), p(2), l(0), l(1), l(2);
        Eigen::SparseLU<Eigen::SparseMatrix<double, Eigen::ColMajor>> tmpSolver;
        tmpSolver.compute(D);
        if(tmpSolver.info()!=Eigen::Success) {
        // decomposition failed
            std::cout << "error" << std::endl;
        }
        Eigen::MatrixXd D_inv_times_gradC(D.rows(), gradC.cols());
        D_inv_times_gradC = tmpSolver.solve(gradC);

        Eigen::MatrixXd lambdaLinearMat(gradC.cols(), D_inv_times_gradC.cols());
        lambdaLinearMat = Eigen::MatrixXd(gradC.transpose() * D_inv_times_gradC);
        Eigen::VectorXd lambda(lambdaLinearMat.rows());
        lambda = lambdaLinearMat.ldlt().solve(c);

        q_iter = q_iter - D_inv_times_gradC * lambda;
        mapStateEigen<0>(world) = q_iter.block(0, 0, x.rows(), 1) - geo_x;
        mapStateEigen<1>(world) = q_iter.block(x.rows(), 0, v.rows(), 1);
        // std::cout << energy_iter << " " << energy_cur << std::endl;
        // std::cout << c.transpose() << std::endl;
        if (c.lpNorm<1>() < 1e-1) break;
    }
}

template<typename DataType, typename MatrixAssembler, typename VectorAssembler>
using TimeStepperEulerImplicitLinearFEPR = TimeStepper<DataType, TimeStepperImplEulerImplicitLinearFEPR<DataType, MatrixAssembler, VectorAssembler> >;





#endif /* TimeStepperEulerImplicitLinearFEPR_h */
