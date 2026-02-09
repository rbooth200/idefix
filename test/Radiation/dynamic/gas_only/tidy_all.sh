for rho in rho*;
do
    cd $rho
    for kappa in kappa*;
    do
        cd $kappa
        rm -rf CMakeCache.txt CMakeFiles/ Makefile build/ cmake_install.cmake  cmake_packages/ generated/ git-state.txt out/ idefix.0.log idefix tmp data.0000.vtk
        cd ..
    done 
    cd ..
done    
