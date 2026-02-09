for rho in rho*;
do
    cd $rho
    for kappa in kappa*;
    do
        cd $kappa
        rm -f CMakeCache.txt cmake_install.cmake git-state.txt Makefile idefix
        cmake ../../../../../../
        make -j8
        ./idefix
    done 
    cd ..
done    
