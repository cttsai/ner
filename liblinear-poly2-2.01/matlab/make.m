% This make.m is for MATLAB and OCTAVE under Windows, Mac, and Unix
function make()
try
	% This part is for OCTAVE
	if(strcmp(Type(1).Name, 'Octave') == 1)
		mex -DPOLY2 libsvmread.c
		mex -DPOLY2 libsvmwrite.c
		mex -DPOLY2 train.c linear_model_matlab.c ../linear.cpp ../tron.cpp ../blas/daxpy.c ../blas/ddot.c ../blas/dnrm2.c ../blas/dscal.c
		mex -DPOLY2 predict.c linear_model_matlab.c ../linear.cpp ../tron.cpp ../blas/daxpy.c ../blas/ddot.c ../blas/dnrm2.c ../blas/dscal.c
	% This part is for MATLAB
	% Add -largeArrayDims on 64-bit machines of MATLAB
	else
		mex -DPOLY2 CFLAGS="\$CFLAGS -std=c99" -largeArrayDims libsvmread.c
		mex -DPOLY2 CFLAGS="\$CFLAGS -std=c99" -largeArrayDims libsvmwrite.c
		mex -DPOLY2 CFLAGS="\$CFLAGS -std=c99" -largeArrayDims train.c linear_model_matlab.c ../linear.cpp ../tron.cpp ../blas/daxpy.c ../blas/ddot.c ../blas/dnrm2.c ../blas/dscal.c
		mex -DPOLY2 CFLAGS="\$CFLAGS -std=c99" -largeArrayDims predict.c linear_model_matlab.c ../linear.cpp ../tron.cpp ../blas/daxpy.c ../blas/ddot.c ../blas/dnrm2.c ../blas/dscal.c
	end
catch err
	fprintf('Error: %s failed (line %d)\n', err.stack(1).file, err.stack(1).line);
	disp(err.message);
	fprintf('=> Please check README for detailed instructions.\n');
end
