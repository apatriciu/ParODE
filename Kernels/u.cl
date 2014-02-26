// the group size(0) is equal with the number of inputs in the system
// the group size(1) should be adjusted such that we have more than 64 work items per group
// the global size(0) is a multiple of local size(0) such that all elements are generated
// the global size(1) is equal with local size(1)
// we assume that we can generate all batch values in one kernel call
__kernel void u(__global fType* uVect, fType tStart, fType tStep, int nElements){
	int nStep = (get_global_id(0) / get_local_size(0)) * get_local_size(1) + get_local_id(1);
	fType tStamp = tStart + tStep * nStep;
	if(nStep < nElements)
		uVect[nStep * get_local_size(0) + get_local_id(0)] = UFunc(tStamp, get_local_id(0));
}
